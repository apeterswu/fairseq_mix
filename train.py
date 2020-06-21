#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def main(args):
    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if not torch.cuda.is_available():
        raise NotImplementedError('Training on CPU is not supported')
    torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    load_dataset_splits(task, ['train', 'valid'])

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    # print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {}'.format(sum(p.numel() for p in model.parameters())))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, dummy_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    # trainer.unset_param()
    # trainer.unset_sen_piece_param()
    # for name, p in model.named_parameters():
    #     print(name, p.requires_grad)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_losses_sen_piece = []
    valid_losses_overall = []
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses, valid_losses_sen_piece, valid_losses_overall = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        # lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses_overall[0])   # train with overall_loss

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0], valid_losses_sen_piece[0], valid_losses_overall[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""

    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr()
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf
    num_batches = len(epoch_itr)
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'loss_sen_piece', 'nll_loss_sen_piece', 'overall_loss', 'overall_nll_loss',
                     'ntokens', 'ntokens_sen_piece', 'nsentences',
                     'sample_size', 'sample_size_sen_piece', 'sample_size_overall']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            if 'loss_sen_piece' in k:
                extra_meters[k].update(v, log_output['sample_size_sen_piece'])
            else:
                extra_meters[k].update(v)
            if 'overall_loss' in k:
                extra_meters[k].update(v, (log_output['sample_size_overall']) / 2.0)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses, valid_losses_sen_piece, valid_overall_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0], valid_losses_sen_piece[0], valid_overall_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'train_loss_sen_piece', 'train_nll_loss_sen_piece',
        'train_overall_loss', 'train_overall_nll_loss',
        'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    stats['loss_sen_piece'] = '{:.3f}'.format(trainer.get_meter('train_loss_sen_piece').avg)
    stats['overall_loss'] = '{:.3f}'.format(trainer.get_meter('train_overall_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    if trainer.get_meter('train_nll_loss_sen_piece').count > 0:
        nll_loss_sen_piece = trainer.get_meter('train_nll_loss_sen_piece').avg
        stats['nll_loss_sen_piece'] = '{:.3f}'.format(nll_loss_sen_piece)
    else:
        nll_loss_sen_piece = trainer.get_meter('train_loss_sen_piece').avg
    if trainer.get_meter('train_overall_nll_loss').count > 0:
        overall_nll_loss = trainer.get_meter('train_overall_nll_loss').avg
        stats['overall_nll_loss'] = '{:.3f}'.format(overall_nll_loss)
    else:
        overall_nll_loss = trainer.get_meter('train_overall_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['ppl_sen_piece'] = get_perplexity(nll_loss_sen_piece)
    stats['overall_ppl'] = get_perplexity(overall_nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = '{:.3f}'.format(trainer.get_meter('loss_scale').avg)
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = round(trainer.get_meter('train_wall').sum)
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    valid_losses_sen_piece = []
    valid_overall_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss', 'valid_loss_sen_piece', 'valid_nll_loss_sen_piece',
                  'valid_overall_loss', 'valid_overall_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'loss_sen_piece', 'nll_loss_sen_piece', 'overall_loss', 'overall_nll_loss',
                         'ntokens', 'ntokens_sen_piece', 'nsentences',
                         'sample_size', 'sample_size_sen_piece', 'sample_size_overall']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats)

        valid_losses.append(stats['valid_loss'])
        valid_losses_sen_piece.append(stats['valid_loss_sen_piece'])
        valid_overall_losses.append(stats['valid_overall_loss'])
    return valid_losses, valid_losses_sen_piece, valid_overall_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    stats['valid_loss_sen_piece'] = trainer.get_meter('valid_loss_sen_piece').avg
    stats['valid_overall_loss'] = trainer.get_meter('valid_overall_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss_sen_piece').count > 0:
        nll_loss_sen_piece = trainer.get_meter('valid_nll_loss_sen_piece').avg
        stats['valid_nll_loss_sen_piece'] = nll_loss_sen_piece
    else:
        nll_loss_sen_piece = trainer.get_meter('valid_loss_sen_piece').avg
    if trainer.get_meter('valid_overall_nll_loss').count > 0:
        overall_nll_loss = trainer.get_meter('valid_overall_nll_loss').avg
        stats['valid_overall_nll_loss'] = overall_nll_loss
    else:
        overall_nll_loss = trainer.get_meter('valid_overall_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    stats['valid_ppl_sen_piece'] = get_perplexity(nll_loss_sen_piece)
    stats['valid_overall_ppl'] = get_perplexity(overall_nll_loss)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best'] = min(save_checkpoint.best, stats['valid_loss'])
    if hasattr(save_checkpoint, 'best_sen_piece'):
        stats['best_sen_piece'] = min(save_checkpoint.best_sen_piece, stats['valid_loss_sen_piece'])
    if hasattr(save_checkpoint, 'best_overall'):
        stats['best_overall'] = min(save_checkpoint.best_overall, stats['valid_overall_loss'])
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss, val_loss_sen_piece, val_overall_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_best_sen_piece.pt'] = (
            val_loss_sen_piece is not None and
            (not hasattr(save_checkpoint, 'best_sen_piece') or val_loss_sen_piece < save_checkpoint.best_sen_piece)
    )
    checkpoint_conds['checkpoint_overall_best.pt'] = (
        val_overall_loss is not None and
        (not hasattr(save_checkpoint, 'best_overall') or val_overall_loss < save_checkpoint.best_overall)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    prev_best_sen_piece = getattr(save_checkpoint, 'best_sen_piece', val_loss_sen_piece)
    prev_best_overall = getattr(save_checkpoint, 'best_overall', val_overall_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    if val_loss_sen_piece is not None:
        save_checkpoint.best_sen_piece = min(val_loss_sen_piece, prev_best_sen_piece)
    if val_overall_loss is not None:
        save_checkpoint.best_overall = min(val_overall_loss, prev_best_overall)
    extra_state = {
        'best': save_checkpoint.best,
        'best_sen_piece': save_checkpoint.best_sen_piece,
        'best_overall': save_checkpoint.best_overall,
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
        'val_loss_sen_piece': val_loss_sen_piece,
        'val_overall_loss': val_overall_loss,
    }

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
            if 'best_sen_piece' in extra_state:
                save_checkpoint.best_sen_piece = extra_state['best_sen_piece']
            if 'best_overall' in extra_state:
                save_checkpoint.best_overall = extra_state['best_overall']
        return True
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


if __name__ == '__main__':
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_port > 0 or args.distributed_init_method is not None:
        from distributed_train import main as distributed_main

        distributed_main(args)
    elif args.distributed_world_size > 1:
        from multiprocessing_train import main as multiprocessing_main

        multiprocessing_main(args)
    else:
        main(args)
