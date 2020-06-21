# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Train a network across multiple GPUs.
"""

from collections import defaultdict, OrderedDict
import contextlib
from itertools import chain

import torch
from fairseq import distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler


class Trainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch):

        if not torch.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion.cuda()
        if args.fp16:
            self._model = model.half().cuda()
        else:
            self._model = model.cuda()

        # initialize meters
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['train_loss_sen_piece'] = AverageMeter()
        self.meters['train_nll_loss_sen_piece'] = AverageMeter()
        self.meters['train_overall_loss'] = AverageMeter()
        self.meters['train_overall_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['valid_loss_sen_piece'] = AverageMeter()
        self.meters['valid_nll_loss_sen_piece'] = AverageMeter()
        self.meters['valid_overall_loss'] = AverageMeter()
        self.meters['valid_overall_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()        # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

        self._lr_scheduler = None
        self._dummy_batch = dummy_batch
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._wrapped_model = None

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        if self.args.fp16:
            if torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, self.model.parameters())

        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            utils.save_state(
                filename, self.args, self.get_model(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = \
            utils.load_model_state(filename, self.get_model())
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state

    def unset_param(self):
        for name, p in self.model.named_parameters():
            if "sen_piece" not in name:
                p.requires_grad = False

    def unset_sen_piece_param(self):
        for name, p in self.model.named_parameters():
            if "sen_piece" in name:
                p.requires_grad = False

    def train_step(self, samples, dummy_batch=False):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.model.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        sample_sizes_sen_piece = []
        sample_sizes_overall = []

        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False
            try:
                # forward
                with torch.autograd.set_detect_anomaly(True):
                    loss, sample_size, loss_sen_piece, sample_size_sen_piece, overall_loss, sample_size_overall, logging_output = self.task.get_loss(
                        self.model, self.criterion, sample,
                    )
                    if ignore_grad:
                        loss *= 0

                    if self.args.distributed_world_size > 1:
                        # only all-reduce gradients in the last backwards pass
                        if i < len(samples) - 1:
                            self.model.need_reduction = False
                        else:
                            self.model.need_reduction = True

                    # backward
                    # self.optimizer.backward(loss)
                    # self.optimizer.backward(loss_sen_piece)
                    self.optimizer.backward(overall_loss)     # train with overall_loss
                    # for name, p in self.model.named_parameters():
                    #     if not p.requires_grad:
                    #         p.grad = torch.cuda.FloatTensor(p.size()).fill_(0.)

                    if not ignore_grad:
                        logging_outputs.append(logging_output)
                        sample_sizes.append(sample_size)
                        sample_sizes_sen_piece.append(sample_size_sen_piece)
                        sample_sizes_overall.append(sample_size_overall)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_sizes, sample_sizes_sen_piece, sample_sizes_overall, ooms = zip(*distributed_utils.all_gather_list(
                [logging_outputs, sample_sizes, sample_sizes_sen_piece, sample_sizes_overall, ooms],
            ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            sample_sizes_sen_piece = list(chain.from_iterable(sample_sizes_sen_piece))
            sample_sizes_overall = list(chain.from_iterable(sample_sizes_overall))
            ooms = sum(ooms)

        if ooms == self.args.distributed_world_size:
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)
        sample_size = self.criterion.__class__.grad_denom(sample_sizes)
        sample_size_sen_piece = self.criterion.__class__.grad_denom(sample_sizes_sen_piece)
        sample_size_overall = self.criterion.__class__.grad_denom(sample_sizes_overall)

        if not all(k in logging_output for k in ['ntokens', 'ntokens_sen_piece', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens, ntokens_sen_piece, and nsentences'
            ).format(self.criterion.__class__.__name__))

        try:
            # normalize grads by sample size
            # self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))
            # self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size_sen_piece))
            self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size_overall / 2.0))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)

            # take an optimization step
            self.optimizer.step()
            self._num_updates += 1

            # update learning rate
            self.lr_scheduler.step_update(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            ntokens_sen_piece = logging_output.get('ntokens_sen_piece', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['oom'].update(ooms)
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            self.meters['train_loss_sen_piece'].update(logging_output.get('loss_sen_piece', 0), sample_size_sen_piece)
            self.meters['train_overall_loss'].update(logging_output.get('overall_loss', 0), sample_size_overall/2.0)
            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
            if 'nll_loss_sen_piece' in logging_output:
                self.meters['train_nll_loss_sen_piece'].update(logging_output.get('nll_loss_sen_piece', 0), ntokens_sen_piece)
            if 'overall_nll_loss' in logging_outputs:
                self.meters['train_overall_nll_loss'].update(logging_output.get('overall_nll_loss', 0), (ntokens + ntokens_sen_piece)/2.0)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, _loss_sen_piece, sample_size_sen_piece, _overall_loss, sample_size_overall, logging_output = self.task.get_loss(
                    self.model, self.criterion, sample,
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size, sample_size_sen_piece, sample_size_overall = {}, 0, 0, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size, sample_size_sen_piece, sample_size_overall = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size, sample_size_sen_piece, sample_size_overall],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
            sample_size_sen_piece = list(sample_size_sen_piece)
            sample_size_overall = list(sample_size_overall)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]
            sample_size_sen_piece = [sample_size_sen_piece]
            sample_size_overall = [sample_size_overall]

        # aggregate logging outputs and sample sizes
        logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_output)
        sample_size = self.criterion.__class__.grad_denom(sample_size)
        sample_size_sen_piece = self.criterion.__class__.grad_denom(sample_size_sen_piece)
        sample_size_overall = self.criterion.__class__.grad_denom(sample_size_overall)

        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        ntokens_sen_piece = logging_output.get('ntokens_sen_piece', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        self.meters['valid_loss_sen_piece'].update(logging_output.get('loss_sen_piece', 0), sample_size_sen_piece)
        self.meters['valid_overall_loss'].update(logging_output.get('overall_loss', 0), sample_size_overall)
        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        if 'nll_loss_sen_piece' in logging_output:
            self.meters['valid_nll_loss_sen_piece'].update(logging_output.get('nll_loss_sen_piece', 0), ntokens_sen_piece)
        if 'overall_nll_loss' in logging_output:
            self.meters['valid_overall_nll_loss'].update(logging_output.get('overall_nll_loss', 0), (ntokens + ntokens_sen_piece)/2.0)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)
