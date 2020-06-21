#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import os
import torch
import sentencepiece as spm

from fairseq import bleu, data, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer


def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    src_dict_sen_piece = task.source_sen_piece_dictionary
    tgt_dict_sen_piece = task.target_sen_piece_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    prefix_path = os.path.split(args.path.split(':')[0])[0]
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,   # default need_attn=False
        )
        if args.fp16:
            model.half()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    if args.score_reference:
        translator = SequenceScorer(models, task.target_dictionary)
    else:
        translator = SequenceGenerator(
            models, task.target_dictionary, task.target_sen_piece_dictionary, beam_size=args.beam, minlen=args.min_len,
            stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
            len_penalty=args.lenpen, unk_penalty=args.unkpen,
            sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
            diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
        )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True
    sp = spm.SentencePieceProcessor()
    # prefix = '/home/v-lijuwu'
    sp.Load(args.senpiece_model)
    with progress_bar.build_progress_bar(args, itr) as t:
        if args.score_reference:
            translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
        else:
            translations = translator.generate_batched_itr(
                t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
            )

        ftgt = open(prefix_path + '/ref_tgt.txt', 'w', encoding='utf-8')
        fbpe_src = open(prefix_path + '/bpe_src.tok', 'w', encoding='utf-8')
        fbpe_hyp = open(prefix_path + '/bpe_trans.tok', 'w', encoding='utf-8')
        fsp_src = open(prefix_path + '/sp_src.detok', 'w', encoding='utf-8')
        fsp_hyp = open(prefix_path + '/trans.txt', 'w', encoding='utf-8')
        fhyp_tok = open(prefix_path + '/hyp_trans.txt', 'w', encoding='utf-8')
        fhyp_tok_ids = open(prefix_path + '/hyp_ids.txt', 'w', encoding='utf-8')
        wps_meter = TimeMeter()
        id = 0
        for sample_id, src_tokens, target_tokens, src_sen_piece_tokens, target_sen_piece_tokens, hypos, hypos_sen_piece in translations:
            # Process input and ground truth
            has_target = target_tokens is not None
            target_tokens = target_tokens.int().cpu() if has_target else None
            target_sen_piece_tokens = target_sen_piece_tokens.int().cpu() if has_target else None

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                src_str_sen_piece = task.dataset(args.gen_subset).src_sen_piece.get_original_text(sample_id)
                tgt_str_sen_piece = task.dataset(args.gen_subset).tgt_sen_piece.get_original_text(sample_id)
            else:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                fbpe_src.write(src_str + '\n')   # write  bpe_token data
                if has_target:
                    target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                src_str_sen_piece = src_dict_sen_piece.string(src_sen_piece_tokens)  # return list, not string
                src_str_sen_piece_list = src_dict_sen_piece.to_list(src_sen_piece_tokens)
                src_str_out = sp.DecodePieces(src_str_sen_piece_list)
                fsp_src.write(src_str_out + '\n')   # write sp_detok data
                if has_target:
                    tgt_str_sen_piece_list = tgt_dict_sen_piece.to_list(target_sen_piece_tokens, escape_unk=True)
                    tgt_str_sen_piece = tgt_dict_sen_piece.string(target_sen_piece_tokens, escape_unk=True)
                    tgt_str_out = sp.DecodePieces(tgt_str_sen_piece_list)
                    ftgt.write(tgt_str_out + '\n')

            if not args.quiet:
                print('S-{}\t{}'.format(sample_id, src_str))
                if has_target:
                    print('T-{}\t{}'.format(sample_id, target_str))
                print('SS-{}\t{}'.format(sample_id, src_str_sen_piece))
                if has_target:
                    print('TS-{}\t{}'.format(sample_id, tgt_str_sen_piece))

            score1 = 0.
            hypo_str1 = ""
            # Process top predictions
            for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):   # args.nbest=1
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                if not args.quiet:
                    print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                    print('P-{}\t{}'.format(
                        sample_id,
                        ' '.join(map(
                            lambda x: '{:.4f}'.format(x),
                            hypo['positional_scores'].tolist(),
                        ))
                    ))

                    if args.print_alignment:
                        print('A-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(lambda x: str(utils.item(x)), alignment))
                        ))

                # Score only the top hypothesis
                if has_target and i == 0:
                    score1 = hypo['score']
                    hypo_str1 = hypo_str
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tokenizer.Tokenizer.tokenize(
                            target_str, tgt_dict, add_if_not_exist=True)
                    scorer.add(target_tokens, hypo_tokens)
                # write bpe_trans to file
                fbpe_hyp.write(hypo_str+'\n')

            score2 = 0.
            # process sen_piece and save translations to file
            for i, hypo in enumerate(hypos_sen_piece[:min(len(hypos_sen_piece), args.nbest)]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str_sen_piece,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict_sen_piece,
                    remove_bpe=None,
                    to_list=True,
                )
                if not args.quiet:
                    print('HS-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                hypo_str_out = sp.DecodePieces(hypo_str)
                fsp_hyp.write(hypo_str_out+'\n')   # detokenized data

                # Score only the top hypothesis
                if has_target and i == 0:
                    score2 = hypo['score']
            if score1 > score2:
                fhyp_tok.write(hypo_str1+'\n')
                fhyp_tok_ids.write(str(id)+'\n')
            id += 1
            wps_meter.update(src_tokens.size(0))
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += 1
    ftgt.close()
    fbpe_src.close()
    fbpe_hyp.close()
    fsp_src.close()
    fsp_hyp.close()
    fhyp_tok.close()
    fhyp_tok_ids.close()
    print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
