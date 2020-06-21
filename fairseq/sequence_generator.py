# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import copy

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder


class SequenceGenerator(object):
    def __init__(
        self, models, tgt_dict, tgt_dict_sen_piece, beam_size=1, minlen=1, maxlen=None, stop_early=True,
        normalize_scores=True, len_penalty=1, unk_penalty=0, retain_dropout=False,
        sampling=False, sampling_topk=-1, sampling_temperature=1,
        diverse_beam_groups=-1, diverse_beam_strength=0.5,
    ):
        """Generates translations of a given source sentence.
        Args:
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
        """
        self.models = models
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.vocab_size_sen_piece = len(tgt_dict_sen_piece)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        max_decoder_len -= 1  # we define maxlen not including the EOS marker
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout

        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'

        if sampling:
            self.search = search.Sampling(tgt_dict, sampling_topk, sampling_temperature)
        elif diverse_beam_groups > 0:
            self.search = search.DiverseBeamSearch(tgt_dict, diverse_beam_groups, diverse_beam_strength)
        else:
            self.search = search.BeamSearch(tgt_dict)

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_batched_itr(
        self, data_itr, beam_size=None, maxlen_a=0.0, maxlen_b=None,
        cuda=False, timer=None, prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen

        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            # model.forward normally channels prev_output_tokens into the decoder
            # separately, but SequenceGenerator directly calls model.encoder
            encoder_input = {
                k: v for k, v in input.items()
                if k != 'prev_output_tokens' and k != 'prev_output_sen_piece_tokens'
            }
            srclen = encoder_input['src_tokens'].size(1)
            srclen_sen_piece = encoder_input['src_sen_piece_tokens'].size(1)
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos, hypos_sen_piece = self.generate(
                    encoder_input,
                    beam_size=beam_size,
                    maxlen=int(maxlen_a*srclen + maxlen_b),   # 200
                    prefix_tokens=s['target'][:, :prefix_size] if prefix_size > 0 else None,
                    prefix_sen_piece_tokens=s['target_sen_piece'][:, :prefix_size] if prefix_size > 0 else None,
                )
            if timer is not None:
                timer.stop(sum(len(h[0]['tokens']) for h in hypos))
            for i, id in enumerate(s['id'].data):
                # remove padding
                src = utils.strip_pad(input['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                src_sen_piece = utils.strip_pad(input['src_sen_piece_tokens'].data[i, :], self.pad)
                ref_sen_piece = utils.strip_pad(s['target_sen_piece'].data[i, :], self.pad) if s['target_sen_piece'] is not None else None
                yield id, src, ref, src_sen_piece, ref_sen_piece, hypos[i], hypos_sen_piece[i]

    def generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None, prefix_sen_piece_tokens=None):
        """Generate a batch of translations.

        Args:
            encoder_input: dictionary containing the inputs to
                model.encoder.forward
            beam_size: int overriding the beam size. defaults to
                self.beam_size
            max_len: maximum length of the generated sequence
            prefix_tokens: force decoder to begin with these tokens
        """
        with torch.no_grad():
            return self._generate(encoder_input, beam_size, maxlen, prefix_tokens, prefix_sen_piece_tokens)

    def _generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None, prefix_sen_piece_tokens=None):
        """See generate
        batch would decrease during the inference period, since early stop (several batch would stop first)
        Therefore, we need to process decoding twice, recalculate the hidden states of both decoders
        (though waste time...a lot, but I don't know better choices since the selection operation during inference).
        """
        src_tokens = encoder_input['src_tokens']
        src_sen_piece_tokens = encoder_input['src_sen_piece_tokens']
        bsz, srclen = src_tokens.size()
        bsz, srclen_sen_piece = src_sen_piece_tokens.size()
        maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)
        beam_size = min(beam_size, self.vocab_size_sen_piece - 1)

        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            if isinstance(model.decoder, FairseqIncrementalDecoder):
                incremental_states[model] = {}
            else:
                incremental_states[model] = None

            # compute the encoder output for each beam
            encoder_out = model.encoder(**encoder_input)
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device)
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, new_order)
            encoder_outs.append(encoder_out)
        saved_encoder_outs = []
        # encoder_outs[0]['encoder_padding_mask_sen_piece'].view(2, 2)
        new_out = {}
        for key, value in encoder_outs[0].items():
            new_out[key] = value.clone() if value is not None else None
        saved_encoder_outs.append(new_out)
        # print(encoder_outs[0]['encoder_out'].size())

        def _oneside_generate(batch_size, encoder_outs, incremental_states, prefix_tokens=None, is_sen_piece=False):
            bsz = batch_size
            # initialize buffers
            scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
            scores_buf = scores.clone()
            tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
            tokens_buf = tokens.clone()
            tokens[:, 0] = self.eos
            attn, attn_buf = None, None
            nonpad_idxs = None

            # list of completed sentences
            finalized = [[] for i in range(bsz)]
            finished = [False for i in range(bsz)]
            worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
            num_remaining_sent = bsz

            # number of candidate hypos per step
            cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

            # offset arrays for converting between different indexing schemes
            bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)  # (0, beam, 2*beam, ..)[bsz, 1]
            cand_offsets = torch.arange(0, cand_size).type_as(tokens)

            # helper function for allocating buffers on the fly
            buffers = {}

            def buffer(name, type_of=tokens):  # noqa
                if name not in buffers:
                    buffers[name] = type_of.new()
                return buffers[name]

            def is_finished(sent, step, unfinalized_scores=None):
                """
                Check whether we've finished generation for a given sentence, by
                comparing the worst score among finalized hypotheses to the best
                possible score among unfinalized hypotheses.
                """
                assert len(finalized[sent]) <= beam_size
                if len(finalized[sent]) == beam_size:
                    if self.stop_early or step == maxlen or unfinalized_scores is None:
                        return True
                    # stop if the best unfinalized score is worse than the worst
                    # finalized one
                    best_unfinalized_score = unfinalized_scores[sent].max()
                    if self.normalize_scores:
                        best_unfinalized_score /= maxlen ** self.len_penalty
                    if worst_finalized[sent]['score'] >= best_unfinalized_score:
                        return True
                return False

            def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
                """
                Finalize the given hypotheses at this step, while keeping the total
                number of finalized hypotheses per sentence <= beam_size.
                Note: the input must be in the desired finalization order, so that
                hypotheses that appear earlier in the input are preferred to those
                that appear later.
                Args:
                    step: current time step
                    bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                        indicating which hypotheses to finalize
                    eos_scores: A vector of the same size as bbsz_idx containing
                        scores for each hypothesis
                    unfinalized_scores: A vector containing scores for all
                        unfinalized hypotheses
                """
                assert bbsz_idx.numel() == eos_scores.numel()

                # clone relevant token and attention tensors
                tokens_clone = tokens.index_select(0, bbsz_idx)
                tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
                tokens_clone[:, step] = self.eos
                attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

                # compute scores per token position
                pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]  # overall scores
                pos_scores[:, step] = eos_scores
                # convert from cumulative to per-position scores
                pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

                # normalize sentence-level scores
                if self.normalize_scores:
                    eos_scores /= (step + 1) ** self.len_penalty

                cum_unfin = []
                prev = 0
                for f in finished:  # bsz
                    if f:
                        prev += 1
                    else:
                        cum_unfin.append(prev)

                sents_seen = set()
                for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                    unfin_idx = idx // beam_size   # unfinished batch id
                    sent = unfin_idx + cum_unfin[unfin_idx]   # corresponds to which source sentence

                    sents_seen.add((sent, unfin_idx))

                    def get_hypo():

                        if attn_clone is not None:
                            # remove padding tokens from attn scores
                            hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                            _, alignment = hypo_attn.max(dim=0)
                        else:
                            hypo_attn = None
                            alignment = None

                        return {
                            'tokens': tokens_clone[i],
                            'score': score,
                            'attention': hypo_attn,  # src_len x tgt_len
                            'alignment': alignment,
                            'positional_scores': pos_scores[i],
                        }

                    if len(finalized[sent]) < beam_size:
                        finalized[sent].append(get_hypo())
                    elif not self.stop_early and score > worst_finalized[sent]['score']:
                        # replace worst hypo for this sentence with new/better one
                        worst_idx = worst_finalized[sent]['idx']
                        if worst_idx is not None:
                            finalized[sent][worst_idx] = get_hypo()

                        # find new worst finalized hypo for this sentence
                        idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                        worst_finalized[sent] = {
                            'score': s['score'],
                            'idx': idx,
                        }

                newly_finished = []
                for sent, unfin_idx in sents_seen:
                    # check termination conditions for this sentence
                    if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                        finished[sent] = True
                        newly_finished.append(unfin_idx)
                return newly_finished

            reorder_state = None
            batch_idxs = None
            # seperate decoding
            for step in range(maxlen + 1):  # one extra step for EOS marker
                # reorder decoder internal states based on the prev choice of beams
                if reorder_state is not None:
                    if batch_idxs is not None:
                        # update beam indices to take into account removed sentences
                        corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                        reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    for i, model in enumerate(self.models):
                        if isinstance(model.decoder, FairseqIncrementalDecoder):
                            model.decoder.reorder_incremental_state(incremental_states[model], reorder_state)
                            model.decoder.reorder_incremental_state(incremental_states[model], reorder_state,
                                                                    sen_piece=True)
                        encoder_outs[i] = model.encoder.reorder_encoder_out(encoder_outs[i], reorder_state)

                if is_sen_piece:
                    lprobs, avg_attn_scores, lprobs_sen_piece, avg_attn_scores_sen_piece = self._decode(
                        None, tokens[:, :step + 1], encoder_outs, incremental_states)
                    # lporbs, avg_attn_scores should be None; Then make the sen_piece results to be lprobs;
                    lprobs, avg_attn_scores = lprobs_sen_piece, avg_attn_scores_sen_piece
                else:
                    lprobs, avg_attn_scores, lprobs_sen_piece, avg_attn_scores_sen_piece = self._decode(
                        tokens[:, :step + 1], None, encoder_outs, incremental_states)

                lprobs[:, self.pad] = -math.inf  # never select pad    # [batch*beam, vocab]
                lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

                # Record attention scores
                if avg_attn_scores is not None:
                    if attn is None:
                        attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
                        attn_buf = attn.clone()
                        nonpad_idxs = src_tokens.ne(self.pad)
                    attn[:, :, step + 1].copy_(avg_attn_scores)

                scores = scores.type_as(lprobs)
                scores_buf = scores_buf.type_as(lprobs)
                eos_bbsz_idx = buffer('eos_bbsz_idx')
                eos_scores = buffer('eos_scores', type_of=scores)
                if step < maxlen:
                    if prefix_tokens is not None and step < prefix_tokens.size(1):   # should be empty
                        probs_slice = lprobs.view(bsz, -1, lprobs.size(-1))[:, 0, :]
                        cand_scores = torch.gather(
                            probs_slice, dim=1,
                            index=prefix_tokens[:, step].view(-1, 1).data
                        ).expand(-1, cand_size)
                        cand_indices = prefix_tokens[:, step].view(-1, 1).expand(bsz, cand_size).data
                        cand_beams = torch.zeros_like(cand_indices)
                    else:
                        cand_scores, cand_indices, cand_beams = self.search.step(
                            step,
                            lprobs.view(bsz, -1, self.vocab_size if not is_sen_piece else self.vocab_size_sen_piece),
                            scores.view(bsz, beam_size, -1)[:, :, :step],
                        )
                        # cand_scores: [batch, 2*beam]
                        # cand_indices: [batch, 2*beam]
                        # cand_beams: [batch, 2*beam]
                        # scores: [batch*beam, len]
                        # tokens: [batch*beam, len]
                else:   # at the end of maxlen
                    # make probs contain cumulative scores for each hypothesis
                    lprobs.add_(scores[:, step - 1].unsqueeze(-1))    # [batch*beam, vocab], accumulated

                    # finalize all active hypotheses once we hit maxlen
                    # pick the hypothesis with the highest prob of EOS right now
                    torch.sort(
                        lprobs[:, self.eos],
                        descending=True,
                        out=(eos_scores, eos_bbsz_idx),
                    )
                    num_remaining_sent -= len(finalize_hypos(
                        step, eos_bbsz_idx, eos_scores))
                    assert num_remaining_sent == 0
                    break

                # cand_bbsz_idx contains beam indices for the top candidate
                # hypotheses, with a range of values: [0, bsz*beam_size),
                # and dimensions: [bsz, cand_size]
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)   # [bsz, 2*beam]

                # finalize hypotheses that end in eos
                eos_mask = cand_indices.eq(self.eos)   # [bsz, 2*beam]

                finalized_sents = set()
                if step >= self.minlen:
                    # only consider eos when it's among the top beam_size indices
                    torch.masked_select(
                        cand_bbsz_idx[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_bbsz_idx,    # selected eos ended beam ids, one-dimension
                    )
                    if eos_bbsz_idx.numel() > 0:
                        torch.masked_select(
                            cand_scores[:, :beam_size],
                            mask=eos_mask[:, :beam_size],
                            out=eos_scores,  # selected scores with eos ended
                        )
                        finalized_sents = finalize_hypos(
                            step, eos_bbsz_idx, eos_scores, cand_scores)   # finished translation sentences with eos
                        num_remaining_sent -= len(finalized_sents)

                assert num_remaining_sent >= 0
                if num_remaining_sent == 0:
                    break
                assert step < maxlen

                if len(finalized_sents) > 0:
                    new_bsz = bsz - len(finalized_sents)

                    # construct batch_idxs which holds indices of batches to keep for the next pass
                    batch_mask = cand_indices.new_ones(bsz)
                    batch_mask[cand_indices.new(finalized_sents)] = 0
                    batch_idxs = batch_mask.nonzero().squeeze(-1)    # non-zero batch ids

                    eos_mask = eos_mask[batch_idxs]
                    cand_beams = cand_beams[batch_idxs]
                    bbsz_offsets.resize_(new_bsz, 1)
                    cand_bbsz_idx = cand_beams.add(bbsz_offsets)

                    cand_scores = cand_scores[batch_idxs]
                    cand_indices = cand_indices[batch_idxs]
                    if prefix_tokens is not None:
                        prefix_tokens = prefix_tokens[batch_idxs]

                    scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                    scores_buf.resize_as_(scores)
                    tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                    tokens_buf.resize_as_(tokens)
                    if attn is not None:
                        attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                        attn_buf.resize_as_(attn)
                    bsz = new_bsz
                else:
                    batch_idxs = None

                # set active_mask so that values > cand_size indicate eos hypos
                # and values < cand_size indicate candidate active hypos.
                # After, the min values per row are the top candidate active hypos
                active_mask = buffer('active_mask')
                torch.add(
                    eos_mask.type_as(cand_offsets) * cand_size,
                    cand_offsets[:eos_mask.size(1)],
                    out=active_mask,
                )

                # get the top beam_size active hypotheses, which are just the hypos
                # with the smallest values in active_mask
                active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
                torch.topk(
                    active_mask, k=beam_size, dim=1, largest=False,
                    out=(_ignore, active_hypos)   # ids of the ranked active_mask in dimension 1
                )

                active_bbsz_idx = buffer('active_bbsz_idx')
                torch.gather(
                    cand_bbsz_idx, dim=1, index=active_hypos,
                    out=active_bbsz_idx,   # [batch, beam_size]
                )
                active_scores = torch.gather(
                    cand_scores, dim=1, index=active_hypos,
                    out=scores[:, step].view(bsz, beam_size),   # update the saved overall scores vector, [batch*beam, len]
                )

                active_bbsz_idx = active_bbsz_idx.view(-1)   # [batch*beam]
                active_scores = active_scores.view(-1)

                # copy tokens and scores for active hypotheses
                torch.index_select(
                    tokens[:, :step + 1], dim=0, index=active_bbsz_idx,  # [batch*beam, len]
                    out=tokens_buf[:, :step + 1],
                )  # select [batch * beam]
                torch.gather(
                    cand_indices, dim=1, index=active_hypos,
                    out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
                )  # select [beam], ids
                if step > 0:
                    torch.index_select(
                        scores[:, :step], dim=0, index=active_bbsz_idx,
                        out=scores_buf[:, :step],
                    )
                torch.gather(
                    cand_scores, dim=1, index=active_hypos,
                    out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
                )

                # copy attention for active hypotheses
                if attn is not None:
                    torch.index_select(
                        attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                        out=attn_buf[:, :, :step + 2],
                    )

                # swap buffers
                tokens, tokens_buf = tokens_buf, tokens
                scores, scores_buf = scores_buf, scores
                if attn is not None:
                    attn, attn_buf = attn_buf, attn

                # reorder incremental state in decoder
                reorder_state = active_bbsz_idx

            # sort by score descending
            for sent in range(len(finalized)):
                finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

            return finalized

        # two separate decoding, however, the dec1 and dec2 will calculate together. Therefore, it costs two times.
        # Time wasted solution. However, I don't know better solutions...
        finalized_sents = _oneside_generate(bsz, encoder_outs, incremental_states, prefix_tokens)

        incremental_states[self.models[0]] = {}
        # print(saved_encoder_outs[0]['encoder_out'].size())
        # saved_encoder_outs[0]['encoder_out'].view(2, 2)
        finalized_sents_sen_piece = _oneside_generate(bsz, saved_encoder_outs, incremental_states, prefix_sen_piece_tokens, is_sen_piece=True)
        return finalized_sents, finalized_sents_sen_piece

    def _decode(self, tokens, tokens_sen_piece, encoder_outs, incremental_states):
        if len(self.models) == 1:
            return self._decode_one(tokens, tokens_sen_piece, self.models[0], encoder_outs[0],
                                    incremental_states, log_probs=True)

        # for multi models
        avg_probs = None
        avg_attn = None
        avg_probs_sen_piece = None
        avg_attn_sen_piece = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn, probs_sen_piece, attn_sen_piece = self._decode_one(tokens, tokens_sen_piece, model,
                                                                            encoder_out, incremental_states, log_probs=False)
            if avg_probs is None:
                avg_probs = probs
                avg_probs_sen_piece = probs_sen_piece
            else:
                avg_probs.add_(probs)
                avg_probs_sen_piece.add_(probs_sen_piece)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                    avg_attn_sen_piece = attn_sen_piece
                else:
                    avg_attn.add_(attn)
                    avg_attn_sen_piece.add_(attn_sen_piece)
        avg_probs.div_(len(self.models))
        avg_probs_sen_piece.div_(len(self.models))
        avg_probs.log_()
        avg_probs_sen_piece.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
            avg_attn_sen_piece.div_(len(self.models))
        return avg_probs, avg_attn, avg_probs_sen_piece, avg_attn_sen_piece

    def _decode_one(self, tokens, tokens_sen_piece, model, encoder_out, incremental_states, log_probs):
        with torch.no_grad():
            if incremental_states[model] is not None:
                decoder_out = list(model.decoder(encoder_out, tokens, tokens_sen_piece,
                                                 incremental_state=incremental_states[model]))
            else:
                decoder_out = list(model.decoder(encoder_out, tokens, tokens_sen_piece))
            if tokens is not None:
                decoder_out[0] = decoder_out[0][:, -1, :]  # get the last step
            if tokens_sen_piece is not None:
                decoder_out[1] = decoder_out[1][:, -1, :]
            attn = decoder_out[2]
            attn_sen_piece = None
            if type(attn) is dict:
                attn = attn['attn'] if tokens is not None else None
                if tokens_sen_piece is not None:
                    attn_sen_piece = attn['attn_sen_piece'] if attn is not None else None
            if attn is not None:
                if type(attn) is dict:
                    attn = attn['attn']
                    attn_sen_piece = attn['attn_sen_piece']
                attn = attn[:, -1, :]
                attn_sen_piece = attn_sen_piece[:, -1, :]
        probs, probs_sen_piece = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs, attn, probs_sen_piece, attn_sen_piece
