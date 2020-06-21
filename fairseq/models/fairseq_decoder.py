# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn as nn
import torch.nn.functional as F


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, encoder_out, prev_output_tokens=None, prev_output_sen_piece_tokens=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the last decoder layer's output of shape
                  `(batch, tgt_len, vocab)`
                - the last decoder layer's attention weights of shape
                  `(batch, tgt_len, src_len)`
        """
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            assert sample is not None and 'target' in sample
            assert sample is not None and 'target_sen_piece' in sample
            out = self.adaptive_softmax.get_log_prob(net_output[0], sample['target'])
            out_sen_piece = self.adaptive_softmax_get_log_prob(net_output[1], sample['target_sen_piece'])
            return out.exp_() if not log_probs else out, out_sen_piece.exp_() if not log_probs else out_sen_piece

        not_both_none = False
        if net_output[0] is not None or net_output[1] is not None:
            not_both_none = True
        assert not_both_none

        logits, logits_sen_piece = None, None
        if net_output[0] is not None:
            logits = net_output[0].float()
        if net_output[1] is not None:
            logits_sen_piece = net_output[1].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1) if logits is not None else None, \
                   F.log_softmax(logits_sen_piece, dim=-1) if logits_sen_piece is not None else None
        else:
            return F.softmax(logits, dim=-1) if logits is not None else None, \
                   F.softmax(logits_sen_piece, dim=-1) if logits_sen_piece is not None else None

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict
