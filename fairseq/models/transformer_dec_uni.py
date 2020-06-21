# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import uniform

from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)


@register_model('transformer_dec_uni')
class TransformerDecUniModel(FairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--all-share', action='store_true', default=False,
                            help='whether to share all the parameters for two embeddings')
        parser.add_argument('--ffn-share', action='store_true', default=False,
                            help='whether to share ffn parameter, suppose that all_share contains this setting,'
                                 'but when all_share is false, we can set ffn_share to be True')
        parser.add_argument('--enc-drop-path-ratio', type=float, default=0.,
                            help='the drop ratio for cross-attention in encoder')
        parser.add_argument('--dec-drop-path-ratio', type=float, default=0.,
                            help='the drop ratio for enc2-attention in decoder')
        parser.add_argument('--token-embed', action='store_true', default=False,
                            help='whether add token embedding for BPE and SP embeddings.')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        src_dict_sen_piece, tgt_dict_sen_piece = task.source_sen_piece_dictionary, task.target_sen_piece_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise RuntimeError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            encoder_embed_tokens_sen_piece = build_embedding(
                src_dict_sen_piece, args.encoder_embed_dim, ''   # no encoder_embed_path
            )
            decoder_embed_tokens_sen_piece = encoder_embed_tokens_sen_piece
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
            encoder_embed_tokens_sen_piece = build_embedding(
                src_dict_sen_piece, args.encoder_embed_dim, ''  # no encoder_embed_path
            )
            decoder_embed_tokens_sen_piece = build_embedding(
                tgt_dict_sen_piece, args.decoder_embed_dim, ''   # no decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, src_dict_sen_piece,
                                     encoder_embed_tokens, encoder_embed_tokens_sen_piece)
        decoder = TransformerDecoder(args, tgt_dict, tgt_dict_sen_piece,
                                     decoder_embed_tokens, decoder_embed_tokens_sen_piece)
        return TransformerDecUniModel(encoder, decoder)


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``True``
    """

    def __init__(self, args, dictionary, dictionary_sen_piece, embed_tokens, embed_tokens_sen_piece, left_pad=True):
        super().__init__(dictionary)
        self.dictionary_sen_piece = dictionary_sen_piece
        self.all_share = args.all_share
        self.dropout = args.dropout
        self.add_token_embed = args.token_embed

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.padding_idx_sen_piece = embed_tokens_sen_piece.padding_idx
        self.padding_idx_sen_piece = self.padding_idx
        self.max_source_positions = args.max_source_positions

        # Lijun: token embedding
        if self.add_token_embed:
            self.BPE_embed = nn.Parameter(torch.Tensor(1, embed_dim))
            self.SP_embed = nn.Parameter(torch.Tensor(1, embed_dim))
            nn.init.normal_(self.BPE_embed, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.SP_embed, mean=0, std=embed_dim ** -0.5)

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.embed_tokens_sen_piece = embed_tokens_sen_piece
        self.embed_positions_sen_piece = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx_sen_piece,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            if not self.all_share:
                self.layer_norm_sen_piece = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_lengths,
                src_sen_piece_tokens=None, src_sen_piece_lengths=None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        x_sen_piece = self.embed_scale * self.embed_tokens_sen_piece(src_sen_piece_tokens)
        # print('src_sen_piece_tokens in encoder, ', src_sen_piece_tokens)

        if self.embed_positions is not None:
            x = x + self.embed_positions(src_tokens)
        if self.add_token_embed:
            x = x + self.BPE_embed
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.embed_positions_sen_piece is not None:
            x_sen_piece = x_sen_piece + self.embed_positions(src_sen_piece_tokens)
        if self.add_token_embed:
            x_sen_piece = x_sen_piece + self.SP_embed
        x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_sen_piece = x_sen_piece.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_padding_mask_sen_piece = src_sen_piece_tokens.eq(self.padding_idx)
        if not encoder_padding_mask_sen_piece.any():
            encoder_padding_mask_sen_piece = None

        # encoder layers
        for layer in self.layers:
            x, x_sen_piece = layer(x, encoder_padding_mask,
                                   x_sen_piece, encoder_padding_mask_sen_piece)

        if self.normalize:
            x = self.layer_norm(x)
            x_sen_piece = self.layer_norm(x_sen_piece) if self.all_share \
                else self.layer_norm_sen_piece(x_sen_piece)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_out_sen_piece': x_sen_piece,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_padding_mask_sen_piece': encoder_padding_mask_sen_piece,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_out_sen_piece'] is not None:
            encoder_out['encoder_out_sen_piece'] = \
                encoder_out['encoder_out_sen_piece'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_padding_mask_sen_piece'] is not None:
            encoder_out['encoder_padding_mask_sen_piece'] = \
                encoder_out['encoder_padding_mask_sen_piece'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'encoder.embed_positions.weights' in state_dict:
                del state_dict['encoder.embed_positions.weights']
            state_dict['encoder.embed_positions._float_tensor'] = torch.FloatTensor(1)
        if utils.item(state_dict.get('encoder.version', torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['encoder.version'] = torch.Tensor([1])
        return state_dict


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
        left_pad (bool, optional): whether the input is left-padded. Default:
            ``False``
    """

    def __init__(self, args, dictionary, dictionary_sen_piece, embed_tokens, embed_tokens_sen_piece,
                 no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dictionary_sen_piece = dictionary_sen_piece
        self.all_share = args.all_share
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.add_token_embed = args.token_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        # Lijun: token embedding
        if self.add_token_embed:
            self.BPE_embed = nn.Parameter(torch.Tensor(1, embed_dim))
            self.SP_embed = nn.Parameter(torch.Tensor(1, embed_dim))
            nn.init.normal_(self.BPE_embed, mean=0, std=embed_dim ** -0.5)
            nn.init.normal_(self.SP_embed, mean=0, std=embed_dim ** -0.5)

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        padding_idx_sen_piece = embed_tokens_sen_piece.padding_idx
        self.embed_tokens_sen_piece = embed_tokens_sen_piece

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False,
                                     uniform=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.embed_positions_sen_piece = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx_sen_piece,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim,
                                      bias=False, uniform=False) if embed_dim != output_embed_dim else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary), output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
            )
            self.adaptive_softmax_sen_piece = AdaptiveSoftmax(
                len(dictionary_sen_piece), output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
            self.embed_out_sen_piece = nn.Parameter(torch.Tensor(len(dictionary_sen_piece), output_embed_dim))
            nn.init.normal_(self.embed_out_sen_piece, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)
            if not self.all_share:
                self.layer_norm_sen_piece = LayerNorm(embed_dim)

    def forward(self, encoder_out=None, prev_output_tokens=None,
                prev_output_sen_piece_tokens=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        """!!!can not both to be None for decoder inputs!"""
        not_both_none = False
        if prev_output_tokens is not None or prev_output_sen_piece_tokens is not None:
            not_both_none = True
        assert not_both_none

        x, x_sen_piece = None, None
        attn, attn_sen_piece = None, None
        inner_states, inner_states_sen_piece = [], []
        if prev_output_tokens is not None:
            # embed positions
            positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None

            if incremental_state is not None:
                prev_output_tokens = prev_output_tokens[:, -1:]
                if positions is not None:
                    positions = positions[:, -1:]

            # embed tokens and positions
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)

            if self.project_in_dim is not None:
                x = self.project_in_dim(x)

            if positions is not None:
                x += positions
            if self.add_token_embed:
                x = x + self.BPE_embed
            x = F.dropout(x, p=self.dropout, training=self.training)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            # attn = None
            inner_states = [x]

        if prev_output_sen_piece_tokens is not None:
            positions_sen_piece = self.embed_positions_sen_piece(
                prev_output_sen_piece_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions_sen_piece is not None else None
            if incremental_state is not None:
                prev_output_sen_piece_tokens = prev_output_sen_piece_tokens[:, -1:]
                if positions_sen_piece is not None:
                    positions_sen_piece = positions_sen_piece[:, -1:]
            # print('prev_output_sen_piece_tokens in decoder, ', prev_output_sen_piece_tokens)

            # embed tokens and positions
            x_sen_piece = self.embed_scale * self.embed_tokens_sen_piece(prev_output_sen_piece_tokens)

            if positions_sen_piece is not None:
                x_sen_piece += positions_sen_piece
            if self.add_token_embed:
                x_sen_piece = x_sen_piece + self.SP_embed
            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)

            x_sen_piece = x_sen_piece.transpose(0, 1)
            # attn_sen_piece = None
            inner_states_sen_piece = [x_sen_piece]

        # decoder layers
        for layer in self.layers:
            x, attn, x_sen_piece, attn_sen_piece = layer(
                x,
                x_sen_piece,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                encoder_out['encoder_out_sen_piece'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask_sen_piece'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
                self_attn_mask_sen_piece=self.buffered_future_mask(x_sen_piece, sen_piece=True) if incremental_state is None else None,
            )
            inner_states.append(x)
            inner_states_sen_piece.append(x_sen_piece)

        if prev_output_tokens is not None:
            if self.normalize:
                x = self.layer_norm(x)
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

            if self.project_out_dim is not None:
                x = self.project_out_dim(x)
            if self.adaptive_softmax is None:
                # project back to size of vocabulary
                x = F.linear(x, self.embed_tokens.weight if self.share_input_output_embed else self.embed_out)
                # if self.share_input_output_embed:
                #     x = F.linear(x, self.embed_tokens.weight)
                # else:
                #     x = F.linear(x, self.embed_out)

        if prev_output_sen_piece_tokens is not None:
            if self.normalize:
                x_sen_piece = self.layer_norm(x_sen_piece) if self.all_share \
                    else self.layer_norm_sen_piece(x_sen_piece)
            # T x B x C -> B x T x C
            x_sen_piece = x_sen_piece.transpose(0, 1)
            if self.adaptive_softmax is None:
                # project back to size of vocabulary
                x_sen_piece = F.linear(x_sen_piece, self.embed_tokens_sen_piece.weight if self.share_input_output_embed
                                                                                       else self.embed_out_sen_piece)

        return x, x_sen_piece, {'attn': attn, 'attn_sen_piece': attn_sen_piece,
                                'inner_states': inner_states, 'inner_states_sen_piece': inner_states_sen_piece}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        min_ = min(self.embed_positions.max_positions(), self.embed_positions_sen_piece.max_positions())
        return min(self.max_target_positions, min_)

    def buffered_future_mask(self, tensor, sen_piece=False):
        dim = tensor.size(0)
        if sen_piece:
            if not hasattr(self, '_future_mask_sen_piece') or self._future_mask_sen_piece is None or self._future_mask_sen_piece.device != tensor.device:
                self._future_mask_sen_piece = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask_sen_piece.size(0) < dim:
                self._future_mask_sen_piece = torch.triu(utils.fill_with_neg_inf(self._future_mask_sen_piece.resize_(dim, dim)), 1)
            return self._future_mask_sen_piece[:dim, :dim]
        else:
            if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
            return self._future_mask[:dim, :dim]

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            if 'decoder.embed_positions.weights' in state_dict:
                del state_dict['decoder.embed_positions.weights']
            state_dict['decoder.embed_positions._float_tensor'] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = 'decoder.layers.{}.layer_norms.{}.{}'.format(i, old, m)
                    if k in state_dict:
                        state_dict['decoder.layers.{}.{}.{}'.format(i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('decoder.version', torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['decoder.version'] = torch.Tensor([1])

        return state_dict


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.all_share = args.all_share
        self.ffn_share = args.ffn_share
        if self.all_share:
            self.ffn_share = True
        self.enc_drop_path_ratio = args.enc_drop_path_ratio
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.ffn_layer_norm = LayerNorm(self.embed_dim)
        if not self.all_share:
            self.self_attn_sen_piece = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.self_attn_sen_piece_layer_norm = LayerNorm(self.embed_dim)
        if not self.ffn_share:
            self.fc1_sen_piece = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
            self.fc2_sen_piece = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
            self.ffn_sen_piece_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before

    def forward(self, x, encoder_padding_mask, x_sen_piece, encoder_padding_mask_sen_piece):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        residual_sen_piece = x_sen_piece
        if not self.all_share:
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
            x_sen_piece = self.maybe_layer_norm(self.self_attn_sen_piece_layer_norm, x_sen_piece, before=True)

            # concat BPE and SP representations
            x_and_x_sen_piece = torch.cat([x, x_sen_piece * self.get_drop_path_ratio()], dim=0)
            x_and_x_sen_piece_padding_mask = torch.cat([encoder_padding_mask, encoder_padding_mask_sen_piece], dim=-1) \
                if encoder_padding_mask is not None and encoder_padding_mask_sen_piece is not None else None
            x_sen_piece_and_x = torch.cat([x_sen_piece, x * self.get_drop_path_ratio()], dim=0)
            x_sen_piece_and_x_padding_mask = torch.cat([encoder_padding_mask_sen_piece, encoder_padding_mask], dim=-1) \
                if encoder_padding_mask_sen_piece is not None and encoder_padding_mask is not None else None

            x, _ = self.self_attn(query=x, key=x_and_x_sen_piece, value=x_and_x_sen_piece,
                                  key_padding_mask=x_and_x_sen_piece_padding_mask)
            x_sen_piece, _ = self.self_attn_sen_piece(query=x_sen_piece, key=x_sen_piece_and_x, value=x_sen_piece_and_x,
                                                      key_padding_mask=x_sen_piece_and_x_padding_mask)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
            x_sen_piece = residual_sen_piece + x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.self_attn_sen_piece_layer_norm, x_sen_piece, after=True)

        else:
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
            x_sen_piece = self.maybe_layer_norm(self.self_attn_layer_norm, x_sen_piece, before=True)

            # concat BPE and SP representations
            x_and_x_sen_piece = torch.cat([x, x_sen_piece * self.get_drop_path_ratio()], dim=0)
            x_and_x_sen_piece_padding_mask = torch.cat([encoder_padding_mask, encoder_padding_mask_sen_piece], dim=-1) \
                if encoder_padding_mask is not None and encoder_padding_mask_sen_piece is not None else None
            x_sen_piece_and_x = torch.cat([x_sen_piece, x * self.get_drop_path_ratio()], dim=0)
            x_sen_piece_and_x_padding_mask = torch.cat([encoder_padding_mask_sen_piece, encoder_padding_mask], dim=-1) \
                if encoder_padding_mask_sen_piece is not None and encoder_padding_mask is not None else None

            x, _ = self.self_attn(query=x, key=x_and_x_sen_piece, value=x_and_x_sen_piece,
                                  key_padding_mask=x_and_x_sen_piece_padding_mask)
            x_sen_piece, _ = self.self_attn(query=x_sen_piece, key=x_sen_piece_and_x, value=x_sen_piece_and_x,
                                            key_padding_mask=x_sen_piece_and_x_padding_mask)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)
            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
            x_sen_piece = residual_sen_piece + x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.self_attn_layer_norm, x_sen_piece, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        residual_sen_piece = x_sen_piece
        x_sen_piece = self.maybe_layer_norm(self.ffn_layer_norm if self.ffn_share else self.ffn_sen_piece_layer_norm,
                                            x_sen_piece, before=True)
        x_sen_piece = F.relu(self.fc1(x_sen_piece) if self.ffn_share else self.fc1_sen_piece(x_sen_piece))
        x_sen_piece = F.dropout(x_sen_piece, p=self.relu_dropout, training=self.training)
        x_sen_piece = self.fc2(x_sen_piece) if self.ffn_share else self.fc2_sen_piece(x_sen_piece)
        x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
        x_sen_piece = residual_sen_piece + x_sen_piece
        x_sen_piece = self.maybe_layer_norm(self.ffn_layer_norm if self.ffn_share else self.ffn_sen_piece_layer_norm,
                                            x_sen_piece, after=True)
        return x, x_sen_piece

    def get_drop_path_ratio(self):
        randp = uniform(0, 1)
        if self.enc_drop_path_ratio != 0.:
            if self.training:
                if randp < self.enc_drop_path_ratio:
                    return 0.
                else:
                    return 1.
            else:
                return 1. - self.enc_drop_path_ratio
        else:
            return 1.

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs.
            Default: ``False``
    """

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.all_share = args.all_share
        self.ffn_share = args.ffn_share
        if self.all_share:
            self.ffn_share = True
        self.dec_drop_path_ratio = args.dec_drop_path_ratio
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        if not self.all_share:
            self.self_attn_sen_piece = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        if not self.all_share:
            self.self_attn_layer_norm_sen_piece = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None

            if not self.all_share:
                self.encoder_attn_sen_piece = None
                self.encoder_attn_layer_norm_sen_piece = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

            if not self.all_share:
                self.encoder_attn_sen_piece = MultiheadAttention(
                    self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                )
                if not self.all_share:
                    self.encoder_attn_layer_norm_sen_piece = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        if not self.ffn_share:
            self.fc1_sen_piece = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2_sen_piece = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
            self.final_layer_norm_sen_piece = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, x_sen_piece, encoder_out, encoder_padding_mask,
                encoder_out_sen_piece, encoder_padding_mask_sen_piece,
                incremental_state, prev_self_attn_state=None, prev_attn_state=None,
                self_attn_mask=None, self_attn_mask_sen_piece=None, self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        """!!!can not be None for both x and x_sen_piece"""
        not_both_none = False
        if x is not None or x_sen_piece is not None:
            not_both_none = True
        assert not_both_none

        if x is not None:
            residual = x
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                incremental_state=incremental_state,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            # decoder self-attn, cross-attn
            # x2, _ = self.self_attn(
            #         query=x_sen_piece, key=x, value=x,
            #         key_padding_mask=self_attn_padding_mask,  # not useful
            #         incremental_state=incremental_state,
            #         static_kv=True,  # reuse the key, instead of concatenate again
            #         need_weights=False,
            #         # attn_mask=self_attn_mask_sen_piece,  # not self-attn, it is cross-attn
            # )
            # x = 0.5 * x1 + 0.5 * x1_sen_piece
            # x_sen_piece = 0.5 * x2 + 0.5 * x2_sen_piece
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        if x_sen_piece is not None:
            residual_sen_piece = x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.self_attn_layer_norm if self.all_share
                                                else self.self_attn_layer_norm_sen_piece, x_sen_piece, before=True)
            if not self.all_share:
                x_sen_piece, _ = self.self_attn_sen_piece(
                    query=x_sen_piece, key=x_sen_piece, value=x_sen_piece,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,
                    sen_piece=True,
                    need_weights=False,
                    attn_mask=self_attn_mask_sen_piece,
                )
                # decoder self-attn, cross-attn
                # x1_sen_piece, _ = self.self_attn_sen_piece(
                #     query=x, key=x_sen_piece, value=x_sen_piece,
                #     key_padding_mask=self_attn_padding_mask,  # not useful
                #     incremental_state=incremental_state,
                #     sen_piece=True,
                #     static_kv=True,   # reuse the key, instead of concatenate again
                #     need_weights=False,
                #     # attn_mask=self_attn_mask,  # not self-attn, it is cross-attn
                # )
            else:
                x_sen_piece, _ = self.self_attn(
                    query=x_sen_piece, key=x_sen_piece, value=x_sen_piece,
                    key_padding_mask=self_attn_padding_mask,
                    incremental_state=incremental_state,  # to make sure the correct saved_key, don't save again
                    sen_piece=True,
                    need_weights=False,
                    attn_mask=self_attn_mask_sen_piece,
                )
                # decoder self-attn, cross-attn
                # x1_sen_piece, _ = self.self_attn(
                #     query=x, key=x_sen_piece, value=x_sen_piece,
                #     key_padding_mask=self_attn_padding_mask,
                #     incremental_state=incremental_state,   # to make sure the correct saved_key, don't save again
                #     sen_piece=True,
                #     static_kv=True,   # reuse the key, instead of concatenate again
                #     need_weights=False,
                #     # attn_mask=self_attn_mask,  # not self-attn, it is cross-attn
                # )
            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
            x_sen_piece = residual_sen_piece + x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.self_attn_layer_norm if self.all_share
                                                else self.self_attn_layer_norm_sen_piece, x_sen_piece, after=True)

        attn = None
        attn_sen_piece = None
        if x is not None:
            if self.encoder_attn is not None:
                residual = x
                x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
                # Second implementation of the decoder
                # decoder, dec1 attn to enc1
                x_and_x_sen_piece = torch.cat([encoder_out, encoder_out_sen_piece * self.get_drop_path_ratio()], dim=0)
                x_and_x_sen_piece_padding_mask = torch.cat([encoder_padding_mask, encoder_padding_mask_sen_piece], dim=-1) \
                    if encoder_padding_mask is not None and encoder_padding_mask_sen_piece is not None else None
                x, attn = self.encoder_attn(
                    query=x,
                    key=x_and_x_sen_piece,
                    value=x_and_x_sen_piece,
                    key_padding_mask=x_and_x_sen_piece_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                )
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x
                x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

                residual = x
                x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, p=self.relu_dropout, training=self.training)
                x = self.fc2(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = residual + x
                x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        if x_sen_piece is not None:
            residual_sen_piece = x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.encoder_attn_layer_norm if self.all_share
                                                else self.encoder_attn_layer_norm_sen_piece, x_sen_piece, before=True)

            x_sen_piece_and_x = torch.cat([encoder_out_sen_piece, encoder_out * self.get_drop_path_ratio()], dim=0)
            x_sen_piece_and_x_padding_mask = torch.cat([encoder_padding_mask_sen_piece, encoder_padding_mask], dim=-1) \
                if encoder_padding_mask_sen_piece is not None and encoder_padding_mask is not None else None
            if not self.all_share:
                x_sen_piece, attn_sen_piece = self.encoder_attn_sen_piece(
                    query=x_sen_piece,
                    key=x_sen_piece_and_x,
                    value=x_sen_piece_and_x,
                    key_padding_mask=x_sen_piece_and_x_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                )
            else:
                x_sen_piece, attn_sen_piece = self.encoder_attn(
                    query=x_sen_piece,
                    key=x_sen_piece_and_x,
                    value=x_sen_piece_and_x,
                    key_padding_mask=x_sen_piece_and_x_padding_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=(not self.training and self.need_attn),
                )

            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
            x_sen_piece = residual_sen_piece + x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.encoder_attn_layer_norm if self.all_share
                                                else self.encoder_attn_layer_norm_sen_piece, x_sen_piece, after=True)

            residual_sen_piece = x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.final_layer_norm if self.ffn_share
                                                else self.final_layer_norm_sen_piece, x_sen_piece, before=True)

            x_sen_piece = F.relu(self.fc1(x_sen_piece) if self.ffn_share else self.fc1_sen_piece(x_sen_piece))
            x_sen_piece = F.dropout(x_sen_piece, p=self.relu_dropout, training=self.training)
            x_sen_piece = self.fc2(x_sen_piece) if self.ffn_share else self.fc2_sen_piece(x_sen_piece)
            x_sen_piece = F.dropout(x_sen_piece, p=self.dropout, training=self.training)
            x_sen_piece = residual_sen_piece + x_sen_piece
            x_sen_piece = self.maybe_layer_norm(self.final_layer_norm if self.ffn_share
                                                else self.final_layer_norm_sen_piece, x_sen_piece, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state, x_sen_piece, _
        return x, attn, x_sen_piece, attn_sen_piece

    def get_drop_path_ratio(self):
        randp = uniform(0, 1)
        if self.dec_drop_path_ratio != 0.:
            if self.training:
                if randp < self.dec_drop_path_ratio:
                    return 0.
                else:
                    return 1.
            else:
                return 1. - self.dec_drop_path_ratio
        else:
            return 1.

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True, uniform=True):
    m = nn.Linear(in_features, out_features, bias)
    if uniform:
        nn.init.xavier_uniform_(m.weight)
    else:
        nn.init.xavier_normal_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.all_parameter = getattr(args, 'all_share', False)
    args.ffn_share = getattr(args, 'ffn_share', False)
    args.enc_drop_path_ratio = getattr(args, 'enc_drop_path_ratio', 0.)
    args.dec_drop_path_ratio = getattr(args, 'dec_drop_path_ratio', 0.)
    args.token_embed = getattr(args, 'token_embed', False)


@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani, et al, 2017)
@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer_dec_uni', 'transformer_dec_uni_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)
