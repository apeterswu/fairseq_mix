# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import numpy as np
import os

from torch.utils.data import ConcatDataset

from fairseq import options
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, IndexedInMemoryDataset,
    IndexedRawTextDataset,
)

from . import FairseqTask, register_task


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`train.py <train>`,
        :mod:`generate.py <generate>` and :mod:`interactive.py <interactive>`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    def __init__(self, args, src_dict, tgt_dict, src_dict_sen_piece=None, tgt_dict_sen_piece=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.src_dict_sen_piece = src_dict_sen_piece
        self.tgt_dict_sen_piece = tgt_dict_sen_piece

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        src_dict_sen_piece = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang + '.sen_piece')))
        tgt_dict_sen_piece = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang + '.sen_piece')))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        assert src_dict_sen_piece.pad() == tgt_dict_sen_piece.pad()
        assert src_dict_sen_piece.eos() == tgt_dict_sen_piece.eos()
        assert src_dict_sen_piece.unk() == tgt_dict_sen_piece.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))
        print('| [{}] dictionary: {} types'.format(args.source_lang + '.sen_piece', len(src_dict_sen_piece)))
        print('| [{}] dictionary: {} types'.format(args.target_lang + '.sen_piece', len(tgt_dict_sen_piece)))

        # return cls(args, src_dict, tgt_dict)
        return cls(args, src_dict, tgt_dict, src_dict_sen_piece, tgt_dict_sen_piece)

    def load_dataset(self, split, combine=False):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedInMemoryDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedInMemoryDataset.exists(path):
                return IndexedInMemoryDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []
        src_datasets_sen_piece = []
        tgt_datasets_sen_piece = []

        data_paths = self.args.data

        for data_path in data_paths:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))
                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if split_exists(split_k, src, tgt, src+'.sen_piece', data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src+'.sen_piece', data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                src_datasets_sen_piece.append(indexed_dataset(prefix+src+'.sen_piece', self.src_dict_sen_piece))
                tgt_datasets_sen_piece.append(indexed_dataset(prefix+tgt+'.sen_piece', self.tgt_dict_sen_piece))
                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets_sen_piece[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)
        assert len(src_datasets_sen_piece) == len(tgt_datasets_sen_piece)
        assert len(src_datasets) == len(src_datasets_sen_piece)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_sen_piece, tgt_dataset_sen_piece = src_datasets_sen_piece[0], tgt_datasets_sen_piece[0]
            src_sizes = src_dataset.sizes
            tgt_sizes = tgt_dataset.sizes
            src_sen_piece_sizes = src_dataset_sen_piece.sizes
            tgt_sen_piece_sizes = tgt_dataset_sen_piece.sizes
        else:
            if self.args.upsample_primary > 1:
                src_datasets.extend([src_datasets[0]] * (self.args.upsample_primary - 1))
                tgt_datasets.extend([tgt_datasets[0]] * (self.args.upsample_primary - 1))
                src_datasets_sen_piece.extend([src_datasets_sen_piece[0]] * (self.args.upsample_primary - 1))
                tgt_datasets_sen_piece.extend([tgt_datasets_sen_piece[0]] * (self.args.upsample_primary - 1))
            src_dataset = ConcatDataset(src_datasets)
            tgt_dataset = ConcatDataset(tgt_datasets)
            src_dataset_sen_piece = ConcatDataset(src_datasets_sen_piece)
            tgt_dataset_sen_piece = ConcatDataset(tgt_datasets_sen_piece)
            src_sizes = np.concatenate([ds.sizes for ds in src_datasets])
            tgt_sizes = np.concatenate([ds.sizes for ds in tgt_datasets])
            src_sen_piece_sizes = np.concatenate([ds.sizes for ds in src_datasets_sen_piece])
            tgt_sen_piece_sizes = np.concatenate([ds.sizes for ds in tgt_datasets_sen_piece])

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_sizes, self.src_dict,
            tgt_dataset, tgt_sizes, self.tgt_dict,
            src_dataset_sen_piece, src_sen_piece_sizes, self.src_dict_sen_piece,
            tgt_dataset_sen_piece, tgt_sen_piece_sizes, self.tgt_dict_sen_piece,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict
    @property
    def source_sen_piece_dictionary(self):
        return self.src_dict_sen_piece

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
    @property
    def target_sen_piece_dictionary(self):
        return self.tgt_dict_sen_piece
