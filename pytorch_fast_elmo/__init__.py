# -*- coding: utf-8 -*-
"""Top-level package for pytorch-fast-elmo."""

__author__ = """Hunt Zhan"""
__email__ = 'huntzhan.dev@gmail.com'
__version__ = '0.6.0'

# To avoid `undefined symbol` error.
import torch

# pylint: disable=no-name-in-module
from pytorch_stateful_lstm import StatefulUnidirectionalLstm
from _pytorch_fast_elmo import ElmoCharacterEncoder, ScalarMix

from pytorch_fast_elmo.utils import (
        batch_to_char_ids,
        load_and_build_vocab2id,
        batch_to_word_ids,
)

from pytorch_fast_elmo.factory import (
        ElmoCharacterEncoderFactory,
        ElmoWordEmbeddingFactory,
        ElmoLstmFactory,
        ElmoVocabProjectionFactory,
)

from pytorch_fast_elmo.model import (
        FastElmoBase,
        FastElmo,
        FastElmoWordEmbedding,
        FastElmoForwardVocabDistrib,
        FastElmoBackwardVocabDistrib,
        FastElmoWordEmbeddingForwardVocabDistrib,
        FastElmoWordEmbeddingBackwardVocabDistrib,
)
