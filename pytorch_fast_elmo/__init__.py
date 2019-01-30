# -*- coding: utf-8 -*-
"""Top-level package for pytorch-fast-elmo."""

__author__ = """Hunt Zhan"""
__email__ = 'huntzhan.dev@gmail.com'
__version__ = '0.2.1'

# To avoid `undefined symbol` error.
import torch
# pylint: disable=no-name-in-module
from _pytorch_fast_elmo import (
        ElmoCharacterEncoder,
        ScalarMix,
)
from pytorch_fast_elmo.restore import (
        ElmoCharacterEncoderRestorer,
        ElmoLstmRestorer,
)
from pytorch_fast_elmo.integrate import FastElmo
