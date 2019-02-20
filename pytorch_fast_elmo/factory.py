"""
Follows AllenNLP.
"""
# pylint: disable=attribute-defined-outside-init

from typing import Dict, Tuple, Any, Optional, List
import json
import math

import torch
import h5py
import numpy as np

from pytorch_stateful_lstm import StatefulUnidirectionalLstm
from _pytorch_fast_elmo import ElmoCharacterEncoder  # pylint: disable=no-name-in-module


def load_options(options_file: Optional[str]):  # type: ignore
    if options_file is None:
        return None
    else:
        with open(options_file) as fin:
            return json.load(fin)


def freeze_parameters(named_parameters: Dict[str, torch.Tensor]) -> None:
    for param in named_parameters.values():
        param.requires_grad = False


class FactoryBase:

    def __init__(
            self,
            options_file: Optional[str],
            weight_file: Optional[str],
    ) -> None:
        self.options = load_options(options_file)
        self.weight_file = weight_file


class ElmoCharacterEncoderFactory(FactoryBase):

    @staticmethod
    def from_scratch(
            char_embedding_cnt: int,
            char_embedding_dim: int,
            filters: List[Tuple[int, int]],
            activation: str,
            num_highway_layers: int,
            output_dim: int,
    ) -> 'ElmoCharacterEncoderFactory':
        factory = ElmoCharacterEncoderFactory(None, None)
        factory.options = {
                'n_characters': char_embedding_cnt,
                'char_cnn': {
                        'embedding': {
                                'dim': char_embedding_dim
                        },
                        'filters': filters,
                        'activation': activation,
                        'n_highway': num_highway_layers,
                },
                'lstm': {
                        'projection_dim': output_dim
                },
        }
        return factory

    def create(self, requires_grad: bool = False) -> ElmoCharacterEncoder:
        assert self.options and 'char_cnn' in self.options

        # Collect parameters for the construction of `ElmoCharacterEncoder`.
        self.char_embedding_cnt = self.options.get('n_characters', 261)
        self.char_embedding_dim = self.options['char_cnn']['embedding']['dim']
        self.filters = self.options['char_cnn']['filters']
        self.activation = self.options['char_cnn']['activation']
        self.num_highway_layers = self.options['char_cnn']['n_highway']
        self.output_dim = self.options['lstm']['projection_dim']

        self.named_parameters: Dict[str, torch.Tensor] = {}

        module = ElmoCharacterEncoder(
                self.char_embedding_cnt,
                self.char_embedding_dim,
                self.filters,
                self.activation,
                self.num_highway_layers,
                self.output_dim,
        )
        self.named_parameters.update(module.named_parameters())

        if self.weight_file:
            self._load_char_embedding()
            self._load_cnn_weights()
            self._load_highway()
            self._load_projection()
        else:
            assert requires_grad

        if not requires_grad:
            freeze_parameters(self.named_parameters)

        return module

    def _load_char_embedding(self) -> None:
        with h5py.File(self.weight_file, 'r') as fin:
            char_embed_weights = fin['char_embed'][...]

        if char_embed_weights.shape != \
                (self.char_embedding_cnt, self.char_embedding_dim):
            raise ValueError('Char embd shape not match. '
                             f'Loaded shape: {char_embed_weights.shape}')

        self.named_parameters['char_embedding.weight'].data[0:, :] = \
                torch.zeros(1, self.char_embedding_dim, dtype=torch.float)
        self.named_parameters['char_embedding.weight'].data[1:, :] = \
                torch.FloatTensor(char_embed_weights)

    def _load_cnn_weights(self) -> None:
        for conv_idx, (kernel_size, out_channels) in enumerate(self.filters):
            with h5py.File(self.weight_file, 'r') as fin:
                weight = fin['CNN'][f'W_cnn_{conv_idx}'][...]
                bias = fin['CNN'][f'b_cnn_{conv_idx}'][...]

            w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != (out_channels, self.char_embedding_dim, kernel_size):
                raise ValueError("Invalid weight file")

            weight_name = f'char_conv_{conv_idx}.weight'
            self.named_parameters[weight_name].data.copy_(torch.FloatTensor(w_reshaped))

            if bias.shape != (out_channels,):
                raise ValueError("Invalid weight file")

            bias_name = f'char_conv_{conv_idx}.bias'
            self.named_parameters[bias_name].data.copy_(torch.FloatTensor(bias))

    def _load_highway(self) -> None:
        """
        Note: `W_carry` and `b_carry` in bilm-tf are weights of transform gate.
        """
        total_out_channels = sum(out_channels for _, out_channels in self.filters)

        for layer_idx in range(self.num_highway_layers):
            with h5py.File(self.weight_file, 'r') as fin:
                w_transform = np.transpose(fin[f'CNN_high_{layer_idx}']['W_transform'][...])
                w_transform_gate = np.transpose(fin[f'CNN_high_{layer_idx}']['W_carry'][...])
                weight = np.concatenate([w_transform, w_transform_gate], axis=0)

                if weight.shape != (total_out_channels * 2, total_out_channels):
                    raise ValueError("Invalid weight file")

                weight_name = f'highway.layers_{layer_idx}.weight'
                self.named_parameters[weight_name].data.copy_(torch.FloatTensor(weight))

                b_transform = fin[f'CNN_high_{layer_idx}']['b_transform'][...]
                b_transform_gate = fin[f'CNN_high_{layer_idx}']['b_carry'][...]
                bias = np.concatenate([b_transform, b_transform_gate], axis=0)

                if bias.shape != (total_out_channels * 2,):
                    raise ValueError("Invalid weight file")

                bias_name = f'highway.layers_{layer_idx}.bias'
                self.named_parameters[bias_name].data.copy_(torch.FloatTensor(bias))

    def _load_projection(self) -> None:
        total_out_channels = sum(out_channels for _, out_channels in self.filters)

        with h5py.File(self.weight_file, 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]

        weight = np.transpose(weight)
        if weight.shape != (self.output_dim, total_out_channels):
            raise ValueError("Invalid weight file")

        weight_name = 'output_proj.weight'
        self.named_parameters[weight_name].data.copy_(torch.FloatTensor(weight))

        if bias.shape != (self.output_dim,):
            raise ValueError("Invalid weight file")

        bias_name = f'output_proj.bias'
        self.named_parameters[bias_name].data.copy_(torch.FloatTensor(bias))


class ElmoWordEmbeddingFactory(FactoryBase):

    @staticmethod
    def from_scratch(
            cnt: int,
            dim: int,
    ) -> 'ElmoWordEmbeddingFactory':
        factory = ElmoWordEmbeddingFactory(None, None)
        factory.options = {
                'n_tokens_vocab': cnt,
                'word_embedding_dim': dim,
        }
        return factory

    def create(
            self,
            requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (embedding, lstm_bos, lstm_eos)
        """
        if self.weight_file:
            # Load `embd_weight` from hdf5 or txt.
            if self.weight_file.endswith(('.h5', '.hdf5')):
                # HDF5 format.
                with h5py.File(self.weight_file, 'r') as fin:
                    assert 'embedding' in fin
                    embd_weight = fin['embedding'][...]

            else:
                # TXT format.
                loaded_cnt = 0
                loaded_dim = 0
                loaded_embds = []
                with open(self.weight_file) as fin:
                    for idx, line in enumerate(fin):
                        fields = line.split()
                        if not fields:
                            continue

                        # L0: <cnt> <dim>
                        if idx == 0 and len(fields) == 2:
                            loaded_dim = int(fields[1])
                            continue

                        token = fields[0]
                        embd = fields[1:]
                        loaded_cnt += 1

                        if loaded_dim == 0:
                            loaded_dim = len(embd)
                        elif loaded_dim != len(embd):
                            raise ValueError(f'Dimension not match on L{idx}: {token}, '
                                             f'should be {loaded_dim}-D.')

                        vec = np.expand_dims(
                                np.asarray(list(map(float, embd))),
                                0,
                        )
                        loaded_embds.append(vec)

                embd_weight = np.concatenate(loaded_embds)

            # Since bilm-tf doesn't include padding,
            # we need to prepend a padding row in index 0.
            self.word_embedding_cnt = embd_weight.shape[0]
            self.word_embedding_dim = embd_weight.shape[1]

            # Check with options if `n_tokens_vocab` exists.
            if 'n_tokens_vocab' in self.options \
                    and self.options['n_tokens_vocab'] != self.word_embedding_cnt:
                raise ValueError('n_tokens_vocab not match')

            embd = torch.zeros(
                    (self.word_embedding_cnt + 1, self.word_embedding_dim),
                    dtype=torch.float,
            )

            embd.data[1:, :].copy_(torch.FloatTensor(embd_weight))
            embd.requires_grad = requires_grad

            lstm_bos_repr = embd.data[1]
            lstm_eos_repr = embd.data[2]

        else:
            assert requires_grad
            assert self.options['n_tokens_vocab'] > 0

            self.word_embedding_cnt = self.options['n_tokens_vocab']
            self.word_embedding_dim = self.options['word_embedding_dim']

            embd = torch.zeros(
                    (self.word_embedding_cnt + 1, self.word_embedding_dim),
                    dtype=torch.float,
            )
            torch.nn.init.normal_(embd)
            embd.requires_grad = True

            # `exec_managed_lstm_bos_eos` should be disabled in this case.
            lstm_bos_repr = None
            lstm_eos_repr = None

        return embd, lstm_bos_repr, lstm_eos_repr


class ElmoLstmFactory(FactoryBase):

    @staticmethod
    def from_scratch(
            num_layers: int,
            input_size: int,
            hidden_size: int,
            cell_size: int,
            cell_clip: float,
            proj_clip: float,
            truncated_bptt: int,
    ) -> 'ElmoLstmFactory':
        factory = ElmoLstmFactory(None, None)
        factory.options = {
                'lstm': {
                        'n_layers': num_layers,
                        'projection_dim': input_size,
                        '_hidden_size': hidden_size,
                        'dim': cell_size,
                        'cell_clip': cell_clip,
                        'proj_clip': proj_clip,
                },
                'unroll_steps': truncated_bptt,
        }
        return factory

    def create(
            self,
            enable_forward: bool = False,
            forward_requires_grad: bool = False,
            enable_backward: bool = False,
            backward_requires_grad: bool = False,
    ) -> Tuple[StatefulUnidirectionalLstm, StatefulUnidirectionalLstm]:
        assert self.options and 'lstm' in self.options

        self.num_layers = self.options['lstm']['n_layers']
        self.input_size = self.options['lstm']['projection_dim']
        self.cell_size = self.options['lstm']['dim']
        self.cell_clip = self.options['lstm']['cell_clip']
        self.proj_clip = self.options['lstm']['proj_clip']
        self.truncated_bptt = self.options.get('unroll_steps', 20)
        self.use_skip_connections = True

        if self.options['lstm'].get('_hidden_size', 0) > 0:
            self.hidden_size = self.options['lstm']['_hidden_size']
        else:
            self.hidden_size = self.input_size

        self.named_parameters: Dict[str, torch.Tensor] = {}

        fwd_lstm = None
        if enable_forward:
            fwd_lstm = StatefulUnidirectionalLstm(
                    go_forward=True,
                    num_layers=self.num_layers,
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    cell_size=self.cell_size,
                    cell_clip=self.cell_clip,
                    proj_clip=self.proj_clip,
                    truncated_bptt=self.truncated_bptt,
                    use_skip_connections=self.use_skip_connections,
            )
            fwd_lstm_named_parameters = fwd_lstm.named_parameters()
            self.named_parameters.update(fwd_lstm_named_parameters)

        bwd_lstm = None
        if enable_backward:
            bwd_lstm = StatefulUnidirectionalLstm(
                    go_forward=False,
                    num_layers=self.num_layers,
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    cell_size=self.cell_size,
                    cell_clip=self.cell_clip,
                    proj_clip=self.proj_clip,
                    truncated_bptt=self.truncated_bptt,
                    use_skip_connections=self.use_skip_connections,
            )
            bwd_lstm_named_parameters = bwd_lstm.named_parameters()
            self.named_parameters.update(bwd_lstm_named_parameters)

        if enable_forward and enable_backward:
            if set(fwd_lstm_named_parameters.keys()) & \
                    set(bwd_lstm_named_parameters.keys()):
                raise ValueError('key conflict.')

        # Load weights.
        if self.weight_file:
            with h5py.File(self.weight_file, 'r') as fin:
                for layer_idx in range(self.num_layers):
                    for direction, prefix in enumerate([
                            'uni_lstm.forward_layer_',
                            'uni_lstm.backward_layer_',
                    ]):
                        good_forward = (direction == 0 and enable_forward)
                        good_backward = (direction == 1 and enable_backward)
                        if good_forward or good_backward:
                            dataset = fin[f'RNN_{direction}']\
                                        ['RNN']\
                                        ['MultiRNNCell']\
                                        [f'Cell{layer_idx}']\
                                        ['LSTMCell']
                            self._load_lstm(prefix + str(layer_idx), dataset)
        else:
            if enable_forward:
                assert forward_requires_grad
            if enable_backward:
                assert backward_requires_grad

        if enable_forward and not forward_requires_grad:
            freeze_parameters(fwd_lstm_named_parameters)

        if enable_backward and not backward_requires_grad:
            freeze_parameters(bwd_lstm_named_parameters)

        return fwd_lstm, bwd_lstm

    def _load_lstm(self, prefix: str, dataset: Any) -> None:
        cell_size = self.cell_size
        input_size = self.input_size

        tf_weights = np.transpose(dataset['W_0'][...])
        torch_weights = tf_weights.copy()

        input_weights = torch_weights[:, :input_size]
        recurrent_weights = torch_weights[:, input_size:]
        tf_input_weights = tf_weights[:, :input_size]
        tf_recurrent_weights = tf_weights[:, input_size:]

        for torch_w, tf_w in [[input_weights, tf_input_weights],
                              [recurrent_weights, tf_recurrent_weights]]:
            torch_w[(1 * cell_size):(2 * cell_size), :] = tf_w[(2 * cell_size):(3 * cell_size), :]
            torch_w[(2 * cell_size):(3 * cell_size), :] = tf_w[(1 * cell_size):(2 * cell_size), :]

        self.named_parameters[prefix + '.input_linearity_weight'].data.copy_(
                torch.FloatTensor(input_weights),)
        self.named_parameters[prefix + '.hidden_linearity_weight'].data.copy_(
                torch.FloatTensor(recurrent_weights),)

        tf_bias = dataset['B'][...]
        tf_bias[(2 * cell_size):(3 * cell_size)] += 1
        torch_bias = tf_bias.copy()
        torch_bias[(1 * cell_size):(2 * cell_size)] = tf_bias[(2 * cell_size):(3 * cell_size)]
        torch_bias[(2 * cell_size):(3 * cell_size)] = tf_bias[(1 * cell_size):(2 * cell_size)]

        self.named_parameters[prefix + '.hidden_linearity_bias'].data.copy_(
                torch.FloatTensor(torch_bias),)

        proj_weights = np.transpose(dataset['W_P_0'][...])

        self.named_parameters[prefix + '.proj_linearity_weight'].data.copy_(
                torch.FloatTensor(proj_weights),)


class ElmoVocabProjectionFactory(FactoryBase):

    @staticmethod
    def from_scratch(
            input_size: int,
            proj_size: int,
    ) -> 'ElmoVocabProjectionFactory':
        factory = ElmoVocabProjectionFactory(None, None)
        factory.options = {
                'lstm': {
                        'projection_dim': input_size
                },
                'n_tokens_vocab': proj_size,
        }
        return factory

    def create(
            self,
            requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (weight, bias) for affine transformation.
        """
        assert self.options \
                and 'n_tokens_vocab' in self.options \
                and 'lstm' in self.options

        self.input_size = self.options['lstm']['projection_dim']
        self.proj_size = self.options['n_tokens_vocab']
        assert self.input_size > 0 and self.proj_size > 0

        # Note: no padding zero.
        weight = torch.zeros(
                (self.proj_size, self.input_size),
                dtype=torch.float,
        )
        bias = torch.zeros(
                (self.proj_size,),
                dtype=torch.float,
        )

        if self.weight_file:
            with h5py.File(self.weight_file, 'r') as fin:
                if 'softmax' not in fin:
                    raise ValueError('softmax not in weight file.')
                loaded_weight = fin['softmax']['W'][...]
                loaded_bias = fin['softmax']['b'][...]

            weight.data.copy_(torch.FloatTensor(loaded_weight))
            weight.requires_grad = requires_grad

            bias.data.copy_(torch.FloatTensor(loaded_bias))
            bias.requires_grad = requires_grad

        else:
            assert requires_grad
            # init.
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight.requires_grad = True

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)  # pylint: disable=protected-access
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(bias, -bound, bound)
            bias.requires_grad = True

        return weight, bias
