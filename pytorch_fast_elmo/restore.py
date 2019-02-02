"""
Follows AllenNLP.
"""
# pylint: disable=attribute-defined-outside-init

from typing import Dict, Tuple, Any, Optional
import json

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


class RestorerBase:

    def __init__(
            self,
            options_file: Optional[str],
            weight_file: Optional[str],
    ) -> None:
        self.options = load_options(options_file)
        self.weight_file = weight_file


class ElmoCharacterEncoderRestorer(RestorerBase):

    def restore(self, requires_grad: bool = False) -> ElmoCharacterEncoder:
        assert self.options and 'char_cnn' in self.options
        assert self.weight_file

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

        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

        if not requires_grad:
            freeze_parameters(self.named_parameters)

        return module

    def _load_char_embedding(self) -> None:
        with h5py.File(self.weight_file, 'r') as fin:
            char_embed_weights = fin['char_embed'][...]

        if char_embed_weights.shape != \
                (self.char_embedding_cnt, self.char_embedding_dim):
            raise ValueError('Char ebd shape not match. '
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


class ElmoWordEmbeddingRestorer(RestorerBase):

    def restore(
            self,
            requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (embedding, lstm_bos, lstm_eos)
        """
        assert self.options is None
        assert self.weight_file

        with h5py.File(self.weight_file, 'r') as fin:
            assert 'embedding' in fin

            ebd_weight = fin['embedding'][...]

            # Since bilm-tf doesn't include padding,
            # we need to prepend a padding row in index 0.
            ebd = torch.zeros(
                    (ebd_weight.shape[0] + 1, ebd_weight.shape[1]),
                    dtype=torch.float,
            )

            ebd.data[1:, :].copy_(torch.FloatTensor(ebd_weight))
            ebd.requires_grad = requires_grad

            lstm_bos_repr = ebd.data[1]
            lstm_eos_repr = ebd.data[2]

            return ebd, lstm_bos_repr, lstm_eos_repr


class ElmoLstmRestorer(RestorerBase):

    def restore(
            self,
            enable_forward: bool = False,
            forward_requires_grad: bool = False,
            enable_backward: bool = False,
            backward_requires_grad: bool = False,
    ) -> Tuple[StatefulUnidirectionalLstm, StatefulUnidirectionalLstm]:
        assert self.options and 'lstm' in self.options
        assert self.weight_file

        self.num_layers = self.options['lstm']['n_layers']
        self.input_size = self.options['lstm']['projection_dim']
        self.hidden_size = self.input_size
        self.cell_size = self.options['lstm']['dim']
        self.cell_clip = self.options['lstm']['cell_clip']
        self.proj_clip = self.options['lstm']['proj_clip']
        self.use_skip_connections = True

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
                    use_skip_connections=self.use_skip_connections,
            )
            bwd_lstm_named_parameters = bwd_lstm.named_parameters()
            self.named_parameters.update(bwd_lstm_named_parameters)

        if enable_forward and enable_backward:
            if set(fwd_lstm_named_parameters.keys()) & \
                    set(bwd_lstm_named_parameters.keys()):
                raise ValueError('key conflict.')

        # Load weights.
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
