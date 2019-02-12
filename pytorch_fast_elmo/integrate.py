"""
Provide helper classes/functions to execute ELMo.
"""
# pylint: disable=no-self-use,arguments-differ,too-many-public-methods

from typing import List, Tuple, Optional, Dict, Union, Any, Set

import torch
from torch.nn.utils.rnn import PackedSequence

from pytorch_fast_elmo.restore import (
        ElmoCharacterEncoderRestorer,
        ElmoWordEmbeddingRestorer,
        ElmoLstmRestorer,
)
from pytorch_fast_elmo import utils
from _pytorch_fast_elmo import ScalarMix  # pylint: disable=no-name-in-module


def _bind_cpp_extension_parameters(
        py_module: torch.nn.Module,
        cpp_module: Any,
        param_prefix: str = '',
        override: bool = False,
        only_trainable: bool = False,
) -> None:
    if isinstance(cpp_module, torch.nn.Module):
        raise TypeError('cpp_module should not be torch.nn.Module.')

    prefix = 'cpp_ext_' + param_prefix
    for name, tensor in cpp_module.named_parameters().items():
        if only_trainable and not tensor.requires_grad:
            continue

        param_name = (prefix + name).replace('.', '_')
        if override and hasattr(py_module, param_name):
            delattr(py_module, param_name)

        py_module.register_parameter(
                param_name,
                torch.nn.Parameter(tensor, requires_grad=tensor.requires_grad),
        )


def _raise_if_kwargs_is_invalid(allowed: Set[str], kwargs: Dict[str, Any]) -> None:
    invalid_keys = set(kwargs) - allowed
    if invalid_keys:
        msg = '\n'.join('invalid kwargs: {}'.format(key) for key in invalid_keys)
        raise ValueError(msg)


class FastElmoBase(torch.nn.Module):  # type: ignore

    SCALAR_MIX_PARAMS = {
            'disable_scalar_mix',
            'num_output_representations',
            'output_representation_dropout',
            'scalar_mix_parameters',
            'do_layer_norm',
    }
    EXEC_PARAMS = {
            'exec_managed_lstm_bos_eos',
            'exec_sort_batch',
    }
    COMMON_PARAMS = SCALAR_MIX_PARAMS | EXEC_PARAMS

    _CHAR_CNN_FILTERS = [
            (1, 32),
            (2, 32),
            (3, 64),
            (4, 128),
            (5, 256),
            (6, 512),
            (7, 1024),
    ]

    def __init__(  # pylint: disable=dangerous-default-value
            self,

            # Generated by bilm-tf.
            options_file: Optional[str],
            weight_file: Optional[str],

            # Controls the behavior of execution.
            exec_managed_lstm_bos_eos: bool = True,
            exec_managed_lstm_reset_states: bool = False,
            exec_sort_batch: bool = True,

            # Controls the behavior of `ScalarMix`.
            disable_scalar_mix: bool = False,
            num_output_representations: int = 1,
            output_representation_dropout: float = 0.0,
            scalar_mix_parameters: Optional[List[float]] = None,
            do_layer_norm: bool = False,

            # Controls the behavior of restorer.
            # Char CNN.
            disable_char_cnn: bool = False,
            char_cnn_requires_grad: bool = False,
            # From scratch.
            char_cnn_char_embedding_cnt: int = 261,
            char_cnn_char_embedding_dim: int = 16,
            char_cnn_filters: List[Tuple[int, int]] = _CHAR_CNN_FILTERS,
            char_cnn_activation: str = 'relu',
            char_cnn_num_highway_layers: int = 2,
            char_cnn_output_dim: int = 512,

            # Word Embedding.
            disable_word_embedding: bool = True,
            word_embedding_weight_file: Optional[str] = None,
            word_embedding_requires_grad: bool = False,
            # From scratch.
            word_embedding_cnt: int = 0,
            word_embedding_dim: int = 512,

            # The Forward LSTM.
            disable_forward_lstm: bool = False,
            forward_lstm_requires_grad: bool = False,
            # The Backward LSTM.
            disable_backward_lstm: bool = False,
            backward_lstm_requires_grad: bool = False,
            # From scratch.
            lstm_num_layers: int = 2,
            lstm_input_size: int = 512,
            lstm_hidden_size: int = 512,
            lstm_cell_size: int = 4096,
            lstm_cell_clip: float = 3.0,
            lstm_proj_clip: float = 3.0,
            lstm_truncated_bptt: int = 20,

            # Provide the BOS/EOS representations of shape `(projection_dim,)`
            # if char CNN is disabled.
            lstm_bos_repr: Optional[torch.Tensor] = None,
            lstm_eos_repr: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.disable_char_cnn = disable_char_cnn
        self.disable_word_embedding = disable_word_embedding
        self.disable_forward_lstm = disable_forward_lstm
        self.disable_backward_lstm = disable_backward_lstm
        self.disable_scalar_mix = disable_scalar_mix

        self.exec_managed_lstm_bos_eos = exec_managed_lstm_bos_eos
        self.exec_managed_lstm_reset_states = exec_managed_lstm_reset_states
        self.exec_sort_batch = exec_sort_batch

        # Char CNN.
        if options_file:
            self.char_cnn_restorer = ElmoCharacterEncoderRestorer(
                    options_file,
                    weight_file,
            )
        else:
            # From scratch.
            self.char_cnn_restorer = ElmoCharacterEncoderRestorer.from_scratch(
                    char_embedding_cnt=char_cnn_char_embedding_cnt,
                    char_embedding_dim=char_cnn_char_embedding_dim,
                    filters=char_cnn_filters,
                    activation=char_cnn_activation,
                    num_highway_layers=char_cnn_num_highway_layers,
                    output_dim=char_cnn_output_dim,
            )

        if not disable_char_cnn:
            self.char_cnn = self.char_cnn_restorer.restore(requires_grad=char_cnn_requires_grad)

        # Word Embedding.
        if options_file:
            self.word_embedding_restorer = ElmoWordEmbeddingRestorer(
                    None,
                    word_embedding_weight_file or weight_file,
            )
        else:
            # From scratch.
            self.word_embedding_restorer = ElmoWordEmbeddingRestorer.from_scratch(
                    cnt=word_embedding_cnt,
                    dim=word_embedding_dim,
            )
            if exec_managed_lstm_bos_eos:
                raise ValueError('exec_managed_lstm_bos_eos should be disabled.')

        if not disable_word_embedding:
            # Not a cpp extension.
            self.word_embedding_weight, lstm_bos_repr, lstm_eos_repr = \
                    self.word_embedding_restorer.restore(requires_grad=word_embedding_requires_grad)

        # LSTM.
        if options_file:
            self.lstm_restorer = ElmoLstmRestorer(
                    options_file,
                    weight_file,
            )
        else:
            # From scratch.
            self.lstm_restorer = ElmoLstmRestorer.from_scratch(
                    num_layers=lstm_num_layers,
                    input_size=lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    cell_size=lstm_cell_size,
                    cell_clip=lstm_cell_clip,
                    proj_clip=lstm_proj_clip,
                    truncated_bptt=lstm_truncated_bptt,
            )

        if not (disable_forward_lstm and disable_backward_lstm):
            self.forward_lstm, self.backward_lstm = self.lstm_restorer.restore(
                    enable_forward=not disable_forward_lstm,
                    forward_requires_grad=forward_lstm_requires_grad,
                    enable_backward=not disable_backward_lstm,
                    backward_requires_grad=backward_lstm_requires_grad,
            )

            # Cache BOS/EOS reprs.
            if exec_managed_lstm_bos_eos:
                if disable_char_cnn:
                    if lstm_bos_repr is None or lstm_eos_repr is None:
                        raise ValueError('BOS/EOS not provided.')
                    self.lstm_bos_repr = lstm_bos_repr
                    self.lstm_eos_repr = lstm_eos_repr

                else:
                    self.lstm_bos_repr, self.lstm_eos_repr = utils.get_bos_eos_token_repr(
                            self.char_cnn_restorer,
                            self.char_cnn,
                    )

        # ScalarMix
        if not disable_scalar_mix:
            self.scalar_mixes: List[ScalarMix] = []

            if scalar_mix_parameters is None:
                scalar_mix_parameters = []

            for _ in range(num_output_representations):
                scalar_mix = ScalarMix(
                        # char cnn + lstm.
                        self.lstm_restorer.num_layers + 1,
                        do_layer_norm=do_layer_norm,
                        initial_scalar_parameters=scalar_mix_parameters,
                        trainable=not scalar_mix_parameters,
                )
                self.scalar_mixes.append(scalar_mix)

            self.repr_dropout = None
            if output_representation_dropout > 0.0:
                self.repr_dropout = torch.nn.Dropout(p=output_representation_dropout)

        # Bind CPU parameters.
        self._bind_parameters()

    def _bind_parameters(self, override: bool = False) -> None:
        # Since `ElmoCharacterEncoder`, `StatefulUnidirectionalLstm`, `ScalarMix` are not
        # instances of `torch.nn.Module`, we need to bind the parameters manually.
        if not self.disable_char_cnn:
            _bind_cpp_extension_parameters(
                    self,
                    self.char_cnn,
                    'char_cnn_',
                    override=override,
                    only_trainable=True,
            )

        if not self.disable_word_embedding \
                and self.word_embedding_weight.requires_grad:
            word_ebd_param_name = 'word_embedding'
            if override and hasattr(self, word_ebd_param_name):
                delattr(self, word_ebd_param_name)
            self.register_parameter(
                    word_ebd_param_name,
                    torch.nn.Parameter(self.word_embedding_weight, requires_grad=True),
            )

        if not self.disable_forward_lstm:
            _bind_cpp_extension_parameters(
                    self,
                    self.forward_lstm,
                    'forward_lstm_',
                    override=override,
                    only_trainable=True,
            )
        if not self.disable_backward_lstm:
            _bind_cpp_extension_parameters(
                    self,
                    self.backward_lstm,
                    'backward_lstm_',
                    override=override,
                    only_trainable=True,
            )

        if not self.disable_scalar_mix:
            for idx, scalar_mix in enumerate(self.scalar_mixes):
                _bind_cpp_extension_parameters(
                        self,
                        scalar_mix,
                        f'scalar_mix_{idx}_',
                        override=override,
                        only_trainable=True,
                )

    def _cpp_ext_cuda(self, cpp_module: Any, device: Optional[int] = None) -> None:
        # TODO: handle optional parameter in cpp extension.
        if device is not None:
            cpp_module.cuda(device)
        else:
            cpp_module.cuda()

    def cuda(self, device=None):  # type: ignore
        # Move all cpp exntensions to GPU.
        if not self.disable_char_cnn:
            self._cpp_ext_cuda(self.char_cnn, device)

        if not self.disable_word_embedding:
            self.word_embedding_weight = self.word_embedding_weight.cuda(device)

        if not self.disable_forward_lstm:
            self._cpp_ext_cuda(self.forward_lstm, device)
        if not self.disable_backward_lstm:
            self._cpp_ext_cuda(self.backward_lstm, device)

        if not self.disable_scalar_mix:
            for scalar_mix in self.scalar_mixes:
                self._cpp_ext_cuda(scalar_mix, device)

        # Override parameter bindings.
        self._bind_parameters(override=True)

        return super().cuda(device)

    def cpu(self):  # type: ignore
        # Move all cpp exntensions to CPU.
        if not self.disable_char_cnn:
            self.char_cnn.cpu()

        if not self.disable_word_embedding:
            self.word_embedding_weight = self.word_embedding_weight.cpu()

        if not self.disable_forward_lstm:
            self.forward_lstm.cpu()
        if not self.disable_backward_lstm:
            self.backward_lstm.cpu()

        if not self.disable_scalar_mix:
            for scalar_mix in self.scalar_mixes:
                scalar_mix.cpu()

        # Also, move BOS/EOS back to CPU.
        if not (self.disable_forward_lstm and self.disable_backward_lstm):
            self.lstm_bos_repr = self.lstm_bos_repr.cpu()
            self.lstm_eos_repr = self.lstm_eos_repr.cpu()

        # Override parameter bindings.
        self._bind_parameters(override=True)

        return super().cpu()

    def _get_lstm_device(self) -> int:
        cpp_ext = None
        if not self.disable_forward_lstm:
            cpp_ext = self.forward_lstm
        elif not self.disable_backward_lstm:
            cpp_ext = self.backward_lstm

        # Assume `cpp_ext` is not None.
        assert cpp_ext is not None
        tensor = cpp_ext.parameters()[0]
        return -1 if not tensor.is_cuda else tensor.get_device()  # type: ignore

    def get_batched_lstm_bos_eos_repr(self, attr_name: str, batch_size: int) -> PackedSequence:
        tensor = getattr(self, attr_name)

        if not tensor.is_cuda:
            # Move to GPU permanently.
            device = self._get_lstm_device()
            if device >= 0:
                tensor = tensor.cuda(device)
                setattr(self, attr_name, tensor)

        batched = tensor.unsqueeze(0).expand(batch_size, -1)
        return PackedSequence(batched, torch.LongTensor([batch_size]))

    def get_batched_lstm_bos_repr(self, batch_size: int) -> PackedSequence:
        return self.get_batched_lstm_bos_eos_repr('lstm_bos_repr', batch_size)

    def get_batched_lstm_eos_repr(self, batch_size: int) -> PackedSequence:
        return self.get_batched_lstm_bos_eos_repr('lstm_eos_repr', batch_size)

    def exec_forward_lstm_bos(self, batch_size: int) -> None:
        batched = self.get_batched_lstm_bos_repr(batch_size)
        with torch.no_grad():
            self.forward_lstm(batched.data, batched.batch_sizes)

    def exec_forward_lstm_eos(self, batch_size: int) -> None:
        batched = self.get_batched_lstm_eos_repr(batch_size)
        with torch.no_grad():
            self.forward_lstm(batched.data, batched.batch_sizes)

    def exec_backward_lstm_bos(self, batch_size: int) -> None:
        batched = self.get_batched_lstm_bos_repr(batch_size)
        with torch.no_grad():
            self.backward_lstm(batched.data, batched.batch_sizes)

    def exec_backward_lstm_eos(self, batch_size: int) -> None:
        batched = self.get_batched_lstm_eos_repr(batch_size)
        with torch.no_grad():
            self.backward_lstm(batched.data, batched.batch_sizes)

    def exec_forward_lstm_permutate_states(self, index: torch.Tensor) -> None:
        self.forward_lstm.permutate_states(index)

    def exec_backward_lstm_permutate_states(self, index: torch.Tensor) -> None:
        self.backward_lstm.permutate_states(index)

    def exec_bilstm_permutate_states(self, index: torch.Tensor) -> None:
        self.exec_forward_lstm_permutate_states(index)
        self.exec_backward_lstm_permutate_states(index)

    def exec_char_cnn(self, inputs: PackedSequence) -> PackedSequence:
        """
        Char CNN.
        """
        output_data = self.char_cnn(inputs.data)
        return PackedSequence(output_data, inputs.batch_sizes)

    def exec_word_embedding(self, inputs: PackedSequence) -> PackedSequence:
        """
        Word embedding.
        """
        output_data = torch.nn.functional.embedding(
                inputs.data,
                self.word_embedding_weight,
                padding_idx=0,
        )
        return PackedSequence(output_data, inputs.batch_sizes)

    def exec_forward_lstm(
            self,
            inputs: PackedSequence,
    ) -> List[PackedSequence]:
        """
        Forward LSTM.
        """
        if self.exec_managed_lstm_bos_eos:
            max_batch_size = int(inputs.batch_sizes.data[0])
            # BOS.
            self.exec_forward_lstm_bos(max_batch_size)
        elif self.exec_managed_lstm_reset_states:
            self.forward_lstm.reset_states()

        # Feed inputs.
        outputs, _ = self.forward_lstm(inputs.data, inputs.batch_sizes)

        if self.exec_managed_lstm_bos_eos:
            # EOS.
            self.exec_forward_lstm_eos(max_batch_size)

        # To list of `PackedSequence`.
        return [PackedSequence(output, inputs.batch_sizes) for output in outputs]

    def exec_backward_lstm(
            self,
            inputs: PackedSequence,
    ) -> List[PackedSequence]:
        """
        Backward LSTM.
        """
        if self.exec_managed_lstm_bos_eos:
            max_batch_size = int(inputs.batch_sizes.data[0])
            # EOS.
            self.exec_backward_lstm_eos(max_batch_size)
        elif self.exec_managed_lstm_reset_states:
            self.backward_lstm.reset_states()

        # Feed inputs.
        outputs, _ = self.backward_lstm(inputs.data, inputs.batch_sizes)

        if self.exec_managed_lstm_bos_eos:
            # BOS.
            self.exec_backward_lstm_bos(max_batch_size)

        # To list of `PackedSequence`.
        return [PackedSequence(output, inputs.batch_sizes) for output in outputs]

    def exec_bilstm(
            self,
            inputs: PackedSequence,
    ) -> List[Tuple[PackedSequence, PackedSequence]]:
        """
        BiLSTM.
        """
        forward_seqs = self.exec_forward_lstm(inputs)
        backward_seqs = self.exec_backward_lstm(inputs)

        return list(zip(forward_seqs, backward_seqs))

    def concat_packed_sequences(
            self,
            packed_sequences: List[Tuple[PackedSequence, PackedSequence]],
    ) -> List[PackedSequence]:
        """
        Concatenate the outputs of fwd/bwd lstms.
        """
        return [
                PackedSequence(
                        torch.cat([fwd.data, bwd.data], dim=-1),
                        fwd.batch_sizes,
                ) for fwd, bwd in packed_sequences
        ]

    def combine_char_cnn_and_bilstm_outputs(
            self,
            char_cnn_packed: PackedSequence,
            bilstm_packed: List[PackedSequence],
    ) -> List[PackedSequence]:
        """
        Combine the outputs of Char CNN & BiLSTM for scalar mix.
        """
        # Simply duplicate the output of char cnn.
        duplicated_char_cnn_packed = PackedSequence(
                torch.cat([char_cnn_packed.data, char_cnn_packed.data], dim=-1),
                char_cnn_packed.batch_sizes,
        )

        combined = [duplicated_char_cnn_packed]
        combined.extend(bilstm_packed)
        return combined

    def exec_scalar_mix(self, packed_sequences: List[PackedSequence]) -> List[PackedSequence]:
        """
        Scalar Mix.
        """
        reprs = []
        for scalar_mix in self.scalar_mixes:
            mixed = scalar_mix([inputs.data for inputs in packed_sequences])
            if self.repr_dropout is not None:
                mixed = self.repr_dropout(mixed)
            reprs.append(PackedSequence(mixed, packed_sequences[0].batch_sizes))
        return reprs

    def exec_bilstm_and_scalar_mix(
            self,
            token_repr: PackedSequence,
    ) -> List[PackedSequence]:
        """
        Common combination.
        """
        # BiLSTM.
        bilstm_repr = self.exec_bilstm(token_repr)
        # Scalar Mix.
        conbimed_repr = self.combine_char_cnn_and_bilstm_outputs(
                token_repr,
                self.concat_packed_sequences(bilstm_repr),
        )
        mixed_reprs = self.exec_scalar_mix(conbimed_repr)
        return mixed_reprs

    def pack_inputs(
            self,
            inputs: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
    ) -> PackedSequence:
        return utils.pack_inputs(inputs, lengths=lengths)

    def unpack_outputs(
            self,
            inputs: PackedSequence,
    ) -> torch.Tensor:
        return utils.unpack_outputs(inputs)

    def unpack_mixed_reprs(
            self,
            mixed_reprs: List[PackedSequence],
    ) -> List[torch.Tensor]:
        """
        Unpack the outputs of scalar mixtures.
        """
        return [self.unpack_outputs(mixed_repr) for mixed_repr in mixed_reprs]

    def to_allennlp_elmo_output_format(
            self,
            unpacks: List[torch.Tensor],
            mask: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        return {'elmo_representations': unpacks, 'mask': mask}

    def preprocess_inputs(
            self,
            inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        lengths = utils.get_lengths_of_zero_padded_batch(inputs)
        original_lengths = lengths
        restoration_index: Optional[torch.Tensor] = None

        if self.exec_sort_batch:
            inputs, permutation_index, restoration_index = \
                    utils.sort_batch_by_length(inputs, lengths)
            lengths = lengths.index_select(0, permutation_index)
            self.exec_bilstm_permutate_states(permutation_index)

        return inputs, lengths, original_lengths, restoration_index

    def postprocess_outputs(
            self,
            unpacked_mixed_reprs: List[torch.Tensor],
            restoration_index: Optional[torch.Tensor],
            inputs: torch.Tensor,
            original_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = utils.generate_mask_from_lengths(
                inputs.shape[0],
                inputs.shape[1],
                original_lengths,
        )
        if self.exec_sort_batch:
            assert restoration_index is not None
            unpacked_mixed_reprs = [
                    tensor.index_select(0, restoration_index) for tensor in unpacked_mixed_reprs
            ]
            self.exec_bilstm_permutate_states(restoration_index)

        return unpacked_mixed_reprs, mask

    def forward(self):  # type: ignore
        raise NotImplementedError()


class FastElmo(FastElmoBase):

    def __init__(
            self,
            options_file: Optional[str],
            weight_file: str,
            **kwargs: Any,
    ) -> None:
        _raise_if_kwargs_is_invalid(
                self.COMMON_PARAMS | {
                        'char_cnn_requires_grad',
                        'forward_lstm_requires_grad',
                        'backward_lstm_requires_grad',
                }, kwargs)
        super().__init__(options_file, weight_file, **kwargs)

    def forward(  # type: ignore
            self,
            inputs: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        The default workflow (same as AllenNLP).

        `inputs` of shape `(batch_size, max_timesteps, max_characters_per_token)
        """
        inputs, lengths, original_lengths, restoration_index = \
                self.preprocess_inputs(inputs)

        packed_inputs = self.pack_inputs(inputs, lengths)
        token_repr = self.exec_char_cnn(packed_inputs)
        mixed_reprs = self.exec_bilstm_and_scalar_mix(token_repr)
        unpacked_mixed_reprs = self.unpack_mixed_reprs(mixed_reprs)

        unpacked_mixed_reprs, mask = self.postprocess_outputs(
                unpacked_mixed_reprs,
                restoration_index,
                inputs,
                original_lengths,
        )
        return self.to_allennlp_elmo_output_format(unpacked_mixed_reprs, mask)


class FastElmoWordEmbedding(FastElmoBase):

    def __init__(
            self,
            options_file: Optional[str],
            weight_file: str,
            **kwargs: Any,
    ) -> None:
        _raise_if_kwargs_is_invalid(
                self.COMMON_PARAMS | {
                        'word_embedding_weight_file',
                        'word_embedding_requires_grad',
                        'forward_lstm_requires_grad',
                        'backward_lstm_requires_grad',
                }, kwargs)

        kwargs['disable_char_cnn'] = True
        kwargs['disable_word_embedding'] = False
        super().__init__(options_file, weight_file, **kwargs)

    def forward(  # type: ignore
            self,
            inputs: torch.Tensor,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        `inputs` of shape `(batch_size, max_timesteps)
        """
        inputs, lengths, original_lengths, restoration_index = \
                self.preprocess_inputs(inputs)

        packed_inputs = self.pack_inputs(inputs, lengths)
        token_repr = self.exec_word_embedding(packed_inputs)
        mixed_reprs = self.exec_bilstm_and_scalar_mix(token_repr)
        unpacked_mixed_reprs = self.unpack_mixed_reprs(mixed_reprs)

        unpacked_mixed_reprs, mask = self.postprocess_outputs(
                unpacked_mixed_reprs,
                restoration_index,
                inputs,
                original_lengths,
        )
        return self.to_allennlp_elmo_output_format(unpacked_mixed_reprs, mask)
