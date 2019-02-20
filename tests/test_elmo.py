from os.path import dirname, join

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import numpy as np
import random

from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.dataset import Batch
from allennlp.nn.util import add_sentence_boundary_token_ids, remove_sentence_boundaries
from allennlp.data.token_indexers.elmo_indexer import (
        ELMoCharacterMapper,
        ELMoTokenCharactersIndexer,
)
from allennlp.modules.elmo import _ElmoCharacterEncoder, _ElmoBiLm, Elmo

from pytorch_fast_elmo import (
        ElmoCharacterEncoder,
        ElmoCharacterEncoderFactory,
        ElmoLstmFactory,
        FastElmo,
        FastElmoWordEmbedding,
        utils,
)

FIXTURES_FODLER = join(dirname(__file__), 'fixtures')
ELMO_OPTIONS_FILE = join(FIXTURES_FODLER, 'options.json')
ELMO_WEIGHT_FILE = join(FIXTURES_FODLER, 'lm_weights.hdf5')

CACHE_VOCAB_FILE = join(FIXTURES_FODLER, 'vocab.txt')
CACHE_EMBD_FILE = join(FIXTURES_FODLER, 'lm_embd.txt')


def test_elmo_character_encoder_simple():
    embedder = ElmoCharacterEncoder(
            char_embedding_cnt=10,
            char_embedding_dim=3,
            filters=[
                    [1, 5],
                    [2, 4],
            ],
            activation='relu',
            num_highway_layers=2,
            output_dim=3,
    )
    inputs = torch.randint(0, 10, (20, 10))
    output = embedder(inputs)
    assert list(output.size()) == [20, 3]


def _sentences_to_ids(sentences):
    indexer = ELMoTokenCharactersIndexer()

    # For each sentence, first create a TextField, then create an instance
    instances = []
    for sentence in sentences:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({'elmo': field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']


def _unpack(tensor, batch_sizes):
    tensor, _ = pad_packed_sequence(
            PackedSequence(tensor, batch_sizes),
            batch_first=True,
    )
    return tensor


def test_elmo_character_encoder_with_allennlp():
    allennlp_embedder = _ElmoCharacterEncoder(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    )
    embedder = ElmoCharacterEncoderFactory(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    ).create()

    allennlp_parameters = [
            '_char_embedding_weights',
            'char_conv_0.bias',
            'char_conv_0.weight',
            'char_conv_1.bias',
            'char_conv_1.weight',
            'char_conv_2.bias',
            'char_conv_2.weight',
            'char_conv_3.bias',
            'char_conv_3.weight',
            'char_conv_4.bias',
            'char_conv_4.weight',
            '_projection.bias',
            '_projection.weight',
    ]
    embedder_parameters = [
            'char_embedding.weight',
            'char_conv_0.bias',
            'char_conv_0.weight',
            'char_conv_1.bias',
            'char_conv_1.weight',
            'char_conv_2.bias',
            'char_conv_2.weight',
            'char_conv_3.bias',
            'char_conv_3.weight',
            'char_conv_4.bias',
            'char_conv_4.weight',
            'output_proj.bias',
            'output_proj.weight',
    ]
    allennlp_parameters_diff = [
            '_highways._layers.0.bias',
            '_highways._layers.0.weight',
            '_highways._layers.1.bias',
            '_highways._layers.1.weight',
    ]
    embedder_parameters_diff = [
            'highway.layers_0.bias',
            'highway.layers_0.weight',
            'highway.layers_1.bias',
            'highway.layers_1.weight',
    ]
    assert len(allennlp_parameters) == len(embedder_parameters)
    assert len(allennlp_parameters_diff) == len(embedder_parameters_diff)

    allennlp_embedder_named_parameters = dict(allennlp_embedder.named_parameters())
    # Same.
    for allennlp_param, embedder_param in zip(allennlp_parameters, embedder_parameters):
        allennlp_w = allennlp_embedder_named_parameters[allennlp_param].data
        embedder_w = embedder.named_parameters()[embedder_param].data

        np.testing.assert_array_equal(embedder_w.numpy(), allennlp_w.numpy())
        assert embedder_w.dtype == allennlp_w.dtype
    # Diff on highway.
    for allennlp_param, embedder_param in zip(allennlp_parameters_diff, embedder_parameters_diff):
        allennlp_w = allennlp_embedder_named_parameters[allennlp_param].data
        embedder_w = embedder.named_parameters()[embedder_param].data

        assert embedder_w.dtype == allennlp_w.dtype
        np.testing.assert_raises(
                AssertionError,
                np.testing.assert_array_equal,
                embedder_w.numpy(),
                allennlp_w.numpy(),
        )

    sentences = [
            ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.'],
            ['The', 'sentence', '.'],
    ]
    # `(2, 7, 50)`
    character_ids = _sentences_to_ids(sentences)

    # AllenNLP.
    out = allennlp_embedder(character_ids)
    allennlp_token_embedding, _ = remove_sentence_boundaries(out['token_embedding'], out['mask'])
    assert list(allennlp_token_embedding.shape) == [2, 7, 16]

    # Ours.
    inputs = pack_padded_sequence(character_ids, [7, 3], batch_first=True)
    out = embedder(inputs.data)
    ours_token_embedding = _unpack(out, inputs.batch_sizes)
    assert list(ours_token_embedding.shape) == [2, 7, 16]

    np.testing.assert_array_almost_equal(
            ours_token_embedding.data.numpy(),
            allennlp_token_embedding.data.numpy(),
    )


def test_elmo_lstm_factory_simple():
    allennlp_elmo_bilm = _ElmoBiLm(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    )

    embedder = ElmoCharacterEncoderFactory(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    ).create()
    fwd_lstm, bwd_lstm = ElmoLstmFactory(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    ).create(
            enable_forward=True, enable_backward=True)

    sentences_1 = [
            ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.'],
            ['The', 'sentence', '.'],
    ]
    sentences_2 = [
            ["This", "is", "a", "sentence"],
            ["Here", "'s", "one"],
            ["Another", "one"],
    ]

    # Internal states should be updated.
    for sentences in [sentences_1, sentences_2] * 10:
        # `(2, 7, 50)`
        character_ids = _sentences_to_ids(sentences)

        # AllenNLP.
        allennlp_out = allennlp_elmo_bilm(character_ids)

        # Ours.
        inputs = character_ids
        _beginning_of_sentence_characters = torch.from_numpy(
                np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1)
        _end_of_sentence_characters = torch.from_numpy(
                np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1)
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
                inputs,
                mask,
                _beginning_of_sentence_characters,
                _end_of_sentence_characters,
        )
        # Pack input.
        lengths = mask_with_bos_eos.sum(dim=-1)
        inputs = pack_padded_sequence(character_ids_with_bos_eos, lengths, batch_first=True)
        char_repr = embedder(inputs.data)
        fwd_lstm_hiddens, _ = fwd_lstm(char_repr, inputs.batch_sizes)
        bwd_lstm_hiddens, _ = bwd_lstm(char_repr, inputs.batch_sizes)
        lstm_hiddens = [
                torch.cat([fwd, bwd], dim=-1)
                for fwd, bwd in zip(fwd_lstm_hiddens, bwd_lstm_hiddens)
        ]
        # Unpack output.
        char_repr = _unpack(char_repr, inputs.batch_sizes)
        duplicated_char_repr = torch.cat(
                [char_repr, char_repr],
                dim=-1,
        ) * mask_with_bos_eos.float().unsqueeze(-1)
        lstm_hiddens = [_unpack(hx, inputs.batch_sizes) for hx in lstm_hiddens]

        # TODO: Investigate the numerical stability issue.
        # np.testing.assert_array_almost_equal(
        #         duplicated_char_repr.data.numpy(),
        #         allennlp_out['activations'][0].data.numpy(),
        # )
        # np.testing.assert_array_almost_equal(
        #         lstm_hiddens[0].data.numpy(),
        #         allennlp_out['activations'][1].data.numpy(),
        # )
        np.testing.assert_array_almost_equal(
                lstm_hiddens[1].data.numpy(),
                allennlp_out['activations'][2].data.numpy(),
        )


def test_fast_elmo_with_allennlp():
    fast = FastElmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            num_output_representations=2,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
    )

    allennlp = Elmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            num_output_representations=2,
            dropout=0.0,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
    )

    sentences_1 = [
            ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.'],
            ['The', 'sentence', '.'],
    ]
    sentences_2 = [
            ["This", "is", "a", "sentence"],
            ["Here", "'s", "one"],
            ["Another", "one"],
    ]

    for sentences in [sentences_1, sentences_2] * 10:
        random.shuffle(sentences)
        character_ids = _sentences_to_ids(sentences)

        fast_out = fast(character_ids)
        allennlp_out = allennlp(character_ids)

        for repr_idx in range(2):
            fast_mixed_repr = fast_out['elmo_representations'][repr_idx]
            allennlp_mixed_repr = allennlp_out['elmo_representations'][repr_idx]
            np.testing.assert_array_almost_equal(
                    fast_mixed_repr,
                    allennlp_mixed_repr,
            )

        np.testing.assert_array_equal(
                fast_out['mask'],
                allennlp_out['mask'],
        )


def test_fast_elmo_with_allennlp_do_layer_norm():
    fast = FastElmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            num_output_representations=1,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
            do_layer_norm=True,
    )

    allennlp = Elmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            num_output_representations=1,
            dropout=0.0,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
            do_layer_norm=True,
    )

    sentences = [
            ['ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.'],
            ['The', 'sentence', '.'],
    ]
    character_ids = _sentences_to_ids(sentences)

    fast_out = fast(character_ids)
    allennlp_out = allennlp(character_ids)

    # Since we don't include the BOS/EOS reprs during layer normalization,
    # the result will be different from AllenNLP's implementation.
    np.testing.assert_raises(
            AssertionError,
            np.testing.assert_array_almost_equal,
            fast_out['elmo_representations'][0],
            allennlp_out['elmo_representations'][0],
    )

    # We can pack BOS/EOS to inputs manually
    _beginning_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1)
    _end_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1)

    mask = ((character_ids > 0).long().sum(dim=-1) > 0).long()
    character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
            character_ids,
            mask,
            _beginning_of_sentence_characters,
            _end_of_sentence_characters,
    )

    # And disable the mock BOS/EOS actions in FastElmo.
    fast.exec_managed_lstm_bos_eos = False
    fast_out_2 = fast(character_ids_with_bos_eos)
    fast_mixed_repr_2, _ = remove_sentence_boundaries(
            fast_out_2['elmo_representations'][0],
            fast_out_2['mask'],
    )

    allennlp_out_2 = allennlp(character_ids)

    np.testing.assert_array_almost_equal(
            fast_mixed_repr_2,
            allennlp_out_2['elmo_representations'][0],
    )


def test_fast_elmo_save_and_load():
    fast_1 = FastElmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    )

    # Change weight and save.
    fast_1.cpp_ext_scalar_mix_0_scalar_0.data.fill_(42.)
    fast_1_state_dict = fast_1.state_dict()

    # Load.
    fast_2 = FastElmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    )
    fast_2.load_state_dict(fast_1_state_dict)

    assert float(fast_2.cpp_ext_scalar_mix_0_scalar_0) == 42.
    assert float(fast_2.scalar_mixes[0].named_parameters()['scalar_0']) == 42.


def test_fast_elmo_word_embedding():
    vocab = utils.load_vocab(CACHE_VOCAB_FILE)

    fast_char_cnn = FastElmo(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
    )

    fast_word_embd = FastElmoWordEmbedding(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            word_embedding_weight_file=CACHE_EMBD_FILE,
    )

    # Test BOS/EOS & other words.
    words = vocab

    embd_repr = fast_word_embd(utils.batch_to_word_ids([words], utils.build_vocab2id(vocab)))
    char_cnn_repr = fast_char_cnn(utils.batch_to_char_ids([words]))

    np.testing.assert_array_almost_equal(
            embd_repr['elmo_representations'][0].data.numpy(),
            char_cnn_repr['elmo_representations'][0].data.numpy(),
    )
