from os.path import dirname, join

import torch
from allennlp.modules.elmo import batch_to_ids
import numpy as np

from pytorch_fast_elmo import utils
from pytorch_fast_elmo import FastElmoBase

FIXTURES_FODLER = join(dirname(__file__), 'fixtures')
ELMO_OPTIONS_FILE = join(FIXTURES_FODLER, 'options.json')
ELMO_WEIGHT_FILE = join(FIXTURES_FODLER, 'lm_weights.hdf5')


def test_batch_to_char_ids():
    sentences = [
            ["This", "is", "a", "sentence"],
            ["Here", "'s", "one"],
            ["Another", "one"],
    ]
    t1 = utils.batch_to_char_ids(sentences)
    t2 = batch_to_ids(sentences)
    np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    sentences = [["one"]]
    t1 = utils.batch_to_char_ids(sentences)
    t2 = batch_to_ids(sentences)
    np.testing.assert_array_equal(t1.numpy(), t2.numpy())


def test_cache_char_cnn_vocab(tmpdir):
    vocab = ['<S>', '</S>', '<UNK>', 'ELMo', 'helps', 'disambiguate', 'ELMo', 'from', 'Elmo', '.']
    vocab_path = tmpdir.join("vocab.txt")
    vocab_path.write('\n'.join(vocab))

    embedding_path = tmpdir.join("embd.txt")

    utils.cache_char_cnn_vocab(
            vocab_path.realpath(),
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            str(embedding_path.realpath()),
            batch_size=2,
            max_characters_per_token=15,
    )

    fast_word_embd = FastElmoBase(
            ELMO_OPTIONS_FILE,
            None,
            disable_word_embedding=False,
            word_embedding_weight_file=str(embedding_path.realpath()),
            # Disable all other components.
            disable_char_cnn=True,
            disable_forward_lstm=True,
            disable_backward_lstm=True,
            disable_scalar_mix=True,
    )
    fast_char_cnn = FastElmoBase(
            ELMO_OPTIONS_FILE,
            ELMO_WEIGHT_FILE,
            # Disable all other components.
            disable_forward_lstm=True,
            disable_backward_lstm=True,
            disable_scalar_mix=True,
    )

    embd_repr = fast_word_embd.exec_word_embedding(
            fast_word_embd.pack_inputs(
                    utils.batch_to_word_ids(
                            [['ELMo', 'helps', '!!!UNK!!!']],
                            utils.load_and_build_vocab2id(vocab_path.realpath()),
                    )))
    char_cnn_repr = fast_char_cnn.exec_char_cnn(
            fast_char_cnn.pack_inputs(
                    utils.batch_to_char_ids(
                            [['ELMo', 'helps', '<UNK>']],
                            max_characters_per_token=15,
                    )))

    np.testing.assert_array_almost_equal(
            embd_repr.data.numpy(),
            char_cnn_repr.data.numpy(),
    )
    np.testing.assert_array_equal(embd_repr.batch_sizes.numpy(), char_cnn_repr.batch_sizes.numpy())


def test_sort_batch_by_length():
    for _ in range(100):
        batch = torch.randn(20, 40)
        lengths = torch.randint(0, 41, (20,), dtype=torch.long)
        mask = utils.generate_mask_from_lengths(20, 40, lengths)
        batch[mask != 1] = 0

        sorted_batch, _, restoration_index = utils.sort_batch_by_length(batch)
        np.testing.assert_array_equal(
                sorted_batch.index_select(0, restoration_index),
                batch,
        )
