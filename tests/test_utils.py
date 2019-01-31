from allennlp.modules.elmo import batch_to_ids
import numpy as np

from pytorch_fast_elmo.utils import batch_to_char_ids


def test_batch_to_char_ids():
    sentences = [
            ["This", "is", "a", "sentence"],
            ["Here", "'s", "one"],
            ["Another", "one"],
    ]
    t1 = batch_to_char_ids(sentences)
    t2 = batch_to_ids(sentences)
    np.testing.assert_array_equal(t1.numpy(), t2.numpy())

    sentences = [["one"]]
    t1 = batch_to_char_ids(sentences)
    t2 = batch_to_ids(sentences)
    np.testing.assert_array_equal(t1.numpy(), t2.numpy())
