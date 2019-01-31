# pylint: disable=no-self-use

import fire
from pytorch_fast_elmo import utils


class Main:

    def cache_char_cnn(  # type: ignore
            self,
            vocab_txt,
            options_file,
            weight_file,
            hdf5_out,
            max_characters_per_token=utils.ElmoCharacterIdsConst.MAX_WORD_LENGTH,
            cuda=False,
            batch_size=256,
    ):
        utils.cache_char_cnn_vocab(
                vocab_txt,
                options_file,
                weight_file,
                hdf5_out,
                max_characters_per_token,
                cuda,
                batch_size,
        )


def main():  # type: ignore
    fire.Fire(Main)
