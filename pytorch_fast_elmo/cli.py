# pylint: skip-file
import cProfile, pstats, io

import fire
from pytorch_fast_elmo import utils, profile


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

    def profile_full(  # type: ignore
            self,
            mode,
            options_file,
            weight_file,
            cuda=False,
            batch_size=32,
            iteration_size=1000,
            word_min=1,
            word_max=20,
            sent_min=1,
            sent_max=30,
            random_seed=10000,
            profiler=False,
            profiler_dump=None,
    ):
        if profiler:
            pr = cProfile.Profile()
            pr.enable()

        profile.profile_full_elmo(
                mode,
                options_file,
                weight_file,
                cuda,
                batch_size,
                iteration_size,
                word_min,
                word_max,
                sent_min,
                sent_max,
                random_seed,
        )

        if profiler:
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()
            with open(profiler_dump, 'w') as fout:
                fout.write(s.getvalue())


def main():  # type: ignore
    fire.Fire(Main)