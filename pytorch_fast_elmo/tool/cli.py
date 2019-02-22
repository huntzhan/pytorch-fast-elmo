# pylint: disable=no-self-use
import cProfile
import pstats
import io

import fire
from pytorch_fast_elmo import utils
from pytorch_fast_elmo.tool import profile, inspect


class Main:

    def cache_char_cnn(  # type: ignore
            self,
            vocab_txt,
            options_file,
            weight_file,
            txt_out,
            max_characters_per_token=utils.ElmoCharacterIdsConst.MAX_WORD_LENGTH,
            cuda_device=-1,
            batch_size=256,
    ):
        utils.cache_char_cnn_vocab(
                vocab_txt,
                options_file,
                weight_file,
                txt_out,
                max_characters_per_token,
                cuda_device,
                batch_size,
        )

    def export_word_embd(  # type: ignore
            self,
            vocab_txt,
            weight_file,
            txt_out,
    ):
        utils.export_word_embd(
                vocab_txt,
                weight_file,
                txt_out,
        )

    def profile_full(  # type: ignore
            self,
            mode,
            options_file,
            weight_file,
            cuda_device=-1,
            cuda_synchronize=False,
            batch_size=32,
            warmup_size=20,
            iteration_size=1000,
            word_min=1,
            word_max=20,
            sent_min=1,
            sent_max=30,
            random_seed=10000,
            profiler=False,
            output_file=None,
    ):
        sstream = io.StringIO()

        if profiler:
            cpr = cProfile.Profile()
            cpr.enable()

        mean, median, stdev = profile.profile_full_elmo(
                mode,
                options_file,
                weight_file,
                cuda_device,
                cuda_synchronize,
                batch_size,
                warmup_size,
                iteration_size,
                word_min,
                word_max,
                sent_min,
                sent_max,
                random_seed,
        )

        sstream.write(f'Finish {iteration_size} iterations.\n')
        sstream.write(f'Mode: {mode}\n')
        sstream.write(f'Duration Mean: {mean}\n')
        sstream.write(f'Duration Median: {median}\n')
        sstream.write(f'Duration Stdev: {stdev}\n\n')

        if profiler:
            cpr.disable()
            pstats.Stats(cpr, stream=sstream).sort_stats('cumulative').print_stats()

        if output_file:
            with open(output_file, 'w') as fout:
                fout.write(sstream.getvalue())
        else:
            print(sstream.getvalue())

    def sample_sentence(  # type: ignore
            self,
            options_file,
            weight_file,
            vocab_txt,
            output_json,
            enable_trace=False,
            go_forward=True,
            no_char_cnn=False,
            char_cnn_maxlen=0,
            next_token_top_k=5,
            sample_size=1,
            sample_constrain_txt=None,
            warm_up_txt=None,
            cuda_device=-1,
    ):
        inspect.sample_sentence(
                options_file,
                weight_file,
                vocab_txt,
                output_json,
                enable_trace,
                no_char_cnn,
                char_cnn_maxlen,
                go_forward,
                next_token_top_k,
                sample_size,
                sample_constrain_txt,
                warm_up_txt,
                cuda_device,
        )


def main():  # type: ignore
    fire.Fire(Main)
