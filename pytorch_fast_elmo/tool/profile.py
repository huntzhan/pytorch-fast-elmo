from typing import List, Any, Tuple
import random
import string
import time
import statistics

import torch
from pytorch_fast_elmo import batch_to_char_ids, FastElmo


class SentenceGenerator:

    def __init__(
            self,
            word_min: int,
            word_max: int,
            sent_min: int,
            sent_max: int,
    ) -> None:
        self.word_min = word_min
        self.word_max = word_max
        self.sent_min = sent_min
        self.sent_max = sent_max

    def generate_sentence(self) -> List[str]:
        return [
                ''.join(
                        random.choices(
                                string.ascii_lowercase,
                                k=random.randint(self.word_min, self.word_max),
                        )) for _ in range(random.randint(self.sent_min, self.sent_max))
        ]

    def generate_batch(self, batch_size: int) -> List[List[str]]:
        return [self.generate_sentence() for _ in range(batch_size)]


def load_fast_elmo(
        options_file: str,
        weight_file: str,
) -> FastElmo:
    return FastElmo(
            options_file,
            weight_file,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
    )


def load_allennlp_elmo(
        options_file: str,
        weight_file: str,
) -> Any:
    from allennlp.modules.elmo import Elmo
    return Elmo(
            options_file,
            weight_file,
            num_output_representations=1,
            dropout=0.0,
            scalar_mix_parameters=[1.0, 1.0, 1.0],
    )


def profile_full_elmo(
        mode: str,
        options_file: str,
        weight_file: str,
        cuda_device: int,
        cuda_synchronize: bool,
        batch_size: int,
        warmup_size: int,
        iteration_size: int,
        word_min: int,
        word_max: int,
        sent_min: int,
        sent_max: int,
        random_seed: int,
) -> Tuple[float, float, float]:
    random.seed(random_seed)

    module: Any = None
    if mode == 'fast-elmo':
        module = load_fast_elmo(options_file, weight_file)
    elif mode == 'allennlp-elmo':
        module = load_allennlp_elmo(options_file, weight_file)
    else:
        raise ValueError('invalid mode')

    sent_gen = SentenceGenerator(
            word_min,
            word_max,
            sent_min,
            sent_max,
    )

    if cuda_device >= 0:
        module = module.cuda(cuda_device)

    durations: List[float] = []

    for idx in range(warmup_size + iteration_size):
        batch = sent_gen.generate_batch(batch_size)
        char_ids = batch_to_char_ids(batch)

        if cuda_device >= 0:
            char_ids = char_ids.cuda(cuda_device)
            if cuda_synchronize:
                torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            module(char_ids)
        if cuda_device >= 0 and cuda_synchronize:
            torch.cuda.synchronize()
        end = time.time()

        if idx >= warmup_size:
            durations.append(end - start)

    mean = statistics.mean(durations)
    median = statistics.median(durations)
    stdev = statistics.stdev(durations)

    return mean, median, stdev
