from typing import Optional, List, Any
import json
import logging
import itertools

import torch
import numpy as np

from pytorch_fast_elmo import (
        batch_to_char_ids,
        load_and_build_vocab2id,
        batch_to_word_ids,
        FastElmoForwardVocabDistrib,
        FastElmoBackwardVocabDistrib,
        FastElmoWordEmbeddingForwardVocabDistrib,
        FastElmoWordEmbeddingBackwardVocabDistrib,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def sample_sentence(
        options_file: str,
        weight_file: str,
        vocab_txt: str,
        output_json: str,
        enable_trace: bool,
        no_char_cnn: bool,
        char_cnn_maxlen: int,
        go_forward: bool,
        next_token_top_k: int,
        sample_size: int,
        sample_constrain_txt: Optional[str],
        warm_up_txt: Optional[str],
        cuda_device: int,
) -> None:
    if no_char_cnn:
        if go_forward:
            fast_elmo_cls = FastElmoWordEmbeddingForwardVocabDistrib
        else:
            fast_elmo_cls = FastElmoWordEmbeddingBackwardVocabDistrib
    else:
        if go_forward:
            fast_elmo_cls = FastElmoForwardVocabDistrib
        else:
            fast_elmo_cls = FastElmoBackwardVocabDistrib

    vocab2id = load_and_build_vocab2id(vocab_txt)
    id2vocab = {token_id: token for token, token_id in vocab2id.items()}

    if no_char_cnn:

        def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
            tensor = batch_to_word_ids(batch, vocab2id)
            if cuda_device >= 0:
                tensor = tensor.cuda(cuda_device)
            return tensor
    else:

        def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
            if char_cnn_maxlen == 0:
                tensor = batch_to_char_ids(batch)
            else:
                tensor = batch_to_char_ids(batch, char_cnn_maxlen)
            if cuda_device >= 0:
                tensor = tensor.cuda(cuda_device)
            return tensor

    elmo = fast_elmo_cls(options_file, weight_file)
    if cuda_device >= 0:
        elmo = elmo.cuda(cuda_device)  # type: ignore

    # Warm up.
    if warm_up_txt:
        sentences_token_ids = []
        with open(warm_up_txt) as fin:
            for line in fin:
                sent = line.split()
                if not sent:
                    continue
                token_ids = batch_to_ids([sent])
                sentences_token_ids.append(token_ids)

        for token_ids in sentences_token_ids:
            elmo(token_ids)

    # Manually deal with BOS/EOS.
    elmo.exec_managed_lstm_bos_eos = False

    if sample_constrain_txt:
        with open(sample_constrain_txt) as fin:
            lines = fin.readlines()
            if not lines:
                raise ValueError('No content in sample_constrain_txt.')
            if len(lines) > 1:
                logging.warning('Multiple lines in sample_constrain_txt, only use the 1st line.')
            sample_constrain_tokens = lines[0].split()
        if not go_forward:
            sample_constrain_tokens.reverse()

    infos: List[Any] = []
    for _ in range(sample_size):
        if go_forward:
            cur_token, end_token = '<S>', '</S>'
        else:
            cur_token, end_token = '</S>', '<S>'

        if sample_constrain_txt:
            for token in itertools.chain([cur_token], sample_constrain_tokens[:-1]):
                elmo(batch_to_ids([[token]]))
            cur_token = sample_constrain_tokens[-1]

        info: List[Any] = []
        while cur_token != end_token:
            batched = batch_to_ids([[cur_token]])
            output, _ = elmo(batched)
            if cuda_device >= 0:
                output = output.cpu()

            probs, indices = torch.topk(output.view(-1), next_token_top_k)
            probs = probs.numpy()
            indices = indices.numpy()

            next_token_id = np.random.choice(indices, p=probs / probs.sum()) + 1
            next_token = id2vocab[next_token_id]

            info_probs = sorted(
                    dict(zip(map(id2vocab.get, indices + 1), probs.tolist())).items(),
                    key=lambda p: p[1],
                    reverse=True,
            )
            info.append({
                    'cur': cur_token,
                    'next': next_token,
                    'probs': info_probs,
            })

            cur_token = next_token

        # Ending.
        elmo(batch_to_ids([[end_token]]))
        # Save info.
        infos.append({'text': ''.join(step['cur'] for step in info)})
        if enable_trace:
            infos[-1]['trace'] = info
        if sample_constrain_txt:
            infos[-1]['text'] = ''.join(sample_constrain_tokens[:-1]) + infos[-1]['text']
        if not go_forward:
            infos[-1]['text'] = infos[-1]['text'][::-1]

    # Output to JSON.
    with open(output_json, 'w') as fout:
        json.dump(infos, fout, ensure_ascii=False, indent=2)
