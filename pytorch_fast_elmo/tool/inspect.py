from typing import Optional, List, Any
import json

import torch
import numpy as np

from pytorch_fast_elmo import (
        # batch_to_char_ids,
        load_and_build_vocab2id,
        batch_to_word_ids,
        # FastElmoForwardVocabDistrib,
        # FastElmoBackwardVocabDistrib,
        FastElmoWordEmbeddingForwardVocabDistrib,
        FastElmoWordEmbeddingBackwardVocabDistrib,
)


def sample_sentence_no_char_cnn(
        options_file: str,
        weight_file: str,
        vocab_txt: str,
        output_json: str,
        go_forward: bool,
        next_token_top_k: int,
        sample_size: int,
        warm_up_txt: Optional[str],
        cuda_device: int,
) -> None:
    if go_forward:
        fast_elmo_cls = FastElmoWordEmbeddingForwardVocabDistrib
    else:
        fast_elmo_cls = FastElmoWordEmbeddingBackwardVocabDistrib

    vocab2id = load_and_build_vocab2id(vocab_txt)
    elmo = fast_elmo_cls(options_file, weight_file)
    if cuda_device >= 0:
        elmo = elmo.cuda(cuda_device)  # type: ignore

    # Warm up.
    if warm_up_txt:
        sentences_word_ids = []
        with open(warm_up_txt) as fin:
            for line in fin:
                sent = line.split()
                if not sent:
                    continue
                word_ids = batch_to_word_ids([sent], vocab2id)
                if cuda_device >= 0:
                    word_ids = word_ids.cuda(cuda_device)
                sentences_word_ids.append(word_ids)

        for word_ids in sentences_word_ids:
            elmo(word_ids)

    # Manually deal with BOS/EOS.
    elmo.exec_managed_lstm_bos_eos = False

    id2vocab = {token_id: token for token, token_id in vocab2id.items()}

    infos: List[Any] = []
    for _ in range(sample_size):
        if go_forward:
            cur_token, end_token = '<S>', '</S>'
        else:
            cur_token, end_token = '</S>', '<S>'

        info: List[Any] = []
        while cur_token != end_token:
            batched = batch_to_word_ids([[cur_token]], vocab2id)
            output, _ = elmo(batched)
            if cuda_device >= 0:
                output = output.cpu()

            probs, indices = torch.topk(output.view(-1), next_token_top_k)
            probs = probs.numpy()
            indices = indices.numpy()

            next_token_id = np.random.choice(indices, p=probs / probs.sum()) + 1
            next_token = id2vocab[next_token_id]

            info.append({
                    'cur':
                    cur_token,
                    'next':
                    next_token,
                    'probs':
                    sorted(
                            dict(zip(map(id2vocab.get, indices + 1), probs.tolist())).items(),
                            key=lambda p: p[1],
                            reverse=True,
                    ),
            })

            cur_token = next_token

        # Ending.
        elmo(batch_to_word_ids([[end_token]], vocab2id))
        # Save info.
        infos.append({
                'text': ''.join(step['cur'] for step in info),
                'trace': info,
        })

    # Output to JSON.
    with open(output_json, 'w') as fout:
        json.dump(infos, fout, ensure_ascii=False, indent=2)
