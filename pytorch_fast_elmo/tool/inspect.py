from typing import Optional, Tuple, List, Dict, Any, Callable
import json
import logging
import itertools

import torch
import numpy as np
import h5py

from pytorch_fast_elmo import (
        batch_to_char_ids,
        load_and_build_vocab2id,
        batch_to_word_ids,
        FastElmoBase,
        FastElmo,
        FastElmoWordEmbedding,
        FastElmoPlainEncoder,
        FastElmoWordEmbeddingPlainEncoder,
        FastElmoForwardVocabDistrib,
        FastElmoBackwardVocabDistrib,
        FastElmoWordEmbeddingForwardVocabDistrib,
        FastElmoWordEmbeddingBackwardVocabDistrib,
)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _generate_vocab2id_id2vocab(vocab_txt: str,) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab2id = load_and_build_vocab2id(vocab_txt)
    id2vocab = {token_id: token for token, token_id in vocab2id.items()}
    return vocab2id, id2vocab


def _generate_batch_to_ids(
        vocab2id: Dict[str, int],
        char_cnn_maxlen: int,
        no_char_cnn: bool,
        cuda_device: int,
) -> Callable[[List[List[str]]], torch.Tensor]:
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

    return batch_to_ids


def _warm_up(
        warm_up_txt: str,
        batch_to_ids: Callable[[List[List[str]]], torch.Tensor],
        elmo: FastElmoBase,
) -> None:
    sentences_token_ids = []
    with open(warm_up_txt) as fin:
        for line in fin:
            sent = line.split()
            if not sent:
                continue
            token_ids = batch_to_ids([sent])
            sentences_token_ids.append(token_ids)

    for token_ids in sentences_token_ids:
        with torch.no_grad():
            elmo(token_ids)


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

    vocab2id, id2vocab = _generate_vocab2id_id2vocab(vocab_txt)
    batch_to_ids = _generate_batch_to_ids(
            vocab2id,
            char_cnn_maxlen,
            no_char_cnn,
            cuda_device,
    )

    elmo = fast_elmo_cls(options_file, weight_file)
    if cuda_device >= 0:
        elmo = elmo.cuda(cuda_device)

    # Warm up.
    if warm_up_txt:
        _warm_up(warm_up_txt, batch_to_ids, elmo)

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
                with torch.no_grad():
                    elmo(batch_to_ids([[token]]))
            cur_token = sample_constrain_tokens[-1]

        info: List[Any] = []
        while cur_token != end_token:
            batched = batch_to_ids([[cur_token]])
            with torch.no_grad():
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
        with torch.no_grad():
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


def encode_sentences(
        options_file: str,
        weight_file: str,
        vocab_txt: str,
        input_txt: str,
        output_hdf5: str,
        no_char_cnn: bool,
        char_cnn_maxlen: int,
        scalar_mix: Optional[Tuple[float]],
        warm_up_txt: Optional[str],
        cuda_device: int,
) -> None:
    if scalar_mix is None:
        if no_char_cnn:
            fast_elmo_cls = FastElmoWordEmbeddingPlainEncoder
        else:
            fast_elmo_cls = FastElmoPlainEncoder

        elmo = fast_elmo_cls(
                options_file,
                weight_file,
        )

    else:
        if no_char_cnn:
            fast_elmo_cls = FastElmoWordEmbedding
        else:
            fast_elmo_cls = FastElmo

        elmo = fast_elmo_cls(
                options_file,
                weight_file,
                scalar_mix_parameters=list(scalar_mix),
        )

    if cuda_device >= 0:
        elmo = elmo.cuda(cuda_device)

    vocab2id, _ = _generate_vocab2id_id2vocab(vocab_txt)
    batch_to_ids = _generate_batch_to_ids(
            vocab2id,
            char_cnn_maxlen,
            no_char_cnn,
            cuda_device,
    )

    # Warm up.
    if warm_up_txt:
        _warm_up(warm_up_txt, batch_to_ids, elmo)

    sentences: List[Tuple[int, List[str]]] = []
    with open(input_txt) as fin:
        for sentence_id, line in enumerate(fin):
            tokens = line.split()
            if not tokens:
                logger.warning('Ignore sentence_id = %s', sentence_id)
                continue
            sentences.append((sentence_id, tokens))

    with h5py.File(output_hdf5, 'w') as fout:
        for sentence_id, tokens in sentences:
            token_ids = batch_to_ids([tokens])

            if scalar_mix is None:
                with torch.no_grad():
                    layer_reprs, _ = elmo(token_ids)
                # (layers, timesteps, hidden_size)
                encoded = torch.cat(layer_reprs, dim=0)
            else:
                with torch.no_grad():
                    out = elmo(token_ids)
                # (1, timesteps, hidden_size)
                encoded = out['elmo_representations'][0]

            if cuda_device >= 0:
                encoded = encoded.cpu()

            fout.create_dataset(
                    str(sentence_id),
                    encoded.shape,
                    dtype='float32',
                    data=encoded.numpy(),
            )
