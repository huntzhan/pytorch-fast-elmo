=================
pytorch-fast-elmo
=================


.. image:: https://img.shields.io/pypi/v/pytorch_fast_elmo.svg
        :target: https://pypi.python.org/pypi/pytorch_fast_elmo

.. image:: https://img.shields.io/travis/cnt-dev/pytorch-fast-elmo.svg
        :target: https://travis-ci.org/cnt-dev/pytorch-fast-elmo

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
        :target: https://travis-ci.org/cnt-dev/pytorch-fast-elmo


Introduction
------------

A fast ELMo implementation with features:

- **Lower execution overhead.** The core components are reimplemented in Libtorch in order to reduce the Python execution overhead (**45%** speedup).
- **A more flexible design.** By redesigning the workflow, the user could extend or change the ELMo behavior easily.

Benchmark
---------

Hardware:

- CPU: i7-7800X
- GPU: 1080Ti

Options:

- Batch size: 32
- Warm up iterations: 20
- Test iterations: 1000
- Word length: [1, 20]
- Sentence length: [1, 30]
- Random seed: 10000

+--------------------------------------+------------------------+------------------------+
| Item                                 | Mean Of Durations (ms) | cumtime(synchronize)%  |
+======================================+========================+========================+
| Fast ELMo (CUDA, no synchronize)     | **31**                 | N/A                    |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CUDA, no synchronize) | 56                     | N/A                    |
+--------------------------------------+------------------------+------------------------+
| Fast ELMo (CUDA, synchronize)        | 47                     | **26.13%**             |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CUDA, synchronize)    | 57                     | 0.02%                  |
+--------------------------------------+------------------------+------------------------+
| Fast ELMo (CPU)                      | 1277                   | N/A                    |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CPU)                  | 1453                   | N/A                    |
+--------------------------------------+------------------------+------------------------+

Usage
-----

Please install **torch==1.0.0** first. Then, simply run this command to install.

.. code-block:: bash

    pip install pytorch-fast-elmo


``FastElmo`` should have the same behavior as AllenNLP's ``ELMo``.

.. code-block:: python

    from pytorch_fast_elmo import FastElmo, batch_to_char_ids

    options_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    elmo = FastElmo(options_file, weight_file)

    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)


Use ``FastElmoWordEmbedding`` if you have disabled ``char_cnn`` in ``bilm-tf``, or have exported the Char CNN representation to a weight file.

.. code-block:: python

    from pytorch_fast_elmo import FastElmoWordEmbedding, load_and_build_vocab2id, batch_to_word_ids

    options_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = '/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    vocab_file = '/path/to/vocab.txt'
    embedding_file = '/path/to/cached_elmo_embedding.hdf5'

    elmo = FastElmoWordEmbedding(
            options_file,
            weight_file,
            # Could be omitted if the embedding weight is in `weight_file`.
            word_embedding_weight_file=embedding_file,
    )
    vocab2id = load_and_build_vocab2id(vocab_file)

    sentences = [['First', 'sentence', '.'], ['Another', '.']]
    word_ids = batch_to_word_ids(sentences, vocab2id)

    embeddings = elmo(word_ids)


CLI commands:

.. code-block:: bash

    # Cache the Char CNN representation.
    fast-elmo cache-char-cnn ./vocab.txt ./options.json ./lm_weights.hdf5 ./lm_embd.hdf5

    # Export word embedding.
    fast-elmo export-word-embd ./vocab.txt ./no-char-cnn.hdf5 ./embd.txt


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
