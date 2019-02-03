=================
pytorch-fast-elmo
=================


.. image:: https://img.shields.io/pypi/v/pytorch_fast_elmo.svg
        :target: https://pypi.python.org/pypi/pytorch_fast_elmo

.. image:: https://img.shields.io/travis/cnt-dev/pytorch-fast-elmo.svg
        :target: https://travis-ci.org/cnt-dev/pytorch-fast-elmo


* Free software: MIT license


Introduction
------------

A fast ELMo implementation with features:

- **Lower execution overhead.** The core components are reimplemented in Libtorch in order to reduce the Python execution overhead significantly (**~45%** speedup).
- **More flexible design.** By redesigning the workflow, the user could extend or change the ELMo model easily. We provide a word embedding ELMo extension for demonstration.

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
| Fast ELMo (CUDA, no synchronize)     | 31                     | N/A                    |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CUDA, no synchronize) | 56                     | N/A                    |
+--------------------------------------+------------------------+------------------------+
| Fast ELMo (CUDA, synchronize)        | 47                     | 26.13%                 |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CUDA, synchronize)    | 57                     | 0.02%                  |
+--------------------------------------+------------------------+------------------------+
| Fast ELMo (CPU)                      | 1277                   | N/A                    |
+--------------------------------------+------------------------+------------------------+
| AllenNLP ELMo (CPU)                  | 1453                   | N/A                    |
+--------------------------------------+------------------------+------------------------+

Usage
-----

TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
