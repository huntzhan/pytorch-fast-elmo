#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from os.path import abspath, dirname


def load_requirements(path):
    with open(path) as fin:
        return [
            line
            for line in map(lambda l: l.strip(), fin.readlines())
            if line and not line.startswith('#')
        ]


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = load_requirements('requirements_prod.txt')
test_requirements = load_requirements('requirements_dev.txt')

setup(
    author="Hunt Zhan",
    author_email='huntzhan.dev@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="None",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pytorch_fast_elmo',
    name='pytorch_fast_elmo',
    packages=find_packages(include=['pytorch_fast_elmo']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/cnt-dev/pytorch-fast-elmo',
    version='0.6.0',
    zip_safe=False,
    # Pytorch Cpp Extension.
    ext_modules=[
        CppExtension(
            '_pytorch_fast_elmo',
            [
                'extension/elmo_character_encoder.cc',
                'extension/scalar_mix.cc',
                'extension/bind.cc',
            ],
            include_dirs=[dirname(abspath(__file__))],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    entry_points={'console_scripts': ['fast-elmo = pytorch_fast_elmo.tool.cli:main']},
)
