#!/usr/bin/env python3
# coding: utf-8
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""
Common utility.
===============
"""

import concurrent.futures as confu
import hashlib
import inspect
import logging
import math
import multiprocessing
import os
import signal
import sys
import unittest
from pathlib import Path
from typing import Callable, ContextManager, List, Sequence, Tuple, cast

import joblib
import matplotlib.pyplot as plt
import numpy as np

from tabulate import tabulate

logger = logging.getLogger(__name__)


###############################################################
# Arg parser.
###############################################################

# def on_ipykernel() -> bool:
#     return 'ipykernel_launcher' in sys.argv[0]
#
#
# def get_arg(n: int, default: str) -> str:
#     args = sys.argv
#     arg = default if on_ipykernel() or len(args) <= n or args[n] is None else args[n]
#     return arg


###############################################################
# Test utils.
###############################################################

# def run_unittest(t: unittest.TestCase):
#     if not on_ipykernel():
#         unittest.main()
#     else:
#         count = 0
#         for tc in map(lambda f: f[0],
#                       filter(lambda f: f[0].startswith('test_'), inspect.getmembers(t, inspect.isroutine))):
#             t.setUp()
#             try:
#                 getattr(t, tc)()
#             except Exception as ex:
#                 print(f'{tc} ({type(t)}) ... ng : {ex}')
#             else:
#                 print(f'{tc} ({type(t)}) ... ok')
#             finally:
#                 count += 1
#             t.tearDown()
#
#         print(f'----------------------------------------------------------------------')
#         print(f'Ran {count} test')


###############################################################
# ML model and data.
###############################################################

# def fetch_dataset(limits: int = -1, *, dataset_name: str = 'mnist_784') -> Tuple[
#     Sequence[np.ndarray], Sequence[np.ndarray]]:
#     """This fetches the dataset.
#     """
#     assert dataset_name in ('mnist_784', 'Fashion-MNIST'), dataset_name
#     dataset = cast(Tuple, Memory('out').cache(fetch_openml)(dataset_name, version=1, return_X_y=True))
#     size = min(len(dataset[0]), limits) if limits >= 0 else len(dataset[0])
#     I = [dataset[0].iloc[idx].to_numpy() for idx in range(size)]
#     O = [np.array([int(dataset[1].iloc[idx])]) for idx in range(size)]
#     return I, O
#
#
# def load_model(path: str, input_shape: Sequence[int], output_shape: Sequence[int]) -> ModelWrapper:
#     """This loads an ONNX model from a given file path.
#     """
#     if path.endswith('.onnx') or path.endswith('.onnx.cache'):
#         model = ONNXModel(path, input_shape, output_shape)
#     elif path.endswith('.tf') or path.endswith('.tf.cache'):
#         model = TFModel(path, input_shape, output_shape)
#     else:
#         raise Exception(f'Un-supported file type : {path}')
#     logger.info(
#         f'Loaded an ONNX model : name = "{os.path.basename(path)}", input_shape = {model.input_shape()}, output_shape = {model.output_shape()}.')
#     return model
#
#
# def get_md5_by_path(cm: ContextManager[Path]) -> str:
#     """This generates a MD5 hash value from a given file path.
#     """
#     hash = hashlib.md5()
#     with cm as path:
#         with open(path, "rb") as f:
#             for chunk in iter(lambda: f.read(4096), b''):
#                 hash.update(chunk)
#         return hash.hexdigest()


###############################################################
# Visualization functions.
###############################################################

def savefig(output: str) -> None:
    assert output, output
    dir = os.path.dirname(output)
    if dir:
        os.makedirs(dir, exist_ok=True)
    plt.savefig(output)
    logger.info(blue(f'Output to "{output}"'))


def imshow(v_I: np.ndarray,
           size: Tuple[int, int] = (7, 7),
           bbox: Tuple[float, float] = (0.0, 1.0),
           *,
           reverse: bool = False
           ) -> None:
    """This plots the image correspond to the given input.
    """
    plt.xlim(0, size[0])
    plt.ylim(size[1], 0)
    plt.xticks([], '')
    plt.yticks([], '')
    plt.imshow(v_I.reshape(size),
               cmap='gray_r' if reverse else 'gray',
               vmin=bbox[0],
               vmax=bbox[1]
               )


def show_all(vl_I: Sequence[np.ndarray], *args,
             title: str = 'Inputs',
             size: Tuple[int, int] = (7, 7),
             bbox: Tuple[float, float] = (0.0, 1.0),
             reverse: bool = False,
             output: str = ''
             ) -> None:
    """This shows the images of the given inputs.
    """
    col = max(1, math.ceil(math.sqrt(len(vl_I))))
    row = math.ceil(len(vl_I) / col)
    fig = plt.figure(figsize=(col, row))
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.8, wspace=1.5, hspace=0.5)
    fig.suptitle(title)

    for no, v_I in enumerate(vl_I):
        x = fig.add_subplot(row, col, no + 1)
        imshow(v_I, size, bbox, reverse=reverse)
        if args:
            x.title.set_text('{}'.format(args[0][no]))

    if output:
        savefig(output)
    else:
        plt.show()

    plt.close()


def red(s: str) -> str:
    return f'\033[31m{s}\033[0m'


def blue(s: str) -> str:
    return f'\033[34m{s}\033[0m'


def DEBUGGING(**kwargs) -> None:
    table = tabulate(kwargs.items(),
                     headers=('KEY', 'VALUE'),
                     tablefmt='github'
                     )
    print(f'\n\033[43m\033[30m{table}\033[0m\n')

if __name__ == '__main__':
    img = np.array([0.25098039215686274,0.7490196078431373,1.0,1.0,1.0,1.0,0.5019607843137255,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5019607843137255,0.5019607843137255,1.0,1.0,0.7490196078431373,0.5019607843137255,0.0,0.0,0.0,0.25098039215686274,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.25098039215686274,1.0])
    show_all([img], title=f'Original',
      bbox=(0, 1),
      output=f'orig.png')

###############################################################
# Parallel processing
###############################################################

# PARALLEL_METHOD_NAMES = ['confu', 'pathos', 'joblib']
#
#
# def parallel(jobs: Sequence[Callable], limit: int = 0, method: str = 'confu') -> List:
#     """This parallelly executes the given jobs, and returns their results as list.
#     e.g.:
#       results = parallel((partial(run, *p) for p in params), method='pathos')
#     """
#     assert method in PARALLEL_METHOD_NAMES, method
#
#     limit = limit if limit > 0 else min(len(jobs), multiprocessing.cpu_count())
#
#     if limit == 1:
#         logger.debug('Start sequential executions.')
#         logger.debug(red(
#             'If you want to start parallel executions by %s, both of the parallelism limit and the number of jobs are greater than 1.'),
#                      method)
#         return [job() for job in jobs]
#
#     elif method == 'confu':
#         logger.debug('Start parallel executions by %s with %s+4 workers.', method, limit)
#         with confu.ThreadPoolExecutor(max_workers=limit + 4) as executor:
#             futures = [executor.submit(job) for job in jobs]
#             results = [f.result() for f in confu.as_completed(futures)]
#         return results
#
#     elif method == 'pathos':
#         logger.debug('Start parallel executions by %s with %s nodes.', method, limit)
#         results = ProcessingPool(nodes=limit).map(lambda job: job(), jobs)
#         return results
#
#     else:  # method == 'joblib':
#         logger.debug('Start parallel executions by %s with %s workers.', method, limit)
#         results = joblib.Parallel(n_jobs=limit, verbose=100)(joblib.delayed(job)() for job in jobs)
#         if results is None:
#             raise Exception(f'Failed to complete the given parallel jobs by joblib!')
#         return results
#
#
# def signalHandler(signal, handler):
#     """This kills all active child processes when parent process is terminating.
#     """
#     for p in multiprocessing.active_children():
#         p.terminate()
#     exit(0)
#
#
# signal.signal(signal.SIGINT, signalHandler)
# signal.signal(signal.SIGTERM, signalHandler)


###############################################################
# Profiling.
###############################################################

# def sizeof(obj):
#     """This profiles memory usage of the given object and its references.
#     """
#     size = sys.getsizeof(obj)
#     if isinstance(obj, dict):
#         return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
#     if isinstance(obj, (list, tuple, set, frozenset)):
#         return size + sum(map(sizeof, obj))
#     if hasattr(obj, '__dict__'):
#         return size + sizeof(obj.__dict__)
#     return size
