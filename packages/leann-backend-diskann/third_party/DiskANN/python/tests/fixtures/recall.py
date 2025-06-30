# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import numpy as np


def calculate_recall(
    result_set_indices: np.ndarray, truth_set_indices: np.ndarray, recall_at: int = 5
) -> float:
    """
    result_set_indices and truth_set_indices correspond by row index. the columns in each row contain the indices of
    the nearest neighbors, with result_set_indices being the approximate nearest neighbor results and truth_set_indices
    being the brute force nearest neighbor calculation via sklearn's NearestNeighbor class.
    :param result_set_indices:
    :param truth_set_indices:
    :param recall_at:
    :return:
    """
    found = 0
    for i in range(0, result_set_indices.shape[0]):
        result_set_set = set(result_set_indices[i][0:recall_at])
        truth_set_set = set(truth_set_indices[i][0:recall_at])
        found += len(result_set_set.intersection(truth_set_set))
    return found / (result_set_indices.shape[0] * recall_at)
