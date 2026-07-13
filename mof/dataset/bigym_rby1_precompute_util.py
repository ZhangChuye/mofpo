from typing import Mapping

import numpy as np


def get_fill_value(dtype):
    if np.issubdtype(dtype, np.floating):
        return np.nan
    return 0


def gather_padded_chunks_from_full_array(
    indices: np.ndarray,
    key_first_k: Mapping[str, int],
    full_array: np.ndarray,
    key: str,
    out_length: int,
) -> np.ndarray:
    out = np.empty((len(indices), out_length) + full_array.shape[1:], dtype=full_array.dtype)
    fill_value = get_fill_value(full_array.dtype)
    first_k = key_first_k.get(key)

    for i, (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx) in enumerate(indices):
        row = out[i]
        row.fill(fill_value)

        buffer_start_idx = int(buffer_start_idx)
        buffer_end_idx = int(buffer_end_idx)
        sample_start_idx = int(sample_start_idx)
        sample_end_idx = int(sample_end_idx)
        n_data = buffer_end_idx - buffer_start_idx
        k_data = n_data if first_k is None else min(int(first_k), n_data)

        prefix_end = min(sample_start_idx, out_length)
        if prefix_end > 0:
            prefix_value = full_array[buffer_start_idx] if k_data > 0 else fill_value
            row[:prefix_end] = prefix_value

        middle_start = min(sample_start_idx, out_length)
        middle_end = min(sample_end_idx, out_length)
        if middle_end > middle_start and k_data > 0:
            valid_mid_len = min(k_data, middle_end - middle_start)
            row[middle_start : middle_start + valid_mid_len] = full_array[
                buffer_start_idx : buffer_start_idx + valid_mid_len
            ]

        if middle_end < out_length:
            if k_data == n_data and n_data > 0:
                suffix_value = full_array[buffer_end_idx - 1]
            else:
                suffix_value = fill_value
            row[middle_end:] = suffix_value

    return out


def gather_padded_timestep_from_full_array(
    indices: np.ndarray,
    key_first_k: Mapping[str, int],
    full_array: np.ndarray,
    key: str,
    timestep: int,
) -> np.ndarray:
    out = np.empty((len(indices),) + full_array.shape[1:], dtype=full_array.dtype)
    fill_value = get_fill_value(full_array.dtype)
    first_k = key_first_k.get(key)

    for i, (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx) in enumerate(indices):
        buffer_start_idx = int(buffer_start_idx)
        buffer_end_idx = int(buffer_end_idx)
        sample_start_idx = int(sample_start_idx)
        sample_end_idx = int(sample_end_idx)
        n_data = buffer_end_idx - buffer_start_idx
        k_data = n_data if first_k is None else min(int(first_k), n_data)

        if timestep < sample_start_idx:
            sample_offset = 0
        elif timestep >= sample_end_idx:
            sample_offset = n_data - 1
        else:
            sample_offset = timestep - sample_start_idx

        if sample_offset < k_data:
            out[i] = full_array[buffer_start_idx + sample_offset]
        else:
            out[i] = fill_value

    return out
