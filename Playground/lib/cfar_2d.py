"""
cfar_2d.py — 2D CA-CFAR (Cell-Averaging Constant False Alarm Rate) detection.

Extends the 1D CFAR from target_detection_dbfs.py to operate on
2D range-Doppler maps. Each cell under test is compared against
the average power in a surrounding rectangular annulus of reference cells,
with guard cells to prevent target self-masking.
"""

import numpy as np
from scipy import ndimage
from functools import lru_cache


@lru_cache(maxsize=32)
def _average_kernel(ref_range, ref_doppler, guard_range, guard_doppler):
    """Return CA-CFAR kernel and reference-cell count."""
    kernel_rows = 2 * (guard_doppler + ref_doppler) + 1
    kernel_cols = 2 * (guard_range + ref_range) + 1
    kernel = np.ones((kernel_rows, kernel_cols), dtype=float)
    r_start = ref_doppler
    r_end = ref_doppler + 2 * guard_doppler + 1
    c_start = ref_range
    c_end = ref_range + 2 * guard_range + 1
    kernel[r_start:r_end, c_start:c_end] = 0.0
    num_ref_cells = float(np.sum(kernel))
    return kernel, num_ref_cells


@lru_cache(maxsize=32)
def _goso_kernels(ref_range, ref_doppler, guard_range, guard_doppler):
    """Return GO/SO-CFAR split kernels and reference-cell counts."""
    kernel_rows = 2 * (guard_doppler + ref_doppler) + 1
    kernel_cols = 2 * (guard_range + ref_range) + 1

    kernel_lead = np.zeros((kernel_rows, kernel_cols), dtype=float)
    kernel_lead[:ref_doppler, :] = 1.0
    kernel_trail = np.zeros((kernel_rows, kernel_cols), dtype=float)
    kernel_trail[-ref_doppler:, :] = 1.0

    kernel_left = np.zeros((kernel_rows, kernel_cols), dtype=float)
    kernel_left[:, :ref_range] = 1.0
    kernel_right = np.zeros((kernel_rows, kernel_cols), dtype=float)
    kernel_right[:, -ref_range:] = 1.0

    kernel_a = np.clip(kernel_lead + kernel_left, 0.0, 1.0)
    kernel_b = np.clip(kernel_trail + kernel_right, 0.0, 1.0)

    r_start = ref_doppler
    r_end = ref_doppler + 2 * guard_doppler + 1
    c_start = ref_range
    c_end = ref_range + 2 * guard_range + 1
    kernel_a[r_start:r_end, c_start:c_end] = 0.0
    kernel_b[r_start:r_end, c_start:c_end] = 0.0

    n_a = float(max(np.sum(kernel_a), 1.0))
    n_b = float(max(np.sum(kernel_b), 1.0))
    return kernel_a, kernel_b, n_a, n_b


def cfar_2d(rd_map, guard_range=2, guard_doppler=2,
            ref_range=4, ref_doppler=4, bias_db=5.0, method='average'):
    """Run 2D CFAR detection on a range-Doppler map.

    Args:
        rd_map: 2D array (doppler_bins, range_bins) in dB scale.
        guard_range: Number of guard cells on each side in range dimension.
        guard_doppler: Number of guard cells on each side in Doppler dimension.
        ref_range: Number of reference cells on each side in range dimension.
        ref_doppler: Number of reference cells on each side in Doppler dimension.
        bias_db: Detection threshold above estimated noise floor (dB).
        method: 'average' (CA-CFAR), 'greatest' (GO-CFAR), 'smallest' (SO-CFAR).

    Returns:
        detections: Boolean mask, same shape as rd_map. True = detection.
        threshold: The adaptive threshold map in dB.
    """
    rows, cols = rd_map.shape
    rd_power = 10 ** (rd_map / 10.0)
    bias_lin = 10 ** (bias_db / 10.0)

    # Build the CFAR kernel: a rectangular annulus
    # Outer window includes reference + guard + CUT
    outer_h = guard_doppler + ref_doppler
    outer_w = guard_range + ref_range
    inner_h = guard_doppler
    inner_w = guard_range

    # Total size of the kernel
    kernel_rows = 2 * outer_h + 1
    kernel_cols = 2 * outer_w + 1

    if method == 'average':
        kernel, num_ref_cells = _average_kernel(
            ref_range, ref_doppler, guard_range, guard_doppler,
        )
        # Convolve to get sum of reference cells at each position
        ref_sum = ndimage.convolve(rd_power, kernel, mode='reflect')
        noise_power = ref_sum / num_ref_cells
        threshold_power = noise_power * bias_lin

    elif method in ('greatest', 'smallest'):
        # Split into leading/trailing halves for both dimensions
        # For GO/SO-CFAR, compute the mean from each half and take max/min

        kernel_a, kernel_b, n_a, n_b = _goso_kernels(
            ref_range, ref_doppler, guard_range, guard_doppler,
        )

        mean_a = ndimage.convolve(rd_power, kernel_a, mode='reflect') / n_a
        mean_b = ndimage.convolve(rd_power, kernel_b, mode='reflect') / n_b

        if method == 'greatest':
            threshold_power = np.maximum(mean_a, mean_b) * bias_lin
        else:
            threshold_power = np.minimum(mean_a, mean_b) * bias_lin
    else:
        raise ValueError(f"Unknown CFAR method: {method}")

    threshold = 10 * np.log10(threshold_power + 1e-20)
    detections = rd_power > threshold_power
    return detections, threshold


def cluster_detections(detections, rd_map, range_axis, velocity_axis):
    """Cluster adjacent detection cells into discrete targets.

    Uses connected-component labeling to group adjacent detections,
    then computes the power-weighted centroid for each cluster.

    Args:
        detections: Boolean 2D mask (doppler_bins, range_bins).
        rd_map: Range-Doppler map in dB (same shape).
        range_axis: 1D array of range values (meters) for each column.
        velocity_axis: 1D array of velocity values (m/s) for each row.

    Returns:
        List of dicts, each with keys:
            'range_m': estimated range in meters
            'velocity_mps': estimated velocity in m/s
            'power_db': peak power in dB
            'pixel_count': number of cells in the cluster
            'range_idx': centroid range index
            'doppler_idx': centroid Doppler index
    """
    if not np.any(detections):
        return []

    # Label connected components (8-connectivity)
    structure = np.ones((3, 3))
    labeled, num_features = ndimage.label(detections, structure=structure)

    targets = []
    for label_id in range(1, num_features + 1):
        mask = labeled == label_id
        pixel_count = np.sum(mask)

        # Power-weighted centroid
        powers_linear = 10 ** (rd_map[mask] / 10.0)  # convert dB power to linear for weighting
        total_power = np.sum(powers_linear)

        doppler_indices, range_indices = np.where(mask)

        centroid_range_idx = np.sum(range_indices * powers_linear) / total_power
        centroid_doppler_idx = np.sum(doppler_indices * powers_linear) / total_power

        # Interpolate to get physical values
        range_m = np.interp(centroid_range_idx, np.arange(len(range_axis)), range_axis)
        velocity_mps = np.interp(centroid_doppler_idx, np.arange(len(velocity_axis)), velocity_axis)

        peak_db = np.max(rd_map[mask])

        targets.append({
            'range_m': float(range_m),
            'velocity_mps': float(velocity_mps),
            'power_db': float(peak_db),
            'pixel_count': int(pixel_count),
            'range_idx': float(centroid_range_idx),
            'doppler_idx': float(centroid_doppler_idx),
        })

    # Sort by power (strongest first)
    targets.sort(key=lambda t: t['power_db'], reverse=True)
    return targets
