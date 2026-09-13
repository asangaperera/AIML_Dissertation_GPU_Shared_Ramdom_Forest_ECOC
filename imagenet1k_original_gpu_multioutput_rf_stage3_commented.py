

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATASET_NAME = "imagenet1k_original"
NUM_CLASSES = 1000
FEATURE_DIM = 768

EXPECTED_TRAIN_SHAPE = (1_281_167, FEATURE_DIM)
EXPECTED_TEST_SHAPE = (50_000, FEATURE_DIM)

DEFAULT_CACHE_DIR = Path(
    "/Scratch/am2618/dinov3_imagenet1k_original/"
    "feature_cache_original/numpy_cache"
)

DEFAULT_OUTPUT_DIR = Path(
    "/Scratch/am2618/dinov3_imagenet1k_original/"
    "results_gpu_multioutput_rf_stage3_final"
)

DEFAULT_SEEDS = (123, 231, 340, 451, 562)
DEFAULT_N_VALUES = (2, 3, 5, 8, 16)
DEFAULT_CODE_LENGTHS = (64, 128, 256, 512)

BACKEND_NAME = "custom_gpu_stage3_shared_tree"
EPS = 1e-6



# ============================================================================
# CUSTOM GPU SHARED-TREE RF — CUDA HISTOGRAM KERNEL
# ----------------------------------------------------------------------------
# This is part of the custom RF implementation. At each tree node it builds
# class histograms for every selected candidate feature on the GPU. These
# histograms are later used by Stage3SharedTreeTrainer._best_split() to score
# all histogram split boundaries with the shared multi-output Gini objective.
# ============================================================================
_HISTOGRAM_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void class_histogram_all_features(
    const float* values,
    const int* labels,
    const float* mins,
    const float* inv_widths,
    unsigned int* hist,
    const long long n_samples,
    const int n_features,
    const int n_bins,
    const int n_classes
) {
    const long long tid =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;

    const long long total =
        n_samples * (long long)n_features;

    if (tid >= total) {
        return;
    }

    const int feature_slot =
        (int)(tid % n_features);

    const long long sample =
        tid / n_features;

    const float x =
        values[tid];

    const float inv =
        inv_widths[feature_slot];

    int bin_id = 0;

    if (inv > 0.0f) {
        bin_id = (int)(
            (x - mins[feature_slot]) * inv
        );

        if (bin_id < 0) {
            bin_id = 0;
        }

        if (bin_id >= n_bins) {
            bin_id = n_bins - 1;
        }
    }

    const int cls =
        labels[sample];

    const long long hist_index =
        (
            (
                (long long)feature_slot
                * n_bins
                + bin_id
            )
            * n_classes
            + cls
        );

    atomicAdd(
        &hist[hist_index],
        1u
    );
}
""",
    "class_histogram_all_features",
)



# ============================================================================
# CUSTOM GPU SHARED-TREE RF — TREE TRAVERSAL KERNEL
# ----------------------------------------------------------------------------
# Traverses ONE trained shared tree for every sample. The feature/threshold
# path is common to all ECOC positions. Once a leaf is reached, the leaf's
# complete L-dimensional symbol vector is copied to the output.
# ============================================================================
_TREE_PREDICT_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void predict_shared_tree(
    const float* X,
    const int* node_feature,
    const float* node_threshold,
    const int* left_child,
    const int* right_child,
    const int* leaf_id,
    const unsigned char* leaf_predictions,
    unsigned char* output,
    const long long n_samples,
    const int feature_dim,
    const int code_len
) {
    const long long sample =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;

    if (sample >= n_samples) {
        return;
    }

    int node = 0;

    while (leaf_id[node] < 0) {
        const int feature =
            node_feature[node];

        const float threshold =
            node_threshold[node];

        const float value =
            X[
                sample
                * (long long)feature_dim
                + feature
            ];

        node = (
            value <= threshold
            ? left_child[node]
            : right_child[node]
        );
    }

    const int lid =
        leaf_id[node];

    const long long leaf_base =
        (long long)lid * code_len;

    const long long out_base =
        sample * (long long)code_len;

    for (int l = 0; l < code_len; ++l) {
        output[out_base + l] =
            leaf_predictions[
                leaf_base + l
            ];
    }
}
""",
    "predict_shared_tree",
)



# ============================================================================
# CUSTOM GPU SHARED-TREE RF — FOREST VOTING KERNEL
# ----------------------------------------------------------------------------
# Each tree predicts one symbol at every ECOC position. This kernel accumulates
# those votes independently across positions so the final forest output remains
# an L-dimensional ECOC symbol vector.
# ============================================================================
_VOTE_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void update_votes(
    const unsigned char* prediction,
    unsigned short* votes,
    const long long total_outputs,
    const int n_symbols
) {
    const long long tid =
        (long long)blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_outputs) {
        return;
    }

    const int symbol =
        (int)prediction[tid];

    const long long vote_index =
        tid * (long long)n_symbols
        + symbol;

    votes[vote_index] +=
        (unsigned short)1;
}
""",
    "update_votes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Stage-3 custom GPU shared-tree Multi-output RF ECOC/N-ary ECOC "
            "on ORIGINAL ImageNet-1K frozen DINOv3 features."
        ),
    )

    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-values", type=int, nargs="+", default=list(DEFAULT_N_VALUES))
    parser.add_argument("--code-lengths", type=int, nargs="+", default=list(DEFAULT_CODE_LENGTHS))
    parser.add_argument("--internal-val-size", type=float, default=0.10)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means unrestricted / None.")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=str, default="sqrt")
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--max-samples", type=float, default=0.70)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--leaf-batch-size", type=int, default=256)

    parser.add_argument("--decoder-sample-batch-size", type=int, default=256)
    parser.add_argument("--decoder-class-batch-size", type=int, default=128)

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save vote tensors every K completed trees. 0 disables checkpoints.",
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="DEBUG ONLY. Final thesis runs must use 0.",
    )

    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--shared-backbone-summary", type=Path, default=None)

    args = parser.parse_args()

    if not 0.0 < args.internal_val_size < 1.0:
        parser.error("--internal-val-size must be in (0,1).")
    if args.n_estimators < 1:
        parser.error("--n-estimators must be >=1.")
    if args.n_estimators > 65535:
        parser.error("--n-estimators must be <=65535.")
    if args.min_samples_leaf < 1:
        parser.error("--min-samples-leaf must be >=1.")
    if args.n_bins < 2:
        parser.error("--n-bins must be >=2.")
    if not 0.0 < args.max_samples <= 1.0:
        parser.error("--max-samples must be in (0,1].")
    if args.leaf_batch_size < 1:
        parser.error("--leaf-batch-size must be >=1.")
    if any(n < 2 or n > 255 for n in args.n_values):
        parser.error("All N values must be in [2,255].")
    if any(L < 1 for L in args.code_lengths):
        parser.error("All code lengths must be >=1.")

    return args


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array, allow_pickle=False)
    os.replace(tmp, path)


def sample_std(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1))


def gpu_name() -> str:
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    return str(name)


def gpu_memory_string() -> str:
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return f"{free_b / 1024**3:.2f}/{total_b / 1024**3:.2f} GiB free"


def print_system_info() -> None:
    print("=" * 110)
    print("SYSTEM / GPU INFORMATION")
    print("=" * 110)
    print("Python       :", sys.version.replace("\n", " "))
    print("Platform     :", platform.platform())
    print("NumPy        :", np.__version__)
    print("pandas       :", pd.__version__)
    print("scikit-learn :", sklearn.__version__)
    print("CuPy         :", cp.__version__)
    print("GPU logical  :", cp.cuda.Device().id)
    print("GPU          :", gpu_name())
    print("GPU memory   :", gpu_memory_string())
    print()


def parse_max_features(value: str, feature_dim: int) -> int:
    value = str(value).strip().lower()
    if value == "sqrt":
        return max(1, int(math.sqrt(feature_dim)))
    if value == "log2":
        return max(1, int(math.log2(feature_dim)))
    try:
        integer = int(value)
    except ValueError as exc:
        raise ValueError("--max-features must be sqrt, log2, or integer.") from exc
    if integer < 1:
        raise ValueError("--max-features integer must be >=1.")
    return min(feature_dim, integer)


def required_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "X_train": cache_dir / "X_train.npy",
        "y_train": cache_dir / "y_train.npy",
        "X_test": cache_dir / "X_val.npy",
        "y_test": cache_dir / "y_val.npy",
    }


def load_and_validate_cache(cache_dir: Path):
    paths = required_cache_paths(cache_dir)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cache files:\n" + "\n".join(missing))

    X_train_all = np.load(paths["X_train"], mmap_mode="r")
    y_train_all = np.load(paths["y_train"], mmap_mode="r")
    X_test_fixed = np.load(paths["X_test"], mmap_mode="r")
    y_test_fixed = np.load(paths["y_test"], mmap_mode="r")

    if X_train_all.shape != EXPECTED_TRAIN_SHAPE:
        raise ValueError(f"Expected {EXPECTED_TRAIN_SHAPE}, found {X_train_all.shape}.")
    if X_test_fixed.shape != EXPECTED_TEST_SHAPE:
        raise ValueError(f"Expected {EXPECTED_TEST_SHAPE}, found {X_test_fixed.shape}.")

    expected_labels = np.arange(NUM_CLASSES)
    if not np.array_equal(np.unique(np.asarray(y_train_all)), expected_labels):
        raise ValueError("Training labels are not exactly 0..999.")
    if not np.array_equal(np.unique(np.asarray(y_test_fixed)), expected_labels):
        raise ValueError("Fixed-test labels are not exactly 0..999.")

    print("=" * 110)
    print("ORIGINAL IMAGENET-1K FROZEN DINOv3 CACHE")
    print("=" * 110)
    print("X_train_all :", X_train_all.shape, X_train_all.dtype)
    print("y_train_all :", y_train_all.shape, y_train_all.dtype)
    print("X_test_fixed:", X_test_fixed.shape, X_test_fixed.dtype)
    print("y_test_fixed:", y_test_fixed.shape, y_test_fixed.dtype)
    print("Feature dim :", FEATURE_DIM)
    print("Scaler      : NONE")
    print()

    return X_train_all, y_train_all, X_test_fixed, y_test_fixed


def make_seed_split(
    *,
    y_train_all,
    seed: int,
    internal_val_size: float,
    max_train_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_train_all, dtype=np.int32)
    all_indices = np.arange(len(labels), dtype=np.int64)

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=float(internal_val_size),
        random_state=int(seed),
        stratify=labels,
    )

    if max_train_samples > 0 and len(train_idx) > max_train_samples:
        train_idx, _ = train_test_split(
            train_idx,
            train_size=int(max_train_samples),
            random_state=int(seed) + 77_777,
            stratify=labels[train_idx],
        )

    return np.sort(train_idx), np.sort(val_idx)


def rows_are_unique(codebook: np.ndarray) -> bool:
    return np.unique(codebook, axis=0).shape[0] == codebook.shape[0]


def make_codebook(
    *,
    num_classes: int,
    code_len: int,
    n_symbols: int,
    seed: int,
    max_tries: int = 100,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if n_symbols == 2:
        base = np.zeros(num_classes, dtype=np.uint8)
        base[: num_classes // 2] = 1
    else:
        base = (np.arange(num_classes, dtype=np.int64) % n_symbols).astype(np.uint8)

    for attempt in range(1, max_tries + 1):
        M = np.empty((num_classes, code_len), dtype=np.uint8)
        for l in range(code_len):
            col = base.copy()
            rng.shuffle(col)
            M[:, l] = col
        if rows_are_unique(M):
            return M
        print(f"Codebook regeneration {attempt}/{max_tries}: duplicate rows.")

    raise RuntimeError("Unable to generate unique ECOC codebook.")


def codebook_diagnostics(codebook: np.ndarray) -> dict[str, float]:
    C = codebook.shape[0]
    dmin = np.inf
    total = 0.0
    pairs = 0
    for i in range(C - 1):
        d = np.count_nonzero(codebook[i + 1 :] != codebook[i], axis=1)
        if d.size:
            dmin = min(dmin, int(d.min()))
            total += float(d.sum())
            pairs += int(d.size)
    return {
        "row_dmin": float(dmin),
        "row_davg": float(total / max(pairs, 1)),
    }



# ============================================================================
# CUSTOM SHARED MULTI-OUTPUT OBJECTIVE
# ----------------------------------------------------------------------------
# The trainer does NOT receive an explicit B x L target matrix. Instead, the
# ECOC codebook is embedded into H and converted into class-agreement matrix A:
#
#   A[c,c'] = fraction of ECOC positions where classes c and c' share a symbol.
#
# A is then used inside _best_split() to compute the exact mean multi-output
# Gini impurity for every candidate split while training ONE shared tree.
# ============================================================================
def build_codebook_embedding_and_agreement(
    *,
    codebook_gpu: cp.ndarray,
    n_symbols: int,
) -> tuple[cp.ndarray, cp.ndarray]:
    C, L = codebook_gpu.shape
    eye = cp.eye(n_symbols, dtype=cp.float32)
    H = eye[codebook_gpu.astype(cp.int32)].reshape(C, L * n_symbols)
    A = (H @ H.T) / float(L)
    return H, A.astype(cp.float32, copy=False)



# ============================================================================
# CUSTOM SHARED TREE REPRESENTATION
# ----------------------------------------------------------------------------
# One FlatSharedTree contains ONE common feature/threshold structure.  It does
# not contain L separate trees.  The multi-output information appears at the
# leaves, where leaf_predictions has shape [n_leaves, L].
# ============================================================================
@dataclass
class FlatSharedTree:
    feature: cp.ndarray
    threshold: cp.ndarray
    left_child: cp.ndarray
    right_child: cp.ndarray
    leaf_id: cp.ndarray
    leaf_predictions: cp.ndarray
    node_count: int
    leaf_count: int
    max_depth_seen: int
    code_len: int

    def predict(self, X: cp.ndarray) -> cp.ndarray:
        n_samples = int(X.shape[0])
        output = cp.empty((n_samples, self.code_len), dtype=cp.uint8)

        threads = 128
        blocks = (n_samples + threads - 1) // threads

        _TREE_PREDICT_KERNEL(
            (blocks,),
            (threads,),
            (
                X,
                self.feature,
                self.threshold,
                self.left_child,
                self.right_child,
                self.leaf_id,
                self.leaf_predictions,
                output,
                np.int64(n_samples),
                np.int32(X.shape[1]),
                np.int32(self.code_len),
            ),
        )
        return output



# ============================================================================
# CORE CUSTOM GPU SHARED-TREE RANDOM FOREST TREE TRAINER
# ----------------------------------------------------------------------------
# This class is the main custom implementation used in Multi-output RF-ECOC.
#
# For one decision tree it:
#   1) samples candidate features at a node;
#   2) builds GPU class histograms;
#   3) scores all candidate thresholds with agreement-matrix multi-output Gini;
#   4) selects ONE feature/threshold split shared across all L ECOC outputs;
#   5) repeats until terminal leaves are reached;
#   6) stores one L-dimensional symbol vector in every leaf.
#
# A forest is obtained by calling this trainer once per bootstrap tree inside
# run_one_configuration().
# ============================================================================
class Stage3SharedTreeTrainer:
    def __init__(
        self,
        *,
        codebook_gpu: cp.ndarray,
        codebook_embedding_gpu: cp.ndarray,
        agreement_gpu: cp.ndarray,
        n_symbols: int,
        n_bins: int,
        max_features_count: int,
        max_depth: int | None,
        min_samples_leaf: int,
        min_gain: float,
        leaf_batch_size: int,
        tree_seed: int,
        progress_every_nodes: int = 5000,
    ):
        self.codebook = codebook_gpu
        self.H = codebook_embedding_gpu
        self.A = agreement_gpu
        self.n_symbols = int(n_symbols)
        self.n_bins = int(n_bins)
        self.max_features_count = int(max_features_count)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.min_gain = float(min_gain)
        self.leaf_batch_size = int(leaf_batch_size)
        self.tree_seed = int(tree_seed)
        self.progress_every_nodes = int(progress_every_nodes)
        self.rng = np.random.default_rng(self.tree_seed)
        self.code_len = int(codebook_gpu.shape[1])

    def _class_counts(self, labels: cp.ndarray) -> cp.ndarray:
        return cp.bincount(
            labels.astype(cp.int32, copy=False),
            minlength=NUM_CLASSES,
        ).astype(cp.float32)

    def _best_split(
        self,
        *,
        X: cp.ndarray,
        y_class: cp.ndarray,
        indices: cp.ndarray,
        parent_counts: cp.ndarray,
    ):
        n_parent = int(indices.size)
        n_features_total = int(X.shape[1])

        features_np = self.rng.choice(
            n_features_total,
            size=self.max_features_count,
            replace=False,
        ).astype(np.int32)

        features = cp.asarray(features_np, dtype=cp.int32)

        values = X[
            indices[:, None],
            features[None, :],
        ]
        values = cp.ascontiguousarray(values, dtype=cp.float32)

        labels = cp.ascontiguousarray(y_class[indices], dtype=cp.int32)

        mins = cp.min(values, axis=0).astype(cp.float32)
        maxs = cp.max(values, axis=0).astype(cp.float32)
        ranges = maxs - mins

        inv_widths = cp.where(
            ranges > 0.0,
            float(self.n_bins) / ranges,
            0.0,
        ).astype(cp.float32)

        hist = cp.zeros(
            (self.max_features_count, self.n_bins, NUM_CLASSES),
            dtype=cp.uint32,
        )

        total_work = int(values.shape[0]) * self.max_features_count
        threads = 256
        blocks = (total_work + threads - 1) // threads

        _HISTOGRAM_KERNEL(
            (blocks,),
            (threads,),
            (
                values,
                labels,
                mins,
                inv_widths,
                hist,
                np.int64(values.shape[0]),
                np.int32(self.max_features_count),
                np.int32(self.n_bins),
                np.int32(NUM_CLASSES),
            ),
        )

        left_counts = cp.cumsum(
            hist,
            axis=1,
            dtype=cp.float32,
        )[:, :-1, :]

        n_left = cp.sum(left_counts, axis=2)
        n_right = float(n_parent) - n_left

        valid = (
            (n_left >= self.min_samples_leaf)
            & (n_right >= self.min_samples_leaf)
            & (ranges[:, None] > 0.0)
        )

        if not bool(cp.any(valid).item()):
            return None

        Q = left_counts.reshape(-1, NUM_CLASSES)

        # ------------------------------------------------------------------
        # SHARED MULTI-OUTPUT SPLIT SCORING
        # ------------------------------------------------------------------
        # Q contains the class-count vector q for every candidate LEFT child.
        # self.A is the ECOC class-agreement matrix.  Therefore q^T A q gives
        # the number of same-symbol class-pair agreements averaged across all
        # L ECOC positions.  Using this quantity gives the mean multi-output
        # Gini impurity WITHOUT training/scoring L separate trees.
        #
        # This is the key step that makes the tree genuinely shared across the
        # ECOC outputs: every candidate feature/threshold is judged jointly by
        # all L positions, and only ONE split is selected for the node.
        # ------------------------------------------------------------------
        # All candidate left quadratic forms in one GEMM.
        QA = Q @ self.A
        left_quad = cp.sum(QA * Q, axis=1)

        total_A = self.A @ parent_counts
        total_quad = cp.dot(parent_counts, total_A)

        # q_right = q_total - q_left:
        # rAr = tAt - 2*qAt + qAq
        cross = Q @ total_A
        right_quad = total_quad - 2.0 * cross + left_quad

        left_n_flat = n_left.reshape(-1)
        right_n_flat = n_right.reshape(-1)

        left_gini = cp.clip(
            1.0 - left_quad / cp.maximum(left_n_flat * left_n_flat, 1.0),
            0.0,
            1.0,
        )

        right_gini = cp.clip(
            1.0 - right_quad / cp.maximum(right_n_flat * right_n_flat, 1.0),
            0.0,
            1.0,
        )

        child_impurity = (
            (left_n_flat / float(n_parent)) * left_gini
            + (right_n_flat / float(n_parent)) * right_gini
        )

        parent_gini = 1.0 - total_quad / float(n_parent * n_parent)
        gains = parent_gini - child_impurity
        gains = cp.where(valid.reshape(-1), gains, -cp.inf)

        best_flat_gpu = cp.argmax(gains)
        best_flat = int(best_flat_gpu.item())
        best_gain = float(gains[best_flat_gpu].item())

        if not np.isfinite(best_gain) or best_gain <= self.min_gain:
            return None

        bins_per_feature = self.n_bins - 1
        feature_slot = best_flat // bins_per_feature
        split_bin = best_flat % bins_per_feature

        best_feature = int(features_np[feature_slot])
        selected_min = float(mins[feature_slot].item())
        selected_max = float(maxs[feature_slot].item())

        width = (selected_max - selected_min) / float(self.n_bins)
        threshold = selected_min + width * float(split_bin + 1)

        split_values = X[indices, best_feature]
        left_mask = split_values <= threshold

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if (
            int(left_indices.size) < self.min_samples_leaf
            or int(right_indices.size) < self.min_samples_leaf
        ):
            return None

        return (
            best_feature,
            float(threshold),
            left_indices,
            right_indices,
            float(best_gain),
        )

    def _derive_leaf_predictions(
        self,
        *,
        y_class: cp.ndarray,
        leaf_indices_parts: list[cp.ndarray],
    ) -> cp.ndarray:
        n_leaves = len(leaf_indices_parts)

        lengths = np.asarray(
            [int(x.size) for x in leaf_indices_parts],
            dtype=np.int64,
        )

        offsets = np.empty(n_leaves + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])

        all_indices = cp.concatenate(leaf_indices_parts).astype(cp.int32, copy=False)

        leaf_predictions = cp.empty(
            (n_leaves, self.code_len),
            dtype=cp.uint8,
        )

        for leaf_start in range(0, n_leaves, self.leaf_batch_size):
            leaf_end = min(leaf_start + self.leaf_batch_size, n_leaves)

            batch_lengths_np = lengths[leaf_start:leaf_end]
            sample_start = int(offsets[leaf_start])
            sample_end = int(offsets[leaf_end])

            batch_indices = all_indices[sample_start:sample_end]
            batch_labels = y_class[batch_indices]

            local_leaf_ids = cp.repeat(
                cp.arange(leaf_end - leaf_start, dtype=cp.int32),
                cp.asarray(batch_lengths_np, dtype=cp.int32),
            )

            flat = (
                local_leaf_ids.astype(cp.int64) * NUM_CLASSES
                + batch_labels.astype(cp.int64)
            )

            class_hist = cp.bincount(
                flat,
                minlength=(leaf_end - leaf_start) * NUM_CLASSES,
            ).reshape(leaf_end - leaf_start, NUM_CLASSES).astype(cp.float32)

            # Convert the class histogram in each leaf into symbol counts for
            # every ECOC position.  self.H is the one-hot representation of the
            # class codebook.  The resulting shape is:
            #
            #     [number_of_leaves_in_batch, L, N]
            #
            # Argmax over N gives ONE predicted symbol for each of the L
            # positions, so every leaf stores an L-dimensional output vector.
            symbol_counts = (class_hist @ self.H).reshape(
                leaf_end - leaf_start,
                self.code_len,
                self.n_symbols,
            )

            leaf_predictions[leaf_start:leaf_end] = cp.argmax(
                symbol_counts,
                axis=2,
            ).astype(cp.uint8)

        return leaf_predictions

    def fit(
        self,
        *,
        X: cp.ndarray,
        y_class: cp.ndarray,
        bootstrap_indices: cp.ndarray,
    ) -> FlatSharedTree:
        node_feature = [-1]
        node_threshold = [0.0]
        left_child = [-1]
        right_child = [-1]
        node_leaf_id = [-1]

        leaf_indices_parts: list[cp.ndarray] = []

        stack: list[tuple[int, cp.ndarray, int]] = [
            (0, bootstrap_indices, 0)
        ]

        node_count = 0
        max_depth_seen = 0

        while stack:
            node_id, indices, depth = stack.pop()

            node_count += 1
            max_depth_seen = max(max_depth_seen, depth)
            n_node = int(indices.size)

            stop = (
                n_node < 2 * self.min_samples_leaf
                or (
                    self.max_depth is not None
                    and depth >= self.max_depth
                )
            )

            labels_node = y_class[indices]

            if not stop:
                label_min = int(cp.min(labels_node).item())
                label_max = int(cp.max(labels_node).item())
                if label_min == label_max:
                    stop = True

            if stop:
                leaf_id = len(leaf_indices_parts)
                node_leaf_id[node_id] = leaf_id
                leaf_indices_parts.append(indices)
            else:
                parent_counts = self._class_counts(labels_node)

                # Select ONE feature/threshold using the shared multi-output
                # criterion.  The returned split is used by every ECOC output
                # position because the tree structure is common/shared.
                split = self._best_split(
                    X=X,
                    y_class=y_class,
                    indices=indices,
                    parent_counts=parent_counts,
                )

                if split is None:
                    leaf_id = len(leaf_indices_parts)
                    node_leaf_id[node_id] = leaf_id
                    leaf_indices_parts.append(indices)
                else:
                    feature, threshold, left_indices, right_indices, _gain = split

                    node_feature[node_id] = feature
                    node_threshold[node_id] = threshold

                    left_id = len(node_feature)
                    right_id = left_id + 1

                    left_child[node_id] = left_id
                    right_child[node_id] = right_id

                    for _ in range(2):
                        node_feature.append(-1)
                        node_threshold.append(0.0)
                        left_child.append(-1)
                        right_child.append(-1)
                        node_leaf_id.append(-1)

                    stack.append((right_id, right_indices, depth + 1))
                    stack.append((left_id, left_indices, depth + 1))

            if (
                self.progress_every_nodes > 0
                and node_count % self.progress_every_nodes == 0
            ):
                print(
                    f"    nodes={node_count:,} "
                    f"leaves={len(leaf_indices_parts):,} "
                    f"stack={len(stack):,} "
                    f"depth={max_depth_seen} "
                    f"GPU={gpu_memory_string()}",
                    flush=True,
                )

        leaf_predictions = self._derive_leaf_predictions(
            y_class=y_class,
            leaf_indices_parts=leaf_indices_parts,
        )

        return FlatSharedTree(
            feature=cp.asarray(np.asarray(node_feature, dtype=np.int32)),
            threshold=cp.asarray(np.asarray(node_threshold, dtype=np.float32)),
            left_child=cp.asarray(np.asarray(left_child, dtype=np.int32)),
            right_child=cp.asarray(np.asarray(right_child, dtype=np.int32)),
            leaf_id=cp.asarray(np.asarray(node_leaf_id, dtype=np.int32)),
            leaf_predictions=leaf_predictions,
            node_count=len(node_feature),
            leaf_count=len(leaf_indices_parts),
            max_depth_seen=max_depth_seen,
            code_len=self.code_len,
        )


def update_votes(
    *,
    prediction: cp.ndarray,
    votes: cp.ndarray,
    n_symbols: int,
) -> None:
    total = int(prediction.size)
    threads = 256
    blocks = (total + threads - 1) // threads

    _VOTE_KERNEL(
        (blocks,),
        (threads,),
        (
            prediction,
            votes,
            np.int64(total),
            np.int32(n_symbols),
        ),
    )


def checkpoint_paths(config_dir: Path) -> dict[str, Path]:
    return {
        "meta": config_dir / "checkpoint_meta.json",
        "val_votes": config_dir / "checkpoint_val_votes.npy",
        "test_votes": config_dir / "checkpoint_test_votes.npy",
    }


def checkpoint_signature(
    *,
    args: argparse.Namespace,
    seed: int,
    n_symbols: int,
    code_len: int,
) -> dict[str, Any]:
    return {
        "backend": BACKEND_NAME,
        "seed": seed,
        "n_symbols": n_symbols,
        "code_len": code_len,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "n_bins": args.n_bins,
        "max_samples": args.max_samples,
        "min_gain": args.min_gain,
        "internal_val_size": args.internal_val_size,
        "max_train_samples": args.max_train_samples,
    }


def signatures_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def save_checkpoint(
    *,
    config_dir: Path,
    signature: dict[str, Any],
    completed_trees: int,
    val_votes: cp.ndarray,
    test_votes: cp.ndarray,
    fit_seconds: float,
    predict_seconds: float,
) -> None:
    paths = checkpoint_paths(config_dir)

    print(f"  Saving checkpoint after tree {completed_trees}...", flush=True)

    cp.cuda.Stream.null.synchronize()
    val_np = cp.asnumpy(val_votes)
    test_np = cp.asnumpy(test_votes)

    atomic_save_npy(paths["val_votes"], val_np)
    atomic_save_npy(paths["test_votes"], test_np)

    atomic_write_json(
        {
            "signature": signature,
            "completed_trees": completed_trees,
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        },
        paths["meta"],
    )


def load_checkpoint(
    *,
    config_dir: Path,
    signature: dict[str, Any],
    expected_val_shape: tuple[int, int, int],
    expected_test_shape: tuple[int, int, int],
):
    paths = checkpoint_paths(config_dir)

    if not all(p.exists() for p in paths.values()):
        return 0, None, None, 0.0, 0.0

    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))

        if not signatures_equal(meta["signature"], signature):
            return 0, None, None, 0.0, 0.0

        val_np = np.load(paths["val_votes"], allow_pickle=False)
        test_np = np.load(paths["test_votes"], allow_pickle=False)

        if tuple(val_np.shape) != expected_val_shape:
            return 0, None, None, 0.0, 0.0
        if tuple(test_np.shape) != expected_test_shape:
            return 0, None, None, 0.0, 0.0

        completed = int(meta["completed_trees"])

        print(f"RESUME checkpoint: {completed} trees already complete.")

        return (
            completed,
            cp.asarray(val_np, dtype=cp.uint16),
            cp.asarray(test_np, dtype=cp.uint16),
            float(meta.get("fit_seconds", 0.0)),
            float(meta.get("predict_seconds", 0.0)),
        )
    except Exception as exc:
        print("WARNING: checkpoint ignored:", exc)
        return 0, None, None, 0.0, 0.0


def remove_checkpoint(config_dir: Path) -> None:
    for path in checkpoint_paths(config_dir).values():
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def validation_column_weights(
    *,
    pred_symbols: cp.ndarray,
    true_symbols: cp.ndarray,
    n_symbols: int,
):
    column_accuracy = cp.mean(
        pred_symbols == true_symbols,
        axis=0,
    ).astype(cp.float32)

    if n_symbols == 2:
        weights = cp.maximum(2.0 * column_accuracy - 1.0, EPS)
    else:
        chance = 1.0 / float(n_symbols)
        weights = cp.maximum(
            (column_accuracy - chance) / (1.0 - chance),
            EPS,
        )

    return weights.astype(cp.float32, copy=False), column_accuracy


def weighted_hamming_decode_gpu(
    *,
    pred_code: cp.ndarray,
    codebook: cp.ndarray,
    weights: cp.ndarray,
    sample_batch_size: int,
    class_batch_size: int,
) -> cp.ndarray:
    n_samples = int(pred_code.shape[0])
    n_classes = int(codebook.shape[0])

    output = cp.empty(n_samples, dtype=cp.int32)

    for sample_start in range(0, n_samples, sample_batch_size):
        sample_end = min(sample_start + sample_batch_size, n_samples)

        pred_batch = pred_code[sample_start:sample_end]
        batch_n = int(pred_batch.shape[0])

        best_distance = cp.full(batch_n, cp.inf, dtype=cp.float32)
        best_class = cp.zeros(batch_n, dtype=cp.int32)

        for class_start in range(0, n_classes, class_batch_size):
            class_end = min(class_start + class_batch_size, n_classes)

            candidate = codebook[class_start:class_end]

            distances = cp.sum(
                (
                    pred_batch[:, None, :]
                    != candidate[None, :, :]
                )
                * weights[None, None, :],
                axis=2,
                dtype=cp.float32,
            )

            local_idx = cp.argmin(distances, axis=1)
            rows = cp.arange(batch_n, dtype=cp.int32)
            local_distance = distances[rows, local_idx]

            better = local_distance < best_distance

            best_distance[better] = local_distance[better]
            best_class[better] = (
                class_start + local_idx[better]
            ).astype(cp.int32)

        output[sample_start:sample_end] = best_class

    return output


def configuration_dir(
    *,
    output_dir: Path,
    seed: int,
    n_symbols: int,
    code_len: int,
) -> Path:
    folder = "binary" if n_symbols == 2 else "nary"
    return output_dir / f"seed_{seed}" / f"{folder}_N{n_symbols}_L{code_len}"


def result_path(
    *,
    output_dir: Path,
    seed: int,
    n_symbols: int,
    code_len: int,
) -> Path:
    return (
        configuration_dir(
            output_dir=output_dir,
            seed=seed,
            n_symbols=n_symbols,
            code_len=code_len,
        )
        / "result.csv"
    )


def existing_result_matches(
    *,
    path: Path,
    args: argparse.Namespace,
    seed: int,
    n_symbols: int,
    code_len: int,
) -> bool:
    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)
        if len(df) != 1:
            return False
        row = df.iloc[0]

        checks = {
            "seed": seed,
            "n_symbols": n_symbols,
            "code_len": code_len,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "n_bins": args.n_bins,
            "max_train_samples": args.max_train_samples,
        }

        for key, expected in checks.items():
            if key not in row.index or int(row[key]) != int(expected):
                return False

        if str(row["rf_backend"]) != BACKEND_NAME:
            return False

        if str(row["max_features"]).strip().lower() != str(args.max_features).strip().lower():
            return False

        if not np.isclose(float(row["max_samples"]), float(args.max_samples)):
            return False

        if not np.isclose(float(row["internal_val_size"]), float(args.internal_val_size)):
            return False

        return True
    except Exception:
        return False



# ============================================================================
# TRAIN ONE CUSTOM SHARED-TREE FOREST FOR ONE (SEED, N, L) CONFIGURATION
# ----------------------------------------------------------------------------
# This function is the main experiment-level entry point for the custom RF.
# It creates ONE forest, not L separate forests. Each bootstrap iteration trains
# one Stage3SharedTreeTrainer tree whose structure is shared across all L ECOC
# positions. The tree then predicts an L-dimensional vector on validation/test
# data and contributes one vote per ECOC position.
# ============================================================================
def run_one_configuration(
    *,
    args: argparse.Namespace,
    seed: int,
    n_symbols: int,
    code_len: int,
    X_train_gpu: cp.ndarray,
    y_train_gpu: cp.ndarray,
    X_val_gpu: cp.ndarray,
    y_val_gpu: cp.ndarray,
    X_test_gpu: cp.ndarray,
    y_test_gpu: cp.ndarray,
) -> pd.DataFrame:
    wall_start = time.perf_counter()

    config_dir = configuration_dir(
        output_dir=args.output_dir,
        seed=seed,
        n_symbols=n_symbols,
        code_len=code_len,
    )
    config_dir.mkdir(parents=True, exist_ok=True)

    setup_start = time.perf_counter()

    codebook_np = make_codebook(
        num_classes=NUM_CLASSES,
        code_len=code_len,
        n_symbols=n_symbols,
        seed=seed + 9000 + n_symbols * 100 + code_len,
    )

    diagnostics = codebook_diagnostics(codebook_np)

    codebook_gpu = cp.asarray(codebook_np, dtype=cp.uint8)

    H_gpu, A_gpu = build_codebook_embedding_and_agreement(
        codebook_gpu=codebook_gpu,
        n_symbols=n_symbols,
    )

    cp.cuda.Stream.null.synchronize()
    setup_seconds = time.perf_counter() - setup_start

    max_depth = None if args.max_depth <= 0 else args.max_depth
    m_try = parse_max_features(args.max_features, FEATURE_DIM)

    method = (
        "Multi-output ECOC(RF)"
        if n_symbols == 2
        else "Multi-output N-ary ECOC(RF)"
    )

    print()
    print("=" * 118)
    print(
        f"Seed={seed} | {method} | N={n_symbols} | L={code_len} | "
        f"trees={args.n_estimators}"
    )
    print("=" * 118)
    print("RF backend          :", BACKEND_NAME)
    print("RF models           : 1")
    print("Train               :", X_train_gpu.shape)
    print("Internal validation :", X_val_gpu.shape)
    print("Fixed final test    :", X_test_gpu.shape)
    print("Trees               :", args.n_estimators)
    print("Max depth           :", "None" if max_depth is None else max_depth)
    print("Min leaf            :", args.min_samples_leaf)
    print("Max features        :", args.max_features, "->", m_try)
    print("Histogram bins      :", args.n_bins)
    print("Bootstrap           : True")
    print("Max samples         :", args.max_samples)
    print("Code row d_min      :", diagnostics["row_dmin"])
    print("Code row d_avg      :", f'{diagnostics["row_davg"]:.3f}')
    print("GPU memory          :", gpu_memory_string())
    print()

    config = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "method": method,
        "n_symbols": n_symbols,
        "code_len": code_len,
        "num_rf_models": 1,
        "shared_tree_structure": True,
        "rf_backend": BACKEND_NAME,
        "n_estimators": args.n_estimators,
        "criterion": "exact_mean_multioutput_gini_via_class_agreement",
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "max_features_count": m_try,
        "n_bins": args.n_bins,
        "bootstrap": True,
        "max_samples": args.max_samples,
        "class_weight": None,
        "standardized": False,
        "feature_dim": FEATURE_DIM,
        "internal_val_size": args.internal_val_size,
        "max_train_samples": args.max_train_samples,
        "decoder": "weighted_hamming",
        "row_dmin": diagnostics["row_dmin"],
        "row_davg": diagnostics["row_davg"],
        "cupy_version": cp.__version__,
        "gpu_name": gpu_name(),
    }

    atomic_write_json(config, config_dir / "config.json")

    signature = checkpoint_signature(
        args=args,
        seed=seed,
        n_symbols=n_symbols,
        code_len=code_len,
    )

    expected_val_shape = (
        int(X_val_gpu.shape[0]),
        code_len,
        n_symbols,
    )

    expected_test_shape = (
        int(X_test_gpu.shape[0]),
        code_len,
        n_symbols,
    )

    if args.overwrite:
        remove_checkpoint(config_dir)

    (
        completed_trees,
        val_votes,
        test_votes,
        fit_seconds,
        predict_seconds,
    ) = load_checkpoint(
        config_dir=config_dir,
        signature=signature,
        expected_val_shape=expected_val_shape,
        expected_test_shape=expected_test_shape,
    )

    if val_votes is None:
        val_votes = cp.zeros(expected_val_shape, dtype=cp.uint16)
        test_votes = cp.zeros(expected_test_shape, dtype=cp.uint16)
        completed_trees = 0
        fit_seconds = 0.0
        predict_seconds = 0.0

    bootstrap_n = max(
        1,
        int(round(int(X_train_gpu.shape[0]) * args.max_samples)),
    )

    for tree_id in range(completed_trees, args.n_estimators):
        tree_seed = (
            seed * 1_000_003
            + n_symbols * 10_007
            + code_len * 101
            + tree_id
        ) % (2**31 - 1)

        print(
            f"Training tree {tree_id + 1}/{args.n_estimators}",
            flush=True,
        )

        gpu_rng = cp.random.RandomState(tree_seed)

        bootstrap_idx = gpu_rng.randint(
            0,
            int(X_train_gpu.shape[0]),
            size=bootstrap_n,
            dtype=cp.int32,
        )

        trainer = Stage3SharedTreeTrainer(
            codebook_gpu=codebook_gpu,
            codebook_embedding_gpu=H_gpu,
            agreement_gpu=A_gpu,
            n_symbols=n_symbols,
            n_bins=args.n_bins,
            max_features_count=m_try,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            min_gain=args.min_gain,
            leaf_batch_size=args.leaf_batch_size,
            tree_seed=tree_seed,
        )

        tree_fit_start = time.perf_counter()

        # Train ONE custom shared tree.  y_train_gpu contains original class
        # labels (0..999).  The ECOC multi-output objective enters through the
        # trainer's codebook embedding H and class-agreement matrix A.
        tree = trainer.fit(
            X=X_train_gpu,
            y_class=y_train_gpu,
            bootstrap_indices=bootstrap_idx,
        )

        cp.cuda.Stream.null.synchronize()

        tree_fit_seconds = time.perf_counter() - tree_fit_start
        fit_seconds += tree_fit_seconds

        print(
            f"  tree={tree_id + 1} "
            f"fit={tree_fit_seconds:.2f}s "
            f"nodes={tree.node_count:,} "
            f"leaves={tree.leaf_count:,} "
            f"depth={tree.max_depth_seen} "
            f"GPU={gpu_memory_string()}",
            flush=True,
        )

        pred_start = time.perf_counter()

        val_tree_pred = tree.predict(X_val_gpu)
        update_votes(
            prediction=val_tree_pred,
            votes=val_votes,
            n_symbols=n_symbols,
        )
        del val_tree_pred

        test_tree_pred = tree.predict(X_test_gpu)
        update_votes(
            prediction=test_tree_pred,
            votes=test_votes,
            n_symbols=n_symbols,
        )
        del test_tree_pred

        cp.cuda.Stream.null.synchronize()

        tree_pred_seconds = time.perf_counter() - pred_start
        predict_seconds += tree_pred_seconds

        print(
            f"  predict/vote={tree_pred_seconds:.2f}s",
            flush=True,
        )

        del tree
        del trainer
        del bootstrap_idx
        del gpu_rng

        cp.get_default_memory_pool().free_all_blocks()

        trees_done = tree_id + 1

        if (
            args.checkpoint_every > 0
            and trees_done % args.checkpoint_every == 0
            and trees_done < args.n_estimators
        ):
            save_checkpoint(
                config_dir=config_dir,
                signature=signature,
                completed_trees=trees_done,
                val_votes=val_votes,
                test_votes=test_votes,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
            )

    # Majority vote independently at each ECOC position.  The result is still
    # an L-dimensional symbol vector per sample and is only converted to the
    # original ImageNet class later by Weighted Hamming decoding.
    val_symbols = cp.argmax(val_votes, axis=2).astype(cp.uint8)
    test_symbols = cp.argmax(test_votes, axis=2).astype(cp.uint8)

    true_val_codes = codebook_gpu[y_val_gpu]
    true_test_codes = codebook_gpu[y_test_gpu]

    weights, val_column_accuracy = validation_column_weights(
        pred_symbols=val_symbols,
        true_symbols=true_val_codes,
        n_symbols=n_symbols,
    )

    decode_start = time.perf_counter()

    val_prediction_gpu = weighted_hamming_decode_gpu(
        pred_code=val_symbols,
        codebook=codebook_gpu,
        weights=weights,
        sample_batch_size=args.decoder_sample_batch_size,
        class_batch_size=args.decoder_class_batch_size,
    )

    test_prediction_gpu = weighted_hamming_decode_gpu(
        pred_code=test_symbols,
        codebook=codebook_gpu,
        weights=weights,
        sample_batch_size=args.decoder_sample_batch_size,
        class_batch_size=args.decoder_class_batch_size,
    )

    cp.cuda.Stream.null.synchronize()
    decode_seconds = time.perf_counter() - decode_start

    metric_start = time.perf_counter()

    y_val_np = cp.asnumpy(y_val_gpu)
    y_test_np = cp.asnumpy(y_test_gpu)
    val_prediction = cp.asnumpy(val_prediction_gpu)
    test_prediction = cp.asnumpy(test_prediction_gpu)

    val_accuracy = accuracy_score(y_val_np, val_prediction) * 100.0
    test_accuracy = accuracy_score(y_test_np, test_prediction) * 100.0

    val_macro_f1 = (
        f1_score(
            y_val_np,
            val_prediction,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )

    test_macro_f1 = (
        f1_score(
            y_test_np,
            test_prediction,
            average="macro",
            zero_division=0,
        )
        * 100.0
    )

    val_symbol_accuracy = float(
        cp.mean(val_symbols == true_val_codes).item() * 100.0
    )

    test_symbol_accuracy = float(
        cp.mean(test_symbols == true_test_codes).item() * 100.0
    )

    metric_seconds = time.perf_counter() - metric_start
    total_seconds = time.perf_counter() - wall_start

    weights_np = cp.asnumpy(weights)
    val_column_accuracy_np = cp.asnumpy(val_column_accuracy)

    atomic_save_npz(
        config_dir / "ecoc_artifacts.npz",
        codebook=codebook_np,
        weights=weights_np.astype(np.float32, copy=False),
        validation_column_accuracy=val_column_accuracy_np.astype(
            np.float32,
            copy=False,
        ),
    )

    row = {
        "dataset": DATASET_NAME,
        "seed": seed,
        "method": method,
        "method_family": "Multi-output RF",
        "base_learner": "Random Forest",
        "decoder": "weighted_hamming",
        "n_symbols": n_symbols,
        "code_len": code_len,
        "num_rf_models": 1,
        "shared_tree_structure": True,
        "rf_backend": BACKEND_NAME,
        "n_estimators": args.n_estimators,
        "criterion": "exact_mean_multioutput_gini_via_class_agreement",
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "max_features_count": m_try,
        "n_bins": args.n_bins,
        "bootstrap": True,
        "max_samples": args.max_samples,
        "internal_val_size": args.internal_val_size,
        "max_train_samples": args.max_train_samples,
        "debug_subsampled": bool(args.max_train_samples > 0),
        "train_samples": int(X_train_gpu.shape[0]),
        "val_samples": int(X_val_gpu.shape[0]),
        "test_samples": int(X_test_gpu.shape[0]),
        "feature_dim": FEATURE_DIM,
        "standardized": False,
        "row_dmin": diagnostics["row_dmin"],
        "row_davg": diagnostics["row_davg"],
        "val_accuracy": float(val_accuracy),
        "test_accuracy": float(test_accuracy),
        "val_macro_f1": float(val_macro_f1),
        "macro_f1": float(test_macro_f1),
        "val_symbol_accuracy": float(val_symbol_accuracy),
        "test_symbol_accuracy": float(test_symbol_accuracy),
        "weight_mean": float(weights_np.mean()),
        "weight_std": float(weights_np.std(ddof=0)),
        "weight_min": float(weights_np.min()),
        "weight_max": float(weights_np.max()),
        "setup_seconds": float(setup_seconds),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "decode_seconds": float(decode_seconds),
        "metric_seconds": float(metric_seconds),
        "total_seconds": float(total_seconds),
        "cupy_version": cp.__version__,
        "gpu_name": gpu_name(),
    }

    result_df = pd.DataFrame([row])
    atomic_write_csv(result_df, config_dir / "result.csv")

    remove_checkpoint(config_dir)

    print()
    print(
        f"RESULT | Val={val_accuracy:.3f}% | "
        f"Test={test_accuracy:.3f}% | "
        f"ValMacroF1={val_macro_f1:.3f}% | "
        f"TestMacroF1={test_macro_f1:.3f}% | "
        f"ValSymbol={val_symbol_accuracy:.3f}% | "
        f"TestSymbol={test_symbol_accuracy:.3f}%"
    )

    print(
        f"TIME   | setup={setup_seconds:.2f}s | "
        f"fit={fit_seconds:.2f}s | "
        f"predict={predict_seconds:.2f}s | "
        f"decode={decode_seconds:.2f}s | "
        f"total={total_seconds:.2f}s"
    )

    print("Saved  :", config_dir / "result.csv")

    return result_df


def collect_completed_results(output_dir: Path) -> pd.DataFrame:
    frames = []

    for path in sorted(output_dir.rglob("result.csv")):
        try:
            df = pd.read_csv(path)
            if len(df) == 1:
                df = df.copy()
                df["result_path"] = str(path)
                frames.append(df)
        except Exception as exc:
            print("WARNING reading", path, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def rebuild_summaries(output_dir: Path) -> pd.DataFrame:
    detailed = collect_completed_results(output_dir)

    if detailed.empty:
        print("No completed result.csv files found.")
        return pd.DataFrame()

    detailed = detailed.sort_values(
        ["n_symbols", "code_len", "seed"]
    ).reset_index(drop=True)

    detailed_csv = (
        output_dir
        / "imagenet1k_original_gpu_multioutput_rf_stage3_detailed_results.csv"
    )

    atomic_write_csv(detailed, detailed_csv)

    group_cols = [
        "dataset",
        "method",
        "method_family",
        "base_learner",
        "decoder",
        "n_symbols",
        "code_len",
        "num_rf_models",
        "shared_tree_structure",
        "rf_backend",
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "max_features_count",
        "n_bins",
        "bootstrap",
        "max_samples",
        "feature_dim",
        "standardized",
        "max_train_samples",
        "debug_subsampled",
    ]

    summary = (
        detailed.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("seed", "count"),
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", sample_std),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", sample_std),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", sample_std),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", sample_std),
            val_symbol_accuracy_mean=("val_symbol_accuracy", "mean"),
            val_symbol_accuracy_std=("val_symbol_accuracy", sample_std),
            test_symbol_accuracy_mean=("test_symbol_accuracy", "mean"),
            test_symbol_accuracy_std=("test_symbol_accuracy", sample_std),
            fit_seconds_mean=("fit_seconds", "mean"),
            fit_seconds_std=("fit_seconds", sample_std),
            predict_seconds_mean=("predict_seconds", "mean"),
            decode_seconds_mean=("decode_seconds", "mean"),
            total_seconds_mean=("total_seconds", "mean"),
            total_seconds_std=("total_seconds", sample_std),
            row_dmin_mean=("row_dmin", "mean"),
            row_davg_mean=("row_davg", "mean"),
        )
        .reset_index()
        .sort_values(["n_symbols", "code_len"])
        .reset_index(drop=True)
    )

    summary_csv = (
        output_dir
        / "imagenet1k_original_gpu_multioutput_rf_stage3_summary.csv"
    )

    atomic_write_csv(summary, summary_csv)

    report = summary[
        [
            "method",
            "n_symbols",
            "code_len",
            "num_rf_models",
            "n_runs",
            "val_accuracy_mean",
            "val_accuracy_std",
            "test_accuracy_mean",
            "test_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "fit_seconds_mean",
            "fit_seconds_std",
            "total_seconds_mean",
            "total_seconds_std",
        ]
    ].copy()

    report.columns = [
        "Method",
        "N",
        "Code Length",
        "No. of RF Models",
        "Runs",
        "Val Accuracy Mean (%)",
        "Val Std",
        "Test Accuracy Mean (%)",
        "Test Std",
        "Macro F1 Mean (%)",
        "Macro F1 Std",
        "Fit Runtime Mean (s)",
        "Fit Runtime Std",
        "Total Runtime Mean (s)",
        "Total Runtime Std",
    ]

    report_csv = (
        output_dir
        / "imagenet1k_original_gpu_multioutput_rf_stage3_report_table.csv"
    )

    atomic_write_csv(report, report_csv)

    selected = (
        summary.sort_values(
            ["n_symbols", "val_accuracy_mean", "code_len"],
            ascending=[True, False, True],
        )
        .groupby("n_symbols", as_index=False, sort=True)
        .head(1)
        .reset_index(drop=True)
    )

    selected_csv = (
        output_dir
        / "imagenet1k_original_gpu_multioutput_rf_stage3_validation_selected.csv"
    )

    atomic_write_csv(selected, selected_csv)

    atomic_write_csv(
        summary[
            [
                "n_symbols",
                "code_len",
                "val_accuracy_mean",
                "val_accuracy_std",
                "test_accuracy_mean",
                "test_accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
            ]
        ],
        output_dir / "code_length_accuracy_comparison.csv",
    )

    atomic_write_csv(
        summary[
            [
                "n_symbols",
                "code_len",
                "test_accuracy_mean",
                "macro_f1_mean",
                "fit_seconds_mean",
                "total_seconds_mean",
            ]
        ],
        output_dir / "runtime_accuracy_comparison.csv",
    )

    print()
    print("=" * 110)
    print("UPDATED COMBINED OUTPUTS")
    print("=" * 110)
    print("Detailed :", detailed_csv)
    print("Summary  :", summary_csv)
    print("Report   :", report_csv)
    print("Selected :", selected_csv)
    print()

    return summary


def final_plot_rows(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    rows = summary.copy()

    if "debug_subsampled" in rows.columns:
        mask = (
            rows["debug_subsampled"]
            .astype(str)
            .str.lower()
            .isin({"false", "0", "0.0"})
        )
        if mask.any():
            rows = rows[mask].copy()

    rows = rows[
        rows["n_symbols"].astype(int).isin(DEFAULT_N_VALUES)
        & rows["code_len"].astype(int).isin(DEFAULT_CODE_LENGTHS)
    ].copy()

    return rows


def plot_summary(*, summary: pd.DataFrame, output_dir: Path) -> None:
    rows = final_plot_rows(summary)
    if rows.empty:
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "val_accuracy_mean",
            "val_accuracy_std",
            "Validation Accuracy (%)",
            "validation_accuracy_vs_code_length.png",
        ),
        (
            "test_accuracy_mean",
            "test_accuracy_std",
            "Test Accuracy (%)",
            "test_accuracy_vs_code_length.png",
        ),
        (
            "macro_f1_mean",
            "macro_f1_std",
            "Macro-F1 (%)",
            "macro_f1_vs_code_length.png",
        ),
        (
            "total_seconds_mean",
            "total_seconds_std",
            "Total Runtime (s)",
            "code_length_vs_runtime.png",
        ),
    ]

    for y_col, err_col, ylabel, filename in plots:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        for n_value, group in rows.groupby("n_symbols", sort=True):
            group = group.sort_values("code_len")
            ax.errorbar(
                group["code_len"],
                group[y_col],
                yerr=group[err_col].fillna(0.0),
                marker="o",
                capsize=3,
                label=f"N={int(n_value)}",
            )

        ax.set_xlabel("Code Length")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs Code Length")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    for n_value in sorted(rows["n_symbols"].unique()):
        group = rows[rows["n_symbols"] == n_value].sort_values("code_len")

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        ax.errorbar(
            group["code_len"],
            group["val_accuracy_mean"],
            yerr=group["val_accuracy_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label="Validation Accuracy",
        )

        ax.errorbar(
            group["code_len"],
            group["test_accuracy_mean"],
            yerr=group["test_accuracy_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label="Test Accuracy",
        )

        ax.errorbar(
            group["code_len"],
            group["macro_f1_mean"],
            yerr=group["macro_f1_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label="Test Macro-F1",
        )

        ax.set_xlabel("Code Length")
        ax.set_ylabel("Score (%)")
        ax.set_title(f"Validation, Test and Macro-F1 (N={int(n_value)})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"val_test_f1_vs_code_length_N{int(n_value)}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for n_value, group in rows.groupby("n_symbols", sort=True):
        ax.scatter(
            group["total_seconds_mean"],
            group["test_accuracy_mean"],
            label=f"N={int(n_value)}",
        )

        for _, row in group.iterrows():
            ax.annotate(
                f"L={int(row['code_len'])}",
                (
                    float(row["total_seconds_mean"]),
                    float(row["test_accuracy_mean"]),
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xlabel("Mean Total Runtime (s)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Runtime vs Test Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "runtime_vs_test_accuracy.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for code_len, group in rows.groupby("code_len", sort=True):
        group = group.sort_values("n_symbols")
        ax.errorbar(
            group["n_symbols"],
            group["test_accuracy_mean"],
            yerr=group["test_accuracy_std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=f"L={int(code_len)}",
        )

    ax.set_xlabel("N")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("N vs Test Accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "n_vs_test_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("Figures  :", figures_dir)


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError("Could not find column. Tried: " + ", ".join(candidates))
    return None


def normalize_shared_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    n_col = pick_column(df, ["n_symbols", "N"])
    l_col = pick_column(df, ["code_len", "Code Length"])
    val_col = pick_column(df, ["val_accuracy_mean", "Val Accuracy Mean (%)"])
    test_col = pick_column(df, ["test_accuracy_mean", "Test Accuracy Mean (%)"])
    f1_col = pick_column(df, ["macro_f1_mean", "Macro F1 Mean (%)"])

    val_std = pick_column(df, ["val_accuracy_std", "Val Std"], required=False)
    test_std = pick_column(df, ["test_accuracy_std", "Test Std"], required=False)
    f1_std = pick_column(df, ["macro_f1_std", "Macro F1 Std"], required=False)
    runtime = pick_column(
        df,
        ["total_seconds_mean", "Total Runtime Mean (s)", "Runtime Mean (s)"],
        required=False,
    )

    out = pd.DataFrame(
        {
            "n_symbols": pd.to_numeric(df[n_col], errors="coerce"),
            "code_len": pd.to_numeric(df[l_col], errors="coerce"),
            "val_accuracy_mean": pd.to_numeric(df[val_col], errors="coerce"),
            "test_accuracy_mean": pd.to_numeric(df[test_col], errors="coerce"),
            "macro_f1_mean": pd.to_numeric(df[f1_col], errors="coerce"),
            "val_accuracy_std": (
                pd.to_numeric(df[val_std], errors="coerce")
                if val_std is not None
                else 0.0
            ),
            "test_accuracy_std": (
                pd.to_numeric(df[test_std], errors="coerce")
                if test_std is not None
                else 0.0
            ),
            "macro_f1_std": (
                pd.to_numeric(df[f1_std], errors="coerce")
                if f1_std is not None
                else 0.0
            ),
            "total_seconds_mean": (
                pd.to_numeric(df[runtime], errors="coerce")
                if runtime is not None
                else np.nan
            ),
        }
    )

    out = out.dropna(
        subset=[
            "n_symbols",
            "code_len",
            "val_accuracy_mean",
            "test_accuracy_mean",
            "macro_f1_mean",
        ]
    )

    out["n_symbols"] = out["n_symbols"].astype(int)
    out["code_len"] = out["code_len"].astype(int)

    return out


def build_shared_comparison(
    *,
    multi_summary: pd.DataFrame,
    shared_path: Path,
    output_dir: Path,
) -> None:
    if not shared_path.exists():
        print("WARNING: Shared-Backbone CSV not found:", shared_path)
        return

    multi = final_plot_rows(multi_summary)
    shared = normalize_shared_summary(shared_path)

    shared = shared[
        shared["n_symbols"].isin(DEFAULT_N_VALUES)
        & shared["code_len"].isin(DEFAULT_CODE_LENGTHS)
    ].copy()

    multi_small = multi[
        [
            "n_symbols",
            "code_len",
            "val_accuracy_mean",
            "val_accuracy_std",
            "test_accuracy_mean",
            "test_accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "total_seconds_mean",
        ]
    ].copy()

    multi_small["comparison_method"] = "Multi-output RF"
    shared["comparison_method"] = "Shared-Backbone"

    combined = pd.concat([shared, multi_small], ignore_index=True)

    comparison_csv = (
        output_dir / "shared_backbone_vs_multioutput_rf_comparison.csv"
    )

    atomic_write_csv(combined, comparison_csv)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        (
            "val_accuracy_mean",
            "val_accuracy_std",
            "Validation Accuracy (%)",
            "validation_accuracy",
        ),
        (
            "test_accuracy_mean",
            "test_accuracy_std",
            "Test Accuracy (%)",
            "test_accuracy",
        ),
        (
            "macro_f1_mean",
            "macro_f1_std",
            "Macro-F1 (%)",
            "macro_f1",
        ),
    ]

    for n_value in DEFAULT_N_VALUES:
        n_df = combined[combined["n_symbols"] == n_value]

        if n_df.empty:
            continue

        for mean_col, std_col, ylabel, suffix in metric_specs:
            fig, ax = plt.subplots(figsize=(8.5, 5.5))

            for method, group in n_df.groupby("comparison_method", sort=True):
                group = group.sort_values("code_len")
                ax.errorbar(
                    group["code_len"],
                    group[mean_col],
                    yerr=group[std_col].fillna(0.0),
                    marker="o",
                    capsize=3,
                    label=method,
                )

            ax.set_xlabel("Code Length")
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"Shared-Backbone vs Multi-output RF: {ylabel} (N={n_value})"
            )
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                figures_dir / f"shared_vs_multioutput_N{n_value}_{suffix}.png",
                dpi=220,
                bbox_inches="tight",
            )

            plt.close(fig)

    print("Comparison:", comparison_csv)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print_system_info()

    run_config = {
        "dataset": DATASET_NAME,
        "backend": BACKEND_NAME,
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "seeds": [int(x) for x in args.seeds],
        "n_values": [int(x) for x in args.n_values],
        "code_lengths": [int(x) for x in args.code_lengths],
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "n_bins": args.n_bins,
        "bootstrap": True,
        "max_samples": args.max_samples,
        "internal_val_size": args.internal_val_size,
        "max_train_samples": args.max_train_samples,
        "decoder": "weighted_hamming",
        "checkpoint_every": args.checkpoint_every,
        "gpu_name": gpu_name(),
        "cupy_version": cp.__version__,
    }

    atomic_write_json(run_config, args.output_dir / "run_config.json")

    if args.summary_only:
        summary = rebuild_summaries(args.output_dir)

        if not args.skip_plots and not summary.empty:
            plot_summary(summary=summary, output_dir=args.output_dir)

            if args.shared_backbone_summary is not None:
                build_shared_comparison(
                    multi_summary=summary,
                    shared_path=args.shared_backbone_summary,
                    output_dir=args.output_dir,
                )

        return

    if args.max_train_samples > 0:
        print()
        print("!" * 110)
        print("DEBUG MODE: --max-train-samples > 0.")
        print("DO NOT report these runs as final ImageNet-1K thesis results.")
        print("!" * 110)
        print()

    (
        X_train_all,
        y_train_all,
        X_test_fixed,
        y_test_fixed,
    ) = load_and_validate_cache(args.cache_dir)

    print("Copying fixed final test set to GPU...")

    X_test_gpu = cp.asarray(
        np.asarray(X_test_fixed, dtype=np.float32)
    )

    y_test_gpu = cp.asarray(
        np.asarray(y_test_fixed, dtype=np.int32)
    )

    print("Fixed test GPU:", X_test_gpu.shape, gpu_memory_string())
    print()

    for seed in args.seeds:
        seed = int(seed)

        print()
        print("#" * 118)
        print(f"PREPARING SEED {seed}")
        print("#" * 118)

        train_idx, val_idx = make_seed_split(
            y_train_all=y_train_all,
            seed=seed,
            internal_val_size=args.internal_val_size,
            max_train_samples=args.max_train_samples,
        )

        print("Materialising seed split from mmap...")

        X_train_np = np.asarray(
            X_train_all[train_idx],
            dtype=np.float32,
        )
        y_train_np = np.asarray(
            y_train_all[train_idx],
            dtype=np.int32,
        )
        X_val_np = np.asarray(
            X_train_all[val_idx],
            dtype=np.float32,
        )
        y_val_np = np.asarray(
            y_train_all[val_idx],
            dtype=np.int32,
        )

        print("Copying seed train/validation to GPU...")

        X_train_gpu = cp.asarray(X_train_np)
        y_train_gpu = cp.asarray(y_train_np, dtype=cp.int32)
        X_val_gpu = cp.asarray(X_val_np)
        y_val_gpu = cp.asarray(y_val_np, dtype=cp.int32)

        del X_train_np, y_train_np, X_val_np, y_val_np
        gc.collect()

        print("Train GPU     :", X_train_gpu.shape)
        print("Validation GPU:", X_val_gpu.shape)
        print("GPU memory    :", gpu_memory_string())

        for n_symbols in args.n_values:
            n_symbols = int(n_symbols)

            for code_len in args.code_lengths:
                code_len = int(code_len)

                saved = result_path(
                    output_dir=args.output_dir,
                    seed=seed,
                    n_symbols=n_symbols,
                    code_len=code_len,
                )

                if (
                    not args.overwrite
                    and existing_result_matches(
                        path=saved,
                        args=args,
                        seed=seed,
                        n_symbols=n_symbols,
                        code_len=code_len,
                    )
                ):
                    print(
                        "SKIP completed | "
                        f"seed={seed} | N={n_symbols} | L={code_len}"
                    )
                    continue

                run_one_configuration(
                    args=args,
                    seed=seed,
                    n_symbols=n_symbols,
                    code_len=code_len,
                    X_train_gpu=X_train_gpu,
                    y_train_gpu=y_train_gpu,
                    X_val_gpu=X_val_gpu,
                    y_val_gpu=y_val_gpu,
                    X_test_gpu=X_test_gpu,
                    y_test_gpu=y_test_gpu,
                )

                summary = rebuild_summaries(args.output_dir)

                if not args.skip_plots and not summary.empty:
                    plot_summary(
                        summary=summary,
                        output_dir=args.output_dir,
                    )

        del X_train_gpu, y_train_gpu, X_val_gpu, y_val_gpu
        del train_idx, val_idx

        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()

    summary = rebuild_summaries(args.output_dir)

    if not args.skip_plots and not summary.empty:
        plot_summary(summary=summary, output_dir=args.output_dir)

        if args.shared_backbone_summary is not None:
            build_shared_comparison(
                multi_summary=summary,
                shared_path=args.shared_backbone_summary,
                output_dir=args.output_dir,
            )

    print()
    print("=" * 110)
    print("ALL REQUESTED STAGE-3 GPU MULTI-OUTPUT RF CONFIGURATIONS COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
