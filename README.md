# AIML_Dissertation_GPU_Shared_Ramdom_Forest_ECOC
Custom GPU shared-tree Random Forest implementation for Multi-output ECOC classification using CuPy/CUDA

CUSTOM GPU SHARED-TREE MULTI-OUTPUT RANDOM FOREST ECOC / N-ARY ECOC
===================================================================

PURPOSE
-------
This script implements the custom Stage-3 GPU shared-tree Random Forest
used for the Multi-output RF-ECOC experiments.

IMPORTANT:
This implementation does NOT use:

    - sklearn.ensemble.RandomForestClassifier
    - sklearn.multioutput.MultiOutputClassifier
    - cuml.ensemble.RandomForestClassifier

for the Multi-output RF-ECOC model.

scikit-learn is used only for utility functions such as:
    - train_test_split
    - accuracy_score
    - f1_score

The Random Forest itself is implemented in this file using CuPy/CUDA.

WHY A CUSTOM IMPLEMENTATION IS USED
-----------------------------------
The required Multi-output RF-ECOC architecture is:

    one Random Forest
        +
    one shared feature/threshold tree structure
        +
    L ECOC output positions

cuML provides GPU Random Forests, but not the required native shared
multi-output classification structure.

scikit-learn supports native multi-output Random Forest classification,
but its Random Forest implementation is CPU-based.

Therefore, this custom Stage-3 implementation was developed to provide:

    shared multi-output RF structure + GPU execution

ARCHITECTURE
------------
For each configuration:

    (seed, N, L)

ONE Random Forest is trained.

Each tree contains:

    - one common feature/threshold tree structure
    - shared across all L ECOC output positions

Each terminal leaf stores:

    - one L-dimensional predicted ECOC symbol vector

Therefore:

    Independent RF-ECOC:
        L independent Random Forests
        -> different tree structures for each ECOC position

    Multi-output RF-ECOC in this script:
        1 Random Forest
        -> tree structures shared across all L ECOC positions

CUSTOM IMPLEMENTATION COMPONENTS
--------------------------------
The main custom RF components in this file are:

1. Stage3SharedTreeTrainer
       Implements training of one shared decision tree.

2. FlatSharedTree
       Stores the trained tree in GPU-friendly flat arrays and performs
       shared-tree prediction.

3. build_codebook_embedding_and_agreement()
       Converts the ECOC codebook into a class-agreement matrix.

4. _HISTOGRAM_KERNEL
       Custom CuPy RawKernel used to construct class histograms for all
       randomly selected features at a node.

5. _TREE_PREDICT_KERNEL
       Custom CuPy RawKernel used to traverse one shared tree for all
       samples.

6. _VOTE_KERNEL
       Custom CuPy RawKernel used to accumulate forest votes for every
       ECOC output position.

SHARED MULTI-OUTPUT SPLIT CRITERION
-----------------------------------
Let A be the class-agreement matrix, where

    A[c,c']

is the fraction of ECOC positions in which classes c and c' have the
same ECOC symbol.

For a node containing class-count vector q, impurity is calculated as:

    G(q) = 1 - (q^T A q) / n^2

where:

    n = sum(q)

This is equivalent to calculating the mean Gini impurity across the
L ECOC output positions.

Therefore, all ECOC positions contribute jointly to the selection of
each feature/threshold split.

TREE TRAINING
-------------
For every tree:

    1. Draw a bootstrap sample.
    2. Select sqrt(d) candidate features at each node.
    3. Build histogram-based candidate splits on the GPU.
    4. Evaluate candidate splits using the agreement-matrix
       multi-output Gini criterion.
    5. Select one feature and one threshold.
    6. Apply that same split to all ECOC output positions.
    7. Continue until stopping criteria are reached.
    8. At each leaf, calculate one L-dimensional ECOC symbol vector.

FOREST PREDICTION
-----------------
Each tree predicts an L-dimensional symbol vector.

Predictions from all trees are combined by majority voting independently
for each ECOC position.

The resulting L-dimensional forest output is decoded to the original
class label using validation-derived Weighted Hamming decoding.

DATA
----
Frozen DINOv3 ViT-B/16 features:

    feature dimension = 768
    number of ImageNet classes = 1000

No feature standardisation is applied for Random Forest experiments.

DEFAULT EXPERIMENT GRID
-----------------------
Seeds:
    123, 231, 340, 451, 562

N:
    2, 3, 5, 8, 16

Code lengths:
    64, 128, 256, 512

Random Forest:
    trees               = 300
    max_depth           = unrestricted
    min_samples_leaf    = 1
    max_features        = sqrt
    histogram bins      = 128
    bootstrap           = True
    max_samples         = 0.70

Decoder:
    validation-derived Weighted Hamming

HOW TO RUN
----------

Example single-configuration test:

    python imagenet1k_original_gpu_multioutput_rf_stage3.py \
        --seeds 123 \
        --n-values 2 \
        --code-lengths 64 \
        --n-estimators 1 \
        --max-train-samples 10000 \
        --skip-plots

IMPORTANT:
    --max-train-samples > 0 is DEBUG ONLY.

For final dissertation experiments:

    --max-train-samples 0

Example final configuration:

    python imagenet1k_original_gpu_multioutput_rf_stage3.py \
        --seeds 123 \
        --n-values 2 \
        --code-lengths 64 \
        --n-estimators 300 \
        --max-depth 0 \
        --min-samples-leaf 1 \
        --max-features sqrt \
        --n-bins 128 \
        --max-samples 0.70 \
        --max-train-samples 0

OUTPUT
------
Each completed configuration saves:

    result.csv
    config.json
    ecoc_artifacts.npz

Checkpoint files are also written during long-running configurations.

BACKEND IDENTIFIER
------------------
The implementation identifies itself as:

    custom_gpu_stage3_shared_tree

This value is stored in each result/configuration file so that the
custom RF results can be distinguished from standard cuML or
scikit-learn Random Forest results.
