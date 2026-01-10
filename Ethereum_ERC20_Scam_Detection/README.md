# Ethereum ERC-20 Scam Detection

This module implements an **ERC-20 token scam detection** pipeline based on **graph-temporal feature engineering** extracted from token transfer activity. It includes:

- **Token transfer extraction** (ERC-20 + ETH native + internal transfers) via Etherscan.
- **Graph and temporal feature extraction** from transaction CSVs.
- **Model training and evaluation** across multiple model families (Balanced RF, MLP, SVM).

The datasets used in this work are not included in the repository. For dataset sharing requests, contact: `mario.trerotola@polito.it`.

### Components

#### 1) Transaction extraction (Etherscan)

- **`transaction_tracker_complete.py`**
  - Fetches three transaction modalities:
    - ERC-20 token transfers (`tokentx`)
    - ETH native transfers (`txlist`)
    - ETH internal transfers (`txlistinternal`)
  - Implements API key rotation and per-key rate limiting.
  - Streams results directly to CSV to reduce memory usage.

- **`extract_history.py`**
  - Batch runner for token contract addresses stored in a CSV (e.g., scam token lists).
  - Supports checkpoint/resume and parallel execution.
  - Produces per-token CSV files of transactions suitable for feature extraction.

Typical inputs:
- A CSV containing contract addresses (column name configurable via constants).
- One or more Etherscan API keys.

#### 2) Feature extraction (graph + temporal + statistics)

- **`extract_features.py`**
  - Loads token transaction CSVs, constructs directed transfer graphs, and computes:
    - Node centralities and distribution statistics.
    - Temporal features (long-term vs short-term behavior).
    - Edge features (transfer value transforms, frequency, recency, accumulation).
    - Transaction-level aggregates (volume, address counts, time gaps, concentration).
  - Supports parallel processing and checkpointing for large corpora.
  - Writes per-source datasets (e.g., scam/licit folders) into `data/dataset_with_features/<source>/features.csv`.

#### 3) Modeling notebooks

- **`scam_detection_brf_hyperparameter_tuning_final.ipynb`**
  - Balanced Random Forest hyperparameter search (`GridSearchCV`), decision threshold tuning on validation set, and full evaluation.

- **`scam_detection_svm_mlp_chainabuse_cmc.ipynb`**
  - Comparative evaluation of BRF/MLP/SVM across multiple dataset constructions.
  - Reuses preprocessing/feature-selection per dataset and tunes decision thresholds on validation data.

### Dependencies and runtime notes

Dependencies are declared at the repository root (`../requirements.txt`).

Operational notes:
- Etherscan API usage requires valid API keys and may be rate limited; the code supports key rotation to mitigate this.
- Large-scale extraction can generate many CSV files; ensure sufficient disk space.

