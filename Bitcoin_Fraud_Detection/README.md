# Bitcoin Fraud Detection

This module implements a **wallet-level fraud detection** pipeline for Bitcoin. It covers:

- **Acquisition** of wallet transaction histories.
- **Feature engineering** (graph/temporal/volume statistics at wallet level).
- **Dataset assembly** and labeling.
- **Model training and evaluation**, including interpretable analysis (CIU, SHAP) and rule extraction (DEXiRE).

Dataset access is not bundled with this repository. For data sharing requests, contact: `mario.trerotola@polito.it`.

### Pipeline overview

#### 1) Data acquisition

- **`wallet_fetcher.ipynb`**
  - Downloads raw wallet transaction JSON from Blockchain.com endpoints (via `https://blockchain.info`), with retry logic and optional rotating proxy.
  - Filters wallet list by class (e.g., Elliptic class `2` for licit, depending on the provided `wallets_classes.csv`).

- **`extract_latest_wallets/extract_latest_wallets.ipynb`**
  - Collects a target number of recent transactions by walking backward from the latest block.
  - Extracts unique addresses from transaction inputs/outputs and persists them to CSV.

#### 2) ChainAbuse scraping and cleaning (optional data source)

- **`chainabuse_fetcher/chainabuse_fetcher.ipynb`**
  - Scrapes Bitcoin scam reports from ChainAbuse using Playwright (headless browser automation).
  - Supports cookie injection and resume from last processed page.

- **`chainabuse_fetcher/clean_data.ipynb`**
  - Validates Bitcoin addresses, filters invalid entries, and deduplicates.
  - Persists a cleaned CSV for downstream use.

#### 3) Feature computation

- **`extract_metrics.ipynb`**
  - Validates downloaded JSON files and computes wallet-level metrics.
  - Produces a tabular feature set (e.g., degrees, unique degrees, volumes, time gaps, activity duration, ratios).

#### 4) Dataset assembly / preprocessing

- **`dataset_assembly.ipynb`**
  - Merges feature tables (e.g., licit vs fraud sources), performs cleaning, removes non-feature columns, and writes the final labeled dataset.

- **`data_exploration.ipynb`**
  - Exploratory analysis of the assembled dataset (distributions, missing values, summary statistics).

#### 5) Modeling and interpretability

- **`rf/random_forest_ciu.ipynb`**
  - Random Forest training with preprocessing pipeline and hyperparameter tuning via `GridSearchCV`.
  - Evaluation via report and confusion matrix.
  - Local explainability with CIU; optional SHAP blocks.

- **`svm/svm.ipynb`**
  - SVM (`SVC(probability=True)`) training with hyperparameter search.
  - Evaluation and optional explainability blocks.

- **`mlp/mlp.ipynb`**
  - MLP classifier with simple architecture search and evaluation on a held-out test set.

- **`confident_learning/confident_learning.ipynb`**
  - Label-noise detection using Cleanlab; focuses on filtering potentially mislabeled fraud samples.
  - Re-trains and compares downstream performance after filtering.

- **`rf/rf_mlp_distill_dexire.ipynb`**
  - Teacher–student setup (Random Forest teacher, MLP student).
  - Synthetic augmentation (SMOTE), confidence filtering, student training.
  - Rule extraction via DEXiRE for symbolic interpretability.

- **`rf/extract rules.ipynb`**
  - Utilities to post-process extracted rules and export them in LaTeX-friendly form (e.g., for paper tables).

### Installation notes

Dependencies are declared at the repository root (`../requirements.txt`).

If you use the ChainAbuse scraper:
- After installing Python packages, run:

```bash
playwright install
```

### Inputs and expected files

Inputs depend on the experiment configuration and your local data layout. Typical inputs include:
- **Wallet list with labels** (e.g., `wallets_classes.csv`).
- **Downloaded wallet JSON files** (created by the fetcher).
- **Feature CSVs** produced by the metrics notebook.

Several notebooks support environment variables for paths and runtime parameters (e.g., proxy URL, dataset path).

