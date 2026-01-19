# Bitcoin Fraud Detection

This module implements a wallet-level fraud detection pipeline for Bitcoin. It covers:

- Acquisition of wallet transaction histories.
- Feature engineering (graph, temporal, and volume statistics per wallet).
- Dataset assembly and labeling.
- Model training and evaluation, including interpretability (CIU, SHAP) and rule extraction (DEXiRE).

Datasets are not included.

### Pipeline overview

#### 1) Data acquisition

- `wallet_fetcher.ipynb`
  - Downloads wallet transaction JSON from Blockchain.com endpoints (`https://blockchain.info`), with retry logic and optional rotating proxies.
  - Filters wallet lists by class (e.g., Elliptic class `2` for licit) using `wallets_classes.csv`.

- `extract_latest_wallets/extract_latest_wallets.ipynb`
  - Collects a target number of recent transactions by walking backward from the latest block.
  - Extracts unique addresses from inputs and outputs and writes them to CSV.

#### 2) ChainAbuse scraping and cleaning (optional)

- `chainabuse_fetcher/chainabuse_fetcher.ipynb`
  - Scrapes Bitcoin scam reports from ChainAbuse using Playwright.
  - Supports cookie injection and resume from the last processed page.

- `chainabuse_fetcher/clean_data.ipynb`
  - Validates Bitcoin addresses, filters invalid entries, and deduplicates.
  - Writes a cleaned CSV for downstream use.

#### 3) Feature computation

- `extract_metrics.ipynb`
  - Validates downloaded JSON files and computes wallet-level metrics.
  - Produces a feature table (degrees, unique degrees, volumes, time gaps, activity duration, ratios).

#### 4) Dataset assembly and preprocessing

- `dataset_assembly.ipynb`
  - Merges feature tables, cleans data, removes non-feature columns, and writes the labeled dataset.

- `data_exploration.ipynb`
  - Exploratory analysis of the dataset (distributions, missing values, summary statistics).

#### 5) Modeling and interpretability

- `rf/random_forest_ciu.ipynb`
  - Random Forest training with preprocessing and `GridSearchCV` tuning.
  - Evaluation with report and confusion matrix.
  - Local explainability with CIU; optional SHAP blocks.

- `svm/svm.ipynb`
  - SVM (`SVC(probability=True)`) training with hyperparameter search.
  - Evaluation and optional explainability blocks.

- `mlp/mlp.ipynb`
  - MLP classifier with simple architecture search and evaluation on a held-out test set.

- `confident_learning/confident_learning.ipynb`
  - Label-noise detection with Cleanlab to filter potentially mislabeled fraud samples.
  - Retrains and compares performance after filtering.

- `rf/rf_mlp_distill_dexire.ipynb`
  - Teacher-student setup (Random Forest teacher, MLP student).
  - Synthetic augmentation (SMOTE), confidence filtering, student training.
  - Rule extraction via DEXiRE.

- `rf/extract rules.ipynb`
  - Post-processes extracted rules and exports them in LaTeX-friendly form.

### Installation notes

Dependencies are declared at the repository root (`../requirements.txt`).

If you use the ChainAbuse scraper, run:

```bash
playwright install
```

### Inputs and expected files

Inputs depend on the experiment configuration and local data layout. Typical inputs:
- Wallet list with labels (e.g., `wallets_classes.csv`).
- Downloaded wallet JSON files (created by the fetcher).
- Feature CSVs produced by the metrics notebook.

Several notebooks support environment variables for paths and runtime parameters (e.g., proxy URL, dataset path).

