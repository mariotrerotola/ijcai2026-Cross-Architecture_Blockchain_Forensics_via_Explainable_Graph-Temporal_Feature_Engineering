# Cross-Architecture Blockchain Forensics via Explainable Graph-Temporal Feature Engineering

This repository contains the code used to build **graph/temporal feature representations** from blockchain activity and to train **interpretable** (and performance-oriented) machine learning models for forensic tasks across two chains:

- **Bitcoin**: wallet-level fraud detection.
- **Ethereum (ERC-20)**: scam token detection.

The project is organized as two main pipelines under:
- `Bitcoin_Fraud_Detection/`
- `Ethereum_ERC20_Scam_Detection/`

### Data availability

The datasets used in this work are **not included** in the repository.  
For dataset sharing and access requests, contact: `mario.trerotola@polito.it`.

### Environment and dependencies

Install dependencies from the root:

```bash
pip install -r requirements.txt
```

Some components have additional system/runtime requirements:
- **Playwright** (used for ChainAbuse scraping): run `playwright install` after `pip install`.

### Repository structure (high level)

- **`Bitcoin_Fraud_Detection/`**: data acquisition (wallet transactions), feature computation (wallet metrics), dataset assembly, and model notebooks (RF/SVM/MLP + explainability).
- **`Ethereum_ERC20_Scam_Detection/`**: token transfer extraction, graph-temporal feature extraction, and model notebooks (BRF/MLP/SVM evaluations and tuning).

### Reproducibility notes

- Several scripts and notebooks accept **environment variables** to override input/output paths or credentials (e.g., API keys, cookie strings, proxy URLs).
- The code is written to support reruns with checkpointing where applicable (e.g., token transfer extraction, ChainAbuse scraping).

### Citation

If you use this repository in academic work, please cite the corresponding paper (to be filled with the final bibliographic entry).

