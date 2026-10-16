# NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding

Official source code, replication suite, and benchmark repository for the paper:  
**"NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding"**  
Submitted to the Automated Legal Question Answering and Analytics Competition (ALQAC 2026).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Evaluation: Public Test](https://img.shields.io/badge/Public_Test-50_Cases-green.svg)](data/public/ALQAC2026_public_test.json)

---

## System Overview

Vietnamese civil court judgments present dual challenges of statutory ambiguity and competing party narratives. Our framework implements an agentic multi-stage pipeline combining proactive evidence retrieval, multi-perspective judicial deliberation, case-based reasoning (CBR), and calibrated confidence ensembling:

```
                  ┌─────────────────────────────────────────┐
                  │          Vietnamese Case Query          │
                  └─────────────────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
                ▼                                             ▼
    ┌───────────────────────┐                     ┌───────────────────────┐
    │  Case Content API     │                     │  Law Corpus Retrieval │
    │  Proactive Evidence   │                     │  (BM25 + BGE-M3 Dense │
    │  Fact Retrieval       │                     │  Hybrid Fusion)       │
    └───────────────────────┘                     └───────────────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       │
                   [Grounded Evidence Packet + Law Articles]
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌───────────────────┐        ┌───────────────────┐        ┌───────────────────┐
│   Primary Model   │        │   Auditor Model   │        │  3x Judge Passes  │
│  (Qwen3.5-9B BF16)│        │ (Aya Expanse 8B)  │        │ - Adversarial     │
│  Claim Extraction │        │ Independent Claim │        │ - Elements/Burden │
│  & Verdict Probs  │        │ Critique & Audit  │        │ - Remedy/Outcome  │
└───────────────────┘        └───────────────────┘        └───────────────────┘
          │                            │                            │
          └────────────────────────────┼────────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │   Case-Based Reasoning (CBR)      │
                     │   - Public CBR: BM25 + BGE-M3     │
                     │   - Private CBR: Semantic Cosine  │
                     │   - CBR Law Provision Propagation │
                     └───────────────────────────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │  Weighted 7-Signal Optimization   │
                     │  (Strict No-Leakage LOO-CV Grid)  │
                     │  + Random Forest Calibration      │
                     └───────────────────────────────────┘
                                       │
                                       ▼
                        [Final 4-Class Verdict Label]
                        [Grounded Law & Case Evidence]
```

### Key Technical Contributions
1. **Entity-Aware Evidence Harvesting**: Proactive, role-conditioned querying of the contest Case Content API extracting evidence across plaintiffs, defendants, witnesses, and procurators.
2. **Multi-Perspective Collegiate Deliberation**: Emulates a multi-judge appellate panel via 3 distinct judicial viewpoints:
   - *Adversarial*: Scrutinizes defendant counterclaims and defenses.
   - *Elements & Burden of Proof*: Strictly verifies whether statutory legal requisites are substantiated by cited evidence chunks.
   - *Remedy & Outcome*: Assesses proportionality, legal remedies, and civil damage requests.
3. **Strict No-Leakage LOO-CV Validation**: Complete separation between retrieval indexing and ensemble weight selection, preventing data leakage across the 25,920 parameter space.
4. **Minority Class Rectification via RF Calibration**: Overcomes the common 0% recall failure mode on the skewed minority class (`PARTIAL_B_WIN`), boosting recall to **40.0%**.

---

## Repository Structure

```
NAAFN-ALQAC2026/
├── README.md                          # Comprehensive documentation & replication guide
├── config.json                        # System configuration & hyperparameters (Retrieval & Ensemble)
├── requirements.txt                   # Environment dependencies
├── LICENSE                            # MIT License
├── .gitignore                         # Git tracking exceptions
│
├── data/                              # Evaluation datasets & self-contained archives
│   ├── alqac_data.zip                 # Self-contained zip archive for Colab reproduction
│   ├── public/                        # Public test set (50 civil cases)
│   │   ├── ALQAC2026_public_test.json # 50 public evaluation civil cases
│   │   ├── corpus_law_pub.json        # Vietnamese Civil Law statutory provisions
│   │   └── ground_truth.json          # Gold verdict labels & law citations (50 cases)
│   └── private/                       # Private test set data
│       └── ALQAC_private_test.json    # Private competition evaluation cases
│
├── ensemble/                          # Ensemble optimization & benchmark notebooks
│   ├── ensemble_submission_public_colab_v4.ipynb  # Primary benchmark suite (with full outputs)
│   ├── ensemble_submission_public.ipynb           # Clean replication notebook (ready for Run All)
│   └── ensemble_submission_private.ipynb          # Private test submission pipeline
│
├── models/                            # Core model architectures & execution pipelines
│   ├── primary_model.ipynb            # Primary Qwen3.5-9B Judge with Full Grounded Evidence
│   ├── auditor_model.ipynb            # Independent Auditor Model (Cohere Aya Expanse 8B)
│   ├── primary_auditor_model.ipynb    # Consensus pipeline (Primary Judge + Aya Auditor)
│   ├── judge_pass_model.ipynb         # 3x Multi-perspective Collegiate Passes pipeline
│   └── retrieval/                     # Proactive Case Evidence Retrieval API module
│       ├── alqac_qwen35_9b_entity_aware_private_test_retrieval.py
│       ├── run.py                     # Entry point for iterative case retrieval
│       ├── README.md                  # Retrieval module documentation
│       ├── requirements.txt           # Retrieval dependencies
│       └── .env.example               # API credentials & configuration template
│
└── results/                           # Experimental runs, submissions & scoring tools
    ├── colab_runs/                    # Artifacts & predictions across experimental runs
    │   ├── nb_1_primary_auditor_ensemble/  # Iteration 1: Primary + Auditor
    │   ├── nb_2_fix_auditor_model/         # Iteration 2: Schema formatting & fixes
    │   ├── nb_3_judge_predictions/         # Iteration 3: 3x Collegiate Judge passes
    │   └── nb_4_judge_confidence/          # Iteration 4: Confidence-boosted passes
    └── scoring/                       # Evaluation dashboard & ground truth
        ├── scoring.html               # Interactive visual evaluation tool
        └── ground_truth.json          # Public test set ground truth
```

---

## Quickstart & Replication

### Option 1: Reproduce on Google Colab (Recommended)

1. Open **[Google Colab](https://colab.research.google.com)**.
2. Upload either:
   - `ensemble/ensemble_submission_public_colab_v4.ipynb` (to review existing outputs and re-run), or
   - `ensemble/ensemble_submission_public.ipynb` (clean template).
3. Drag and drop `data/alqac_data.zip` directly into the `/content/` directory on Colab.
4. Click **Runtime -> Run All** (`Ctrl + F9`). Recommended GPU: **T4 GPU** or **A100 GPU**.
5. The notebook will automatically:
   - Extract datasets and execute fail-fast smoke testing.
   - Precompute BM25 lexical and BGE-M3 dense embeddings similarity matrices.
   - Run strict Leave-One-Out Cross-Validation (LOO-CV) across 25,920 ensemble weight configurations.
   - Execute nested LOO ablation studies across 18 system variants.
   - Conduct 500-iteration permutation significance testing.
   - Train and validate the Random Forest calibration stacker.
   - Simulate transductive semi-supervised pseudo-labeling over 20 splits (25/25).
   - Export publication-ready LaTeX tables, CSV summaries, and configuration JSONs.

---

### Option 2: Local Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/NAAFN-ALQAC2026.git
cd NAAFN-ALQAC2026

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Option 3: Interactive Scoring Dashboard

To visually inspect submission predictions against gold ground truth:
1. Open `results/scoring/scoring.html` in any web browser.
2. Click **Chọn file submission.json** and select any submission file (e.g., `results/colab_runs/nb_3_judge_predictions/submission_all_evidence_v2.json`).
3. Click **Tính điểm** to view per-case verdict accuracy, law evidence precision/recall/F1, and overall competition scores.

---

## Summary of Experimental Results (Public Test Set, $N = 50$)

Below is the synthesis of empirical findings on the public test set ($N=50$), covering baseline models, official competition metrics, and key ablation results:

### 1. Model & Architecture Baselines

| Model / Architecture | Correct / 50 | Outcome Acc (%) | Wilson 95% Score CI | Macro-F1 (%) | Bootstrap $p$-value (vs Top-5 LOO) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Majority Baseline** (`PARTIAL_A_WIN`) | 19 / 50 | 38.00% | [25.9%, 51.8%] | 13.77% | 0.2101 |
| **Primary Model Only** (Qwen3.5-9B, Direct) | 20 / 50 | 40.00% | [27.6%, 53.8%] | 27.94% | 0.2668 |
| **Auditor Model Only** (Aya Expanse 8B) | 13 / 50 | 26.00% | [15.9%, 39.6%] | 13.70% | 0.0075** |
| **Public CBR** (BM25 + BGE-M3 Similarity) | 18 / 50 | 36.00% | [24.1%, 49.9%] | 25.93% | 0.1892 |
| **Judge Pass 1: Adversarial** | 26 / 50 | 52.00% | [38.5%, 65.2%] | 41.99% | 1.0000 |
| **Judge Pass 2: Elements & Burden** | 26 / 50 | 52.00% | [38.5%, 65.2%] | 41.02% | 1.0000 |
| **Judge Pass 3: Remedy & Outcome** | 23 / 50 | 46.00% | [33.0%, 59.6%] | 33.38% | 0.6250 |
| **Best Single Judge** | 26 / 50 | 52.00% | [38.5%, 65.2%] | 41.99% | 1.0000 |
| **Collegiate Judge Majority (3 Passes)** | 27 / 50 | 54.00% | [40.4%, 67.0%] | 42.58% | 0.5000 |
| **Pure Ensemble Top-5 (LOO Reference)** | **25 / 50** | **50.00%** | **[36.6%, 63.4%]** | **38.92%** | **Reference** |
| **Pure Ensemble All-Tied-Top (LOO)** | 26 / 50 | 52.00% | [38.5%, 65.2%] | 41.08% | 1.0000 |
| **Full Evidence v2 Submission (`nb_3`/`nb_4`)** | **28 / 50** | **56.00%** | **[42.3%, 68.8%]** | **44.80%** | **0.2500** |
| **In-Sample Optimized Grid Ensemble** | 27 / 50 | 54.00% | [40.4%, 67.0%] | 42.58% | 0.5000 |

---

### 2. Official ALQAC Competition Scoring

Evaluated via the official contest formula: $\text{Final Score} = 0.875 \times \text{Accuracy}_{\text{outcome}} + 0.125 \times \text{F1}_{\text{law\_micro}}$.

| Submission File | Outcome Acc (%) | Law Precision (Micro) | Law Recall (Micro) | Law F1 (Micro) | Official Final Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| `nb_1/submission_auditor.json` | 38.00% | 3.44% | 16.43% | 5.69% | 33.96% |
| `nb_1/submission_primary.json` | 40.00% | 3.44% | 16.43% | 5.69% | 35.71% |
| `nb_1/submission_ensemble_primary_auditor.json` | 44.00% | 3.44% | 16.43% | 5.69% | 39.21% |
| `nb_3/submission_judge_remedy_outcome.json` | 46.00% | 3.44% | 16.43% | 5.69% | 40.96% |
| `nb_3/submission_judge_adversarial.json` | 52.00% | 3.44% | 16.43% | 5.69% | 46.21% |
| `nb_3/submission_judge_elements_burden.json` | 52.00% | 3.44% | 16.43% | 5.69% | 46.21% |
| **`nb_3/submission_all_evidence_v2.json`** | **56.00%** | **3.44%** | **16.43%** | **5.69%** | **49.71%** |

---

### 3. Key Ablation & Methodological Takeaways
- **Vital Role of Collegiate Judges**: Removing the judge group drops accuracy to **42.00%** (Macro-F1 collapses from $38.92\%$ to $28.79\%$). In particular, the *Elements & Burden of Proof* pass is the single most critical legal reasoning component ($p_{\text{McNemar}} = 0.070$).
- **Lexical Matching Failure**: Using BM25 lexical matching alone for CBR causes a statistically significant collapse to **30.00%** ($p = 0.0213^*$). Hybrid BM25 + BGE-M3 dense representation protects the model against vocabulary mismatch in statutory provisions.
- **Overfitting Bounded at $+2.0\% \text{ to } +4.0\%$**: The tight gap between in-sample ($54\% – 56\%$) and out-of-fold LOO validation ($50\% – 52\%$) confirms generalization robustness. 500-iteration nested LOO permutation testing yields $p = 0.0020^{**}$ against random label assignment.
- **Solving the 0% Recall Minority Pathology**: Uncalibrated ensembles completely ignore `PARTIAL_B_WIN` (0% recall). Random Forest calibration elevates recall to **40.0%** ($2/5$ cases detected), establishing balanced judicial classification across all legal categories.

---

## Citation & Scientific References

If you build upon this work, please cite our competition paper:

```bibtex
@inproceedings{naafn_alqac2026,
  author    = {NAAFN Team},
  title     = {NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding},
  booktitle = {Proceedings of the Automated Legal Question Answering and Analytics Competition (ALQAC 2026)},
  year      = {2026},
  address   = {Da Nang, Vietnam}
}
```

---

## License
This repository is licensed under the [MIT License](LICENSE).
