# NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding

Official source code and replication repository for the paper:  
**"NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding"**  
Submitted to ALQAC 2026 (Automated Legal Question Answering and Analytics Competition).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏛️ System Overview

Our framework addresses the dual challenges of statutory ambiguity and multi-perspective judicial reasoning in Vietnamese civil judgments through an end-to-end agentic workflow:

```
                  +-----------------------------------------+
                  |         Vietnamese Case Query           |
                  +-----------------------------------------+
                                       |
                +----------------------+----------------------+
                |                                             |
                v                                             v
    +-----------------------+                     +-----------------------+
    | Case Content API      |                     | Law Corpus Retrieval  |
    | Proactive Evidence    |                     | (BM25 + BGE-M3 + Qwen |
    | Fact Retrieval        |                     | RRF Hybrid Fusion)    |
    +-----------------------+                     +-----------------------+
                |                                             |
                +----------------------+----------------------+
                                       |
                   [Grounding Context + Evidence Chunks]
                                       |
          +----------------------------+----------------------------+
          |                            |                            |
          v                            v                            v
+-------------------+        +-------------------+        +-------------------+
|   Primary Model   |        |   Auditor Model   |        | 3x Judge Passes   |
| (Qwen3.5-9B/AYA)  |        | (Indep. Critique) |        | - Adversarial     |
| Probabilities P(c)|        | Label + Confidence|        | - Elements Burden |
|                   |        |                   |        | - Remedy Outcome  |
+-------------------+        +-------------------+        +-------------------+
          |                            |                            |
          +----------------------------+----------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Case-Based Reasoning (CBR)       |
                     |  - Public CBR: BM25 + BGE-M3      |
                     |  - Private CBR: Semantic Cosine   |
                     |  - CBR Law Propagation            |
                     +-----------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |  Weighted 7-8 Signal Ensemble     |
                     |  + Random Forest Calibration      |
                     +-----------------------------------+
                                       |
                                       v
                        [Final 4-Class Verdict Label]
                        [Law & Case Evidence IDs]
```

### Key Highlights
- **Entity-Aware Evidence Retrieval**: Iterative, agentic probing of the contest Case Content API with party-specific role extraction.
- **Multi-Perspective Judge Simulation**: Emulates collegiate panel deliberations via three distinct legal perspectives (*Adversarial*, *Elements & Burden of Proof*, *Remedy & Outcome*).
- **Strict No-Leakage LOO-CV**: Leave-One-Out Cross-Validation ensuring zero data contamination between retrieval indexes and prediction validation.
- **Confidence Calibration**: Non-linear calibration with Random Forest mitigating skewed minority class distributions (`PARTIAL_B_WIN`).

---

## 📂 Repository Structure

```
NAAFN-ALQAC2026/
├── README.md                          # Comprehensive documentation & replication guide
├── requirements.txt                   # Environment dependencies
├── LICENSE                            # MIT License
│
├── data/                              # Evaluation dataset & pre-computed artifacts
│   ├── ALQAC2026_public_test.json     # 50 public evaluation civil cases
│   ├── corpus_law_pub.json            # Vietnamese Civil Law corpus provisions
│   ├── public_embeddings_bge_m3.npy   # Dense BGE-M3 embeddings (1024-dim)
│   ├── primary_predictions.json       # Base primary model probabilities
│   ├── auditor_predictions.json       # Auditor model predictions & confidence
│   ├── ground_truth.json              # Gold labels for the 50 public cases
│   ├── alqac_data.zip                 # Self-contained zip archive for Colab reproduction
│   └── judge_passes/                  # Multi-agent simulated judge passes
│       ├── adversarial/               # Adversarial perspective (50 cases)
│       ├── elements_burden/           # Elements & burden of proof (50 cases)
│       └── remedy_outcome/            # Remedy appropriateness (50 cases)
│
├── models/                            # Core model architectures & execution notebooks
│   ├── retrieval/                     # Proactive Case Evidence Retrieval API
│   │   ├── alqac_qwen35_9b_entity_aware_private_test_retrieval.py
│   │   ├── run.py                     # Entry point for evidence retrieval
│   │   ├── README.md                  # Retrieval module docs
│   │   ├── requirements.txt           # Retrieval dependencies
│   │   └── .env.example               # Configuration template
│   ├── primary_model.ipynb            # Primary Qwen3.5-9B Judge with Full Grounded Evidence
│   ├── primary_auditor_model.ipynb    # End-to-end consensus model (Primary Judge + Aya Auditor)
│   ├── auditor_model.ipynb            # Standalone Independent Auditor & Verification model
│   └── judge_pass_model.ipynb         # 3x Multi-perspective Judge Passes pipeline
│
├── ensemble/                          # Ensemble & Calibration systems
│   ├── ensemble_submission_public_colab_v4.ipynb  # Primary replication notebook (Colab ready)
│   ├── ensemble_submission_public_colab_v3.ipynb  # Baseline Colab notebook
│   ├── ensemble_submission_public.ipynb          # Public evaluation pipeline
│   └── ensemble_submission_private.ipynb         # Full private test submission pipeline
│
├── results/                           # Competition submissions & evaluation metrics
│   ├── submissions/                   # Validated submission JSON files
│   │   ├── submission_ensemble_primary_auditor.json
│   │   └── submission_all_evidence_v2.json
│   ├── colab_runs/                    # Output logs across experimental iterations (nb_1 - nb_4)
│   └── scoring/                       # Local evaluation tool & ground truth
│       ├── scoring.html               # Interactive visualization dashboard
│       └── ground_truth.json
│
└── docs/                              # Research documentation & architecture artifacts
    ├── architecture_figure.tex        # LaTeX TikZ diagram code for paper
    ├── SystemDescription.txt          # Competition technical summary
    └── RETRIEVAL_WEIGHTS_REPORT.md    # Retrieval ablation & fusion report
```

---

## 🚀 Quickstart & Replication

### Option 1: Reproduce on Google Colab (Recommended)

1. Open **[Google Colab](https://colab.research.google.com)**.
2. Upload `ensemble/ensemble_submission_public_colab_v4.ipynb`.
3. Drag and drop `data/alqac_data.zip` into `/content/` on Colab.
4. In Colab, click **Runtime -> Run All** (`Ctrl + F9`).
5. The notebook will automatically:
   - Run Smoke Test verification.
   - Execute strict Leave-One-Out Cross-Validation (LOO-CV).
   - Compute Wilson 95% Score Confidence Intervals and Paired Bootstrap Tests (10,000 resamples).
   - Run Full Ablation Studies & Overfitting Diagnostics.
   - Train and evaluate the Random Forest Calibration module.
   - Export publication-ready LaTeX tables and CSV reports into `/content/results/`.

---

### Option 2: Local Setup & Execution

#### 1. Environment Setup
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

#### 2. Run Evidence Retrieval API (Optional / Data Collection)
```bash
cd models/retrieval
cp .env.example .env
# Edit .env with your ALQAC_TOKEN and file paths
python run.py
```

#### 3. Run Local Scoring & Visualizer
To inspect predictions against ground truth using the interactive HTML dashboard:
- Open `results/scoring/scoring.html` in any modern web browser.
- Select `results/submissions/submission_ensemble_primary_auditor.json` to view per-case accuracy, confusion matrix, and evidence recall.

---

## 📊 Summary of Experimental Results (Public Test Set)

| Method / Configuration | Accuracy (%) | 95% Wilson CI | Macro-F1 | p-value (vs Pure Ens) |
|:-----------------------|:------------:|:-------------:|:--------:|:---------------------:|
| Primary Model (Only)   | 50.00%       | [36.6%, 63.4%] | 0.443    | 0.0078**              |
| Auditor Model (Only)   | 46.00%       | [32.9%, 59.7%] | 0.398    | 0.0020**              |
| 3x Judge Passes (Avg)  | 56.00%       | [42.3%, 68.8%] | 0.485    | 0.0312*               |
| Public CBR Hybrid      | 48.00%       | [34.8%, 61.5%] | 0.412    | 0.0039**              |
| **Pure Weighted Ens**  | **74.00%**   | **[60.4%, 84.1%]** | **0.697** | **Baseline**          |
| **RF-Calibrated Ens**  | **74.00%**   | **[60.4%, 84.1%]** | **0.718** | **0.0000***           |

*Results reproduced under strict Leave-One-Out validation on the 50 public cases.*

---

## 📝 Citation

If you use this codebase or framework in your research, please cite our paper:

```bibtex
@inproceedings{naafn_alqac2026,
  title={NAAFN@ALQAC 2026: An Agentic Ensemble Framework for Vietnamese Legal Case Outcome Prediction with Evidence Grounding},
  author={NAAFN Team},
  booktitle={Proceedings of the Automated Legal Question Answering and Analytics Competition (ALQAC 2026)},
  year={2026}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
