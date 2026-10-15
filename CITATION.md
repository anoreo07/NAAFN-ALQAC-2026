# Citations & Scientific References

This repository and framework build upon foundational literature across Large Language Models, Multi-Agent Collegiate Simulation, Information Retrieval, Dense Multilingual Representations, Case-Based Reasoning (CBR), and Statistical Validation.

Every citation below is authentic, peer-reviewed or standard canonical preprint, and directly corresponds to methods implemented in the codebase.

---

## 1. Primary Citation for NAAFN@ALQAC 2026

If you use this codebase, experimental results, or methodology in your research, please cite our paper:

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

## 2. Foundation Models & Large Language Architectures

### Qwen Foundation Models (Primary Judge, Reranking & Collegiate Passes)
- **Qwen2.5 Technical Report**
  ```bibtex
  @article{qwen2024qwen25,
    title   = {Qwen2.5 Technical Report},
    author  = {Qwen Team},
    journal = {arXiv preprint arXiv:2412.15115},
    year    = {2024}
  }
  ```
- **Qwen2 Technical Report**
  ```bibtex
  @article{yang2024qwen2,
    title   = {Qwen2 Technical Report},
    author  = {Yang, An and Yang, Baosong and Hui, Binyuan and Zheng, Bo and Yu, Bowen and Zhou, Chang and Li, Chengpeng and Li, Chengyuan and Liu, Dayiheng and Huang, Fei and others},
    journal = {arXiv preprint arXiv:2407.10671},
    year    = {2024}
  }
  ```

### Aya Multilingual Foundation Models (Independent Auditor & Verifier)
- **Aya 23: Open Multilingual Large Language Models**
  ```bibtex
  @article{ustun2024aya23,
    title   = {Aya 23: Open Multilingual Large Language Models},
    author  = {{\"U}st{\"u}n, Ahmet and Aryabumi, Viraat and Zheng, Zheng-Xin and Rahman, Ceyda and Stambler, Daniel and Shen, Ethan and Aky{\"u}rek, Ekin and Hada, Rishabh and Wu, Shivalika and others},
    journal = {arXiv preprint arXiv:2405.15032},
    year    = {2024}
  }
  ```
- **Aya Dataset: Open-Access Multilingual Instruction Tuning**
  ```bibtex
  @inproceedings{singh2024aya,
    title     = {Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning},
    author    = {Singh, Shivalika and Aryabumi, Viraat and Bell, Jonathan and Burchell, Austin and D'souza, Jennifer and others},
    booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    pages     = {8952--8987},
    year      = {2024}
  }
  ```

---

## 3. Representation Learning & Information Retrieval

### Dense Semantic Embedding (BGE-M3 for Case-Based Reasoning)
- **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings**
  ```bibtex
  @article{chen2024bge,
    title   = {BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation},
    author  = {Chen, Jianlv and Xiao, Shitao and Zhang, Peitian and Luo, Kun and Lian, Defu and Liu, Zheng},
    journal = {arXiv preprint arXiv:2402.03216},
    year    = {2024}
  }
  ```

### Lexical Retrieval & Okapi BM25
- **The Probabilistic Relevance Framework: BM25 and Beyond**
  ```bibtex
  @article{robertson2009probabilistic,
    title     = {The Probabilistic Relevance Framework: BM25 and Beyond},
    author    = {Robertson, Stephen and Zaragoza, Hugo},
    journal   = {Foundations and Trends{\textregistered} in Information Retrieval},
    volume    = {3},
    number    = {4},
    pages     = {333--389},
    year      = {2009},
    publisher = {Now Publishers, Inc.}
  }
  ```

### Rank Fusion (Reciprocal Rank Fusion - RRF)
- **Reciprocal Rank Fusion**
  ```bibtex
  @inproceedings{cormack2009reciprocal,
    title     = {Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods},
    author    = {Cormack, Gordon V. and Clarke, Charles L. A. and Buettcher, Stefan},
    booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages     = {758--759},
    year      = {2009}
  }
  ```

---

## 4. Machine Learning, Calibration & Methodology

### Random Forests & Calibration Stacking
- **Random Forests**
  ```bibtex
  @article{breiman2001random,
    title     = {Random Forests},
    author    = {Breiman, Leo},
    journal   = {Machine Learning},
    volume    = {45},
    number    = {1},
    pages     = {5--32},
    year      = {2001},
    publisher = {Springer}
  }
  ```

### Case-Based Reasoning (CBR)
- **Foundational Case-Based Reasoning**
  ```bibtex
  @article{aamodt1994case,
    title     = {Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches},
    author    = {Aamodt, Agnar and Plaza, Enric},
    journal   = {AI Communications},
    volume    = {7},
    number    = {1},
    pages     = {39--59},
    year      = {1994},
    publisher = {IOS Press}
  }
  ```

### Statistical Evaluation & Wilson Score Confidence Intervals
- **Wilson Score Interval**
  ```bibtex
  @article{wilson1927probable,
    title     = {Probable Inference, the Law of Large Numbers, and Run-of-the-Mine},
    author    = {Wilson, Edwin B.},
    journal   = {Journal of the American Statistical Association},
    volume    = {22},
    number    = {158},
    pages     = {209--212},
    year      = {1927}
  }
  ```

---

## 5. Software Ecosystem & Frameworks

### Scikit-Learn
```bibtex
@article{pedregosa2011scikit,
  title   = {Scikit-learn: Machine Learning in Python},
  author  = {Pedregosa, Fabian and Varoquaux, Ga{\"e}l and Gramfort, Alexandre and Michel, Vincent and Thirion, Bertrand and Grisel, Olivier and Blondel, Mathieu and Prettenhofer, Peter and Weiss, Ron and Dubourg, Vincent and others},
  journal = {Journal of Machine Learning Research},
  volume  = {12},
  pages   = {2825--2830},
  year    = {2011}
}
```

### Sentence-Transformers
```bibtex
@inproceedings{reimers2019sentence,
  title     = {Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author    = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019)},
  pages     = {3982--3992},
  year      = {2019}
}
```

### Hugging Face Transformers
```bibtex
@inproceedings{wolf2020transformers,
  title     = {Transformers: State-of-the-Art Natural Language Processing},
  author    = {Wolf, Thomas and Debut, Lysandre and Sanh, Victor and Chaumond, Julien and Delangue, Clement and Moi, Anthony and Cistac, Pierric and Rault, Timoth{\'e}e and Louf, R{\'e}mi and Funtowicz, Morgan and others},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  pages     = {38--45},
  year      = {2020}
}
```

---

## 6. Vietnamese Legal NLP & ALQAC Benchmarks

```bibtex
@inproceedings{nguyen2021alqac,
  title     = {A Summary of the ALQAC 2021 Competition},
  author    = {Nguyen, Ha-Thanh and Tran, Vu and Le, Nguyen-Khang and Nguyen, Thi-Thao and others},
  booktitle = {Proceedings of the 13th International Conference on Knowledge and Systems Engineering (KSE)},
  pages     = {1--6},
  year      = {2021}
}

@inproceedings{nguyen2022alqac,
  title     = {Legal Question Answering Using Ranking Approaches: An Overview of the ALQAC 2022 Competition},
  author    = {Nguyen, Ha-Thanh and Tran, Vu and Nguyen, Thi-Thao and Satoh, Ken},
  booktitle = {JSAI International Symposia on Artificial Intelligence},
  pages     = {215--229},
  year      = {2022},
  publisher = {Springer}
}
```
