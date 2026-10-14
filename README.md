# 🔍 LLM Text Source Classification
### *Distinguishing Human-Authored Text from 11 Large Language Model Architectures*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

The rapid proliferation of generative artificial intelligence has made it increasingly challenging to verify the origin and provenance of digital text. While general AI detectors typically formulate the task as a binary classification problem (*AI vs. Human*), practical forensics, intellectual property verification, and academic integrity require fine-grained **source attribution**: determining *which specific model* generated a given text.

This repository implements and benchmarks an end-to-end multi-class attribution pipeline evaluated on a balanced corpus of **996,000 text samples** across **12 distinct classes**:
- **1 Human-written benchmark**
- **11 LLM Architectures**: `ChatGPT`, `GPT-2`, `GPT-3`, `GPT-4`, `LLaMA-Chat`, `Mistral`, `Mistral-Chat`, `Cohere`, `Cohere-Chat`, `MPT`, and `MPT-Chat`.

---

## 🏆 Key Results

Across three distinct experimental tracks spanning classical frequency models, sentence embeddings, and deep sequential networks, our **Bidirectional GRU with Additive (Bahdanau) Attention** achieved the overall highest performance:

| Track | Architecture | Representations / Features | Test Accuracy | Macro F1 |
|:---|:---|:---|:---:|:---:|
| **Track 3 (Sequential / DL)** | **BiGRU + Additive Attention** | **Learned Token Embeddings (256-d)** | **87.13%** | **0.871** |
| **Track 3 (Sequential / DL)** | BiLSTM + Additive Attention | Frozen BERT WordPiece (768-d) | **81.44%** | 0.814 |
| **Track 1 (Classical / Hybrid)** | Deep Neural Network (4096-2048) | TF-IDF (N-gram 1–3, 50k features) | **81.19%** | 0.820 |
| **Track 1 (Classical / Hybrid)** | LinearSVC | TF-IDF (Hashing $2^{18}$ buckets) | **77.80%** | 0.786 |
| **Track 1 (Classical / Hybrid)** | LightGBM | Word2Vec (100-d) + Stylometrics | **76.95%** | 0.768 |
| **Track 3 (Transformer Finetune)**| BERT (Top 2 layers unfrozen) + DNN | `bert-base-uncased` | **69.56%** | 0.696 |
| **Track 3 (Transformer LoRA)** | E5 + LoRA Adapters ($r=8$) | `intfloat/e5-base` | **66.36%** | 0.663 |
| **Track 2 (Sentence Embeddings)**| Frozen E5-base-v2 + MLP | Sentence-level 768-d vectors | **63.01%** | 0.628 |
| **Track 1 (Classical / Hybrid)** | MLP | Word2Vec (100-d) + Stylometrics | **60.26%** | 0.595 |
| **Track 2 (Sentence Embeddings)**| Random Forest | BGE-base-en-v1.5 / E5-base-v2 | **57.72%** | 0.570 |

---

## 🧠 Why Recurrent Attention Outperformed Transformers

1. **Syntactic Rhythm vs. Semantic Content**: Sentence transformers (`MiniLM`, `MPNet`, `BGE`, `E5`) are pre-trained to cluster documents by *topic semantics*. In this dataset, all 12 classes discuss overlapping open-domain topics; frozen sentence embeddings inadvertently suppress the fine-grained token-to-token transition quirks that reveal generative signatures.
2. **Dynamic Additive Attention**: Instead of mean-pooling tokens, our custom additive attention layer with padding masking ($-10^9$) allows the network to assign sharp attention weights to anomalous punctuation distributions, contractions (`'ve`), formatting markers, and connective transitions (`furthermore`, `overall`).
3. **Efficiency Under Compute Constraints**: The BiGRU model (~2.6M parameters) converges rapidly and trains to full convergence within resource budgets, whereas fine-tuning 110M+ parameter transformers over 1M texts required early truncation on available GPUs.

---

## 🔬 Experimental Tracks

```
                               ┌───────────────────────────────────────────────┐
                               │           Input Text (996,000 samples)        │
                               └──────────────────────┬────────────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         │                                            │                                            │
         ▼                                            ▼                                            ▼
┌───────────────────────────┐             ┌───────────────────────┐             ┌─────────────────────────────┐
│ Track 1: Classical/Hybrid │             │ Track 2: Sentence     │             │ Track 3: Deep Sequential &  │
│ Feature Engineering       │             │ Transformers          │             │ Attention Architectures     │
├───────────────────────────┤             ├───────────────────────┤             ├─────────────────────────────┤
│ • TF-IDF (Hashing / N-gram│             │ • all-MiniLM-L6-v2    │             │ • BiGRU + Additive Attention│
│ • Word2Vec (Skip-gram 100)│             │ • all-mpnet-base-v2   │             │ • BiLSTM + BERT Embeddings  │
│ • Stylometrics (TTR, etc.)│             │ • BAAI/bge-base-en    │             │ • BERT Fine-Tuning + DNN    │
│ • SVM, LightGBM, Deep NN  │             │ • intfloat/e5-base-v2 │             │ • E5 + LoRA Adapters (r=8)  │
└───────────────────────────┘             └───────────────────────┘             └─────────────────────────────┘
```

### 1. Data Analysis & Preprocessing
- **Percentile Truncation**: Exploratory analysis revealed a heavy right tail (max 13,261 words). A **512-word threshold** preserves >99% of samples in full, eliminating outlier distortion while aligning with standard transformer windows.
- **Deduplication**: Addressed cross-split exact matches across train, validation, and test partitions to ensure zero evaluation contamination.
- **Length Bias Discovery**: While TF-IDF + DNN reached 81.19%, frequency models partially exploit document-length disparities across classes (e.g., GPT-3 averages 131 words; LLaMA-Chat averages 247 words) rather than pure stylistic traits.

### 2. Recurrent Attention Architecture (`bigru_additive_attention.ipynb`)
- **WordPiece Tokenization**: 384 max sequence length.
- **Trainable Embedding Layer**: 256-dimensional learned vector space.
- **Encoder**: 2-layer Bidirectional GRU (256 hidden units per direction $\to$ 512 total).
- **Additive Attention Mechanism**:
  $$\alpha_t = \frac{\exp(\mathbf{v}^\top \tanh(W \mathbf{h}_t + b))}{\sum_{k} \exp(\mathbf{v}^\top \tanh(W \mathbf{h}_k + b))}, \quad \mathbf{c} = \sum_{t} \alpha_t \mathbf{h}_t$$
- **Classification Head**: $\mathbf{c} \to \text{Linear}(512 \to 256) \to \text{LayerNorm} \to \text{ReLU} \to \text{Dropout}(0.4) \to \text{Linear}(256 \to 12)$.
- **Scheduler**: `CosineAnnealingLR` with AdamW optimizer.

### 3. Key Attribution Failure Modes & Confusion Patterns
- **High Separability**: `GPT-4` ($F1 = 0.817$) and `LLaMA-Chat` ($F1 = 0.769$) are readily distinguishable due to structured discourse markers and distinct syntactic cadence.
- **Family Entanglement**: `Cohere` vs. `Cohere-Chat` ($F1 \approx 0.57$) represent the most difficult boundary due to identical base tokenizers and shared foundational training data.
- **The Human Asymmetry**: The classifier exhibits **90.5% precision** but **58.3% recall** on human text—demonstrating high confidence when predicting human authorship, but occasionally misclassifying standard, formal human writing as LLM-generated.

---

## 📂 Repository Structure

```
.
├── docs/
│   ├── Project_Report.pdf                     # Complete 34-page technical report
│   └── Project_Presentation.pdf               # Project presentation deck
│
├── eda_and_preprocessing/
│   ├── data_analysis_and_eda.ipynb            # Dataset distributions, length analysis, token frequencies
│   ├── data_cleaning_and_transforms.ipynb     # Outlier removal, text truncation, and sampling
│   └── sequence_length_windowing.ipynb        # Sequence normalization and windowing transforms
│
├── classical_and_hybrid_models/
│   ├── tfidf_deep_neural_network.ipynb        # TF-IDF (1-3 ngrams) + Deep Neural Network (81.19%)
│   ├── tfidf_word2vec_lightgbm_svm_mlp.ipynb  # Hybrid TF-IDF + Word2Vec + Stylometric features
│   ├── tfidf_nn_ngram3.ipynb                  # TF-IDF multi-layer neural network variations
│   └── tfidf_nn_baseline.ipynb                # Sparse matrix generator and baseline dense net
│
├── sentence_embeddings/
│   ├── sentence_transformers_benchmark.ipynb  # MiniLM, MPNet, BGE, and E5 across 9 classifiers
│   └── sbert_tfidf_svm.ipynb                  # SBERT + TF-IDF concatenated SVM pipeline
│
├── recurrent_attention_models/
│   ├── bigru_additive_attention.ipynb         # Top model: BiGRU + Additive Attention (87.13%)
│   └── bilstm_attention_bert_embeddings.ipynb # BiLSTM with frozen BERT embeddings (81.44%)
│
└── transformer_models/
    ├── bert_finetune_dnn.py                   # Partial BERT fine-tuning (top 2 layers) + 5-layer DNN
    ├── bert_rnn_baseline.py                   # BERT representation + recurrent classification head
    └── e5_embeddings_lora_dnn.py              # E5 contrastive embeddings + LoRA fine-tuning
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for Track 3 deep learning pipelines)

### 2. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/aaditya-kr5892/llm-source-classification.git
cd llm-source-classification
pip install -r requirements.txt
```

*(Or install core dependencies directly)*:
```bash
pip install torch transformers datasets scikit-learn lightgbm gensim pandas numpy matplotlib seaborn
```

### 3. Running the Best Performing Model (BiGRU + Additive Attention)
Open and run `recurrent_attention_models/bigru_additive_attention.ipynb` in Jupyter Notebook, Google Colab, or Kaggle.

### 4. Running Transformer Fine-Tuning
Execute the standalone fine-tuning scripts with automatic FP16 mixed precision:
```bash
python transformer_models/bert_finetune_dnn.py
python transformer_models/e5_embeddings_lora_dnn.py
```

---

## 👥 Authors & Acknowledgments

- **Aaditya Kumar** ([@aaditya-kr5892](https://github.com/aaditya-kr5892))
- **Affan Pathan**
- **Ayush Yadav**
- **Rohit Patil**
- **Arjun Gangwar**

Supervised by **Sujeet Kumar** (PhD Scholar, Department of Computer Science & Engineering, IIT Dharwad).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
