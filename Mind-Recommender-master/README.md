# News Recommendation System with NRMS on MIND-small

## 1. Introduction

News recommendation is the task of ranking candidate articles for each reader based on their interests, recent behavior, and the available article catalog. A strong recommender helps readers discover relevant stories while handling fast-changing content, short article lifetimes, and sparse user histories.

The Microsoft News Dataset (MIND) is a large-scale benchmark for news recommendation. It contains anonymized user behavior logs, clicked news histories, impression-level candidate lists, article metadata, and entity/relation embeddings. This project targets the MIND-small split, which is compact enough for local experimentation while preserving the core ranking problem.

The goal of this project is to build an end-to-end PyTorch implementation of NRMS (Neural News Recommendation with Multi-Head Self-Attention). The system preprocesses MIND titles, initializes word embeddings from GloVe, trains with negative sampling, and evaluates impression-level ranking quality with AUC, MRR, nDCG@5, and nDCG@10.

## 2. Setup Instructions

Create and activate a Python environment:

```bash
cd mind-recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download MIND-small from Microsoft and place the extracted files in this layout:

```text
data/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
├── MINDsmall_dev/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
└── glove/
    └── glove.6B.300d.txt
```

Typical download commands:

```bash
curl -L -o MINDsmall_train.zip https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip
curl -L -o MINDsmall_dev.zip "https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_dev.zip"
unzip MINDsmall_train.zip -d data/MINDsmall_train
unzip MINDsmall_dev.zip -d data/MINDsmall_dev
```

Download GloVe 6B from Stanford, extract it, and copy `glove.6B.300d.txt` to `data/glove/`.

Run notebooks:

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_training.ipynb
```

Train from the command line:

```bash
python src/train.py --epochs 5 --batch-size 64
```

Evaluate the best checkpoint:

```bash
python src/evaluate.py --checkpoint models/nrms_best.pt
```

## 3. Methodology

### Data Preprocessing Pipeline

The preprocessing flow starts by loading `news.tsv` and `behaviors.tsv` using stable MIND column names. Article titles are tokenized with NLTK's `TreebankWordTokenizer`, lowercased, and encoded as fixed-length sequences of 30 tokens. The vocabulary is built from training titles with a minimum frequency of 2, using `<PAD>` and `<UNK>` special tokens.

GloVe 300-dimensional embeddings initialize the word embedding matrix. Tokens found in GloVe use pretrained vectors; unknown vocabulary tokens are randomly initialized; the padding token is initialized to zeros.

Training examples are created from impression logs. For each clicked candidate, the loader samples 4 negative candidates from the same impression, producing a 5-way classification problem. User histories are truncated to the most recent 50 clicked articles and padded when necessary.

### Model Architecture: NRMS

The model follows the NRMS design:

- The news encoder maps title tokens to GloVe embeddings, applies dropout, contextualizes tokens with multi-head self-attention, and pools the sequence with additive attention.
- The user encoder receives vectors for clicked news in the user's history, applies multi-head self-attention across the history, and uses additive attention to produce a single user preference vector.
- The prediction layer computes dot products between the user vector and candidate news vectors, returning one logit per candidate.

### Training Process

Training uses `CrossEntropyLoss`, where the target is the index of the clicked item among one positive and four sampled negatives. Optimization uses Adam with gradient clipping for stability. Checkpoints are saved to `models/nrms_last.pt` every epoch and `models/nrms_best.pt` whenever dev MRR improves. Training history is saved to `results/training_history.json`.

## 4. Results

| Metric | Result | Interpretation |
| --- | --- | --- |
| AUC | 0.635 | The model ranks clicked articles above non-clicked articles more often than random, showing useful personalization signal. |
| MRR | 0.296 | On average, clicked articles appear reasonably high in the ranked candidate list, though there is still room for stronger top-rank precision. |
| nDCG@5 | 0.327 | The top 5 recommendations contain meaningful clicked-item signal, which is important because users usually inspect only the first few articles. |
| nDCG@10 | 0.382 | Ranking quality improves when considering the top 10, suggesting relevant articles are often present but not always placed at the very top. |

The ranking metrics indicate that the model performs better than random recommendation and learns useful patterns from title semantics and user click histories. The gap between nDCG@5 and nDCG@10 suggests that the model often identifies relevant articles but could improve at placing them in the very first few positions.

## 5. Error Analysis

Cold start remains a central limitation. Users with no or very short histories cannot be represented well by a history-only user encoder. The current implementation pads empty histories, which is safe computationally but weak semantically. A production system should add popularity, freshness, location, and contextual features for cold-start traffic.

Sparse user history can also make attention unstable because the model has little evidence about long-term preference. This affects casual readers and new users especially. Possible mitigations include category-level user profiles, session-based signals, pretrained news encoders, and hybrid collaborative/content features.

Category imbalance can bias recommendations toward dominant topics in the catalog and click logs. If sports or news-heavy categories dominate impressions, the model may under-rank niche interests. Useful countermeasures include balanced negative sampling, per-category evaluation, diversity constraints, and calibration layers.

## 6. Conclusion

This project implements a complete NRMS pipeline for MIND-small: data loading, vocabulary construction, GloVe initialization, negative sampling, PyTorch modeling, checkpointed training, evaluation, and exploratory notebooks.

The main strength of NRMS is its attention-based representation learning. It can learn which title words matter for article meaning and which clicked articles matter most for user preference. Its main weaknesses are dependence on user click history, limited cold-start behavior, and the absence of richer article/context features.

Promising improvements include adding abstract text, using transformer text encoders, incorporating entity embeddings, adding recency and popularity features, tuning negative sampling, and running systematic hyperparameter sweeps.
