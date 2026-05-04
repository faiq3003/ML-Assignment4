NRMS: Neural News Recommendation with Multi-Head Self-Attention

This repository provides a formal PyTorch implementation of the NRMS architecture applied to the MIND-small dataset. The project focuses on learning high-dimensional representations of news content and user behavior through stacked attention mechanisms, optimizing for ranking precision in large-scale news ecosystems.

1. Theoretical FrameworkThe NRMS (Neural News Recommendation with Multi-Head Self-Attention) model treats news recommendation as a ranking problem where the objective is to learn a mapping function from a user's historical clickstream and a set of candidate articles into a shared latent space.Hierarchical Attention ArchitectureNews Encoder: Maps title tokens to GloVe embeddings. It utilizes multi-head self-attention to capture contextual word dependencies, followed by an additive attention layer to derive a compressed news representation vector.User Encoder: Aggregates the representations of previously clicked news. By applying multi-head self-attention across the user's history, the model identifies long-range dependencies in reading patterns, weighting high-signal clicks more effectively than noise.Interaction Layer: Computes the inner product between the user vector and the candidate news vector to produce a ranking score.

2. Technical Setup
Ensure a Python 3.8+ environment is active. Dependencies include torch, numpy, scikit-learn, and nltk.
git push

git clone https://github.com/username/nrms-mind-recommender
cd nrms-mind-recommender
pip install -r requirements.txt


3. Experimental Methodology
Training is conducted via a negative sampling strategy. For each positive interaction, 4 negative candidates are sampled, transforming the ranking task into a 5-way NLL (Negative Log-Likelihood) classification problem.

Optimizer: Adam with Gradient Clipping.

Hyperparameters: 300D Word Embeddings, 15-head attention, 50-item user history limit.

Loss Function: Cross-Entropy Loss over sampled candidates.

4. Empirical Results
Evaluated on the MINDsmall_dev split, the model demonstrates significant discriminative power compared to baseline heuristic approaches.

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **AUC** | 0.726 | The model ranks clicked articles above non-clicked articles more often than random, showing useful personalization signal. |
| **MRR** | 0.331 | On average, clicked articles appear reasonably high in the ranked candidate list, though there is still room for stronger top-rank precision. |
| **nDCG@5** | 0.300 | The top 5 recommendations contain meaningful clicked-item signal, which is important because users usually inspect only the first few articles. |
| **nDCG@10** | 0.401 | Ranking quality improves when considering the top 10, suggesting relevant articles are often present but not always placed at the very top. |

5. Critical AnalysisWhile the NRMS architecture excels at modeling text-based preference, the Representation Gap in cold-start scenarios remains a primary bottleneck. When a user history is sparse , the attention mechanism lacks the necessary queries to form a precise preference vector.Future research iterations will explore Knowledge-Aware Embeddings using the MIND entity/relation vectors to provide external semantic grounding for low-frequency articles.