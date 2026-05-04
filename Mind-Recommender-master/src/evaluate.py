from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data_loader import (
    combine_news,
    create_eval_dataloader,
    encode_news_titles,
    load_glove_embeddings,
    load_news,
)
from model import NRMSModel


def mrr_score(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    ranked_labels = labels[order]
    positive_ranks = np.where(ranked_labels == 1)[0] + 1
    if len(positive_ranks) == 0:
        return 0.0
    return float(np.mean(1.0 / positive_ranks))


def ndcg_score(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(scores)[::-1][:k]
    gains = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(labels)[::-1][:k]
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum(ideal * ideal_discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def compute_ranking_metrics(group_labels: Iterable[np.ndarray], group_scores: Iterable[np.ndarray]) -> Dict[str, float]:
    aucs: List[float] = []
    mrrs: List[float] = []
    ndcg5s: List[float] = []
    ndcg10s: List[float] = []

    for labels, scores in zip(group_labels, group_scores):
        labels = np.asarray(labels)
        scores = np.asarray(scores)
        if len(np.unique(labels)) == 2:
            aucs.append(float(roc_auc_score(labels, scores)))
        mrrs.append(mrr_score(labels, scores))
        ndcg5s.append(ndcg_score(labels, scores, k=5))
        ndcg10s.append(ndcg_score(labels, scores, k=10))

    return {
        "AUC": float(np.mean(aucs)) if aucs else float("nan"),
        "MRR": float(np.mean(mrrs)) if mrrs else float("nan"),
        "nDCG@5": float(np.mean(ndcg5s)) if ndcg5s else float("nan"),
        "nDCG@10": float(np.mean(ndcg10s)) if ndcg10s else float("nan"),
    }


@torch.no_grad()
def evaluate_model(
    model: NRMSModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        history = batch["history"].to(device)
        candidates = batch["candidates"].to(device)
        scores = model(history, candidates).squeeze(0).detach().cpu().numpy()
        labels = batch["labels"].numpy()
        all_scores.append(scores)
        all_labels.append(labels)

    return compute_ranking_metrics(all_labels, all_scores)


def load_model_from_checkpoint(checkpoint_path: Path | str, device: torch.device) -> tuple[NRMSModel, Dict[str, int], Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})
    model = NRMSModel(
        embedding_matrix=checkpoint["embedding_matrix"],
        num_heads=model_config.get("num_heads", 15),
        attention_dim=model_config.get("attention_dim", 200),
        dropout=model_config.get("dropout", 0.2),
        freeze_embeddings=model_config.get("freeze_embeddings", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["vocab"], checkpoint


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate a trained NRMS model on MIND-small dev.")
    parser.add_argument("--checkpoint", type=Path, default=project_root / "models" / "nrms_best.pt")
    parser.add_argument("--train-news", type=Path, default=project_root / "data" / "MINDsmall_train" / "news.tsv")
    parser.add_argument("--dev-news", type=Path, default=project_root / "data" / "MINDsmall_dev" / "news.tsv")
    parser.add_argument("--dev-behaviors", type=Path, default=project_root / "data" / "MINDsmall_dev" / "behaviors.tsv")
    parser.add_argument("--glove", type=Path, default=project_root / "data" / "glove" / "glove.6B.300d.txt")
    parser.add_argument("--history-size", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint.exists():
        model, vocab, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
        max_title_length = checkpoint.get("max_title_length", 30)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Run src/train.py first.")

    train_news = load_news(args.train_news)
    dev_news = load_news(args.dev_news)
    news_title_map = encode_news_titles(combine_news(train_news, dev_news), vocab, max_length=max_title_length)

    dataloader = create_eval_dataloader(
        behaviors_path=args.dev_behaviors,
        news_title_map=news_title_map,
        history_size=args.history_size,
    )
    metrics = evaluate_model(model, dataloader, device)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
