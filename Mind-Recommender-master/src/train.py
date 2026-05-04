from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from data_loader import (
    build_vocabulary,
    combine_news,
    create_eval_dataloader,
    create_train_dataloader,
    encode_news_titles,
    load_glove_embeddings,
    load_news,
    set_seed,
)
from evaluate import evaluate_model
from model import NRMSModel


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train NRMS on MIND-small.")
    parser.add_argument("--train-news", type=Path, default=project_root / "data" / "MINDsmall_train" / "news.tsv")
    parser.add_argument("--train-behaviors", type=Path, default=project_root / "data" / "MINDsmall_train" / "behaviors.tsv")
    parser.add_argument("--dev-news", type=Path, default=project_root / "data" / "MINDsmall_dev" / "news.tsv")
    parser.add_argument("--dev-behaviors", type=Path, default=project_root / "data" / "MINDsmall_dev" / "behaviors.tsv")
    parser.add_argument("--glove", type=Path, default=project_root / "data" / "glove" / "glove.6B.300d.txt")
    parser.add_argument("--models-dir", type=Path, default=project_root / "models")
    parser.add_argument("--results-dir", type=Path, default=project_root / "results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--max-title-length", type=int, default=30)
    parser.add_argument("--negative-sampling-ratio", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=15)
    parser.add_argument("--attention-dim", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-embeddings", action="store_true")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: NRMSModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    vocab: Dict[str, int],
    embedding_matrix: torch.Tensor,
    max_title_length: int,
    args: argparse.Namespace,
    metrics: Dict[str, float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab": vocab,
        "embedding_matrix": embedding_matrix.detach().cpu(),
        "max_title_length": max_title_length,
        "metrics": metrics or {},
        "model_config": {
            "num_heads": args.num_heads,
            "attention_dim": args.attention_dim,
            "dropout": args.dropout,
            "freeze_embeddings": args.freeze_embeddings,
        },
    }
    torch.save(checkpoint, path)


def train_one_epoch(
    model: NRMSModel,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for history, candidates, labels in tqdm(dataloader, desc="Training"):
        history = history.to(device)
        candidates = candidates.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(history, candidates)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(1, total_examples)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_news = load_news(args.train_news)
    dev_news = load_news(args.dev_news)
    vocab = build_vocabulary(train_news, min_frequency=2)
    all_news = combine_news(train_news, dev_news)
    news_title_map = encode_news_titles(all_news, vocab, max_length=args.max_title_length)
    embedding_matrix = torch.from_numpy(load_glove_embeddings(args.glove, vocab, embedding_dim=300, seed=args.seed))

    train_loader = create_train_dataloader(
        behaviors_path=args.train_behaviors,
        news_title_map=news_title_map,
        batch_size=args.batch_size,
        history_size=args.history_size,
        negative_sampling_ratio=args.negative_sampling_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    dev_loader = create_eval_dataloader(
        behaviors_path=args.dev_behaviors,
        news_title_map=news_title_map,
        history_size=args.history_size,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NRMSModel(
        embedding_matrix=embedding_matrix,
        num_heads=args.num_heads,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    best_mrr = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        metrics = evaluate_model(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
            f"AUC={metrics['AUC']:.4f} | MRR={metrics['MRR']:.4f} | "
            f"nDCG@5={metrics['nDCG@5']:.4f} | nDCG@10={metrics['nDCG@10']:.4f}"
        )

        save_checkpoint(
            args.models_dir / "nrms_last.pt",
            model,
            optimizer,
            epoch,
            vocab,
            embedding_matrix,
            args.max_title_length,
            args,
            metrics,
        )
        if metrics["MRR"] > best_mrr:
            best_mrr = metrics["MRR"]
            save_checkpoint(
                args.models_dir / "nrms_best.pt",
                model,
                optimizer,
                epoch,
                vocab,
                embedding_matrix,
                args.max_title_length,
                args,
                metrics,
            )

    with (args.results_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()
