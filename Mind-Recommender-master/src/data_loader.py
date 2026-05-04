from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import TreebankWordTokenizer
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def set_seed(seed: int = 42) -> None:
    """Make Python, NumPy, and PyTorch behavior reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_file(path: Path) -> None:
    """Raise a helpful error when a required data artifact is unavailable."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Required file not found or empty: {path}\n"
            "Download MIND-small and GloVe, then place the files in the project data/ "
            "directory as described in README.md."
        )
    with path.open("r", encoding="utf-8", errors="ignore") as file:
        first_line = file.readline()
    if first_line.startswith("PLACEHOLDER:"):
        raise FileNotFoundError(
            f"Required file is still a placeholder: {path}\n"
            "Replace it with the official MIND-small or GloVe file before running."
        )


def load_news(news_path: Path | str) -> pd.DataFrame:
    """Load MIND news.tsv with stable column names."""
    path = Path(news_path)
    require_file(path)
    return pd.read_csv(path, sep="\t", names=NEWS_COLUMNS, quoting=3, keep_default_na=False)


def load_behaviors(behaviors_path: Path | str) -> pd.DataFrame:
    """Load MIND behaviors.tsv with stable column names."""
    path = Path(behaviors_path)
    require_file(path)
    return pd.read_csv(path, sep="\t", names=BEHAVIOR_COLUMNS, quoting=3, keep_default_na=False)


_TOKENIZER = TreebankWordTokenizer()


def tokenize_title(title: str) -> List[str]:
    """Tokenize a news title using NLTK's Treebank tokenizer."""
    return [token.lower() for token in _TOKENIZER.tokenize(str(title))]


def build_vocabulary(
    news_df: pd.DataFrame,
    min_frequency: int = 2,
    max_vocab_size: Optional[int] = None,
) -> Dict[str, int]:
    """Build a word-to-index vocabulary from news titles."""
    counter: Counter[str] = Counter()
    for title in news_df["title"]:
        counter.update(tokenize_title(title))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    words = [word for word, freq in counter.items() if freq >= min_frequency]
    words = sorted(words, key=lambda word: (-counter[word], word))
    if max_vocab_size is not None:
        words = words[: max(0, max_vocab_size - len(vocab))]

    for word in words:
        vocab[word] = len(vocab)
    return vocab


def encode_title(title: str, vocab: Dict[str, int], max_length: int = 30) -> np.ndarray:
    """Convert one title into a fixed-length integer sequence."""
    token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize_title(title)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([vocab[PAD_TOKEN]] * (max_length - len(token_ids)))
    return np.asarray(token_ids, dtype=np.int64)


def encode_news_titles(
    news_df: pd.DataFrame,
    vocab: Dict[str, int],
    max_length: int = 30,
) -> Dict[str, np.ndarray]:
    """Encode every news title into a lookup table keyed by news_id."""
    return {
        row.news_id: encode_title(row.title, vocab, max_length=max_length)
        for row in news_df.itertuples(index=False)
    }


def load_glove_embeddings(
    glove_path: Path | str,
    vocab: Dict[str, int],
    embedding_dim: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Create an embedding matrix initialized from GloVe where available."""
    path = Path(glove_path)
    require_file(path)

    rng = np.random.default_rng(seed)
    scale = 0.1
    matrix = rng.normal(0.0, scale, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[vocab[PAD_TOKEN]] = np.zeros(embedding_dim, dtype=np.float32)

    needed = set(vocab)
    found = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            pieces = line.rstrip().split(" ")
            if len(pieces) != embedding_dim + 1:
                continue
            word = pieces[0]
            if word in needed:
                matrix[vocab[word]] = np.asarray(pieces[1:], dtype=np.float32)
                found += 1

    coverage = found / max(1, len(vocab) - 2)
    print(f"Loaded GloVe vectors for {found:,}/{len(vocab):,} tokens ({coverage:.1%}).")
    return matrix


def parse_history(history: str) -> List[str]:
    """Parse the space-separated history column."""
    if not history:
        return []
    return [news_id for news_id in history.split(" ") if news_id]


def parse_impressions(impressions: str) -> List[Tuple[str, int]]:
    """Parse the impressions column into (news_id, clicked_label) pairs."""
    parsed: List[Tuple[str, int]] = []
    for item in impressions.split(" "):
        if not item:
            continue
        news_id, label = item.rsplit("-", 1)
        parsed.append((news_id, int(label)))
    return parsed


def sample_negative_candidates(
    impressions: Sequence[Tuple[str, int]],
    negative_sampling_ratio: int,
    rng: random.Random,
) -> List[Tuple[List[str], int]]:
    """Create fixed-size candidate sets with one positive and K negatives."""
    positives = [news_id for news_id, label in impressions if label == 1]
    negatives = [news_id for news_id, label in impressions if label == 0]
    samples: List[Tuple[List[str], int]] = []

    if not positives or not negatives:
        return samples

    for positive in positives:
        if len(negatives) >= negative_sampling_ratio:
            chosen_negatives = rng.sample(negatives, negative_sampling_ratio)
        else:
            chosen_negatives = [rng.choice(negatives) for _ in range(negative_sampling_ratio)]

        candidate_ids = [positive] + chosen_negatives
        rng.shuffle(candidate_ids)
        label_index = candidate_ids.index(positive)
        samples.append((candidate_ids, label_index))
    return samples


class MINDTrainDataset(Dataset):
    """Training dataset that performs NRMS negative sampling from impressions."""

    def __init__(
        self,
        behaviors_path: Path | str,
        news_title_map: Dict[str, np.ndarray],
        history_size: int = 50,
        negative_sampling_ratio: int = 4,
        seed: int = 42,
    ) -> None:
        self.behaviors = load_behaviors(behaviors_path)
        self.news_title_map = news_title_map
        self.history_size = history_size
        self.max_title_length = len(next(iter(news_title_map.values())))
        self.pad_title = np.zeros(self.max_title_length, dtype=np.int64)
        self.samples: List[Tuple[List[str], List[str], int]] = []
        rng = random.Random(seed)

        for row in self.behaviors.itertuples(index=False):
            history_ids = parse_history(row.history)
            impressions = parse_impressions(row.impressions)
            for candidate_ids, label_index in sample_negative_candidates(
                impressions, negative_sampling_ratio, rng
            ):
                self.samples.append((history_ids, candidate_ids, label_index))

        if not self.samples:
            raise ValueError(
                "No training samples were created. Check that behaviors.tsv contains "
                "impressions with both clicked and non-clicked items."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_history(self, history_ids: Sequence[str]) -> np.ndarray:
        recent = list(history_ids)[-self.history_size :]
        encoded = [self.news_title_map.get(news_id, self.pad_title) for news_id in recent]
        padding = [self.pad_title] * (self.history_size - len(encoded))
        return np.stack(padding + encoded)

    def _encode_candidates(self, candidate_ids: Sequence[str]) -> np.ndarray:
        encoded = [self.news_title_map.get(news_id, self.pad_title) for news_id in candidate_ids]
        return np.stack(encoded)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_ids, candidate_ids, label_index = self.samples[index]
        history = torch.as_tensor(self._encode_history(history_ids), dtype=torch.long)
        candidates = torch.as_tensor(self._encode_candidates(candidate_ids), dtype=torch.long)
        label = torch.as_tensor(label_index, dtype=torch.long)
        return history, candidates, label


class MINDImpressionDataset(Dataset):
    """Validation/test dataset that preserves full impression candidate groups."""

    def __init__(
        self,
        behaviors_path: Path | str,
        news_title_map: Dict[str, np.ndarray],
        history_size: int = 50,
    ) -> None:
        self.behaviors = load_behaviors(behaviors_path)
        self.news_title_map = news_title_map
        self.history_size = history_size
        self.max_title_length = len(next(iter(news_title_map.values())))
        self.pad_title = np.zeros(self.max_title_length, dtype=np.int64)
        self.samples: List[Tuple[str, List[str], List[str], List[int]]] = []

        for row in self.behaviors.itertuples(index=False):
            impressions = parse_impressions(row.impressions)
            if not impressions:
                continue
            candidate_ids = [news_id for news_id, _ in impressions]
            labels = [label for _, label in impressions]
            self.samples.append((row.impression_id, parse_history(row.history), candidate_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_history(self, history_ids: Sequence[str]) -> np.ndarray:
        recent = list(history_ids)[-self.history_size :]
        encoded = [self.news_title_map.get(news_id, self.pad_title) for news_id in recent]
        padding = [self.pad_title] * (self.history_size - len(encoded))
        return np.stack(padding + encoded)

    def _encode_candidates(self, candidate_ids: Sequence[str]) -> np.ndarray:
        encoded = [self.news_title_map.get(news_id, self.pad_title) for news_id in candidate_ids]
        return np.stack(encoded)

    def __getitem__(self, index: int) -> Dict[str, object]:
        impression_id, history_ids, candidate_ids, labels = self.samples[index]
        return {
            "impression_id": impression_id,
            "history": torch.as_tensor(self._encode_history(history_ids), dtype=torch.long),
            "candidates": torch.as_tensor(self._encode_candidates(candidate_ids), dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "candidate_ids": candidate_ids,
        }


def impression_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate one impression at a time to support variable candidate counts."""
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 for impression-level evaluation.")
    item = batch[0]
    return {
        "impression_id": item["impression_id"],
        "history": item["history"].unsqueeze(0),
        "candidates": item["candidates"].unsqueeze(0),
        "labels": item["labels"],
        "candidate_ids": item["candidate_ids"],
    }


def create_train_dataloader(
    behaviors_path: Path | str,
    news_title_map: Dict[str, np.ndarray],
    batch_size: int = 64,
    history_size: int = 50,
    negative_sampling_ratio: int = 4,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a shuffled training DataLoader."""
    dataset = MINDTrainDataset(
        behaviors_path=behaviors_path,
        news_title_map=news_title_map,
        history_size=history_size,
        negative_sampling_ratio=negative_sampling_ratio,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )


def create_eval_dataloader(
    behaviors_path: Path | str,
    news_title_map: Dict[str, np.ndarray],
    history_size: int = 50,
    num_workers: int = 0,
) -> DataLoader:
    """Create an impression-level evaluation DataLoader."""
    dataset = MINDImpressionDataset(
        behaviors_path=behaviors_path,
        news_title_map=news_title_map,
        history_size=history_size,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=impression_collate_fn,
    )


def save_processed_artifacts(
    output_dir: Path | str,
    vocab: Dict[str, int],
    news_title_map: Dict[str, np.ndarray],
) -> None:
    """Persist vocabulary and encoded title lookup for notebooks or experiments."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab, file, indent=2)
    np.savez_compressed(output / "encoded_news_titles.npz", **news_title_map)


def load_processed_artifacts(input_dir: Path | str) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """Load vocabulary and encoded title lookup created by save_processed_artifacts."""
    path = Path(input_dir)
    with (path / "vocab.json").open("r", encoding="utf-8") as file:
        vocab = json.load(file)
    encoded = np.load(path / "encoded_news_titles.npz")
    news_title_map = {news_id: encoded[news_id] for news_id in encoded.files}
    return vocab, news_title_map


def combine_news(train_news: pd.DataFrame, dev_news: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Combine train/dev news rows while removing duplicate news IDs."""
    if dev_news is None:
        return train_news.drop_duplicates("news_id").reset_index(drop=True)
    return (
        pd.concat([train_news, dev_news], ignore_index=True)
        .drop_duplicates("news_id")
        .reset_index(drop=True)
    )


def iter_titles(news_df: pd.DataFrame) -> Iterable[str]:
    """Expose titles as an iterable for notebooks and diagnostics."""
    return news_df["title"].astype(str).tolist()
