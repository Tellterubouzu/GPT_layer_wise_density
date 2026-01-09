from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import DataCollatorForLanguageModeling


def build_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    seq_len: int,
    shuffle: bool,
    text_column: str = "text",
) -> DataLoader:
    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples[text_column], return_attention_mask=False)

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        concatenated = []
        for tokens in examples["input_ids"]:
            concatenated.extend(tokens)
        total_length = (len(concatenated) // seq_len) * seq_len
        if total_length == 0:
            return {"input_ids": []}
        result = {
            "input_ids": [
                concatenated[i : i + seq_len]
                for i in range(0, total_length, seq_len)
            ]
        }
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    grouped = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(grouped, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)


class StreamingDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_config: Optional[str],
        split: str,
        tokenizer,
        seq_len: int,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        self.ds = load_dataset(dataset_path, dataset_config, split=split, streaming=True)
        if shuffle_buffer and shuffle_buffer > 0:
            self.ds = self.ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        buf = []
        for idx, ex in enumerate(self.ds):
            global_idx = self.rank * num_workers + worker_id
            global_stride = self.world_size * num_workers
            if idx % global_stride != global_idx:
                continue
            toks = self.tokenizer(ex["text"], return_attention_mask=False, add_special_tokens=True)["input_ids"]
            buf.extend(toks)
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len:]


def build_streaming_dataloader(
    dataset_path: str,
    dataset_config: Optional[str],
    split: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    num_workers: int = 0,
    shuffle_buffer: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = StreamingDataset(
        dataset_path=dataset_path,
        dataset_config=dataset_config,
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    def collate_fn(batch):
        return {"input_ids": torch.stack(batch, dim=0)}

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)


class TokenLimitedLoader:
    def __init__(self, dataloader: DataLoader, max_tokens: Optional[int]) -> None:
        self.dataloader = dataloader
        self.max_tokens = max_tokens

    def __iter__(self):
        seen = 0
        for batch in self.dataloader:
            input_ids = batch["input_ids"]
            tokens = int(input_ids.numel())
            if self.max_tokens is not None and seen + tokens > self.max_tokens:
                break
            seen += tokens
            yield batch
            if self.max_tokens is not None and seen >= self.max_tokens:
                break
