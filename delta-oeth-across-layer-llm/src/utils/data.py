from typing import Any, Dict

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def build_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    block_size: int,
    shuffle: bool,
    text_column: str = "text",
) -> DataLoader:
    def tokenize_fn(examples: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(examples[text_column], return_attention_mask=False)

    def group_texts(examples: Dict[str, Any]) -> Dict[str, Any]:
        concatenated = []
        for tokens in examples["input_ids"]:
            concatenated.extend(tokens)
        total_length = (len(concatenated) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": []}
        result = {
            "input_ids": [
                concatenated[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
        }
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    grouped = tokenized.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(grouped, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
