# file: load_data.py

import json
from torch.utils.data import Dataset


class MiniworldTextDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=64):
        super().__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                text = item["prompt"] + item["completion"]
                self.data.append(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # For next-token prediction, labels = input_ids (shifted internally,
        # but we'll keep it simple: model will do next-step on the same IDs)
        # We'll do a manual shift in the training loop for RNN/CNN if needed.
        return {"input_ids": input_ids, "attention_mask": attention_mask}
