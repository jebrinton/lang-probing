from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

class SentenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence_text"]
        tags = item["tags"]
        language = item["language"]

        # Tokenize the sentence
        encoded = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # padding happens later in the collate_fn
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "tags": tags,
            "language": language,
            "sentence_text": sentence,
        }

def collate_fn(batch, tokenizer):
    # Pad dynamically
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    padded = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_masks},
        padding=True,
        return_tensors="pt"
    )

    languages = [item["language"] for item in batch]
    tags = [item["tags"] for item in batch]
    sentence_texts = [item["sentence_text"] for item in batch]
    
    return {
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "tags": tags,
        "language": languages,
        "sentence_text": sentence_texts,
    }
