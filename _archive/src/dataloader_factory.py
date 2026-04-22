import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

# We still need the Dataset class we defined previously
class ActivationDataset(Dataset):
    """
    Dataset that tokenizes sentences on the fly.
    """
    def __init__(self, samples, tokenizer):
        """
        Args:
            samples (list): A list of strings with the sentences.
            tokenizer: A Hugging Face tokenizer.
        """
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Tokenizes a single sentence. Padding is handled by the DataLoader's collate_fn.
        """
        # The tokenizer returns a dictionary with 'input_ids', 'attention_mask', etc.
        return self.tokenizer(self.samples[idx], truncation=True, max_length=512)


# Here is the new Factory class to achieve your "dream" workflow
class SentenceDataLoaderFactory:
    """
    A factory class to simplify the creation of a DataLoader for tokenized sentences.
    
    This class encapsulates the logic of creating a Dataset, a DataCollator,
    and a DataLoader, providing a clean interface.
    """
    def __init__(self, samples, tokenizer, batch_size=8):
        """
        Args:
            samples (list): A list of strings with the sentences.
            tokenizer: A configured Hugging Face tokenizer.
            batch_size (int): The desired batch size for the DataLoader.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._dataloader = None  # To cache the dataloader once created

    @property
    def dataloader(self):
        """
        Returns a fully configured DataLoader instance.
        
        The DataLoader is created on the first access and then cached.
        """
        if self._dataloader is None:
            # This logic runs only the first time you access the property
            print("Constructing DataLoader for the first time...")
            
            # 1. Create the Dataset instance
            dataset = ActivationDataset(self.samples, self.tokenizer)
            
            # 2. Create the data collator for dynamic padding
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # 3. Create and cache the DataLoader
            self._dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=data_collator
            )
        return self._dataloader

