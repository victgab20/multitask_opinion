import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("PATH")

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")


def load_data(nome_arquivo):
    with open(f"{path}\\{nome_arquivo}", "rb") as f:
        return pickle.load(f)

class OpiniaoDataset(Dataset):
    def __init__(self, df, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = df["helpfulness"].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded_text = tokenizer(
            self.texts[idx], 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )

        text_tensor = encoded_text["input_ids"].squeeze(0)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return text_tensor, label_tensor

def create_dataloader(df, batch_size=32):
    dataset = OpiniaoDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)