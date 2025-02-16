import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from dotenv import load_dotenv
import os
import re
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import nltk
import pandas as pd
nltk.download('rslp')


path = r"C:\Users\victo\Downloads\corpus\labeled"

# Inicializando o tokenizer e ferramentas de NLP
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
nltk.download("stopwords")
stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

# Função de pré-processamento
def processing_data(df: pd.DataFrame):
    # Remover duplicatas e valores nulos
    df = df.drop_duplicates(subset=["text"])
    df = df.dropna(subset=["text"])
    
    # Filtrar textos com menos de 3 palavras
    df = df[df["text"].str.split().apply(len) > 3]
    
    # Limpeza do texto: remoção de espaços extras, conversão para minúsculas, remoção de caracteres especiais
    df["text"] = df["text"].str.strip().str.lower()
    df["text"] = df["text"].apply(lambda x: re.sub(r"[^a-zA-Zá-úÁ-Ú\s]", "", x))
    
    # Remover stopwords
    df["text"] = df["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
    
    # Stemming
    df["text"] = df["text"].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))
    
    return df

# Função para carregar os dados
def load_data(nome_arquivo):
    with open(f"{path}\\{nome_arquivo}", "rb") as f:
        return pickle.load(f)

# Classe Dataset para opiniões
class OpiniaoDataset(Dataset):
    def __init__(self, df, max_length=128):
        # Pré-processar os dados antes de armazenar
        df = processing_data(df)
        self.texts = df["text"].tolist()
        self.labels = df["helpfulness"].astype(float).tolist()  # Corrigido para garantir que os rótulos são inteiros
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

        text_tensor = encoded_text["input_ids"].squeeze(0).to(torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return text_tensor, label_tensor

# Função para criar o DataLoader
def create_dataloader(df, batch_size=32):
    dataset = OpiniaoDataset(df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
