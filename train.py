import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from transformers import BertTokenizer

from src.tensorboard.polarity.board import TensorBoardLogger
from src.logs.versioning.log import Logger


class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df["text"].tolist()
        self.stars = df["stars"].tolist()
        self.helpfulness = df["helpfulness"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True
        )

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "stars": torch.tensor(self.stars[idx], dtype=torch.long),
            "helpfulness": torch.tensor(self.helpfulness[idx], dtype=torch.float)
        }

path = r"C:\Users\victo\Downloads\corpus\labeled"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_data(nome_arquivo):
    with open(f"{path}\\{nome_arquivo}", "rb") as f:
        return pickle.load(f)

def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r'[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Função de pré-processamento para o DataFrame
def pre_processing(df: pd.DataFrame):
    df = df.loc[lambda x: x["stars"] != 3]
    df["stars"] = df["stars"].apply(lambda x: 1 if x > 3 else 0)
    df["text"] = df["text"].apply(preprocess_text)
    df = df[df['text'].notnull() & (df['text'] != '')]
    return df

train_apps = load_data("train_apps.pkl")
dev_apps = load_data("dev_apps.pkl")
test_apps = load_data("test_apps.pkl")

train_filmes = load_data("train_filmes.pkl")
dev_filmes = load_data("dev_filmes.pkl")
test_filmes = load_data("test_filmes.pkl")

train_data = pre_processing(train_apps[["stars", "text", "helpfulness"]])
dev_data = pre_processing(dev_apps[["stars", "text", "helpfulness"]])
test_data = pre_processing(test_apps[["stars", "text", "helpfulness"]])

train_dataset = ReviewDataset(train_data, tokenizer)
dev_dataset = ReviewDataset(dev_data, tokenizer)
test_dataset = ReviewDataset(test_data, tokenizer)

BATCH_SIZE = 16

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_size):
        super(MultiTaskClassifier, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        self.polarity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.utility_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        polarity_out = self.polarity_head(shared)
        utility_out = self.utility_head(shared)
        return polarity_out, utility_out

# Função de cálculo de métricas
def calcular_metricas(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="macro"),
    )

# Função de treinamento
def train(model, train_loader, criterion, optimizer, device, logger, tb_logger, epoch):
    model.train()  # Coloca o modelo em modo de treinamento
    total_loss = 0
    y_true_polarity, y_pred_polarity = [], []
    y_true_utility, y_pred_utility = [], []

    # Iteração pelos batches de dados
    for batch in train_loader:
        # Acessando os dados diretamente do dicionário retornado pelo Dataset
        X_batch = batch["input_ids"].to(device).float()  # Input_ids
        attention_mask = batch["attention_mask"].to(device)  # Máscara de atenção
        y_polarity = batch["stars"].to(device).float()   # Rótulos de polaridade (estrelas)
        y_utility = batch["helpfulness"].to(device).float()   # Rótulos de utilidade (helpfulness)

        optimizer.zero_grad()  # Zera os gradientes antes de realizar a retropropagação
        outputs_polarity, outputs_utility = model(X_batch)  # Passa os dados pelo modelo

        # Calcula a perda (loss) para ambas as tarefas
        loss_polarity = criterion(outputs_polarity, y_polarity.unsqueeze(1))  # Polaridade
        loss_utility = criterion(outputs_utility, y_utility.unsqueeze(1))  # Utilidade
        loss = loss_polarity + loss_utility  # Soma as perdas

        loss.backward()  # Realiza a retropropagação
        optimizer.step()  # Atualiza os pesos do modelo

        total_loss += loss.item()  # Acumula a perda total

        # Armazena as predições e os valores verdadeiros para cálculo das métricas
        y_true_polarity.extend(y_polarity.cpu().numpy())
        y_pred_polarity.extend((outputs_polarity >= 0.5).float().cpu().numpy())
        
        y_true_utility.extend(y_utility.cpu().numpy())
        y_pred_utility.extend((outputs_utility >= 0.5).float().cpu().numpy())

    # Cálculo das métricas para a tarefa de polaridade
    y_pred_polarity = y_pred_polarity.detach().cpu().numpy() if isinstance(y_pred_polarity, torch.Tensor) else np.array(y_pred_polarity)
    polarity_acc = accuracy_score(y_true_polarity, (y_pred_polarity >= 0.5).astype(int))
    polarity_precision, polarity_recall, polarity_f1, _ = precision_recall_fscore_support(
    y_true_polarity, (y_pred_polarity >= 0.5).astype(int), average='binary')

    # Cálculo das métricas para a tarefa de utilidade
    utility_acc = accuracy_score(y_true_utility, (y_pred_utility >= 0.5).astype(int))
    utility_precision, utility_recall, utility_f1, _ = precision_recall_fscore_support(
        y_true_utility, (y_pred_utility >= 0.5).astype(int), average='binary')

    # Registra as métricas e a perda total no logger e no TensorBoard
    logger.log(f"Epoch {epoch+1}: Train Loss: {total_loss/len(train_loader):.4f}")
    logger.log(f"Polaridade - Acc: {polarity_acc:.4f} | Precision: {polarity_precision:.4f} | Recall: {polarity_recall:.4f} | F1: {polarity_f1:.4f}")
    logger.log(f"Utilidade - Acc: {utility_acc:.4f} | Precision: {utility_precision:.4f} | Recall: {utility_recall:.4f} | F1: {utility_f1:.4f}")

    # Adiciona logs ao TensorBoard
    tb_logger.add_scalar('Loss/Train', total_loss / len(train_loader), epoch)
    tb_logger.add_scalar('Accuracy/Polaridade', polarity_acc, epoch)
    tb_logger.add_scalar('F1/Polaridade', polarity_f1, epoch)
    tb_logger.add_scalar('Accuracy/Utilidade', utility_acc, epoch)
    tb_logger.add_scalar('F1/Utilidade', utility_f1, epoch)

    return total_loss / len(train_loader), (polarity_acc, polarity_precision, polarity_recall, polarity_f1), (utility_acc, utility_precision, utility_recall, utility_f1)


    metrics_polarity = calcular_metricas(y_true_polarity, y_pred_polarity)
    metrics_utility = calcular_metricas(y_true_utility, y_pred_utility)
    
    tb_logger.log_metrics(epoch, total_loss, metrics_polarity[0], 0, 0, metrics_polarity[1], metrics_polarity[2], metrics_polarity[3], metrics_utility[1], metrics_utility[2], metrics_utility[3])
    
    logger.log(f"Epoch {epoch+1} Train Loss: {total_loss:.4f} | Polarity: {metrics_polarity} | Utility: {metrics_utility}")
    return total_loss / len(train_loader), metrics_polarity, metrics_utility

# Função de avaliação
def evaluate(model, dev_loader, criterion, device, logger, tb_logger, epoch):
    model.eval()  # Coloca o modelo em modo de avaliação
    total_loss = 0
    y_true_polarity, y_pred_polarity = [], []
    y_true_utility, y_pred_utility = [], []

    with torch.no_grad():
        # Itera pelos batches de dados no dev_loader
        for batch in dev_loader:
            # Acessando os dados diretamente do dicionário retornado pelo Dataset
            X_batch = batch["input_ids"].to(device).float()  # Input_ids
            attention_mask = batch["attention_mask"].to(device)  # Máscara de atenção
            y_polarity = batch["stars"].to(device).float()   # Rótulos de polaridade (estrelas)
            y_utility = batch["helpfulness"].to(device).float()   # Rótulos de utilidade (helpfulness)

            # Passa os dados pelo modelo
            outputs_polarity, outputs_utility = model(X_batch)

            # Calcula a perda (loss) para ambas as tarefas
            loss_polarity = criterion(outputs_polarity, y_polarity.unsqueeze(1))  # Polaridade
            loss_utility = criterion(outputs_utility, y_utility.unsqueeze(1))  # Utilidade
            loss = loss_polarity + loss_utility  # Soma as perdas

            total_loss += loss.item()  # Acumula a perda total

            # Armazena as predições e os valores verdadeiros para cálculo das métricas
            y_true_polarity.extend(y_polarity.cpu().numpy())
            y_pred_polarity.extend((outputs_polarity >= 0.5).float().cpu().numpy())
            y_true_utility.extend(y_utility.cpu().numpy())
            y_pred_utility.extend((outputs_utility >= 0.5).float().cpu().numpy())

    # Calcula as métricas para as tarefas de polaridade e utilidade
    metrics_polarity = calcular_metricas(y_true_polarity, y_pred_polarity)
    metrics_utility = calcular_metricas(y_true_utility, y_pred_utility)

    # Registra as métricas e a perda total no logger e no TensorBoard
    tb_logger.log_metrics(epoch, 0, 0, total_loss, 0, metrics_polarity[1], metrics_polarity[2], metrics_polarity[3], metrics_utility[1], metrics_utility[2], metrics_utility[3])
    
    logger.log(f"Epoch {epoch+1} Dev Loss: {total_loss:.4f} | Polarity: {metrics_polarity} | Utility: {metrics_utility}")

    return total_loss / len(dev_loader), metrics_polarity, metrics_utility


# Função de treinamento e validação
def train_and_validate(train_loader, dev_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch in train_loader:
        # Acessa os dados de entrada, considerando que o batch é um dicionário
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # O input_size será o número de características (tamanho da sequência de input)
        input_size = input_ids.shape[1]
        
        # Interrompe após o primeiro batch para não iterar desnecessariamente
        break

    # Criação do modelo com o input_size obtido
    model = MultiTaskClassifier(input_size=input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    logger = Logger()
    tb_logger = TensorBoardLogger()

    for epoch in range(epochs):
        train_loss, train_polarity_metrics, train_utility_metrics = train(model, train_loader, criterion, optimizer, device, logger, tb_logger, epoch)
        dev_loss, dev_polarity_metrics, dev_utility_metrics = evaluate(model, dev_loader, criterion, device, logger, tb_logger, epoch)
        logger.log(f"Epoch {epoch+1}/{epochs}")
        logger.log(f"Train Loss: {train_loss:.4f} | Polarity: {train_polarity_metrics} | Utility: {train_utility_metrics}")
        logger.log(f"Dev Loss: {dev_loss:.4f} | Polarity: {dev_polarity_metrics} | Utility: {dev_utility_metrics}\n")
    
    tb_logger.close()
    return model

print("entrou")
epochs = 10
lr = 0.001
model = train_and_validate(train_loader, dev_loader, epochs=epochs, lr=lr)