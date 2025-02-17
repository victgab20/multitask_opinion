import torch
from torch.utils.data import DataLoader
from src.models.polarity_model import PolarityClassifier
from src.tensorboard.polarity.board import TensorBoardLogger
from src.logs.versioning.log import Logger
import time
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

path = r"C:\Users\victo\Downloads\corpus\labeled"


def calcular_metricas(y_true, y_pred):
    return (
        precision_score(y_true, y_pred, average="macro"),
        recall_score(y_true, y_pred, average="macro"),
        f1_score(y_true, y_pred, average="macro"),
    )

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

def tokenize_texts(df: pd.DataFrame, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
    X = vectorizer.fit_transform(df["text"]).toarray()
    y = df["stars"].values
    return X, y, vectorizer

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    y_true_list, y_pred_list = [], []

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predictions = (outputs >= 0.5).float().cpu().numpy()
        y_true_list.extend(y_batch.cpu().numpy())
        y_pred_list.extend(predictions)

    accuracy = accuracy_score(y_true_list, y_pred_list)
    precision, recall, f1 = calcular_metricas(y_true_list, y_pred_list)
    loss = total_loss / len(train_loader)

    return loss, accuracy, precision, recall, f1

def evaluate(model, dev_loader, criterion, device):
    model.eval()
    dev_loss = 0
    y_true_list, y_pred_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            dev_loss += loss.item()

            predictions = (outputs >= 0.5).float().cpu().numpy()
            y_true_list.extend(y_batch.cpu().numpy())
            y_pred_list.extend(predictions)

    dev_accuracy = accuracy_score(y_true_list, y_pred_list)
    dev_precision, dev_recall, dev_f1 = calcular_metricas(y_true_list, y_pred_list)
    dev_loss = dev_loss / len(dev_loader)

    return dev_loss, dev_accuracy, dev_precision, dev_recall, dev_f1

# Função de treinamento e validação
def train_and_validate(train_loader, dev_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Criando o modelo, critério de perda e otimizador
    model = PolarityClassifier(input_size=train_loader.dataset.X.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger = Logger()
    tensorboard_logger = TensorBoardLogger()

    for epoch in range(epochs):
        start_time = time.time()

        # Treinamento
        train_loss, train_accuracy, train_precision, train_recall, train_f1 = train(model, train_loader, criterion, optimizer, device)

        # Avaliação
        dev_loss, dev_accuracy, dev_precision, dev_recall, dev_f1 = evaluate(model, dev_loader, criterion, device)

        epoch_time = time.time() - start_time
        logger.log(f"Época {epoch+1}/{epochs} - "
                   f"Perda Treino: {train_loss:.4f} - Acurácia Treino: {train_accuracy:.4f} - "
                   f"Precisão Treino: {train_precision:.4f} - Recall Treino: {train_recall:.4f} - F1 Treino: {train_f1:.4f} - "
                   f"Perda Validação: {dev_loss:.4f} - Acurácia Validação: {dev_accuracy:.4f} - "
                   f"Precisão Validação: {dev_precision:.4f} - Recall Validação: {dev_recall:.4f} - F1 Validação: {dev_f1:.4f} - "
                   f"Tempo: {epoch_time:.2f}s")

        # Logar as métricas no TensorBoard
        tensorboard_logger.log_metrics(epoch, train_loss, train_accuracy, dev_loss, dev_accuracy,
                                       train_precision, train_recall, train_f1,
                                       dev_precision, dev_recall, dev_f1)

    tensorboard_logger.close()

    return model
# Carregamento e pré-processamento dos dados
train_apps = load_data("train_apps.pkl")
dev_apps = load_data("dev_apps.pkl")
test_apps = load_data("test_apps.pkl")

train_filmes = load_data("train_filmes.pkl")
dev_filmes = load_data("dev_filmes.pkl")
test_filmes = load_data("test_filmes.pkl")

a = pre_processing(train_apps[["stars", "text"]])
b = pre_processing(dev_apps[["stars", "text"]])
c = pre_processing(test_apps[["stars","text"]])

X_train, y_train, vectorizer = tokenize_texts(a)
X_dev, y_dev, _ = tokenize_texts(b, vectorizer)
X_test, y_test, _ = tokenize_texts(c, vectorizer)

# Criando Datasets e DataLoaders
train_dataset = TextDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

dev_dataset = TextDataset(X_dev, y_dev)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False)

test_dataset = TextDataset(X_test,y_test)
test_loader = DataLoader(test_dataset,batch_size=128, shuffle=False)


trained_model = train_and_validate(train_loader, test_loader, epochs=20, lr=0.001)
