

import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import TensorDataset, DataLoader
from src.models.polarity_model import PolarityClassifier
from src.tensorboard.polarity.board import TensorBoardLogger
from src.logs.versioning.log import Logger

import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        predictions = (outputs >= 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += y_batch.size(0)

    accuracy = correct / total
    loss = total_loss / len(train_loader)
    return loss, accuracy

def evaluate(model, dev_loader, criterion, device):
    model.eval()
    dev_loss = 0
    dev_correct = 0
    dev_total = 0

    with torch.no_grad():
        for X_batch, y_batch in dev_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            dev_loss += loss.item()

            predictions = (outputs >= 0.5).float()
            dev_correct += (predictions == y_batch).sum().item()
            dev_total += y_batch.size(0)

    dev_accuracy = dev_correct / dev_total
    dev_loss = dev_loss / len(dev_loader)
    return dev_loss, dev_accuracy




def train_and_validate(train_loader, dev_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Criando o modelo, critério de perda e otimizador
    model = PolarityClassifier(input_size=train_loader.dataset.X.shape[1]).to(device)  # Acessando tamanho de entrada do dataset
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Criar instâncias dos loggers
    logger = Logger()
    tensorboard_logger = TensorBoardLogger()

    # Loop de épocas
    for epoch in range(epochs):
        start_time = time.time()

        # Treinamento
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)

        # Avaliação
        dev_loss, dev_accuracy = evaluate(model, dev_loader, criterion, device)

        epoch_time = time.time() - start_time
        logger.log(f"Época {epoch+1}/{epochs} - "
                   f"Perda Treino: {train_loss:.4f} - Acurácia Treino: {train_accuracy:.4f} - "
                   f"Perda Validação: {dev_loss:.4f} - Acurácia Validação: {dev_accuracy:.4f} - "
                   f"Tempo: {epoch_time:.2f}s")

        # Logar as métricas no TensorBoard
        tensorboard_logger.log_metrics(epoch, train_loss, train_accuracy, dev_loss, dev_accuracy)

    tensorboard_logger.close()

    return model

