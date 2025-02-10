import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class PolarityClassifier(nn.Module):
    def __init__(self, input_size):
        super(PolarityClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32) 
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return self.sigmoid(x)


def train_and_validate(X_train, y_train, X_dev, y_dev, epochs=10, batch_size=32, lr=0.001):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_dev_tensor = torch.tensor(X_dev, dtype=torch.float32)
    y_dev_tensor = torch.tensor(y_dev, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    dev_dataset = TensorDataset(X_dev_tensor, y_dev_tensor)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    model = PolarityClassifier(input_size=X_train.shape[1])

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            predictions = (outputs >= 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        train_accuracy = correct / total
        train_loss = total_loss / len(train_loader)

        model.eval()
        dev_loss = 0
        dev_correct = 0
        dev_total = 0
        with torch.no_grad():
            for X_batch, y_batch in dev_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                dev_loss += loss.item()

                predictions = (outputs >= 0.5).float()
                dev_correct += (predictions == y_batch).sum().item()
                dev_total += y_batch.size(0)

        dev_accuracy = dev_correct / dev_total
        dev_loss = dev_loss / len(dev_loader)

        epoch_time = time.time() - start_time
        print(f"Época {epoch+1}/{epochs} - "
              f"Perda Treino: {train_loss:.4f} - Acurácia Treino: {train_accuracy:.4f} - "
              f"Perda Validação: {dev_loss:.4f} - Acurácia Validação: {dev_accuracy:.4f} - "
              f"Tempo: {epoch_time:.2f}s")

    return model

def evaluate_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    criterion = nn.BCELoss()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            predictions = (outputs >= 0.5).float()
            all_predictions.append(predictions)
            all_labels.append(y_batch)

            test_correct += (predictions == y_batch).sum().item()
            test_total += y_batch.size(0)

    # Concatenar todas as previsões e rótulos após o loop
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print("Acurácia:", accuracy_score(all_labels.numpy(), all_predictions.numpy()))
    print("Precisão:", precision_score(all_labels.numpy(), all_predictions.numpy()))
    print("Recall:", recall_score(all_labels.numpy(), all_predictions.numpy()))
    print("F1-Score:", f1_score(all_labels.numpy(), all_predictions.numpy()))
    
    test_accuracy = test_correct / test_total
    test_loss = test_loss / len(test_loader)

    print(f"Resultados no Conjunto de Teste: Perda = {test_loss:.4f}, Acurácia = {test_accuracy:.4f}")
