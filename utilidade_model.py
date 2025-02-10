import torch
import torch.nn as nn
import torch.optim as optim
from dataset import load_data, create_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class OpiniaoCNN(nn.Module):
    def __init__(self):
        super(OpiniaoCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(32 * 25, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

def train_model(model, train_loader, dev_loader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Época [{epoch+1}/{epochs}], Perda: {total_loss/len(train_loader):.4f}")
        evaluate_model(model, dev_loader)

def evaluate_model(model, dev_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in dev_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts).squeeze()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Acurácia de validação: {correct / total:.4f}")

train_data = load_data("train_apps.pkl")
dev_data = load_data("dev_apps.pkl")

train_loader = create_dataloader(train_data)
dev_loader = create_dataloader(dev_data)

model = OpiniaoCNN().to(device)

train_model(model, train_loader, dev_loader, epochs=10)
