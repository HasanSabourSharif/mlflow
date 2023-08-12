import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 2

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist_data', train=True, download=True, transform=transform),
                                           batch_size=BATCH_SIZE, shuffle=True)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
losses=[]
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}')


torch.save(model.state_dict(), 'simple_model.pth')



with mlflow.start_run() as run:
    mlflow.log_param("model_type", "SimpleNN")
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.pytorch.log_model(model, "models")


with mlflow.start_run():
    for epoch in range(0, 3):
        mlflow.log_metric(key="quality", value=2 * epoch, step=epoch)
        mlflow.log_metric(key="accuracy", value=2 * epoch, step=epoch)
        mlflow.log_metric(key="recall", value=2 * epoch, step=epoch)
        mlflow.log_metric(key="precision", value=2 * epoch, step=epoch)
        mlflow.log_metric(key="F1 score", value=2 * epoch, step=epoch)
    mlflow.log_metric()

    

