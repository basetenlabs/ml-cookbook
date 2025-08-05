import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

class MNISTClassifier(nn.Module): # TODO: make configurable?
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4608, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model:
    def __init__(self, **kwargs):
        self.model = MNISTClassifier()
        self.checkpoint_dir = os.environ.get('BT_CHECKPOINT_DIR', './checkpoints')


    def load(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.checkpoint_dir, 'model.pth')
                )
            )
        self.model.to(DEVICE)
        self.model.eval()

    def predict(self, inputs):
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
