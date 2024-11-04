import torch
from torch import nn
import numpy as np
from train import test, MyDataset
from generate_data import generate_data
from env import DISTRIBUTION_TYPES, SAMPLE_SIZE, TEST_SIZE
from torch.utils.data import Dataset, DataLoader

model = nn.Sequential(
        nn.Linear(in_features=SAMPLE_SIZE, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=len(DISTRIBUTION_TYPES)+2),
        )

# Load existing thing in, and run some tests
weights_path = 'model_weights.pth'
model.load_state_dict(torch.load(weights_path))

raw_test_data = generate_data(count=TEST_SIZE, sample_size=SAMPLE_SIZE)
test_samples = np.array([elem[0] for elem in raw_test_data])
test_labels = np.array([elem[1] for elem in raw_test_data])
test_dataset = MyDataset(test_samples, test_labels)
test_dataloader = DataLoader(test_dataset)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
loss_fn = nn.MSELoss()
test(test_dataloader, model, loss_fn, device)
