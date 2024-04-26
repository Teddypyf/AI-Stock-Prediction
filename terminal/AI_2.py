import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt # Used to draw loss curves
from tqdm import tqdm  # Used to display progress bar

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  # Read CSV file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use the first 8 inputs as features and the 9th value as the label
        features = torch.tensor(self.data.iloc[idx, :8].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, 8], dtype=torch.float32)
        return features, label

# Define neural network model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 32) 
        self.fc11 = nn.Linear(32, 1)  
        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc11(x)
        return x

def train_model(learning_rate=0.0002, num_epochs=200, batch_size=8192):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    # Prepare dataset and data loader
    dataset = CustomDataset('data.csv')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function and optimizer, and move the model to the GPU
    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # List used to store loss values
    losses = []

    # Training model
    with tqdm(total=len(data_loader)*num_epochs, desc=f'Training', unit='batch') as pbar:  # Create a progress bar
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))  # Reshape labels to match output shape
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})  # Show loss value on progress bar
                pbar.update()
            epoch_loss /= len(dataset)
            losses.append(epoch_loss)

    # Draw a loss curve
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved.')

if __name__ == "__main__":
    # Get hyperparameters from user input
    learning_rate_input = input("Enter learning rate (default=0.0002): ").strip()
    learning_rate = float(learning_rate_input) if learning_rate_input else 0.0002

    num_epochs_input = input("Enter number of epochs (default=200): ").strip()
    num_epochs = int(num_epochs_input) if num_epochs_input else 200

    batch_size_input = input("Enter batch size (default=8192): ").strip()
    batch_size = int(batch_size_input) if batch_size_input else 8192

    # Train model with specified hyperparameters
    train_model(learning_rate, num_epochs, batch_size)
