import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt # Used to draw loss curves
from tqdm import tqdm  # Used to display progress bar
import streamlit as st

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
    st.write(f'Using {device}')

    # Prepare dataset and data loader
    dataset = CustomDataset('data.csv')
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function and optimizer, and move the model to the GPU
    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # List used to store loss values
    losses = []

    # Initialize progress bar
    progress_bar = st.progress(0)

    # Training model
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))  # Reshape labels to match output shape
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)

            # Update progress bar
            progress = ((i + 1) + epoch * len(data_loader)) / (num_epochs * len(data_loader))
            progress_bar.progress(progress)

        epoch_loss /= len(dataset)
        losses.append(epoch_loss)

    # Draw a loss curve
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    st.pyplot(fig)

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    st.write('Model saved.')

    # Display final loss
    st.write(f'Final Loss: {losses[-1]}')

def main():
    st.title('Train Model')

    learning_rate = st.number_input('Enter learning rate', value=0.0002)
    num_epochs = st.number_input('Enter number of epochs', value=200)
    batch_size = st.number_input('Enter batch size', value=8192)

    if st.button('Train Model'):
        train_model(learning_rate, num_epochs, batch_size)

if __name__ == "__main__":
    main()
