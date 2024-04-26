import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  
import streamlit as st

# Load the definition of the neural network model
from AI import Model

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use the first 8 inputs as features
        features = torch.tensor(self.data.iloc[idx, :8].values, dtype=torch.float32)
        return features

def load_model_and_predict(input_data):
    # Load model
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Enter the first 8 features and predict the 9th value
    with torch.no_grad():
        prediction = model(input_data)
    
    return prediction.item()

def main():
    st.title("Neural Network Prediction")

    # Load dataset
    dataset = CustomDataset('data.csv')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform predictions
    predictions = []
    progress_bar = st.progress(0)
    with st.spinner('Predicting...'):
        for i, input_data in enumerate(tqdm(data_loader, total=len(data_loader))):  # Add progress bar
            prediction = load_model_and_predict(input_data)
            predictions.append(prediction)
            progress_bar.progress((i + 1) / len(data_loader))

    st.success("Prediction completed.")

    # Calculate moving average
    window_size = 7
    rolling_average = pd.Series(predictions).rolling(window=window_size, min_periods=1).mean()

    # Load the last column of the data.csv
    original_data = pd.read_csv('data.csv')
    actual_values = original_data.iloc[:, -1].values

    # Create a new DataFrame to save predicted values, moving average and actual values
    prediction_df = pd.DataFrame({'Actual': actual_values, 'Predicted': predictions, 'Rolling Average': rolling_average})

    # Save predicted values, moving average and actual values to a new CSV file
    prediction_df.to_csv('predictions.csv', index=False)
    st.success("The prediction results are saved to the predictions.csv file.")

if __name__ == "__main__":
    main()
