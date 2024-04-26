import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 加载神经网络模型的定义
from AI_2 import Model

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将前8个输入作为特征
        features = torch.tensor(self.data.iloc[idx, :8].values, dtype=torch.float32)
        return features

def load_model_and_predict(input_data):
    # 加载模型
    model = Model()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # 输入前8个特征，预测第9个数值
    with torch.no_grad():
        prediction = model(input_data)
    
    return prediction.item()

if __name__ == "__main__":
    # 加载数据集
    dataset = CustomDataset('data.csv')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 预测值列表
    predictions = []

    # 加载模型并进行预测
    for input_data in data_loader:
        prediction = load_model_and_predict(input_data)
        predictions.append(prediction)

    # 加载原始数据的最后一列
    original_data = pd.read_csv('data.csv')
    actual_values = original_data.iloc[:, -1].values

    # 创建新的DataFrame保存预测值和实际值
    prediction_df = pd.DataFrame({'Actual': actual_values, 'Predicted': predictions})

    # 将预测值和实际值保存到新的CSV文件
    prediction_df.to_csv('predictions.csv', index=False)
    print("预测结果已保存到 predictions.csv 文件中。")
