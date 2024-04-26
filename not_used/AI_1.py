import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 将前8个输入作为特征，第9个数值作为标签
        features = torch.tensor(self.data.iloc[idx, :8].values, dtype=torch.float32)
        label = torch.tensor(self.data.iloc[idx, 8], dtype=torch.float32)
        return features, label

# 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # 输入特征数为8，隐藏层1节点数为16
        self.fc2 = nn.Linear(16, 32)  # 隐藏层1节点数为16，隐藏层2节点数为32
        self.fc3 = nn.Linear(32, 1)   # 隐藏层2节点数为32，输出为1

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 设置超参数
learning_rate = 0.01
num_epochs = 100
batch_size = 8

# 准备数据集和数据加载器
dataset = CustomDataset('data_with_ma.csv')
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、损失函数和优化器
model = Model()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))  # 对标签进行reshape以匹配输出形状
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型并进行预测
model = Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 输入前8个特征，预测第9个数值
input_data = torch.tensor([166.2100067138672,166.39999389648438,164.0800018310547,165.0,67772100,170.52857099260603,170.2074996948242,182.8384996032715], dtype=torch.float32)
prediction = model(input_data)
print(f'Prediction: {prediction.item()}')
