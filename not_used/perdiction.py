import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader


from streamlit.AI_2 import Model


# 加载模型并进行预测
model = Model()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# 输入前8个特征，预测第9个数值
input_data = torch.tensor([401.8299865722656,408.0,401.79998779296875,403.7799987792969,30657700,404.432852608817,392.03799896240236,358.6740017700195], dtype=torch.float32)
prediction = model(input_data)
print(f'Prediction: {prediction.item()}')
# 403.0346984863281


# 输入前8个特征，预测第9个数值
input_data = torch.tensor([406.9599914550781,415.32000732421875,397.2099914550781,397.5799865722656,47871100,403.7357090541295,390.3789993286133,357.9789016723633], dtype=torch.float32)
prediction = model(input_data)
print(f'Prediction: {prediction.item()}')
# 396.84613037109375