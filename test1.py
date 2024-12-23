import torch
from model import Rock_CNN

# 模型路径
model_path = r'E:\Deep_learning\Photo\model_pic\dataset\out\rock_937_0.5939849624060151.pth'

# 检查路径是否存在
import os
if not os.path.exists(model_path):
    print(f"Model path {model_path} does not exist.")
else:
    print(f"Loading model from {model_path}")

# 加载模型
model = Rock_CNN(11)  # 根据你的模型定义结构
model.load_state_dict(torch.load(model_path))  # 加载模型的权重

# 将模型移到设备上（如果有GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 计算模型的参数总数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f'Total trainable parameters: {total_params}')
