import torch
from model import Rock_CNN
from train import count_outnum

# 数据集
train_path = r'D:\vscode_code\Deep_learning\Photo\model_pic\dataset\data\train'
#val_path = r'D:\vscode_code\Deep_learning\study_CNN\MNIST_data\val'
test_path = r'D:\vscode_code\Deep_learning\Photo\model_pic\dataset\data\test'


out_num = count_outnum(train_path) if count_outnum(train_path) == count_outnum(test_path) else None
model = Rock_CNN(out_num)


# 打印整个模型的结构
print(model)
