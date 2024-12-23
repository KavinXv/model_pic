import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from model1 import Rock_CNN
# from model2 import Rock_CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Rock_CNN(out_num=11)  
# conv5
# model.load_state_dict(torch.load(r'D:\vscode_code\Deep_learning\Photo\model_pic\dataset\out_conv5\rock_150_0.5526.pth')) 
model.load_state_dict(torch.load(r'D:\vscode_code\Deep_learning\Photo\model_pic\dataset\out\rock_937_0.5939849624060151.pth')) 
model.to(device) 
model.eval()  


transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

test_path = r'D:\vscode_code\Deep_learning\Photo\model_pic\dataset\data\test'
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)  
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  


criterion = nn.CrossEntropyLoss()  


@torch.no_grad()
def predict(model, test_loader, device, criterion):
    losses = []  
    correct = 0  
    total = 0 

    for epoch in range(test_epochs): 
        for batch_idx, (data, label) in enumerate(test_loader): 
            
            data, label = data.to(device), label.to(device) 

            output = model(data) 
            loss = criterion(output, label) 
            losses.append(loss.item()) 

            # 计算准确率
            _, predicted = torch.max(output, 1)  
            correct += (predicted == label).sum().item()  
            total += label.size(0) 

            print("正在评估...")
            print(f"Epoch {epoch + 1}/{test_epochs} "
                  f"| 批次 {batch_idx+1}/{len(test_loader)} "
                  f"| 损失: {loss.item():.4f} "
                  f"| 准确率: {correct / total:.4f}"
            )

    avg_loss = sum(losses) / len(losses) 
    accuracy = correct / total  

    return avg_loss, accuracy 


test_epochs = 1  
avg_loss, accuracy = predict(model, test_loader, device, criterion)
print(f"平均损失: {avg_loss:.4f}, 准确率: {accuracy:.4f}")
