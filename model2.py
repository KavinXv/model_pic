import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import numpy as np

# 封装conv2d,bn,SiLu
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, d=1):
        '''
        in_channels: 输入通道
        out_channels: 输出通道
        k: kernel_size, 卷积核大小
        s: stride, 滑动步长
        p: padding, (k-1)*d//2
        g: groups, 分组卷积,这里没有分组
        d: dilation,
        '''
        super().__init__()
        # 如果 p 为 None，设置默认填充为 0
        p = (k - 1) * d // 2 if p is None else p
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p,
                               dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)

        return x

class Bottleneck(nn.Module):
    '''
    Bottleneck模块
    首先是加入了残差,当shortcut = True时,进行残差连接
    其次是卷积过程中多加了一层
    '''
    def __init__(self, in_channels, out_channels, shortcut=True, k=3, e=0.5):
        '''
        shortcut: 决定是否残差,一般都是True
        e: 缩放因子
        '''
        super().__init__()
        # 中间卷积的输出通道
        mid_channels = int(e * out_channels)
        self.cv1 = Conv(in_channels, mid_channels, k, 1)
        self.cv2 = Conv(mid_channels, out_channels, k, 1)
        self.is_shortcut = shortcut and (in_channels == out_channels)

    def forward(self, x):
        x1 = x
        x = self.cv1(x)
        x = self.cv2(x)

        # 如果用残差就返回两个的和,否则就只返回经过两次卷积的结果
        return x + x1 if self.is_shortcut else x

class C2f(nn.Module):
    '''
    C2f模块
    将输入的x分成两部分x1、x2,一部分进入多层Bottleneck,输出为x1
    其中x1是由每一层的Bottleneck的输出拼接的,所以会有n+2
    n+2就是n层Bottleneck加上最开始的x1,x2
    再进行卷积然后输出
    '''
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, e=0.5):
        super().__init__()
        self.mid_channels = int(out_channels * e)
        self.cv1 = Conv(in_channels, 2*self.mid_channels)
        self.cv2 = Conv((n+2)*self.mid_channels, out_channels)
        self.Bottleneck_list = nn.ModuleList(Bottleneck(self.mid_channels, self.mid_channels, shortcut, e=1.0)
                                             for _ in range(n))
    
    def forward(self, x):
        x1, x2 = self.cv1(x).chunk(2, 1)
        y = [x1, x2]
        # 对 x2 应用所有 Bottleneck 模块，并将结果添加到 y 中
        for bottleneck in self.Bottleneck_list:  # 使用 for 循环遍历 Bottleneck 模块
            x2 = bottleneck(x2)  # 更新 x2
            y.append(x2)  # 将每个 Bottleneck 的输出添加到 y 中
        x = torch.cat(y, 1)
        x = self.cv2(x)

        return x

class C2f_block(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        # assert n > 0,
        super().__init__()
        self.block = nn.ModuleList([C2f(in_channels, out_channels) 
                                    for _ in range(n)])
    
    def forward(self, x):
        for c in self.block:
            x = c(x)

        return x


# 通道注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        # assert in_channels > 0,
        
        # 自适应平均池化层，将每个通道的特征图进行全局平均池化，输出大小为 (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 自适应最大池化层，将每个通道的特征图进行全局最大池化，输出大小为 (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # MLP（多层感知机），包含两个1x1卷积层。第一个卷积层用于降维，第二个用于升维。
        # 其中 in_planes 表示输入通道数，ratio 是降维比例（默认 16 倍降维）
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),  # 通道降维
            nn.ReLU(),  # 使用 ReLU 激活函数
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)  # 通道升维，恢复原始通道数
        )
        
        # Sigmoid 激活函数，将注意力权重映射到 [0,1] 之间
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播函数
    def forward(self, x):
        # 对输入 x 进行全局平均池化，输出尺寸为 (B, C, 1, 1)，再通过 MLP 处理
        avg_out = self.mlp(self.avg_pool(x))
        
        # 对输入 x 进行全局最大池化，输出尺寸为 (B, C, 1, 1)，再通过 MLP 处理
        max_out = self.mlp(self.max_pool(x))
        
        # 将平均池化和最大池化的结果相加
        out = avg_out + max_out
        
        # 使用 Sigmoid 函数生成通道注意力权重，返回形状为 (B, C, 1, 1)
        return self.sigmoid(out)

# 空间注意力机制模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # 确保 kernel_size 只能是 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        
        # 根据 kernel_size 设置 padding 大小，kernel_size 为 7 时 padding 为 3，kernel_size 为 3 时 padding 为 1
        padding = 3 if kernel_size == 7 else 1
        
        # 卷积层，将输入的通道数从 2 降到 1，kernel_size 用于控制卷积核大小
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
        # Sigmoid 激活函数，将注意力权重映射到 [0,1] 之间
        self.sigmoid = nn.Sigmoid()
    
    # 前向传播函数
    def forward(self, x):
        # 这里处理的都是所有通道的同一个像素点
        # 比如说avg_out就是将所有通道上同一个像素点的值加起来求平均

        # 在通道维度上计算输入的平均池化，得到形状为 (B, 1, H, W) 的张量
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 在通道维度上计算输入的最大池化，得到形状为 (B, 1, H, W) 的张量
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 将平均池化和最大池化结果在通道维度上拼接，得到形状为 (B, 2, H, W)
        x = torch.cat([avg_out, max_out], dim=1)
        
        # 使用卷积层将通道数从 2 降到 1，得到形状为 (B, 1, H, W)
        x = self.conv1(x)
        
        # 使用 Sigmoid 函数生成空间注意力权重，返回形状为 (B, 1, H, W)
        return self.sigmoid(x)

# CBAM 模块：包含通道注意力机制和空间注意力机制
class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # 初始化通道注意力模块
        self.ca = ChannelAttention(in_channels, ratio)
        
        # 初始化空间注意力模块
        self.sa = SpatialAttention(kernel_size)
    
    # 初始化权重
    def init_weights(self):
        # 遍历所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层的权重使用 Kaiming 初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                
                # 如果卷积层有偏置，则将偏置初始化为 0
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm 层的权重初始化为 1，偏置初始化为 0
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层的权重使用正态分布初始化，方差为 0.001，偏置初始化为 0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    # 前向传播函数
    def forward(self, x):
        # 通过通道注意力机制生成的特征图，x 为输入特征图，ca(x) 为通道注意力权重
        # 将输入特征图与通道注意力权重相乘，得到输出特征图 out
        # 通过通道注意力机制计算出的权重。这一过程强调重要的特征通道，抑制不重要的通道。
        out = x * self.ca(x)
        
        # 通过空间注意力机制生成的特征图，将通道注意力后的特征图与空间注意力权重相乘
        # 最终得到经过 CBAM 模块处理的输出特征图 result
        # 通过空间注意力机制计算出的权重，用于强调特征图中重要的空间区域。
        result = out * self.sa(out)
        
        # 返回最终的特征图
        return result

class Rock_CNN(nn.Module):
    def __init__(self, out_num):
        super().__init__()
        if out_num == None:
            print('error')
        self.out_num = out_num
        self.backbone = nn.Sequential(
            Conv(3, 16, 3, 2),  
            #CBAM(16),
            nn.Dropout(0.5),
            Conv(16, 32, 3, 2),  
            #C2f_block(32, 32, 3),    
            nn.Dropout(0.5),
            Conv(32, 64, 3, 2), 
            #C2f_block(64, 64, 6),    
            Conv(64, 128, 3, 2),
            nn.Dropout(0.5),
            #C2f_block(128, 128, 6),     
            Conv(128, 256, 3, 2),
            #CBAM(256),
            #nn.Dropout(0.5),
            #C2f_block(256, 256, 1)     
        )

        # 添加全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),  # 将特征图展平
            nn.Linear(102400 * 1 * 1, 64),  # 假设输出的特征图是 (batch_size, 1024, 1, 1)
            nn.ReLU(),      # 激活函数
            nn.Dropout(0.5),  # Dropout 层
            nn.Linear(64, out_num)  # 输出层，out_num 是类别数量
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return x



'''
# 示例用法
if __name__ == "__main__":
    # 输入特征图（3通道，3x3）
    input = np.array([[[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]],
                      
                      [[9, 8, 7],
                       [6, 5, 4],
                       [3, 2, 1]],
                      
                      [[1, 3, 5],
                       [7, 9, 11],
                       [13, 15, 17]]])

    conv = Conv(3, 6)
    bottleneck = Bottleneck(3, 6)
    c2f = C2f(3, 6)
    cbam = CBAM(3, 1)
    # 将输入转换为 PyTorch 张量，并添加批次维度
    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)  # 添加批次维度

    output = cbam(input)

    print(output)

    out_num = 10  # 例如，10 类输出

    # 创建 Rock_CNN 实例
    model = Rock_CNN(out_num)

    # 生成一个随机输入张量，形状为 (batch_size, channels, height, width)
    # 例如，批大小为 4，输入图像为 3 个通道（RGB），尺寸为 256x256
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 256, 256)

    # 进行前向传播
    output = model(input_tensor)

    # 打印输出结果的形状
    print("Output shape:", output.shape)
'''
        
                
            