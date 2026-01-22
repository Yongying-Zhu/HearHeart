import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 第一层：3通道输入（Mel + STFT + CQT）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        
        # 64 (CNN) + 15 (wide features) = 79
        self.lin = nn.Linear(in_features=79, out_features=3)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, wide_features):
        x = self.pool1(self.bn1(self.relu1(self.conv1(x))))
        x = self.pool2(self.bn2(self.relu2(self.conv2(x))))
        x = self.pool3(self.bn3(self.relu3(self.conv3(x))))
        x = self.pool4(self.bn4(self.relu4(self.conv4(x))))
        x = self.ap(x)
        x = x.view(x.shape[0], -1)  # 64维
        x = torch.cat([x, wide_features], dim=1)  # 64 + 15 = 79维
        x = self.lin(x)
        return x
