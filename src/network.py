# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Sequential(
                nn.ReflectionPad1d(1),
                nn.Conv2d(1, 4, kernel_size = 3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                nn.Dropout(p=.2),

                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size = 3),
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(8),
                nn.Dropout(p=0.2),

                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size = 3),
                nn.ReLU(inplace = True),
                nn.BatchNorm2d(8),
                nn.Dropout2d(p=.2),
                )

        self.fc1 = nn.Sequential(
                nn.Linear(8 * 100 * 100, 500),
                nn.ReLU(inplace = True),

                nn.Linear(500, 500),
                nn.ReLU(inplace = True),

                nn.Linear(500, 5)
                )

    def forward_once(self, x):
        output = self.conv1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)

        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2



