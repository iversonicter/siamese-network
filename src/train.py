# Author: wang yongjie
# Email : yongjie.wang@ntu.edu.sg

import sys
sys.path.append("../preprocess")
sys.path.append("../util")

from create_train import *
from loader import *
from network import *
from loss import *
from torch.autograd import Variable
import matplotlib.pyplot as plt

dataset = "/home/yongjie/github/siamese-network/data/orl_faces/"
file_list = create_filelist(dataset)
image_list, label_list = create_pair(file_list, 10000)

transform = transforms.Compose([transforms.Resize((100, 100)),
    transforms.ToTensor()])

train_data = Face(image_list, label_list, transform = transform)
train_loader = Data.DataLoader(dataset = train_data, batch_size = 1024, shuffle = True, num_workers = 20)

net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0005)

counter = []
loss_history = []
iteration_number = 0

if __name__ == "__main__":
    for epoch in range(0, 20):
        for i, data in enumerate(train_loader, 0):
            img0, img1, label = data
            img0, img1, label = Variable(img0.float()).cuda(), Variable(img1.float()).cuda(), Variable(label.float()).cuda()
            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Epoch:{}\tcurrent loss:{}\n".format(epoch, loss.data.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.data.item())


    plt.plot(counter, loss_history)
    plt.show()
    torch.save(net.state_dict(), "../model/SiameseNet.pth")


