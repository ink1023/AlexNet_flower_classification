import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import AlexNet
import os
import json
import time

# 第1个GPU或者使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

# 训练集 和 测试集 采用的transform
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 载入训练集
data_root = os.path.abspath(os.path.join(os.getcwd()))
image_path = data_root + "/flower_data/"
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 保存一下这个数据集都有哪些分类到文件
# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())  # 交换索引和索引值的身份
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

# 构造训练集的DataLoader
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

# 载入验证集
validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
# 构造验证集的DataLoader
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))


net = AlexNet(num_classes=5, init_weights=True)  # 构造一个新模型
#net.load_state_dict(torch.load("./AlexNet.pth"))
net.to(device)  # 将数据迁移到GPU（如果有的话）
loss_function = nn.CrossEntropyLoss()  # 损失函数
#pata = list(net.parameters())
optimizer = optim.Adam(net.parameters(), lr=0.0002)  # 梯度下降优化器

save_path = './AlexNet.pth'
best_acc = 0.0  # 历史最高准确率（每一个epoch得到的模型都有一个新的准确率）
for epoch in range(10):
    net.train()  # 训练模式开启
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()  # 各参数梯度置零，否则梯度会累加
        outputs = net(images.to(device))  # 将数据正向过一遍网络，得到一个输出
        loss = loss_function(outputs, labels.to(device))  # 计算输出值和期望值的某种误差
        loss.backward()  # 计算 loss函数 对于各权重参数的偏导数，存入各参数的梯度属性
        optimizer.step()  # 梯度下降法更新各参数


        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)  # 本epoch的进度
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)  # 本epoch的所用时间

    net.eval()  # 验证模式开启
    acc = 0.0
    with torch.no_grad():
        for data_test in validate_loader:  # 对于验证集中的每一个batch
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))  # 经过我们的网络获得输出
            predict_y = torch.max(outputs, dim=1)[1]  # 对于本batch的每一张图片，取得网络输出的结果中最大值那个位置的索引
            acc += (predict_y == test_labels.to(device)).sum().item()  # 将索引和labels比对，求本batch出有多少个预测正确的，将这个个数累加
        accurate_test = acc / val_num  # 算出当前epoch在验证集中的预测准确率
        if accurate_test > best_acc:  # 如果本batch的准确率比以前都高
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)  # 就保存当前模型
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, acc / val_num))

print('Finished Training')