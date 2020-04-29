import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 载入图片
img = Image.open("./pgy.jpg")
plt.imshow(img)

img = data_transform(img)  # [H, W, Channel] --> [Channel, H, W]
img = torch.unsqueeze(img, dim=0)  # [Channel, H, W] --> [1, Channel, H, W]

# 读取分类标签文本
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)


model = AlexNet(num_classes=5)

model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()  # 此模式dropout使模型中的操作失效

with torch.no_grad():  # 只预测是不需要反向传播的，也就不需要梯度数据，使用no_grad()不生成梯度数据可以节约内存
    output = torch.squeeze(model(img))  # 去掉batch维度 （这里batch=1）
    predict = torch.softmax(output, dim=0)  # 变成概率分布
    predict_cla = torch.argmax(predict).numpy()  # 取最大值那个位置的索引
print(class_indict[str(predict_cla)], predict[predict_cla].item())
plt.show()