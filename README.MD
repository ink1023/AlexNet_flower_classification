# CNN样例: 使用AlexNet模型对于花朵图片进行分类

## 说明

```train.py```是训练脚本
```GUI.py```是图形界面

```AlexNet.pth```是已经训练好的模型

有3张测试用的图片 ```pgy.jpg```/```xrk.jpg```/```yjx.jpg```



## 下载训练集

如果你想自己训练一遍，可以下载下边这个鲜花训练集

<img src="https://s1.ax1x.com/2020/04/29/J7AgOg.png" alt="image-20200429164204119" style="zoom: 33%;" />

**训练用的数据集体积太大，放到了蓝奏云网盘上**

分块压缩包，注意改成正确的后缀名，然后放到一起解压那个 ```.zip```文件

https://lanzous.com/ic2kbsd  (后缀名改成```.z02```)

https://lanzous.com/ic2k8pc (后缀名改成```.z01```)

https://lanzous.com/ic2jzzi (后缀名不用改，和前两个放在一起，解压本压缩文件)

<img src="https://s1.ax1x.com/2020/04/29/J7ArfP.png" alt="image-20200429164644862" style="zoom: 67%;" />

将解压出来的 ```train``` 和 ```val``` 文件夹放进 ```flower_data``` 文件夹





## 使用预训练好的模型文件

下载下来解压出```AlexNet.pth```文件和```GUI.py```放到同一级目录下即可

https://lanzous.com/ic2kxda





## 图形界面```GUI.py```使用方法

1. 先加载训练好的模型文件```AlexNet.pth```

   <img src="https://s1.ax1x.com/2020/04/29/J7A6l8.png" alt="image-20200429160949071" style="zoom: 50%;" />

   

   2. 再去网上找一个鲜花图片（```雏菊```、``` 蒲公英```、```玫瑰```、```向日葵```、```郁金香``` 之一），要求```.jpg```格式，在程序中加载这个图片

      <img src="https://s1.ax1x.com/2020/04/29/J7AySf.png" alt="image-20200429161147338" style="zoom:50%;" />

      

      3. 点击识别按钮进行识别预测

         <img src="https://s1.ax1x.com/2020/04/29/J7ARmQ.png" alt="image-20200429161409886" style="zoom:50%;" />

      

   

