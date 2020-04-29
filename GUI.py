import os
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import AlexNet

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('识别一下是什么花')

        self.frame_left = tk.Frame(self.root, padx=1, pady=5, bg="#aaaaaa")
        self.frame_left.pack(padx=5, pady=10, fill="y", side=tk.LEFT)

        self.frame_right = tk.Frame(self.root, padx=5, pady=5)
        self.frame_right.pack(padx=5, pady=10, fill="y", side=tk.LEFT)

        self.button_loadModel = tk.Button(self.frame_right, text='加载模型', command=self.loadModel, width=15, height=2)
        self.button_loadModel.pack(fill="x")

        self.button_loadImage = tk.Button(self.frame_right, text='加载图像', command=self.loadimg, width=15, height=2)
        self.button_loadImage.pack(fill="x")
        self.button_loadImage.config(state=tk.DISABLED)

        self.button_predict = tk.Button(self.frame_right, text='识别一下', command=self.predict, width=15, height=2)
        self.button_predict.pack(fill="x")
        self.button_predict.config(state=tk.DISABLED)

        self.label_info = tk.Label(self.frame_right, font=('宋体', 12), justify=tk.LEFT, padx=2, pady=30)
        self.label_info.pack(fill="x")
        self.label_info.config(text="请载入一个模型文件")

        self.canvas = tk.Canvas(self.frame_left, bg='gray', height=300, width=400)
        self.canvas.pack(fill='x', expand='yes')

        self.root.mainloop()

    def load_image_to_canvas(self, file_path):
        """把给定路径的图像加载入self.img 并绘制到canvas"""
        def resize(w_box, h_box, pil_image):  # 参数是：要适应的窗口宽、高、Image.open后的图片
            w, h = pil_image.size  # 获取图像的原始大小
            f1 = 1.0 * w_box / w
            f2 = 1.0 * h_box / h
            factor = min([f1, f2])
            width = int(w * factor)
            height = int(h * factor)
            return pil_image.resize((width, height), Image.ANTIALIAS)

        try:
            img = Image.open(file_path)
            self.img = img
            img_w, img_h = img.size
            if img_w>400:
                img_w = 400
                img_h = img_h * (400/img_w)
                img = resize(img_w, img_h, img)
            self.pil_img = ImageTk.PhotoImage(img)  # PhotoImage返回的对象必须一直被引用着，一旦失去引用，canvas上的图像立即消失
            self.canvas.update()  # 获取宽高之前要先对于这个组件update()
            x, y = 0, (self.canvas.winfo_height()-img.size[1]) / 2
            self.canvas.create_image(x, y, anchor='nw', image=self.pil_img)
        except Exception as e:
            self.label_info.config(text="图片载入出错")
        finally:
            self.button_predict.config(state=tk.NORMAL)
            self.label_info.config(text="图片已载入\n点击预测按钮")

    def predict(self):
        """根据已载入的模型进行识别"""
        class_dict = {
            "0": "雏菊",
            "1": "蒲公英",
            "2": "玫瑰",
            "3": "向日葵",
            "4": "郁金香"
        }
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = data_transform(self.img)  # [H, W, Channel] --> [Channel, H, W]
        img = torch.unsqueeze(img, dim=0)  # [Channel, H, W] --> [1, Channel, H, W]
        with torch.no_grad():
            output = torch.squeeze(self.model(img))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()  # 最大值位置索引
            print(predict_cla)
        class_str = class_dict[str(predict_cla)]
        prob_str = "%.1f" % (predict[predict_cla].item()*100)
        self.label_info.config(text=f"类别：{class_str}\n可能性：{prob_str}%")



    def loadModel(self):
        """载入指定的模型"""

        default_dir = os.getcwd()
        modelPath = askopenfilename(title='选择一个模型文件',
                                    initialdir=(os.path.expanduser(default_dir)),
                                    filetypes=[('pth文件', '*.pth'), ('All Files', '*')])
        if modelPath == "":
            return

        try:
            self.label_info.config(text="载入模型中……")
            model = AlexNet(num_classes=5)
            model.load_state_dict(torch.load(modelPath))
            model.eval()
            self.model = model
        except Exception as e:
            self.label_info.config(text="模型载入出错")
        finally:
            self.button_loadImage.config(state=tk.NORMAL)
            self.label_info.config(text="请打开一张图片")

    def loadimg(self):
        """载入指定的jpg图片"""
        default_dir = os.getcwd()
        photoPath = askopenfilename(title='打开一个照片（jpg格式）',
                                    initialdir=(os.path.expanduser(default_dir)),
                                    filetypes=[('jpg文件', '*.jpg'), ('All Files', '*')])
        if photoPath=="":
            return
        self.load_image_to_canvas(photoPath)



if __name__ == '__main__':
    win = MainWindow()