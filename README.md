
[^_^]:
  主标题
# 图像处理基础（一）

[^_^]:
  二级标题
## 2021年3月

[^_^]:
  三级级标题
### 课程大纲
- 1、图像处理的概念与基础操作
- 2、OpenCV库进阶操作
- 3、图像分类任务概念导入
- 4、PaddleClas数据增强代码解析
- 5、参考资料
------在这里撰写你的笔记------  
### 1、图像处理的概念与基本操作
#### 图片、GIF、视频

![](https://ai-studio-static-online.cdn.bcebos.com/e51b24965be04f87a6c4aefff4bcaaf4f629ba9e84514e1095a20f6c9d47b522)
![](https://ai-studio-static-online.cdn.bcebos.com/0cc0db1f324346f49cc32c10faf7e9f95a28f591163f4494b35e4d67d1130d6a)
![](https://ai-studio-static-online.cdn.bcebos.com/66d068fd30e449f4ba64fb30692d224b6af91d82b8c5475085a63e2e0001bf2b)
![](https://ai-studio-static-online.cdn.bcebos.com/dbad81fc63bb42a8bb54369c2a8ccc77c29adda20b1e4aac83a3758dd3086f2e)
![](https://ai-studio-static-online.cdn.bcebos.com/ea8435831b0f497a869bb52c741de13dd2fd10ed912944db9b3d1798af02b9e2)

[趣味视频项目](https://aistudio.baidu.com/aistudio/projectdetail/1176901?channelType=0&channel=0)

#### 像素：画面中最小的点
问题：图片在计算机中长啥样？
##### 灰度图片
```
# 加载一张手写数字的灰度图片
# 从Paddle2.0内置数据集中加载手写数字数据集，本文第3章会进一步说明
from paddle.vision.datasets import MNIST
# 选择测试集
mnist = MNIST(mode='test')
# 遍历手写数字的测试集
for i in range(len(mnist)):
    # 取出第一张图片
    if i == 0:
        sample = mnist[i]
        # 打印第一张图片的形状和标签
        print(sample[0].size, sample[1])
```
- 灰度值与光学三原色（RGB）：红、绿、蓝(靛蓝)。光学三原色混合后，组成像素点的显示颜色，三原色同时相加为白色，白色属于无色系（黑白灰）中的一种。
- 参考资料：[常用RGB颜色表](http://jsxzjh.bokee.com/3744988.html)

![https://bkimg.cdn.bcebos.com/pic/a8014c086e061d9582f99a307bf40ad162d9ca7f?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxNTA=,g_7,xp_5,yp_5/format,f_auto](https://bkimg.cdn.bcebos.com/pic/a8014c086e061d9582f99a307bf40ad162d9ca7f?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxNTA=,g_7,xp_5,yp_5/format,f_auto)

![https://bkimg.cdn.bcebos.com/pic/b3fb43166d224f4af2641ca204f790529922d186?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxNTA=,g_7,xp_5,yp_5/format,f_auto](https://bkimg.cdn.bcebos.com/pic/b3fb43166d224f4af2641ca204f790529922d186?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxNTA=,g_7,xp_5,yp_5/format,f_auto)

#### 分辨率=画面水平方向的像素值 * 画面垂直方向的像素值
##### 屏幕分辨率
例如，屏幕分辨率是1024×768，也就是说设备屏幕的水平方向上有1024个像素点，垂直方向上有768个像素点。像素的大小是没有固定长度的，不同设备上一个单位像素色块的大小是不一样的。   

例如，尺寸面积大小相同的两块屏幕，分辨率大小可以是不一样的，分辨率高的屏幕上面像素点（色块）就多，所以屏幕内可以展示的画面就更细致，单个色块面积更小。而分辨率低的屏幕上像素点（色块）更少，单个像素面积更大，可以显示的画面就没那么细致。
##### 图像分辨率
例如，一张图片分辨率是500x200，也就是说这张图片在屏幕上按1:1放大时，水平方向有500个像素点（色块），垂直方向有200个像素点（色块）。

在同一台设备上，图片分辨率越高，这张图片1:1放大时，图片面积越大；图片分辨率越低，这张图片1:1缩放时，图片面积越小。（可以理解为图片的像素点和屏幕的像素点是一个一个对应的）。  

但是，在屏幕上把图片超过100%放大时，为什么图片上像素色块也变的越大，其实是设备通过算法对图像进行了像素补足，我们把图片放的很大后看到的一块一块的方格子，虽然理解为一个图像像素，但是其实是已经补充了很多个屏幕像素；同理，把图片小于100%缩小时，也是通过算法将图片像素进行减少。

> 作者：自由自在的心
> 链接：https://www.zhihu.com/question/21149600/answer/29971490

![file](https://ai-studio-static-online.cdn.bcebos.com/a9a7da1ad6074a55b836cf6a8f06b7dbf1fb6196ccd948f7941f563c35f891c7)

#### 图像的基本概念
- 常见图片格式：jpg、png、gif、psd、tiff、bmp等
- 参考资料：[几种常见图片格式的区别](https://zhuanlan.zhihu.com/p/143649897)

#### 使用OpenCV加载并保存图片
- 加载图片，显示图片，保存图片
- OpenCV函数：`cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`

##### 说明

大部分人可能都知道电脑上的彩色图是以RGB(红-绿-蓝，Red-Green-Blue)颜色模式显示的，但OpenCV中彩色图是以B-G-R通道顺序存储的，灰度图只有一个通道。
> OpenCV默认使用BGR格式，而RGB和BGR的颜色转换不同，即使转换为灰度也是如此。一些开发人员认为R+G+B/3对于灰度是正确的，但最佳灰度值称为亮度（luminosity），并且具有公式：0.21*R+0.72*G+0.07*B

图像坐标的起始点是在左上角，所以行对应的是y，列对应的是x。

##### 加载图片

使用`cv2.imread()`来读入一张图片：
- 参数1：图片的文件名

    - 如果图片放在当前文件夹下，直接写文件名就行了，如'lena.jpg'
    - 否则需要给出绝对路径，如'D:\OpenCVSamples\lena.jpg'

- 参数2：读入方式，省略即采用默认值

    - `cv2.IMREAD_COLOR`：彩色图，默认值(1)
    - `cv2.IMREAD_GRAYSCALE`：灰度图(0)
    - `cv2.IMREAD_UNCHANGED`：包含透明通道的彩色图(-1)

> 经验之谈：路径中不能有中文，没有加载成功的话是不会报错的，`print(img)`的结果为None，后面处理才会报错。

