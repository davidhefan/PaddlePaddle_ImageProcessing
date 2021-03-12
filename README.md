
[^_^]:
  主标题
# 图像处理基础

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

![图2](https://ai-studio-static-online.cdn.bcebos.com/0cc0db1f324346f49cc32c10faf7e9f95a28f591163f4494b35e4d67d1130d6a "图2")
![图3](https://ai-studio-static-online.cdn.bcebos.com/66d068fd30e449f4ba64fb30692d224b6af91d82b8c5475085a63e2e0001bf2b "图3")
![图4](https://ai-studio-static-online.cdn.bcebos.com/dbad81fc63bb42a8bb54369c2a8ccc77c29adda20b1e4aac83a3758dd3086f2e "图4")
![图5](https://ai-studio-static-online.cdn.bcebos.com/ea8435831b0f497a869bb52c741de13dd2fd10ed912944db9b3d1798af02b9e2 "图5")

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

### 2、OpenCV库进阶操作
#### 图像基本操作

学习ROI感兴趣区域，通道分离合并等基本操作。

##### ROI

[ROI](https://baike.baidu.com/item/ROI/1125333#viewPageContent)：Region of Interest，感兴趣区域。。

截取ROI非常简单，指定图片的范围即可

##### 通道分割与合并

彩色图的BGR三个通道是可以分开单独访问的，也可以将单独的三个通道合并成一副图像。分别使用`cv2.split()`和`cv2.merge()`

#### 颜色空间转换
最常用的颜色空间转换如下：
* RGB或BGR到灰度（COLOR_RGB2GRAY，COLOR_BGR2GRAY）
* RGB或BGR到YcrCb（或YCC）（COLOR_RGB2YCrCb，COLOR_BGR2YCrCb）
* RGB或BGR到HSV（COLOR_RGB2HSV，COLOR_BGR2HSV）
* RGB或BGR到Luv（COLOR_RGB2Luv，COLOR_BGR2Luv）
* 灰度到RGB或BGR（COLOR_GRAY2RGB，COLOR_GRAY2BGR）

> 经验之谈：颜色转换其实是数学运算，如灰度化最常用的是：`gray=R*0.299+G*0.587+B*0.114`。

> 参考资料：[OpenCV中的颜色空间](https://zhuanlan.zhihu.com/p/112790325)

##### 特定颜色物体追踪

[HSV](https://baike.baidu.com/item/HSV/547122)是一个常用于颜色识别的模型，相比BGR更易区分颜色，转换模式用`COLOR_BGR2HSV`表示。

> 经验之谈：OpenCV中色调H范围为[0,179]，饱和度S是[0,255]，明度V是[0,255]。虽然H的理论数值是0°~360°，但8位图像像素点的最大值是255，所以OpenCV中除以了2，某些软件可能使用不同的尺度表示，所以同其他软件混用时，记得归一化。

相关参考知识：
- [RGB、HSV和HSL颜色空间](https://zhuanlan.zhihu.com/p/67930839)

![https://pic4.zhimg.com/v2-e9f9c843e7d60e8f7aa7de1cd61d1818_1440w.jpg?source=172ae18b](https://pic4.zhimg.com/v2-e9f9c843e7d60e8f7aa7de1cd61d1818_1440w.jpg?source=172ae18b)

现在，我们实现一个使用HSV来只显示视频中蓝色物体的例子，步骤如下：

1. 捕获视频中的一帧
2. 从BGR转换到HSV
3. 提取蓝色范围的物体
4. 只显示蓝色物体

其中，`bitwise_and()`函数暂时不用管，后面会讲到。那蓝色的HSV值的上下限lower和upper范围是怎么得到的呢？其实很简单，我们先把标准蓝色的BGR值用`cvtColor()`转换下

结果是[120, 255, 255]，所以，我们把蓝色的范围调整成了上面代码那样。

> 经验之谈：[Lab](https://baike.baidu.com/item/Lab/1514615)颜色空间也经常用来做颜色识别，有兴趣的同学可以了解下。

#### 阈值分割
- 使用固定阈值、自适应阈值和Otsu阈值法"二值化"图像
- OpenCV函数：`cv2.threshold()`, `cv2.adaptiveThreshold()`

###### 固定阈值分割

固定阈值分割很直接，一句话说就是像素点值大于阈值变成一类值，小于阈值变成另一类值。

`cv2.threshold()`用来实现阈值分割，ret是return value缩写，代表当前的阈值。函数有4个参数：

- 参数1：要处理的原图，**一般是灰度图**
- 参数2：设定的阈值
- 参数3：最大阈值，一般为255
- 参数4：阈值的方式，主要有5种，详情：[ThresholdTypes](https://docs.opencv.org/4.0.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)
  - 0: THRESH_BINARY  当前点值大于阈值时，取Maxval,也就是第四个参数，否则设置为0
  - 1: THRESH_BINARY_INV 当前点值大于阈值时，设置为0，否则设置为Maxval
  - 2: THRESH_TRUNC 当前点值大于阈值时，设置为阈值，否则不改变
  - 3: THRESH_TOZERO 当前点值大于阈值时，不改变，否则设置为0
  - 4:THRESH_TOZERO_INV  当前点值大于阈值时，设置为0，否则不改变
>参考资料：[基于opencv的固定阈值分割_自适应阈值分割](https://blog.csdn.net/naibozhuan3744/article/details/78561574)

> 经验之谈：很多人误以为阈值分割就是[二值化](https://baike.baidu.com/item/%E4%BA%8C%E5%80%BC%E5%8C%96)。从上图中可以发现，两者并不等同，阈值分割结果是两类值，而不是两个值。

##### 自适应阈值

看得出来固定阈值是在整幅图片上应用一个阈值进行分割，*它并不适用于明暗分布不均的图片*。 `cv2.adaptiveThreshold()`自适应阈值会每次取图片的一小部分计算阈值，这样图片不同区域的阈值就不尽相同。它有5个参数，其实很好理解，先看下效果：

- 参数1：要处理的原图
- 参数2：最大阈值，一般为255
- 参数3：小区域阈值的计算方式
    - `ADAPTIVE_THRESH_MEAN_C`：小区域内取均值
    - `ADAPTIVE_THRESH_GAUSSIAN_C`：小区域内加权求和，权重是个高斯核
- 参数4：阈值方式（跟前面讲的那5种相同）
- 参数5：小区域的面积，如11就是11*11的小块
- 参数6：最终阈值等于小区域计算出的阈值再减去此值

建议读者调整下参数看看不同的结果。

##### Otsu阈值

在前面固定阈值中，我们是随便选了一个阈值如127，那如何知道我们选的这个阈值效果好不好呢？答案是：不断尝试，所以这种方法在很多文献中都被称为经验阈值。[Otsu阈值法](https://baike.baidu.com/item/otsu/16252828)就提供了一种自动高效的二值化方法。

##### 小结

- `cv2.threshold()`用来进行固定阈值分割。固定阈值不适用于光线不均匀的图片，所以用 `cv2.adaptiveThreshold()`进行自适应阈值分割。
- 二值化跟阈值分割并不等同。针对不同的图片，可以采用不同的阈值方法。

## 图像几何变换

- 实现旋转、平移和缩放图片
- OpenCV函数：`cv2.resize()`, `cv2.flip()`, `cv2.warpAffine()`

##### 缩放图片

缩放就是调整图片的大小，使用`cv2.resize()`函数实现缩放。可以按照比例缩放，也可以按照指定的大小缩放：
我们也可以指定缩放方法`interpolation`，更专业点叫插值方法，默认是`INTER_LINEAR`，全部可以参考：[InterpolationFlags](https://docs.opencv.org/4.0.0/da/d54/group__imgproc__transform.html#ga5bb5a1fea74ea38e1a5445ca803ff121)

缩放过程中有五种插值方式：
* cv2.INTER_NEAREST 最近邻插值
* cv2.INTER_LINEAR 线性插值
* cv2.INTER_AREA 基于局部像素的重采样，区域插值
* cv2.INTER_CUBIC 基于邻域4x4像素的三次插值
* cv2.INTER_LANCZOS4 基于8x8像素邻域的Lanczos插值

##### 翻转图片
镜像翻转图片，可以用`cv2.flip()`函数：
其中，参数2 = 0：垂直翻转(沿x轴)，参数2 > 0: 水平翻转(沿y轴)，参数2 < 0: 水平垂直翻转。

##### 平移图片

要平移图片，我们需要定义下面这样一个矩阵，tx,ty是向x和y方向平移的距离：

$$
 M = \left[
 \begin{matrix}
   1 & 0 & t_x \newline
   0 & 1 & t_y 
  \end{matrix}
  \right] 
$$

平移是用仿射变换函数`cv2.warpAffine()`实现的

#### 绘图功能
- 绘制各种几何形状、添加文字
- OpenCV函数：`cv2.line()`, `cv2.circle()`, `cv2.rectangle()`, `cv2.ellipse()`, `cv2.putText()`

绘制形状的函数有一些共同的参数，提前在此说明一下：
- img：要绘制形状的图片
- color：绘制的颜色
  - 彩色图就传入BGR的一组值，如蓝色就是(255,0,0)
  - 灰度图，传入一个灰度值就行
- thickness：线宽，默认为1；**对于矩形/圆之类的封闭形状而言，传入-1表示填充形状**
- lineType：线的类型。默认情况下，它是8连接的。cv2.LINE_AA 是适合曲线的抗锯齿线。

##### 画线
画直线只需指定起点和终点的坐标就行

##### 添加文字

使用`cv2.putText()`添加文字，它的参数也比较多，同样请对照后面的代码理解这几个参数：

- 参数2：要添加的文本
- 参数3：文字的起始坐标（左下角为起点）
- 参数4：字体
- 参数5：文字大小（缩放比例）

##### 小结

- `cv2.line()`画直线，`cv2.circle()`画圆，`cv2.rectangle()`画矩形，`cv2.ellipse()`画椭圆，`cv2.polylines()`画多边形，`cv2.putText()`添加文字。
- 画多条直线时，`cv2.polylines()`要比`cv2.line()`高效很多。
- 要在图像中打上中文，可以用PIL库结合OpenCV实现。

#### 图像间数学运算
- 图片间的数学运算，如相加、按位运算等
- OpenCV函数：`cv2.add()`, `cv2.addWeighted()`, `cv2.bitwise_and()`
##### 图片相加
要叠加两张图片，可以用`cv2.add()`函数，相加两幅图片的形状（高度/宽度/通道数）必须相同。numpy中可以直接用res = img + img1相加，但这两者的结果并不相同：

如果是二值化图片（只有0和255两种值），两者结果是一样的（用numpy的方式更简便一些）。
##### 图像混合
图像混合`cv2.addWeighted()`也是一种图片相加的操作，只不过两幅图片的权重不一样，γ相当于一个修正值：
$$
dst = \alpha\times img1+\beta\times img2 + \gamma
$$

##### 按位操作

按位操作包括按位与/或/非/异或操作，有什么用途呢？

如果将两幅图片直接相加会改变图片的颜色，如果用图像混合，则会改变图片的透明度，所以我们需要用按位操作。首先来了解一下
[掩膜](https://baike.baidu.com/item/%E6%8E%A9%E8%86%9C/8544392?fr=aladdin)（mask）的概念：掩膜是用一副二值化图片对另外一幅图片进行局部的遮挡

##### 小结

- `cv2.add()`用来叠加两幅图片，`cv2.addWeighted()`也是叠加两幅图片，但两幅图片的权重不一样。
- `cv2.bitwise_and()`, `cv2.bitwise_not()`, `cv2.bitwise_or()`, `cv2.bitwise_xor()`分别执行按位与/或/非/异或运算。掩膜就是用来对图片进行全局或局部的遮挡。

#### 平滑图像

- 模糊/平滑图片来消除图片噪声
- OpenCV函数：`cv2.blur()`, `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`

##### 滤波与模糊

关于滤波和模糊：

- 它们都属于卷积，不同滤波方法之间只是卷积核不同（对线性滤波而言）
- 低通滤波器是模糊，高通滤波器是锐化

低通滤波器就是允许低频信号通过，在图像中边缘和噪点都相当于高频部分，所以低通滤波器用于去除噪点、平滑和模糊图像。高通滤波器则反之，用来增强图像边缘，进行锐化处理。

> 常见噪声有[椒盐噪声](https://baike.baidu.com/item/%E6%A4%92%E7%9B%90%E5%99%AA%E5%A3%B0/3455958?fr=aladdin)和[高斯噪声](https://baike.baidu.com/item/%E9%AB%98%E6%96%AF%E5%99%AA%E5%A3%B0)，椒盐噪声可以理解为斑点，随机出现在图像中的黑点或白点；高斯噪声可以理解为拍摄图片时由于光照等原因造成的噪声。

##### 均值滤波

均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值，用`cv2.blur()`实现，如3×3的卷积核：

$$
 kernel = \frac{1}{9}\left[
 \begin{matrix}
   1 & 1 & 1 \newline
   1 & 1 & 1 \newline
   1 & 1 & 1
  \end{matrix}
  \right]
$$

```python
img = cv2.imread('lena.jpg')
blur = cv2.blur(img, (3, 3))  # 均值模糊
```

##### 方框滤波

方框滤波跟均值滤波很像，如3×3的滤波核如下：

$$
k = a\left[
 \begin{matrix}
   1 & 1 & 1 \newline
   1 & 1 & 1 \newline
   1 & 1 & 1
  \end{matrix}
  \right]
$$

用`cv2.boxFilter()`函数实现，当可选参数normalize为True的时候，方框滤波就是均值滤波，上式中的a就等于1/9；normalize为False的时候，a=1，相当于求区域内的像素和。

##### 高斯滤波

前面两种滤波方式，卷积核内的每个值都一样，也就是说图像区域中每个像素的权重也就一样。高斯滤波的卷积核权重并不相同：中间像素点权重最高，越远离中心的像素权重越小。

显然这种处理元素间权值的方式更加合理一些。图像是2维的，所以我们需要使用[2维的高斯函数](https://en.wikipedia.org/wiki/Gaussian_filter)，比如OpenCV中默认的3×3的高斯卷积核：

$$
k = \left[
 \begin{matrix}
   0.0625 & 0.125 & 0.0625 \newline
   0.125 & 0.25 & 0.125 \newline
   0.0625 & 0.125 & 0.0625
  \end{matrix}
  \right]
$$
OpenCV中对应函数为`cv2.GaussianBlur(src,ksize,sigmaX)`:
参数3 σx值越大，模糊效果越明显。高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。均值滤波与高斯滤波的对比结果如下（均值滤波丢失的细节更多）

##### 中值滤波

[中值](https://baike.baidu.com/item/%E4%B8%AD%E5%80%BC)又叫中位数，是所有数排序后取中间的值。中值滤波就是用区域内的中值来代替本像素值，所以那种孤立的斑点，如0或255很容易消除掉，适用于去除椒盐噪声和斑点噪声。中值是一种非线性操作，效率相比前面几种线性滤波要慢。

##### 双边滤波
模糊操作基本都会损失掉图像细节信息，尤其前面介绍的线性滤波器，图像的边缘信息很难保留下来。然而，边缘（edge）信息是图像中很重要的一个特征，所以这才有了[双边滤波](https://baike.baidu.com/item/%E5%8F%8C%E8%BE%B9%E6%BB%A4%E6%B3%A2)。用`cv2.bilateralFilter()`函数实现：可以看到，双边滤波明显保留了更多边缘信息。

#### 边缘检测

> [Canny J . A Computational Approach To Edge Detection[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1986, PAMI-8(6):679-698.](https://www.computer.org/cms/Computer.org/Transactions%20Home%20Pages/TPAMI/PDFs/top_ten_6.pdf)

- Canny边缘检测的简单概念
- OpenCV函数：`cv2.Canny()`

Canny边缘检测方法常被誉为边缘检测的最优方法：

`cv2.Canny()`进行边缘检测，参数2、3表示最低、高阈值，下面来解释下具体原理。

> 经验之谈：之前我们用低通滤波的方式模糊了图片，那反过来，想得到物体的边缘，就需要用到高通滤波。

##### Canny边缘检测

Canny边缘提取的具体步骤如下：

1. 使用5×5高斯滤波消除噪声：

边缘检测本身属于锐化操作，对噪点比较敏感，所以需要进行平滑处理。
$$
K=\frac{1}{256}\left[
 \begin{matrix}
   1 & 4 & 6 & 4 & 1 \newline
   4 & 16 & 24 & 16 & 4  \newline
   6 & 24 & 36 & 24 & 6  \newline
   4 & 16 & 24 & 16 & 4  \newline
   1 & 4 & 6 & 4 & 1
  \end{matrix}
  \right]
$$
2. 计算图像梯度的方向：

首先使用Sobel算子计算两个方向上的梯度$ G_x $和$ G_y $，然后算出梯度的方向：
$$
\theta=\arctan(\frac{G_y}{G_x})
$$
保留这四个方向的梯度：0°/45°/90°/135°，有什么用呢？我们接着看。

3. 取局部极大值：

梯度其实已经表示了轮廓，但为了进一步筛选，可以在上面的四个角度方向上再取局部极大值

4. 滞后阈值：

经过前面三步，就只剩下0和可能的边缘梯度值了，为了最终确定下来，需要设定高低阈值：
- 像素点的值大于最高阈值，那肯定是边缘
- 同理像素值小于最低阈值，那肯定不是边缘
- 像素值介于两者之间，如果与高于最高阈值的点连接，也算边缘，所以上图中C算，B不算

Canny推荐的高低阈值比在2:1到3:1之间。

##### 先阈值分割后检测
其实很多情况下，阈值分割后再检测边缘，效果会更好。

##### 小结

- Canny是用的最多的边缘检测算法，用`cv2.Canny()`实现。

#### 腐蚀与膨胀

- 了解形态学操作的概念
- 学习膨胀、腐蚀、开运算和闭运算等形态学操作
- OpenCV函数：`cv2.erode()`, `cv2.dilate()`, `cv2.morphologyEx()`

##### 啥叫形态学操作

形态学操作其实就是**改变物体的形状**，比如腐蚀就是"变瘦"，膨胀就是"变胖"。

> 经验之谈：形态学操作一般作用于二值化图，来连接相邻的元素或分离成独立的元素。**腐蚀和膨胀是针对图片中的白色部分！**

##### 腐蚀

腐蚀的效果是把图片"变瘦"，其原理是在原图的小区域内取局部最小值。因为是二值化图，只有0和255，所以小区域内有一个是0该像素点就为0。

这样原图中边缘地方就会变成0，达到了瘦身目的

OpenCV中用`cv2.erode()`函数进行腐蚀，只需要指定核的大小就行

##### 膨胀

膨胀与腐蚀相反，取的是局部最大值，效果是把图片"变胖"

##### 开/闭运算

先腐蚀后膨胀叫开运算（因为先腐蚀会分开物体，这样容易记住），其作用是：分离物体，消除小区域。这类形态学操作用`cv2.morphologyEx()`函数实现

#### 使用OpenCV摄像头与加载视频
学习打开摄像头捕获照片、播放本地视频、录制视频等。
- 打开摄像头并捕获照片
- 播放本地视频，录制视频
- OpenCV函数：`cv2.VideoCapture()`, `cv2.VideoWriter()`


##### 打开摄像头

要使用摄像头，需要使用`cv2.VideoCapture(0)`创建VideoCapture对象，参数0指的是摄像头的编号，如果你电脑上有两个摄像头的话，访问第2个摄像头就可以传入1，依此类推。

``` python
# 打开摄像头并灰度化显示
import cv2

capture = cv2.VideoCapture(0)

while(True):
    # 获取一帧
    ret, frame = capture.read()
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
```

`capture.read()`函数返回的第1个参数ret(return value缩写)是一个布尔值，表示当前这一帧是否获取正确。`cv2.cvtColor()`用来转换颜色，这里将彩色图转成灰度图。

另外，通过`cap.get(propId)`可以获取摄像头的一些属性，比如捕获的分辨率，亮度和对比度等。propId是从0~18的数字，代表不同的属性，完整的属性列表可以参考：[VideoCaptureProperties](https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)。也可以使用`cap.set(propId,value)`来修改属性值。比如说，我们在while之前添加下面的代码：

``` python
# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
width, height = capture.get(3), capture.get(4)
print(width, height)

# 以原分辨率的一倍来捕获
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)
```

> 经验之谈：某些摄像头设定分辨率等参数时会无效，因为它有固定的分辨率大小支持，一般可在摄像头的资料页中找到。

##### 播放本地视频

跟打开摄像头一样，如果把摄像头的编号换成视频的路径就可以播放本地视频了。回想一下`cv2.waitKey()`，它的参数表示暂停时间，所以这个值越大，视频播放速度越慢，反之，播放速度越快，通常设置为25或30。

```python
# 播放本地视频
capture = cv2.VideoCapture('demo_video.mp4')

while(capture.isOpened()):
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)
    if cv2.waitKey(30) == ord('q'):
        break
```

##### 录制视频

之前我们保存图片用的是`cv2.imwrite()`，要保存视频，我们需要创建一个`VideoWriter`的对象，需要给它传入四个参数：

- 输出的文件名，如'output.avi'
- 编码方式[FourCC](https://baike.baidu.com/item/fourcc/6168470?fr=aladdin)码
- 帧率[FPS](https://baike.baidu.com/item/FPS/3227416)
- 要保存的分辨率大小

FourCC是用来指定视频编码方式的四字节码，所有的编码可参考[Video Codecs](http://www.fourcc.org/codecs.php)。如MJPG编码可以这样写： `cv2.VideoWriter_fourcc(*'MJPG')`或`cv2.VideoWriter_fourcc('M','J','P','G')`

```python
capture = cv2.VideoCapture(0)

# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 25., (640, 480))

while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
```

##### 小结

- 使用`cv2.VideoCapture()`创建视频对象，然后在循环中一帧帧显示图像。参数传入数字时，代表打开摄像头，传入本地视频路径时，表示播放本地视频。
- `cap.get(propId)`获取视频属性，`cap.set(propId,value)`设置视频属性。
- `cv2.VideoWriter()`创建视频写入对象，用来录制/保存视频。

### 3、图像分类任务概念导入
#### 计算机视觉中的图像分类任务
##### 图像分类的基本任务
![](https://ai-studio-static-online.cdn.bcebos.com/b2242b73369b4af0846b5149c823679d79560e02eb3c4eb79fecd6f75c59bf90)
> 对人类来说，识别猫和狗是件非常容易的事。但对计算机来说，即使是一个精通编程的高手，也很难轻松写出具有通用性的程序（比如：假设程序认为体型大的是狗，体型小的是猫，但由于拍摄角度不同，可能一张图片上猫占据的像素比狗还多）。

在早期的图像分类任务中，通常是先人工提取图像特征，再用机器学习算法对这些特征进行分类，分类的结果强依赖于特征提取方法，往往只有经验丰富的研究者才能完成。  

![](https://ai-studio-static-online.cdn.bcebos.com/01179d17c9f74570b8a618d6123261ce6e10344f11c84dda8e47d44c1eb4fc81)

在这种背景下，基于神经网络的特征提取方法应运而生。Yann LeCun是最早将卷积神经网络应用到图像识别领域的，其主要逻辑是使用卷积神经网络提取图像特征，并对图像所属类别进行预测，通过训练数据不断调整网络参数，最终形成一套能自动提取图像特征并对这些特征进行分类的网络。  
  
![](https://ai-studio-static-online.cdn.bcebos.com/1ccd30567304415d98b0b373ec641a3d00f76d803f194ea4b14aa85ce85bf7bb)
##### 人工智能与深度学习
![](https://ai-studio-static-online.cdn.bcebos.com/62c9013cdaa94bd89128b9d322cf37a0038d8245bc3f4c3fa2e355932d5460f9)
> 图源：公众号tuputech

![](https://ai-studio-static-online.cdn.bcebos.com/2e51a81e3a7e4f94b7c6c1b56e695391bcffc0ff6b034200b140a6e1067b6c05)
> 图源《如何创造可信的AI》
##### 计算机视觉的子任务
![https://ai-studio-static-online.cdn.bcebos.com/d65f1ebcb0054dcb81a8eb50223adc529bb9b63265ab467d931a5df5b2864122](https://ai-studio-static-online.cdn.bcebos.com/d65f1ebcb0054dcb81a8eb50223adc529bb9b63265ab467d931a5df5b2864122)
* Image Classification： 图像分类，用于识别图像中物体的类别（如：bottle、cup、cube）。
* Object Localization： 目标检测，用于检测图像中每个物体的类别，并准确标出它们的位置。 
* Semantic Segmentation： 图像语义分割，用于标出图像中每个像素点所属的类别，属于同一类别的像素点用一个颜色标识。
* Instance Segmentation： 实例分割，值得注意的是，目标检测任务只需要标注出物体位置，实例分割任务不仅要标注出物体位置，还需要标注出物体的外形轮廓。

#### 图像分类问题的经典数据集
##### MNIST手写数字识别
> MNIST是一个手写体数字的图片数据集，该数据集来由美国国家标准与技术研究所（National Institute of Standards and Technology (NIST)）发起整理，一共统计了来自250个不同的人手写数字图片，其中50%是高中生，50%来自人口普查局的工作人员。该数据集的收集目的是希望通过算法，实现对手写数字的识别。

![](https://ai-studio-static-online.cdn.bcebos.com/eee28910dfb340649c245742059ae2a5677ee5d0b129414e826f622fbb10a1f2)

[数据集链接](http://yann.lecun.com/exdb/mnist/)

#### Cifar数据集
##### CIFAR-10
CIFAR-10数据集由10个类的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。  
数据集分为五个训练批次和一个测试批次，每个批次有10000个图像。测试批次包含来自每个类别的恰好1000个随机选择的图像。训练批次以随机顺序包含剩余图像，但一些训练批次可能包含来自一个类别的图像比另一个更多。总体来说，五个训练集之和包含来自每个类的正好5000张图像。  
以下是数据集中的类，以及来自每个类的10个随机图像：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200404185441124.png)  
这些类完全相互排斥。汽车和卡车之间没有重叠。“汽车”包括轿车，SUV，这类东西。“卡车”只包括大卡车。都不包括皮卡车。

[CIFAR-10 python版本](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) 

##### CIFAR-100
CIFAR-100数据集就像CIFAR-10，除了它有100个类，每个类包含600个图像。，每类各有500个训练图像和100个测试图像。CIFAR-100中的100个类被分成20个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）  
以下是CIFAR-100中的类别列表：

| 超类 | 类别 |
| --- | --- |
| 水生哺乳动物 | 海狸，海豚，水獭，海豹，鲸鱼 |
| 鱼 | 水族馆的鱼，比目鱼，射线，鲨鱼，鳟鱼 |
| 花卉 | 兰花，罂粟花，玫瑰，向日葵，郁金香 |
| 食品容器 | 瓶子，碗，罐子，杯子，盘子 |
| 水果和蔬菜 | 苹果，蘑菇，橘子，梨，甜椒 |
| 家用电器 | 时钟，电脑键盘，台灯，电话机，电视机 |
| 家用家具 | 床，椅子，沙发，桌子，衣柜 |
| 昆虫 | 蜜蜂，甲虫，蝴蝶，毛虫，蟑螂 |
| 大型食肉动物 | 熊，豹，狮子，老虎，狼 |
| 大型人造户外用品 | 桥，城堡，房子，路，摩天大楼 |
| 大自然的户外场景 | 云，森林，山，平原，海 |
| 大杂食动物和食草动物 | 骆驼，牛，黑猩猩，大象，袋鼠 |
| 中型哺乳动物 | 狐狸，豪猪，负鼠，浣熊，臭鼬 |
| 非昆虫无脊椎动物 | 螃蟹，龙虾，蜗牛，蜘蛛，蠕虫 |
| 人 | 宝贝，男孩，女孩，男人，女人 |
| 爬行动物 | 鳄鱼，恐龙，蜥蜴，蛇，乌龟 |
| 小型哺乳动物 | 仓鼠，老鼠，兔子，母老虎，松鼠 |
| 树木 | 枫树，橡树，棕榈，松树，柳树 |
| 车辆1 | 自行车，公共汽车，摩托车，皮卡车，火车 |
| 车辆2 | 割草机，火箭，有轨电车，坦克，拖拉机 |

[CIFAR-100 python版本](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)  

#### ImageNet数据集
* ImageNet数据集是一个计算机视觉数据集，是由斯坦福大学的李飞飞教授带领创建。该数据集包合 14,197,122张图片和21,841个Synset索引。Synset是WordNet层次结构中的一个节点，它又是一组同义词集合。ImageNet数据集一直是评估图像分类算法性能的基准。  
* ImageNet 数据集是为了促进计算机图像识别技术的发展而设立的一个大型图像数据集。2016 年ImageNet 数据集中已经超过干万张图片，每一张图片都被手工标定好类别。ImageNet 数据集中的图片涵盖了大部分生活中会看到的图片类别。ImageNet最初是拥有超过100万张图像的数据集。如图下图所示，它包含了各种各样的图像，并且每张图像都被关联了标签（类别名）。每年都会举办使用这个巨大数据集的ILSVRC图像识别大赛。

[http://image-net.org/download-imageurls](http://image-net.org/download-imageurls)

![file](https://ai-studio-static-online.cdn.bcebos.com/b66a423c93214f2887a221f8fa8065a8eb3a88133d094419b0026373ddae0b3a)

### 4、PaddleClas数据增强代码解析与实战

'''
class RandFlipImage(object):
    """ random flip image 随机翻转图片
        flip_code:
            1: Flipped Horizontally 水平翻转
            0: Flipped Vertically 上下翻转
            -1: Flipped Horizontally & Vertically 水平、上下翻转
    """

    def __init__(self, flip_code=1):
        # 设置一个翻转参数，1、0或-1
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        # 随机生成0或1（即是否翻转）
        if random.randint(0, 1) == 1:
            return cv2.flip(img, self.flip_code)
        else:
            return img
'''

'''
class RandCropImage(object):
    """ random crop image """
    """ 随机裁剪图片 """

    def __init__(self, size, scale=None, ratio=None, interpolation=-1):

        self.interpolation = interpolation if interpolation >= 0 else None
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]
        if self.interpolation is None:
            return cv2.resize(img, size)
        else:
            return cv2.resize(img, size, interpolation=self.interpolation)
'''

### 5、参考资料
- [面向初学者的OpenCV-Python教程](http://codec.wang/#/opencv/)
- [OpenCV学习—OpenCV图像处理基本操作](https://www.bilibili.com/video/BV1VC4y1h7wq?p=2)
- [OpenCV 4计算机视觉项目实战（原书第2版）](https://github.com/PacktPublishing/Learn-OpenCV-4-By-Building-Projects-Second-Edition)
