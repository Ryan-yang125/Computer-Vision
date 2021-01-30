<h1><center> HW01-制作无声小短片</center></h1>

<center> 3180101941 杨锐</center>

## 软件开发说明

- @Windows
- Python == 3.7.6

- Numpy == 1.8.1
- Opencv-python == 4.2.0.34
- Pyinstaller == 4.1(用于打包python文件 -> 可执行文件)

## 算法设计思路

视频分为三个部分：

- 片头：浙大照片+个人照片+个人信息
- 简笔画：吃豆人+ 豆子
- 片尾：随机射线

下面分别详细讲述实现

### 片头

> 浙大照片 -> 向下平移出 -> 个人照片 -> 个人信息 -> 向左平移出

首先读取一张图片，并初始化视频参数：

![截屏2020-11-30 10.22.23](https://i.loli.net/2020/11/30/NDvQyALc1wxeV8U.png)

对图片进行*resize*，静态写入两秒视频。

使用`np.float32([[1,0,x],[0,1,y]])`设定平移矩阵，使用`cv2.warpAffine()`每一帧更新图片并读入视频。

在每一帧读入视频的同时，使用`cv2.imshow`展示当前图片，并使用`cv2.waitkey()`等待用户键入空格来暂停/继续（这意味着我们对图片的处理也暂停了）

![截屏2020-11-30 10.31.21](https://i.loli.net/2020/11/30/yTD7pfWdqCMFhxK.png)

读入个人图片的思路完全一样，增加一个`cv2.putText()`添加个人信息

### 简笔画

> 吃豆人出现  > 第一个豆子> 第二个豆子> ... > 吃豆人发射射线

由于要绘制多个豆子，而他们的区别仅在于位置和颜色的不同，因此封装成一个类并配合`@staticmethod`实现代码复现：

```python
class AnimateDraw:
    @staticmethod
    def drawPacman(img, pacmanBodyCenter):
    @staticmethod
    def drawBeans(img, beanHeadCenter, bodyColor):
    @staticmethod
    def drawAnimate(video,fps):
```

**吃豆人**：使用`cv2.ellipse()`绘制一个3/4的扇形，使用`cv2.circle`绘制眼睛

```python
def drawPacman(img, pacmanBodyCenter):
    cv2.ellipse(img, pacmanBodyCenter, (80, 80),
                0, 30, 330, (0, 255, 255), -1)
    pacmanEyeCenter = (110, 200)
    cv2.circle(img, pacmanEyeCenter, 12, (255, 255, 255), -1)
```

**豆子**：由于豆子涉及的点数较多，如果将每一个坐标写死，显然不利于复用，因此通过传入参数`beanHeadCenter`和`bodysize`计算出其余坐标，再使用`cv2.fillPoly()`绘制并填充多边形

![截屏2020-11-30 11.03.22](https://i.loli.net/2020/11/30/Pvb2hIdr96kFaw1.png)

**绘制动画：**

通过每隔一定帧调用`AnimateDraw.drawBeans()`函数，并传入更新后的坐标和颜色，实现间隔出现，其他绘制和之前思路一致

![截屏2020-11-30 11.10.16](https://i.loli.net/2020/11/30/IApY4ltPkaV5NRd.png)

### 结尾

> 发射随机射线杀死豆子，结尾出现Thanks

![截屏2020-11-30 11.13.07](https://i.loli.net/2020/11/30/6I1XbdjcUBRr9qe.png)

## 实验结果展示和分析

### 开头动画

![截屏2020-11-30 20.30.58](https://i.loli.net/2020/11/30/FKtiT6QEsqGRMNg.png)

### 个人信息

![截屏2020-11-30 20.31.42](https://i.loli.net/2020/11/30/fqNAu69xaihgDyv.png)

### 吃豆人

![截屏2020-11-30 20.32.20](../../../../../Library/Application%20Support/typora-user-images/%E6%88%AA%E5%B1%8F2020-11-30%2020.32.20.png)

### 动画

![截屏2020-11-30 20.32.50](https://i.loli.net/2020/11/30/QfFea5ktCOoNlvA.png)

### 结尾

![截屏2020-11-30 20.33.42](https://i.loli.net/2020/11/30/RPgUIw2XZEdphF8.png)

完整版见exe/main.exe运行结果或*output.mp4*

## 编程体会

- 熟悉了opencv读/写图片的基本操作
- 熟悉了opencv读/写视频的基本操作
- 熟悉了opencv绘制图形的基本操作
- 熟悉了opencv图形变换（平移、旋转、缩放）的基本操作

## 个人照片

![person](https://i.loli.net/2020/11/30/AzOkMmwGsvygJ2r.jpg)