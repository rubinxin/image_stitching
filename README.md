# 图片拼接算法Image Stitching Python

实现图片拼接的python代码

## 依赖项

安装以下依赖项目:
 - Python >= 3.6.0
 - opencv-python=4.6.0
 - matplotlib

## 拼接图片
在terminal输入以下代码进行拼接，`-path=`设为需拼接的图片所在文件夹，结果会保存成`board_rgb.jpg` 和`board_white.jpg`在图片文件夹里。
```
python image_stitching.py -path='/Users/binxinru/Documents/StartUp/Projects/Datasets/smtv1' -nr=4 -nc=4
```
