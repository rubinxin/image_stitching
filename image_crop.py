import cv2
import os
import matplotlib.pyplot as plt

# crop 0_0 into 6 subpatches
img_folder = './aoi_images/check/pcb/0-0'
image_name = 'board_white1'
refFilename = os.path.join(img_folder, f"{image_name}.jpg")
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

# 是否将图切成多块
multiple_boards = False

# 切成几行
nrows = 2
# 切成几列
ncols = 3

# 将一块图切成好几块
if multiple_boards:
    # 单块子图的宽度
    width_sub = 8100
    # 单块子图的高度
    height_sub = 6000

    for image_name_color in [f"{image_name}_white.jpg", f"{image_name}_rgb.jpg"]:
        refFilename = os.path.join(img_folder, f"{image_name_color}.jpg")
        img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        for i in range(nrows):
            for j in range(ncols):
                x_min = max([height_sub*i-100,0])
                x_max = height_sub*(i+1)+200
                y_min = max([width_sub*j-100,0])
                y_max = width_sub*(j+1)+200

                img_sub = img[int(x_min): int(x_max),int(y_min): int(y_max)]
                output_name = os.path.join(img_folder, f"{image_name_color}_crop_{i}-{j}.jpg")
                cv2.imwrite(output_name, img_sub)

else:
    # 将一块图扣中心区域
    # 子图中心点的横坐标
    x_sub = 1000
    # 子图中心点的纵坐标
    y_sub = 500
    # 子图的宽度
    width_sub = 8100
    # 子图的高度
    height_sub = 6000

    for image_name_color in [f"{image_name}_white.jpg", f"{image_name}_rgb.jpg"]:
        refFilename = os.path.join(img_folder, f"{image_name_color}.jpg")
        img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

        x_min = int(x_sub-width_sub/2)
        x_max = int(x_sub+width_sub/2)
        y_min = int(y_sub+height_sub/2)
        y_max = int(y_sub+height_sub/2)

        img_sub = img[int(x_min): int(x_max), int(y_min): int(y_max)]
        output_name = os.path.join(img_folder, f"{image_name_color}_crop.jpg")
        cv2.imwrite(output_name, img_sub)