import cv2
import os
import matplotlib.pyplot as plt

# crop 0_0 into 6 subpatches
img_folder = './aoi_images/check/pcb/0-0'
image_name = 'board_white1'
refFilename = os.path.join(img_folder, f"{image_name}.jpg")
img = cv2.imread(refFilename, cv2.IMREAD_COLOR)

height_sub, width_sub = 5926, 8180
height_sub, width_sub = 6000, 8100

for i in range(2):
    for j in range(3):
        x_min = max([height_sub*i-100,0])
        x_max = height_sub*(i+1)+200
        y_min = max([width_sub*j-100,0])
        y_max = width_sub*(j+1)+200

        img_sub = img[int(x_min): int(x_max),int(y_min): int(y_max)]
        output_name = os.path.join(img_folder, f"{image_name}_{i}-{j}.jpg")
        cv2.imwrite(output_name, img_sub)
