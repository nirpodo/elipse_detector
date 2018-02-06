import cv2
import numpy as np
import csv
import os
img_dir="elipse_images"
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
img_size = 64
total_images = 10000
dict={}

with open ("outputs.csv", 'wb') as f:
    for i in range (total_images):
        img = np.ones((img_size, img_size, 3), np.uint8) * 255
        center_x = np.random.randint(2,img_size-2)
        center_y = np.random.randint(2,img_size-2)
        rotation_angle = np.random.randint(0, 360)
        major_axis = np.random.randint(2,min(center_x, center_y, img_size-center_x, img_size-center_y)+3)  # for getting the entire elipse in the image
        minor_axis = np.random.randint(2,major_axis+3)
        R = np.random.randint(255)
        B = np.random.randint(255)
        G = np.random.randint(255)
        img = cv2.ellipse(img,(center_x,center_y),(major_axis,minor_axis),rotation_angle,0,360,(R,G,B),-1)
        cv2.imwrite(img_dir+'/img'+str(i)+'.jpeg', img)
        file=csv.writer(f,delimiter='\t')
        file.writerow([center_x, center_y, rotation_angle, major_axis, minor_axis])

