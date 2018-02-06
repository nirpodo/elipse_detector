import cv2
import numpy as np

img_dir = "elipse_images"

def files_2_np(total_number_of_inputs):
    outputs = np.genfromtxt('outputs.csv', delimiter='\t')
    #outputs = outputs[:,0:2]
    inputs=np.zeros((12288,total_number_of_inputs))
    for i in range(total_number_of_inputs):
        pic = cv2.imread(img_dir+'/img'+str(i)+'.jpeg')
        inputs[:,i] = np.reshape(pic, np.product(pic.shape))
    inputs-=np.mean(inputs, axis=0)#for zero centred data
    return inputs, outputs

def next_batch(X, y, batch_size, end_point_last_batch):
#    X -= np.mean(X, axis=0)  # for zero centred data
    return X[:,end_point_last_batch:batch_size+end_point_last_batch], y[end_point_last_batch:batch_size+end_point_last_batch,:]


