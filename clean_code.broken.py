import cv2
import numpy as np
import csv
import os
import torch
from torch.autograd import Variable

img_dir = "elipse_images_test"
img_size = 64
#batch_size = 5000

if not os.path.exists(img_dir):
    os.makedirs(img_dir)


def create_data(batch_size):
    batch_start = len(next(os.walk(img_dir))[2])

    with open ("outputs_test.csv", 'a') as f:
        for i in range (batch_start, batch_size+batch_start):
            img = np.ones((img_size, img_size, 3), np.uint8) * 255
            center_x = np.random.randint(5,img_size-6)
            center_y = np.random.randint(5,img_size-6)
            rotation_angle = np.random.randint(0, 360)
            major_axis = np.random.randint(5,min(center_x, center_y, img_size-center_x, img_size-center_y)+3)  # for getting the entire elipse in the image
            minor_axis = np.random.randint(4,major_axis)
            R = np.random.randint(240)
            B = np.random.randint(240)
            G = np.random.randint(240)
            img = cv2.ellipse(img,(center_x,center_y),(major_axis,minor_axis),rotation_angle,0,360,(R,G,B),-1)
            cv2.imwrite(img_dir+'/img'+str(i)+'.jpeg', img)
            file=csv.writer(f,delimiter='\t')
            file.writerow([center_x, center_y, rotation_angle, major_axis, minor_axis])


def next_batch(batch_size):
    batch_start = len(next(os.walk(img_dir))[2])
    create_data(batch_size)
    outputs = np.genfromtxt('outputs_test.csv', delimiter='\t', skip_header=batch_start)
    outputs = outputs[:, 0:2]

    inputs=np.zeros((4096,batch_size))
    for i in range(batch_start,batch_size+batch_start):
        pic = cv2.imread(img_dir+'/img'+str(i)+'.jpeg', 0)
        _, pic = cv2.threshold(pic, 250, 128, cv2.THRESH_BINARY_INV)#thersholding the image - no need for color or even grey scale
        #pic=255-pic
        inputs[:,i] = np.reshape(pic, np.product(pic.shape))
#    inputs -= np.mean(x_train)  # zero centred data
#    inputs = Variable(torch.from_numpy(inputs), requires_grad=False)
#    inputs = inputs.float()
#    inputs = torch.transpose(inputs, 0, 1)  # fitting dimensions
#    outputs = Variable(torch.from_numpy(outputs), requires_grad=False)
#    outputs = outputs.float()
    return inputs, outputs


epochs = 500
N, D_in, H1, H2, H3, D_out = 1000, 4096, 200, 100, 50, 2

x_np_tr , y_np_tr = next_batch(N)
x_np_tr-=np.mean(x_np_tr)#zero centred data
x_tr = Variable(torch.from_numpy(x_np_tr), requires_grad=False)
x_tr=x_tr.float()
x_tr = torch.transpose(x_tr,0,1)#fitting dimensions
y_tr = Variable(torch.from_numpy(y_np_tr), requires_grad=False)
y_tr=y_tr.float()

x_np_val , y_np_val = next_batch(N)
x_np_val-=np.mean(x_np_tr)#zero centred data
x_val = Variable(torch.from_numpy(x_np_val), requires_grad=False)
x_val=x_val.float()
x_val = torch.transpose(x_val,0,1)#fitting dimensions
y_val = Variable(torch.from_numpy(y_np_val), requires_grad=False)
y_val = y_val.float()

twoLnet = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
    torch.nn.BatchNorm1d(H1),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
    torch.nn.Linear(H1,H2),
    torch.nn.BatchNorm1d(H2),
    torch.nn.ReLU(),
    torch.nn.Linear(H2, H3),
    torch.nn.ReLU(),
    torch.nn.Linear(H3, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=True)
#loss_arr = np.zeros(epochs)


learning_rate = 5e-2
optimizer = torch.optim.Adam(twoLnet.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
for t in range(epochs):
    y_tr_pred = twoLnet(x_tr)
    loss_tr = loss_fn(y_tr_pred, y_tr)
    #loss_arr[t]=loss_tr

    loss_val = loss_fn(twoLnet(x_val), y_val)
    if t % 10 == 0:
        print(t, loss_tr.data[0], loss_val.data[0])


    optimizer.zero_grad()
    loss_tr.backward()
    optimizer.step()
    scheduler.step()


torch.save(twoLnet, "models/2_outputs(B&W)_test.pkl")

