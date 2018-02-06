import torch
from data_process import files_2_np, next_batch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

N, D_in, H, D_out, =1000, 12288, 50, 5
X, Y = files_2_np(10000)
x_data , y_data = next_batch(X, Y, N, 2000)
x_val = Variable(torch.from_numpy(x_data), requires_grad=False)
x_val=x_val.float()
x_val = torch.transpose(x_val,0,1)#fitting dimensions
y_val = Variable(torch.from_numpy(y_data), requires_grad=False)
y_val = y_val.float()

x_data , y_data = next_batch(X, Y, N, 0)
x_tr = Variable(torch.from_numpy(x_data))
x_tr = x_tr.float()
x_tr = torch.transpose(x_tr,0,1)#fitting dimensions
y_tr = Variable(torch.from_numpy(y_data))
y_tr=y_tr.float()

model = torch.load("models/2_L_net_2_dropout_1_Relu_lr-3.pkl")
#model.eval()

loss_fn = torch.nn.MSELoss(size_average=True)

y_pred_val = model(x_val)
loss_val = loss_fn(y_pred_val, y_val)
print loss_val.data[0]

#print y_pred_val
#print y_val
y_pred_tr = model(x_tr)
loss_tr = loss_fn(y_pred_tr , y_tr)
print loss_tr.data[0]


#predicted = model(Variable(torch.from_numpy(x_tr))).data.numpy()
#plt.plot(x_tr.data.numpy(), y_tr.data.numpy(), 'ro', label='Original data')
#plt.plot(x_tr.data.numpy(), y_pred_tr.data.numpy(), label='Fitted line')
#plt.legend()
#plt.show()





