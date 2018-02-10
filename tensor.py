import torch
from torch.autograd import Variable
import numpy as np
from data_process import files_2_np, next_batch
import matplotlib.pyplot as plt


epochs = 500
N, D_in, H1, H2, H3, D_out = 1000, 4096, 200, 100, 50, 2
X, Y = files_2_np(5000)

x_np_tr , y_np_tr = next_batch(X, Y, 2000, 0)
x_np_tr-=np.mean(x_np_tr)#zero centred data
x_tr = Variable(torch.from_numpy(x_np_tr), requires_grad=False)
x_tr=x_tr.float()
x_tr = torch.transpose(x_tr,0,1)#fitting dimensions
y_tr = Variable(torch.from_numpy(y_np_tr), requires_grad=False)
y_tr=y_tr.float()

x_np_val , y_np_val = next_batch(X, Y, N, 3000)
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


torch.save(twoLnet, "models/2_outputs(B&W).pkl")


#plt.plot(loss_arr)
#plt.legend()
#plt.show()