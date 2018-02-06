import torch
from torch.autograd import Variable
import numpy as np
from data_process import files_2_np, next_batch
import matplotlib.pyplot as plt


epochs = 500
N, D_in, H1, H2, D_out, = 2000, 12288, 500,100, 5
X, Y = files_2_np(10000)
x_data , y_data = next_batch(X, Y, N, 0)
x = Variable(torch.from_numpy(x_data), requires_grad=False)
x=x.float()
x = torch.transpose(x,0,1)#fitting dimensions
y = Variable(torch.from_numpy(y_data), requires_grad=False)
y=y.float()

twoLnet = torch.nn.Sequential(
    torch.nn.Linear(D_in, H1),
#    torch.nn.BatchNorm1d(H1),
    torch.nn.Dropout(),
    torch.nn.ReLU(),
    torch.nn.Dropout(),
#    torch.nn.Linear(H1,H2),
#    torch.nn.ReLU(),
    torch.nn.Linear(H1, D_out),
)

loss_fn = torch.nn.MSELoss(size_average=True)
learning_rate = 1e-3
optimizer = torch.optim.Adam(twoLnet.parameters(), lr=learning_rate)
loss_arr = np.zeros(epochs)
for t in range(epochs):
    y_pred = twoLnet(x)

    loss = loss_fn(y_pred, y)
    loss_arr[t]=loss
    if t % 10 == 0:
        print(t, loss.data[0])

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

torch.save(twoLnet, "models/2_L_net_2_dropout_1_Relu_lr-3.pkl")


plt.plot(loss_arr)
plt.legend()
plt.show()