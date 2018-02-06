import numpy as np
import matplotlib.pyplot as plt
import torch

x=np.random.rand(20)
y=np.random.rand(20)

#plt.plot(x_tr.data.numpy(), y_tr.data.numpy(), 'ro', label='Original data')
plt.plot(x,y, label='Fitted line')
plt.legend()
plt.show()