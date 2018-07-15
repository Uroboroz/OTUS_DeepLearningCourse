
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# параметры распределений
mu0, sigma0 = -2., 1.
mu1, sigma1 = 3., 2.


# In[3]:


def sample(d0, d1, n=32):
    x0 = d0.sample((n,))
    x1 = d1.sample((n,))
    y0 = torch.zeros((n, 1))
    y1 = torch.ones((n, 1))
    return torch.cat([x0, x1], 0), torch.cat([y0, y1], 0)


# In[24]:


d0 = torch.distributions.MultivariateNormal(torch.zeros(10), torch.eye(10))
d1 = torch.distributions.MultivariateNormal(torch.ones(10), torch.eye(10))


# In[29]:


layer = nn.Linear(10, 1)
print([p.data[0] for p in layer.parameters()])
opt = optim.RMSprop(lr=1e-2, params=list(layer.parameters()))


# In[30]:


log_freq = 500
for i in range(10000):
    if i%log_freq == 0:
        with torch.no_grad():
            x, y = sample(d0, d1, 100000)
            out = F.sigmoid(layer(x))
            loss = F.binary_cross_entropy(out, y)
        print('Ошибка после %d итераций: %f' %(i/log_freq, loss))
    x, y = sample(d0, d1, 1024)
    out = F.sigmoid(layer(x))
    loss = F.binary_cross_entropy(out, y)
    loss.backward()
    opt.step()


# In[ ]:


x_scale = np.linspace(-10, 10, 5000)
d0_pdf = stats.norm.pdf(x_scale, mu0, sigma0) 
d1_pdf = stats.norm.pdf(x_scale, mu1, sigma1)
x_tensor = torch.tensor(x_scale.reshape(-1, 1), dtype=torch.float)
with torch.no_grad():
    dist = F.sigmoid(layer(x_tensor)).numpy()


# In[ ]:


plt.plot(x_scale, d0_pdf*2, label='d0') # умножение на 2 для красоты графиков, на распределения не влияет
plt.plot(x_scale, d1_pdf*2, label='d1')
plt.plot(x_scale, dist.flatten(), label='pred')
plt.legend();


# In[ ]:


opt = optim.SGD(lr=1e-7, params=list(layer.parameters()))

