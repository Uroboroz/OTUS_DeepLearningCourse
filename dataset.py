
# coding: utf-8

# In[1]:


import torch
from torch import utils
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def plot_mnist(images, shape):
    fig = plt.figure(figsize=shape[::-1], dpi=80)
    for j in range(1, len(images) + 1):
        ax = fig.add_subplot(shape[0], shape[1], j)
        ax.matshow(images[j - 1][0], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()


# In[3]:


path='./MNIST_data'


# In[44]:


mnist_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1555,), (0.3292,)),
           ])


# In[45]:


train_data = datasets.MNIST(path, train=True, download=True, transform=mnist_transform)
test_data = datasets.MNIST(path, train=False, download=True, transform=mnist_transform)
print("mean =", torch.mean(train_data[1][0]), "\nstd =", torch.std(train_data[1][0]), train_data[0][0])


# In[41]:


train_data = datasets.MNIST(path, train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(path, train=False, download=True, transform=transforms.ToTensor())
print("mean =", torch.mean(train_data[1][0]), "\nstd =", torch.std(train_data[1][0]), train_data[0][0])


# In[37]:


train_data[0][0]


# In[12]:


images = [train_data[i][0] for i in range(50)]


# In[13]:


plot_mnist(images, (5, 10))


# In[ ]:


images[0][0].shape


# In[ ]:


train_loader = utils.data.DataLoader(train_data, batch_size=50, shuffle=True)


# In[ ]:


batch_x, batch_y = next(iter(train_loader))


# In[ ]:


batch_x.shape


# In[ ]:


batch_y

