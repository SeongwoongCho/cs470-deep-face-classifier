#!/usr/bin/env python
# coding: utf-8

# ### Install library

# In[1]:


get_ipython().system('pip install opencv-python')
get_ipython().system('pip install facenet-pytorch')


# In[2]:


import cv2
import sys
import os
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


# ### Rename images

# In[3]:


file_path = '/home/jovyan/cs470-deep-face-classifier/full_data/raw/'


# In[4]:


for i in ['dog','cat', 'bear', 'dinosaur', 'bald', 'rabbit']:
    file_path_i = file_path + i
    filenames_i = os.listdir(file_path_i)
    print(filenames_i)


# ### Face detect

# In[14]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

for i in ['dog','cat', 'bear', 'dinosaur', 'rabbit', 'bald']:
    file_path_i = file_path + i
    filenames_i = os.listdir(file_path_i)
    try:
        os.mkdir('/home/jovyan/cs470-deep-face-classifier/full_data/updated/'+i)
    except:
        pass
    for j in filenames_i:
        file_path_j = file_path_i + '/'+j
        files = os.listdir(file_path_j)
        try:
            os.mkdir('/home/jovyan/cs470-deep-face-classifier/full_data/updated/'+i+'/'+j)
        except:
            continue
        m=1
        for k in files:
            src = file_path_j + '/' + k
            try:
                img = Image.open(src)
                boxes, _ = mtcnn.detect(img)
                if len(boxes) != 1:
                    continue
            except:
                continue
            
            x, y, width, height = boxes[0]
            im = np.array(img)
            convert  = im[int(y):int(height),int(x):int(width)]
            dst = '/home/jovyan/cs470-deep-face-classifier/full_data/updated/'+i+'/'+j+'/'+str(m)+'.jpg'
            try:
                gr_im= Image.fromarray(convert).save(dst)
            except:
                continue
            m += 1


# In[ ]:





# In[19]:


path = '/home/jovyan/cs470-deep-face-classifier/datas/raw/bear/김구라/17.jpg'
img = Image.open(path)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)

boxes, _ = mtcnn.detect(img)


plt.imshow(img)

ax = plt.gca()

x, y, width, height = boxes[0]

rect = plt.Rectangle((x, y), width-x, height-y, fill=False, color='green')

ax.add_patch(rect)

plt.show()

im = np.array(img)

convert  = im[int(y):int(height),int(x):int(width)]
gr_im= Image.fromarray(convert).save('/home/jovyan/cs470-deep-face-classifier/new_data/updated/dog/1.jpg')


# In[ ]:




