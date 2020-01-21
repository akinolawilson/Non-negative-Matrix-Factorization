import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm as c

x = np.load('reducedX.npy')

def colNormalisation(X):
    for i in range(np.shape(X)[1]):
        X[:,i] = X[:,i]/sum(X[:,i])
        
    return X
xnorm = colNormalisation(x)
#%%
fig = plt.figure(figsize=(10,10))
ax = fig.gca()

im = np.reshape(xnorm[:,50], (42,38)) # dimensions of reduced images: 42 by 38

ax.imshow(im,cmap=c.gray)
#%%
import preprocessing as p 

B, Xp, Xpnorm = p.preprocess(im)

#%%
fig = plt.figure(figsize=(10,10))
ax1 = fig.addsubplots()
 # dimensions of reduced images: 42 by 38
ax.imshow(Xpnorm,cmap=c.gray)
#%%
def toGreyScale(image):
    grey = 0.2989 *image[:,:,0] + 0.5870 *image[:,:,1] + 0.1140 *image[:,:,2]
    return(grey)

a = np.where()
#%%

plt.rcParams['font.size'] = 11

im1 = plt.imread('preReduction.png')
im2 = plt.imread('postReduction.png')
im3 = plt.imread('postPreporcessing.png')

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(1,3, 1)
ax2 = fig.add_subplot(1,3, 2)
ax3 = fig.add_subplot(1,3, 3)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')

ax1.imshow(im1)
ax2.imshow(im2)
ax3.imshow(im3)
ax1.title.set_text('Image size 180 by 200 pixels')
ax2.title.set_text('Image size 38 by 42 pixels')
ax3.title.set_text('Post preprocessing: Image size 38 by 42 pixels')