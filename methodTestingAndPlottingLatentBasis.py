#
#
# Hyperparameter selection for local NMF
#
#
#
import numpy as np
x = np.load('reducedX.npy')
#%%
import matplotlib.pyplot as plt
import matplotlib.cm as c 


X = np.load('reducedX.npy')
#%%
plt.imshow(im)

im = np.reshape(X[:,50], (42,38)) # dimensions of reduced images: 42 by 38

plt.imshow(im,cmap=c.gray)
#%%

Xsmall = X[:,:25]


#%%
recon = w @ h
#%%
im = np.reshape(recon[:,1], (42,38)) # dimensions of reduced images: 42 by 38

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as c 
#%%
plt.imshow(im,cmap=c.gray)
#%%
imL0 = w[:,0]
imL1 = w[:,1]
imL2 = w[:,2]
imL3 = w[:,3]
imL4 = w[:,4] 

fig = plt.figure(figsize=(20,5))

ax0 = fig.add_subplot(1,5,1)
im0 = np.reshape(imL0, (42,38))
ax0.imshow(im0,cmap=c.gray)

ax1 = fig.add_subplot(1,5,2)
im1 = np.reshape(imL1, (42,38))
ax1.imshow(im1,cmap=c.gray)

ax2 = fig.add_subplot(1,5,3)
im2 = np.reshape(imL2, (42,38))
ax2.imshow(im2,cmap=c.gray)

ax3 = fig.add_subplot(1,5,4)
im3 = np.reshape(imL3, (42,38))
ax3.imshow(im3,cmap=c.gray)

ax4 = fig.add_subplot(1,5,5)
im4 = np.reshape(imL4, (42,38))
ax4.imshow(im4,cmap=c.gray)
#%%
imL0 = h[0,:]
imL1 = h[1,:]
imL2 = h[2,:]
imL3 = h[3,:]
#imL4 = w[:,4] 

fig = plt.figure(figsize=(20,5))

ax0 = fig.add_subplot(1,5,1)
im0 = np.reshape(imL0, (42,38))
ax0.imshow(im0,cmap=c.gray)

ax1 = fig.add_subplot(1,5,2)
im1 = np.reshape(imL1, (42,38))
ax1.imshow(im1,cmap=c.gray)

ax2 = fig.add_subplot(1,5,3)
im2 = np.reshape(imL2, (42,38))
ax2.imshow(im2,cmap=c.gray)

ax3 = fig.add_subplot(1,5,4)
im3 = np.reshape(imL3, (42,38))
ax3.imshow(im3,cmap=c.gray)