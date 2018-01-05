
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

from PIL import Image
from scipy.misc import imread
from scipy.misc import imsave
import numpy as np
from scipy.stats import multivariate_normal
import math


# In[2]:


## Read image
data_path = './Dataset/hw3_img.jpg'
img = Image.open(data_path) # Image object
image = np.array(img) # Array

# get shape
im_w = image.shape[0]
im_h = image.shape[1]

resize_ratio = 0.1
img_resize = img.resize((int(im_h*resize_ratio), int(im_w*resize_ratio))) # Image object
image_resize = np.array(img_resize) # Array

# get shape
im_w_re = image_resize.shape[0]
im_h_re = image_resize.shape[1]

## Normalize pixel value to 0 - 1
im = image/255.0

## Avoid sigular matrix of Cov_mat
bias = 0.0001


# print(im_w,im_h)
# print(im_w_re,im_h_re)
# print(image.shape)
# print(image_resize.shape)


# In[3]:


# setting para.
K_cluster = 10
means = np.zeros((K_cluster,3))
for i in range(5):
    for j in range(3):
        means[i,j] = i
k_ary = np.zeros((im_w_re,im_h_re))


# In[4]:


# kmeans
# for k means
# means = np.zeros((1,3,2))
# means[0,:,1]=np.array([1.0, 1.0, 1.0])


def k_means(k_ary, image, means):    
    # init
    init_means = np.zeros_like(means)
    init_cov = np.zeros((K_cluster,3,3))
    init_pi = np.zeros(K_cluster)  
    
    # cal k-array
    for i in range(im_w_re):
        for j in range(im_h_re):
            dist = np.zeros(K_cluster)
            for k in range(K_cluster):
                dist[k] = np.sum((image[i,j,:]-means[k,:])**2)
            k_ary[i,j] = np.argmin(dist)  

    # start update means
    for k in range(K_cluster):
        k_cluster_list = np.where(k_ary == k)
        k_lst_L = len(k_cluster_list[0])
        
        # cal for pi
        init_pi[k] = float(k_lst_L)/float((im_w_re*im_h_re))
        
        # use for create mean 
        if k_lst_L != 0:
            R = image[:,:,0]
            G = image[:,:,1]
            B = image[:,:,2]
            init_means[k,0] = float(R[k_cluster_list].sum())/k_lst_L
            init_means[k,1] = float(G[k_cluster_list].sum())/k_lst_L
            init_means[k,2] = float(B[k_cluster_list].sum())/k_lst_L
            
        # cal Cov.
        N_k = k_lst_L
        val = init_cov[k,:,:]
        
        for i in range(im_w_re):
            for j in range(im_h_re):
                if k_ary[i,j] == k:
                    val += np.dot((image[i,j,:] - init_means[k,:]).reshape((3,1)), (image[i,j,:] - init_means[k,:]).reshape((1,3)))
        init_cov[k,:,:] = val/N_k

    return k_ary, init_means, init_cov, init_pi


IterN = 100
for i in range(IterN):
    k_ary, means, cov, pi = k_means(k_ary, image_resize, means)

print ('======================================')
print ('kmeans - means = \n',means)
print ('kmeans - cov = \n',cov)
print ('kmeans - pi = \n',pi)

# print(means[0,:])
# print(cov[0,:,:])


# In[5]:


# for EM algorithm   
likelihood = []
iter_num = 10

def Cal_mean(image, means, gamma_mat):
    new_means = np.zeros((K_cluster,3))   
    
    for k in range(K_cluster):
        r = 0.0
        g = 0.0
        b = 0.0
        N_k = np.sum(gamma_mat[:,:,k])
        for i in range(im_w_re):
            for j in range(im_h_re):
                r += image[i,j,0]*gamma_mat[i,j,k]
                g += image[i,j,1]*gamma_mat[i,j,k]
                b += image[i,j,2]*gamma_mat[i,j,k]
        new_means[k,:] = np.array([r/N_k, g/N_k, b/N_k])
    return new_means

def Cal_cov(image, cov, means, gamma_mat):
    new_cov = np.zeros_like(cov)
    
    for k in range(K_cluster):
        val = new_cov[k,:,:]
        N_k = np.sum(gamma_mat[:,:,k])
        
        for i in range(im_w_re):
            for j in range(im_h_re):
                tmp = np.dot((image[i,j,:] - means[k,:]).reshape((3,1)), (image[i,j,:] - means[k,:]).reshape((1,3)))
                val += gamma_mat[i,j,k]*tmp
        new_cov[k,:,:] = val/N_k
    return new_cov

def Cal_pi(pis,gamma_mat):
    new_pis = np.zeros_like(pis)
    
    for k in range(K_cluster):
        N_k = np.sum(gamma_mat[:,:,k])
        new_pis[k] = N_k/float(im_w_re*im_h_re)
    return new_pis
    
def Cal_gamma(pixel, k, means, cov, pis):
    denom = 0.0
    gamma = 0
    nom = pis[k]*multivariate_normal.pdf(pixel, means[k,:], cov[k,:,:]+bias)
    for i in range(K_cluster):
        denom += pis[i]*multivariate_normal.pdf(pixel, means[i,:], cov[i,:,:]+bias)
    
    gamma = nom/denom
    return gamma

def EM_(image, means, covars, pis, k_array):
    
    # E step
    gamma_mat = np.zeros((im_w_re, im_h_re, K_cluster))
    for i in range(im_w_re):
        for j in range(im_h_re):     
            for k in range(K_cluster):
                gamma_mat[i,j,k] = Cal_gamma(image[i,j,:], k, means, covars, pis)
    
    # M step
    new_means = Cal_mean(image, means, gamma_mat)
    new_covars = Cal_cov(image, covars, new_means, gamma_mat)
    new_pis = Cal_pi(pis, gamma_mat)
    
    return gamma_mat,new_means, new_covars, new_pis

def log_likelihood(image, means, covars, pis):
    val = 0.0
    for i in range(im_w_re):
        for j in range(im_h_re):
            tmp = 0.0
            for k in range(K_cluster):
                # reshape mean to a vector
                mean_vec = means[k,:].reshape(-1)
                tmp += pis[k]*multivariate_normal.pdf(image[i,j,:], mean_vec, covars[k,:,:]+bias)
            val += np.log(tmp)
    return val

print('Start EM Algorithm')
for i in range(iter_num):
    print('iter = ',i)
    
    gamma_mat, means, cov, pi = EM_(image_resize, means, cov, pi, k_ary)
    
    print ('EM - means = \n',means)
    print ('EM - cov = \n',cov)
    print ('EMs - pi = \n',pi)
    print ('=========================================')
    
    likelihood.append(log_likelihood(image_resize, means, cov, pi))
    


# In[6]:


print(means)


# In[7]:


# reconstruct image
def construct_image(image, means, gamma_matrixs):
    k_num = gamma_matrixs.shape[2]
    new_image = np.zeros_like(image)
    for i in range(im_w_re):
        for j in range(im_h_re):
            compare_ls = []
            for k in range(k_num):
                compare_ls.append(gamma_matrixs[i,j,k])
            min_idx = np.argmax(compare_ls)
            new_image[i,j,0] = means[min_idx,0]
            new_image[i,j,1] = means[min_idx,1]
            new_image[i,j,2] = means[min_idx,2]
    return new_image

new_image = construct_image(image_resize, means, gamma_mat)    

# plot log likelihood & image
plt.figure()
plt.plot(range(len(likelihood)),likelihood)
plt.title('log likelihood')

plt.figure()
plt.imshow(image_resize)
plt.title('resize image')

plt.figure()
plt.imshow(new_image)
plt.title('construct image')

# save fig
plt.savefig('./K_%s_reconstructImg.png'%str(K_cluster))

plt.show()


# In[8]:


# print(new_pis)
#now [0.51673384  0.48326616]


# In[9]:


# print(np.array([[1,2,3],[1,2,3]]).sum())


# In[10]:


# export to python file
get_ipython().system('jupyter nbconvert --to python Problem3_Gaussian_Mixture_Model.ipynb')

