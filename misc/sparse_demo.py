import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.util import view_as_windows as viewW
from skimage.util import view_as_blocks as viewB
from build_dct_unitary_dictionary import build_dct_unitary_dictionary
from show_dictionary import show_dictionary
from batch_thresholding import batch_thresholding
from compute_stat import compute_stat
from unitary_dictionary_learning import unitary_dictionary_learning

#CCH 20200520 widthen display output
import pandas as pd
desired_width = 260
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 10)
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': lambda x: " {0:7.3f}".format(x)})

#%% Part A: Data Construction and Parameter-Setting

# Read an image
im = np.array(Image.open('barbara.png'))
#im = np.array(Image.open('brain4.jpg'))
#im = np.array(Image.open('brain4_r01.jpg'))

# Show the image
plt.figure(0)
plt.imshow(im,'gray') 
plt.title('Original image')
#plt.show()

# Patch dimensions [height, width]
dim = 6
patch_size = [dim, dim]

#Create the patches
all_patches = viewW(im, patch_size)
# Number of patches to train on
num_train_patches = 10000
# Number of patches to test on
num_test_patches = 5000

# Set the seed for the random generator
seed = 20827
 
# Set a fixed random seed to reproduce the results
np.random.seed(seed)

# TODO: Create a training set by choosing a random subset 
# of 'num_train_patches' taken from 'all_patches'
# Write your code here... train_patches = ???

train_patches_idx = np.random.permutation((512-dim)*(512-dim))

train_patches = np.zeros((dim*dim, num_train_patches))

for i in range(num_train_patches):
    tp_x = np.floor_divide(train_patches_idx[i], 512 - dim)
    tp_y = np.mod(train_patches_idx[i], 512 - dim)
    train_patches[:,i] = all_patches[tp_x, tp_y].flatten()
    if i==12:
        plt.figure(1)
        plt.title('0th training patch')
        plt.imshow(all_patches[tp_x, tp_y], 'gray')
        print('train patch0:', train_patches[:,i] )

# TODO: Create a test set by choosing another random subset
# of 'num_test_patches' taken from the remaining patches
# Write your code here... test_patches = ???

test_patches = np.zeros((dim*dim, num_test_patches))
test_patches_idx = np.random.permutation((512-dim)*(512-dim))

for i in range(num_test_patches):
    tp_x = np.floor_divide(test_patches_idx[i], 512 - dim)
    tp_y = np.mod(test_patches_idx[i], 512 - dim)
    test_patches[:,i] = all_patches[tp_x, tp_y].flatten()


# TODO: Initialize the dictionary
# Write your code here... D_DCT = build_dct_unitary_dictionary( ? )

D_DCT = build_dct_unitary_dictionary(patch_size)
#print('D_DCT', D_DCT[:,1])

# Show the unitary DCT dictionary
plt.figure(3)
# plt.subplot(3, 2, 1)
show_dictionary(D_DCT)
plt.title('Unitary DCT Dictionary')

#CCH 20210129 sneak preview
#plt.show()

# TODO: Set K - the cardinality of the solution.
# This will serve us later as the stopping criterion of the pursuit
# Write your code here... K = ???

K=4
 
#%% Part B: Compute the Representation Error Obtained by the DCT Dictionary
 
# Compute the representation of each patch that belongs to the training set using Thresholding
est_train_patches_dct, est_train_coeffs_dct = batch_thresholding(D_DCT, train_patches, K)
# print('coef:', est_train_coeffs_dct[:,0])

# Compute the representation of each patch that belongs to the test set using Thresholding
est_test_patches_dct, est_test_coeffs_dct = batch_thresholding(D_DCT, test_patches, K)


# Compute and display the statistics
print('\n\nDCT dictionary: Training set, ')
compute_stat(est_train_patches_dct, train_patches, est_train_coeffs_dct)

print('DCT dictionary: Testing  set, ')
compute_stat(est_test_patches_dct, test_patches, est_test_coeffs_dct)

print('\n\n')


#----------------------------------------------------------------------------------------------------------------------


#%% Part C: Procrustes Dictionary Learning

# TODO: Set the number of training iterations
# Write your code here... T = ???

T=20

# Train a dictionary via Procrustes analysis
D_learned, mean_error, mean_cardinality = unitary_dictionary_learning(train_patches, D_DCT, T, K)

# Show the dictionary
plt.figure(1)
# plt.subplot(1, 2, 2)
show_dictionary(D_learned)
plt.title("Learned Unitary Dictionary")
#plt.show()

# Show the representation error and the cardinality as a function of the learning iterations
plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(np.arange(T), mean_error, linewidth=2.0)
plt.ylabel("Average Representation Error")
plt.xlabel("Learning Iteration")
plt.subplot(1, 2, 2)
plt.plot(np.arange(T), mean_cardinality, linewidth=2.0)
plt.ylabel('Average Number of Non-Zeros')
plt.ylim((K-1, K+1))
plt.xlabel('Learning Iteration')


# Compute the representation of each signal that belong to the training set using Thresholding
est_train_patches_learning, est_train_coeffs_learning = batch_thresholding(D_learned, train_patches, K)

# Compute the representation of each signal that belong to the testing set using Thresholding
est_test_patches_learning, est_test_coeffs_learning = batch_thresholding(D_learned, test_patches, K)

# Compute and display the statistics
print('\n\nLearned dictionary: Training set, ')
compute_stat(est_train_patches_learning, train_patches, est_train_coeffs_learning)
print('Learned dictionary: Testing  set, ')
compute_stat(est_test_patches_learning, test_patches, est_test_coeffs_learning)
print('\n\n')

plt.show()