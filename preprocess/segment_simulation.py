#Creates segmented Vpeak dataset given a halo catalog from the Bolshoi-Planck simulation

#Imports
import os
import numpy as np
import itertools
import pickle
from tqdm import tqdm
np.random.seed(230);
#Load data
data = open('../data/experiment1/bolshoiplanck_halos_vpeak_150_snap_100231.bin' , 'rb')
bolshoiplanck_halos_vpeak_150_snap_100231 = pickle.load(data)

#Set hyperparameters
Nsplit = 250
scale_box = 25
size_of_box = 250

#Assign halos to boxes
pos = np.array([bolshoiplanck_halos_vpeak_150_snap_100231[item] for item in ['x','y','z']]).T
count, _ = np.histogramdd(pos, bins=(Nsplit,Nsplit,Nsplit))
H, _ = np.histogramdd(pos, bins=(Nsplit,Nsplit,Nsplit), weights=bolshoiplanck_halos_vpeak_150_snap_100231['vpeak'])
H = H/count
H[np.where(count==0)] = 0
#Segment simulation
n_split_dim = int(float(size_of_box)/scale_box)
outdir = '/home/users/chto/course/umml/data/experiment1.1/datavector_original/'
split_probability=0.95
try:
    traindir = outdir+"train_umML/"
    testdir = outdir+"test_umML/"
    os.mkdir(traindir)
    os.mkdir(testdir)
except:
    None
for i in tqdm(range(n_split_dim)):
    for j in range(n_split_dim):
        for k in range(n_split_dim):
            name = "bolshoiplanck_halos_vpeak_150_snap_100231_{0}_{1}_{2}_scale_box_{3}".format(i, j, k, scale_box)
            vpeak = H[i*scale_box:(i+1)*scale_box, j*scale_box:(j+1)*scale_box, k*scale_box:(k+1)*scale_box]
            offset_pos  = np.array([i*scale_box, j*scale_box, k*scale_box])
            templist = [temp for temp in range(scale_box)]
            pos = offset_pos + list(itertools.product(templist, templist, templist))
            if np.random.uniform(0,1)>split_probability:
                outdir = testdir
            else:
                outdir = traindir
            np.save(outdir + name, np.hstack((pos,vpeak.flatten().reshape(-1,1))))
