#Creates segmented Vpeak dataset given a halo catalog from the Bolshoi-Planck simulation

#Imports
import os
import numpy as np
import itertools
import pickle
from tqdm import tqdm
import pandas as pd
np.random.seed(230);
#Load data
data = 'merged_vpeak_150_RF.cache'
UM_halos = pd.read_csv(data)
print(UM_halos[['Mpredict','obs_SM']])
#Set hyperparameters
Nsplit = 250
scale_box = 25
size_of_box = 250
h70 = 0.7

#Assign halos to boxes
pos = np.array(UM_halos[['X','Y','Z']])
count, _ = np.histogramdd(pos, bins=(Nsplit,Nsplit,Nsplit))
H, _ = np.histogramdd(pos, bins=(Nsplit,Nsplit,Nsplit), weights=UM_halos['Vpeak'])
HSM, _ = np.histogramdd(pos, bins=(Nsplit,Nsplit,Nsplit), weights=np.log10(UM_halos['Mpredict']/0.678**2))
H = H/count
H[np.where(count==0)] = 0
HSM = HSM/count
HSM[np.where(count==0)] = 0

#Segment simulation
n_split_dim = int(float(size_of_box)/scale_box)
outdir = '/home/users/chto/course/umml/data/experiment1.5_RF/datavector_original/'
split_probability=0.95
try:
    traindir = outdir+"train_umML/"
    testdir = outdir+"test_umML/"
    os.mkdir(traindir)
    os.mkdir(testdir)
except:
    None
def binstellarmass(sm):
   h = 0.678
   bins = np.linspace(9,12.4,35)
   result, _ = np.histogram(sm[np.isfinite(sm)], bins)
   return (result/float(scale_box)**3)[12:26]
smf = np.loadtxt("moustakas_z0.01_z0.20.smf")
for i in tqdm(range(n_split_dim)):
    for j in range(n_split_dim):
        for k in range(n_split_dim):
            name = "bolshoiplanck_halos_vpeak_150_snap_100231_{0}_{1}_{2}_scale_box_{3}".format(i, j, k, scale_box)
            labels = "bolshoiplanck_halos_vpeak_150_snap_100231_{0}_{1}_{2}_scale_box_{3}_labels".format(i, j, k, scale_box)
            vpeak = H[i*scale_box:(i+1)*scale_box, j*scale_box:(j+1)*scale_box, k*scale_box:(k+1)*scale_box]
            sm = HSM[i*scale_box:(i+1)*scale_box, j*scale_box:(j+1)*scale_box, k*scale_box:(k+1)*scale_box]
            offset_pos  = np.array([i*scale_box, j*scale_box, k*scale_box])
            templist = [temp for temp in range(scale_box)]
            pos = offset_pos + list(itertools.product(templist, templist, templist))
            if np.random.uniform(0,1)>split_probability:
                outdir = testdir
            else:
                outdir = traindir
            np.save(outdir + name, np.hstack((pos,vpeak.flatten().reshape(-1,1))))
            #np.save(outdir + labels, np.hstack((pos,sm.flatten().reshape(-1,1))))
            print(np.max(smf[:,3:5], axis=1).shape, )
            np.save(outdir + labels, np.hstack((binstellarmass(sm.flatten()).reshape(-1,1), np.max(smf[:,3:5], axis=1).reshape(-1,1)[12:26])))
