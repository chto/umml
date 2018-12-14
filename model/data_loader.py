import random
import os
import numpy as np
#from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
# loader for evaluation, no horizontal flip

class UMDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.transform = transform
        self.filenames_original = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames_original if f.endswith('npy')]
        labelfile = [os.path.join(data_dir,f) for f in self.filenames_original if f.endswith('smf')]

        print(labelfile)
        print(data_dir)
        assert len(labelfile)==1
        labelvector = np.loadtxt(labelfile[0])[:,2:] 
        self.labels=self.transform(np.array([labelvector[:,0],np.array([np.max(temp) for temp in zip(labelvector[:,1], labelvector[:,2])])]))
    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx data and label in the dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        data  = np.load(self.filenames[idx])  
        n_each_side = int(round(len(data)**(1/3.)))
        data = data[:,3].reshape(n_each_side, n_each_side, n_each_side)
        data = self.transform(data)
        return data.float(), self.labels.float()

class UMDataset_UMlabel(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """

        self.transform = transform
        self.filenames_original = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames_original if f.endswith('25.npy')]
        self.labelfile = [f[:-4]+"_labels.npy" for f in self.filenames]

        print(self.labelfile)
        print(data_dir)
        assert len(self.labelfile)==len(self.filenames)
    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx data and label in the dataset.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        data  = np.load(self.filenames[idx])  
        n_each_side = int(round(len(data)**(1/3.)))
        data = data[:,3].reshape(n_each_side, n_each_side, n_each_side)
        data = self.transform(data)
        
        label = np.load(self.labelfile[idx])
        return data.float(),self.transform(np.swapaxes(label,0,1)).float() 



def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_umML".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(UMDataset_UMlabel(path, torch.from_numpy), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda, drop_last=True)
            else:
                dl = DataLoader(UMDataset_UMlabel(path, torch.from_numpy), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda, drop_last=True)

            dataloaders[split] = dl

    return dataloaders
