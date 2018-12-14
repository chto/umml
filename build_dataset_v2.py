import argparse
import random
import os

from tqdm import tqdm
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")


def movefile(filename, output_dir):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    shutil.move(filename, os.path.join(output_dir, filename.split('/')[-1]))
    labelname = filename[:-4]+"_labels.npy"
    shutil.move(labelname, os.path.join(output_dir, labelname.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_umML')
    test_data_dir = os.path.join(args.data_dir, 'test_umML')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('25.npy')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('25.npy')]

    # Split the images in 'train_signs' into 80% train and 20% val
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.9 * len(filenames))
    train_filenames = filenames[:split]
    val_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_umML'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            movefile(filename, output_dir_split)

    print("Done building dataset")
