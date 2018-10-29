from data_preprocess import PROC_DATA_DIR, DATA_TYPE
images_dir = '%s/%s/images/'%(PROC_DATA_DIR, DATA_TYPE)
masks_dirs = '%s/%s/masks/'%(PROC_DATA_DIR, DATA_TYPE)



import os
import random
import numpy as np
import skimage.io as io

def getRandomFile(path):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  files = os.listdir(path)
  index = random.randrange(0, len(files))
  return files[index]

def gen_fn():
    while True:
        # Want an infinite generator
        img_file = getRandomFile(images_dir)
        mask_file = masks_dirs + img_file + ".npy"
        img = io.imread(fname=images_dir + img_file)
        mask = np.load(mask_file)
        # img => (256,256,3); mask => (256,256,)
        yield img, mask

    

def test_generator_fn():
    pass