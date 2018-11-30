from data_preprocess import PROC_DATA_DIR, DATA_TYPE
images_dir = '%s/%s/images/'%(PROC_DATA_DIR, DATA_TYPE)
masks_dirs = '%s/%s/masks/'%(PROC_DATA_DIR, DATA_TYPE)

import os
import random
import numpy as np
import skimage.io as io
import cv2

def getRandomFile(path):
  """
  Returns a random filename, chosen among the files of the given path.
  """
  files = os.listdir(path)
  index = random.randrange(0, len(files))
  return files[index]

def generator_fn(num_processes=4, batch_size=128):
  while True:
    files = [getRandomFile(images_dir) for i in range(batch_size)]
    mask_files = [masks_dirs + f + ".npy" for f in files]
    img_mask_pairs = [(io.imread(fname=images_dir + img_file), np.load(mask_file)) for img_file, mask_file in zip(files, mask_files)]
    for img, mask in img_mask_pairs:
      if img.shape == (256, 256, 3):
        yield cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC), cv2.resize(mask, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    