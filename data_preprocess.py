from cocoapi.PythonAPI.pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import os
import time

DATA_TYPE = "val2017"
DATA_DIR=""
annFile='annotations/instances_{}.json'.format(DATA_TYPE)
PROC_DATA_DIR = "proc"
SIZE = 256

def main():
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds )
    print("Person images : ", len(imgIds))
    images = coco.loadImgs(imgIds)
    count = 0
    iter_count = 0
    start_time = time.time()
    os.makedirs('%s/%s/images/'%(PROC_DATA_DIR,DATA_TYPE), exist_ok=True)
    os.makedirs('%s/%s/masks/'%(PROC_DATA_DIR,DATA_TYPE), exist_ok=True)
    for img in images:
        iter_count +=1
        if iter_count % 100 == 0:
            print("Time since started : " , time.time() - start_time)
        if img['height']  < SIZE or img['width'] < SIZE:
            continue
        count +=1
        I = io.imread('%s%s/%s'%(DATA_DIR,DATA_TYPE,img['file_name']))
        cropped = I[:SIZE][:SIZE]
        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(ids=annotation_ids)
        masks = np.array([coco.annToMask(ann) for ann in annotations])
        cropped_masks = [mask[:SIZE][:SIZE] for mask in masks]
        io.imsave('%s/%s/images/%s'%(PROC_DATA_DIR,DATA_TYPE,img['file_name']), cropped)
        np.save('%s/%s/masks/%s'%(PROC_DATA_DIR,DATA_TYPE,img['file_name']), cropped_masks)
    print("Processed " + str(count) + " images")


if __name__ == '__main__':
    main()