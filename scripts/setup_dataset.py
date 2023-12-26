# Script for taking the images and put them in a single folder.
from glob import glob
from pprint import pprint
import shutil

if __name__ == '__main__':
    #base_addr = "/home/amaldonado/Code/image-retrival-system/syntetic_dataset_3/*/*.jpg"
    base_addr = "/home/amaldonado/Code/cv-order-validation/dataset/ds1/test/*.jpg"
    
    files = glob(base_addr)
    
    pprint(len(files))

    output_addr = "/home/amaldonado/Datasets/MB/DS1"
    lag = 319
    for i, f in enumerate(files):
        shutil.copyfile(f, output_addr+"/img_{}.jpg".format(i+lag))
