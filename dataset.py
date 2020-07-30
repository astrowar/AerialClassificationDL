
#https://onedrive.live.com/?authkey=%21AAqO0B6SeejPkr0&cid=42EC9A19F3DE58D8&id=42EC9A19F3DE58D8%2176404&parId=42EC9A19F3DE58D8%2132701&action=locate

import tensorflow as tf
import numpy as np
import PIL
import os
import multiprocessing
from functools import *
import itertools
import os
from glob import glob

def get_all_files_forest():
    files = glob("AID/forest/" + '/*.jpg', recursive=True)
    #print(files)
    if len(files) < 1 :
        raise  FileNotFoundError()
    return files

def get_all_files_SparseResidential():
    files = glob("AID/SparseResidential/" + '/*.jpg', recursive=True)
    #print(files)
    if len(files) < 1 :
        raise  FileNotFoundError()
    return files

def get_all_files_BareLand():
    files = glob("AID/BareLand/" + '/*.jpg', recursive=True)
    #print(files)
    if len(files) < 1 :
        raise  FileNotFoundError()
    return files
 


def getdata(filename,dsize):     
    im = PIL.Image.open(filename).convert("RGB")
    imm = im.resize([dsize,dsize])
    data =  np.asarray(imm)*(1.0/255.0)
    yield   data
    yield np.asarray(imm.rotate(90))*(1.0/255.0)
    yield np.asarray(imm.rotate(180))*(1.0/255.0)


files_forest = get_all_files_forest()
files_SparseResidential = get_all_files_SparseResidential()
files_BareLand = get_all_files_BareLand()

#local_images_names  =[ "thumb/thumbnails128x128/" + th  for th in os.listdir("thumb/thumbnails128x128/")[ 0:1024*60]  ]
def genDatasetForest(dsize):
    load_image = partial(getdata, dsize=dsize)
    for h in files_forest:        
        for limage in load_image(h):
            yield    2.0*np.array(limage) - 1.0


def genDatasetSparseResidential(dsize):
    load_image = partial(getdata, dsize=dsize)
    for h in files_SparseResidential:        
        for limage in load_image(h):
            yield    2.0*np.array(limage) - 1.0

def genDatasetBareLand (dsize):
    load_image = partial(getdata, dsize=dsize)
    for h in files_BareLand:        
        for limage in load_image(h):
            yield    2.0*np.array(limage) - 1.0




def getImageDataSetForest(dsize):
    return list( genDatasetForest(dsize))
    #load_image_ds = partial(genDatasetForest, dsize=dsize)
    #return tf.data.Dataset.from_generator(   load_image_ds , output_types = ( tf.float32))

def getImageDataSetSparseResidential(dsize):
    return list( genDatasetSparseResidential(dsize))
    #load_image_ds = partial(genDatasetSparseResidential, dsize=dsize)
    #return tf.data.Dataset.from_generator(   load_image_ds , output_types = ( tf.float32))

def getImageDataSetBareLand(dsize):
    return list( genDatasetBareLand(dsize))
    #load_image_ds = partial(genDatasetBareLand, dsize=dsize)
    #return tf.data.Dataset.from_generator(   load_image_ds , output_types = ( tf.float32))    

