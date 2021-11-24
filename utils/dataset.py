import numpy as np
import glob
import cv2
import xml.etree.ElementTree as ET
import gdown
import os
from zipfile import ZipFile

"""

API for a standardizing a validation or test set to evaluate against using a DepthAI model

type: Type of dataset to read
image_dir: Directory containing images
label_dir: Directory containing labels in VOC format
sample_size: Number of images to randomly sample. None indicates the inclusion of the entire validation set
seed: Numpy random seed used to consistently generate samples if applicable

"""

class Dataset:

    def __init__(self, nn_type, image_dir, label_dir, sample_size=None, seed=None):
        self.types = ['classification', 'detection']

        if nn_type in self.types:
            self.nn_type = nn_type
        else:
            raise ValueError(f"Dataset type {nn_type} is not valid. Valid types include {self.types}")

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sample_size = sample_size
        self.seed = seed

        # the following will initialize when reading dataset
        self.y = None
        self.max_class = None

        cv2_extensions = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm', 'ppm', 'pxm', 'pnm', 'pfm', \
                          'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic'] # https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
        # include upper case versions
        cv2_extensions = cv2_extensions + [ext.upper() for ext in cv2_extensions]

        self.image_paths = []
        for ext in cv2_extensions:
            self.image_paths += glob.glob(self.image_dir + f'/*.{ext}') # finds any supported cv2 image
        self.label_paths = glob.glob(self.label_dir + '/*.xml') # only XML format is accepted

        # check that each image has a corresponding label
        if len(self.image_paths) != len(self.label_paths):
            raise Exception(f"Number of images ({len(self.image_paths)}) does not match number of labels ({len(self.label_paths)})")
        # sort paths such that indicies correspond to the same granule
        self.image_paths = sorted(self.image_paths)
        self.label_paths = sorted(self.label_paths)

        # set a seed for consistent sampling
        if self.seed:
            np.random.seed(self.seed)

        # randomly sample according to sample size
        if self.sample_size:
            idxs = np.random.choice(range(len(self.image_paths)), self.sample_size)
            self.image_paths = list(np.array(self.image_paths)[idxs])
            self.label_paths = list(np.array(self.label_paths)[idxs])

        image_granules = [path.split('/')[-1].split('.')[0] for path in self.image_paths]
        label_granules = [path.split('/')[-1].split('.')[0] for path in self.label_paths]
        for img_g, lab_g in list(zip(image_granules, label_granules)):
            if img_g != lab_g:
                raise Exception(f"{img_g} in {self.image_dir} does not match {lab_g} in {self.label_dir}")


    """
    Initializes numeric labels for a classification task
    """

    def read_classification(self):
        if self.nn_type != 'classification':
            raise Exception(f"Type {self.nn_type} does not match classification")

        self.y = np.zeros(len(self.label_paths)).astype(np.int64)
        for i, path in enumerate(self.label_paths):
            tree = ET.parse(self.label_paths[i])
            root = tree.getroot()
            self.y[i] = int(root.find("id").text)

    """
    Initializes numeric labels for a detection task

    Reads data in the conventional [label, xmin, ymin, xmax, ymax] VOC format.
    """

    def read_detection(self, alternate_coco=False):
        if self.nn_type != 'detection':
            raise ValueError(f"Type {self.nn_type} does not match detection")

        self.max_class = 0
        self.y = [] # list: each image can contain multiple objects
        for i in range(len(self.label_paths)):
            tree = ET.parse(self.label_paths[i])
            root = tree.getroot()
            objs = root.findall("object")
            label = np.zeros((len(objs), 6)) # [label, area, xmin, ymin, xmax, ymax]
            for j, obj in enumerate(objs):
                if alternate_coco: class_id = float(obj.find("id91").text)
                else: class_id = float(obj.find("id").text) # for other data
                if class_id > self.max_class: self.max_class = class_id
                label[j,0] = class_id
                label[j,1] = float(obj.find("area").text)
                bbox = obj.find("bndbox")
                label[j,2] = float(bbox.find("xmin").text)
                label[j,3] = float(bbox.find("ymin").text)
                label[j,4] = float(bbox.find("xmax").text)
                label[j,5] = float(bbox.find("ymax").text)
            self.y.append(label)



"""

Method for downloading a standard dataset from Luxonis Google Drive.

"""

def download_dataset(name):
    # check the name of the dataset
    dataset_types = ['imagenet_val', 'coco_val']
    if name not in ['imagenet_val', 'coco_val']:
        raise ValueError(f"Dataset ({name}) must be in {dataset_types}")

    # match dataset to google drive download
    if name == 'imagenet_val':
        path = os.getcwd()+'/imagenet_val'
        image_path = path+'/images'
        label_path = path+'/labels'

        # check if dataset is already cached
        if os.path.exists(image_path) and os.path.exists(label_path):
            return image_path, label_path

        # otherwise download from google drive
        if not os.path.exists(path):
            os.mkdir(path)
        url = 'https://drive.google.com/uc?id=13GzeMHflmBvGHqwn_k9_xGyq99IjsgOW'
        output = path+'/imagenet_val.zip'
        gdown.download(url, output, quiet=False)
        with ZipFile(output, 'r') as z:
            z.extractall(path=path)
        os.remove(output)

    if name == 'coco_val':
        path = os.getcwd()+'/coco_val'
        image_path = path+'/images'
        label_path = path+'/labels'

        # check if dataset is already cached
        if os.path.exists(image_path) and os.path.exists(label_path):
            return image_path, label_path

        # otherwise download from google drive
        if not os.path.exists(path):
            os.mkdir(path)
        url = 'https://drive.google.com/uc?id=1dMZMvK6NCq2503EbEeWC9HZ-q6EQJ8TX'
        output = path+'/coco_val.zip'
        gdown.download(url, output, quiet=False)
        with ZipFile(output, 'r') as z:
            z.extractall(path=path)
        os.remove(output)

    # return new image directory and label directory
    return image_path, label_path
