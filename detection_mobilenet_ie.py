from utils.ie_backend import IEModel
from utils.dataset import Dataset, download_dataset
from utils.object_detection_helpers import mAP, iou, nms
import depthai as dai
import cv2
import numpy as np
import argparse
import blobconverter
from shutil import rmtree

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--bin', type=str, help="Path to the bin file for model", required=True)
parser.add_argument('-x', '--xml', type=str, help="Path to the bin file for model", required=True)
parser.add_argument('-s', '--input_shape', type=int, nargs='+', help="List of ints", required=True)
parser.add_argument('-c', '--chip', default='CPU', type=str, help="Name of the chip to evaluate on (CPU or MYRIAD)")
parser.add_argument('-z', '--zoo_type', type=str, default='intel', help="Zoo type")
parser.add_argument('-d', '--dataset', type=str, default=None, help="Dataset type")
parser.add_argument('-i', '--image_dir', type=str, help="Path to image directory for local dataset")
parser.add_argument('-l', '--label_dir', type=str, help="Path to label directory for local dataset")
parser.add_argument('-sh', '--shaves', type=int, default=6, help="Number of shaves to use for blob")
parser.add_argument('-ss', '--sample_size', type=int, default=None, help="Number of shaves to use for blob")
parser.add_argument('-se', '--seed', type=int, default=None, help="Number of shaves to use for blob")
parser.add_argument('-ns', '--no_save', action="store_true", help="Do not save a dataset downloaded online locally")
args = parser.parse_args()

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

# download dataset
if args.dataset:
    print("Downloading dataset...")
    image_dir, label_dir = download_dataset(args.dataset)
else:
    image_dir = args.image_dir
    label_dir = args.label_dir

# create model
print("Creating model...")
# create model
model = IEModel(model_xml=args.xml,
                model_bin=args.bin,
                device_name=args.chip)

# create dataset
print("Creating dataset...")
dataset = Dataset(
    nn_type='detection',
    image_dir=image_dir,
    label_dir=label_dir,
    sample_size=args.sample_size,
    seed=args.seed
)

# the alternate_coco flag here denotes that the 91 class encoding is used instead of the 80 class encoding for COCO
dataset.read_detection(alternate_coco=True)

class_confs = [[] for _ in range(int(dataset.max_class))]
class_hits = [[] for _ in range(int(dataset.max_class))]

# evaluate model on the dataset
print("Evaluating... this may take some time...\n")
for i, image_path in enumerate(dataset.image_paths):

    # read image
    img = cv2.imread(image_path)

    # do the necessary preprocessing
    img = cv2.resize(img, (args.input_shape[1], args.input_shape[2]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis = 0)

    # inference
    output = model.infer(img)

    # select the right output
    output = output["DetectionOutput"][0][0]

    stop_idx = np.where(output == -1)[0][0]
    output = output[:stop_idx, :]

    output = np.delete(output, [0], axis=1)
    output = nms(output)

    # loop through the predicted boxes
    for k in range(dataset.y[i].shape[0]):
        y = dataset.y[i][k, :]
        found = False
        for j in range(output.shape[0]):
            yhat = output[j, :]
            # if the labels match
            if yhat[0] == y[0]:
                if iou(yhat[-4:], y[-4:]) > 0.5:
                    # hit!
                    class_confs[int(y[0])-1].append(yhat[1])
                    class_hits[int(y[0])-1].append(1)
                    found = True
        if found == False:
            # miss!
            class_confs[int(y[0])-1].append(0.0)
            class_hits[int(y[0])-1].append(0)

map = mAP(class_confs, class_hits)

print("--- Results ---")
print(f"mAP: {map[0]}")

# delete local data
if args.dataset and args.no_save:
    # remove the parent directory
    rm_path = image_dir.split('/images')[0]
    rmtree(rm_path)
