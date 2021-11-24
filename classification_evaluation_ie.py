from utils.ie_backend import IEModel
from utils.dataset import Dataset, download_dataset
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
    nn_type="classification",
    image_dir=image_dir,
    label_dir=label_dir,
    sample_size=args.sample_size,
    seed=args.seed
)

dataset.read_classification()

# evaluate model on the dataset
print("Evaluating... this may take some time...\n")
total_sum1, total_sum5 = 0, 0
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
    output = output["1796"][0]

    # do the necessary postprocessing
    sm = softmax(output)
    result1 = np.argmax(sm)
    result5 = np.argsort(sm)[-5:]

    # measure top-1
    total_sum1 += result1 == dataset.y[i]
    total_sum5 += dataset.y[i] in result5

print("--- Results ---")
print(f"Top-1: {total_sum1 / len(dataset.y)}")
print(f"Top-5: {total_sum5 / len(dataset.y)}")

# delete local data
if args.dataset and args.no_save:
    # remove the parent directory
    rm_path = image_dir.split('/images')[0]
    rmtree(rm_path)
