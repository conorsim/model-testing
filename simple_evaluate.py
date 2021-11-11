from utils.oak_backend import OakModel
import depthai as dai
import cv2
import numpy as np

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

# create model
model = OakModel("examples/micronet_m0_224x224.blob", 2, dai.OpenVINO.VERSION_2021_4)

# example images taken from pexels
image_paths = ["examples/strawberry1.png", "examples/strawberry2.png", "examples/strawberry3.png"]
gts = [949, 949, 949]
total_sum1, total_sum5 = 0, 0
for i, image_path in enumerate(image_paths):

    # read image
    img = cv2.imread(image_path)

    # do the necessary preprocessing
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # inference
    output = model.infer(img)

    # do the necessary postprocessing
    sm = softmax(output)
    result1 = np.argmax(sm)
    result5 = np.argsort(sm)

    # measure top-1
    total_sum1 += result1 == gts[i]
    total_sum5 += gts[i] in result5

print(f"Top-1: {total_sum1 / len(gts)}")
print(f"Top-5: {total_sum5 / len(gts)}")