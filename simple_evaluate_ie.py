from utils.ie_backend import IEModel
import cv2
import numpy as np

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

# create model
model = IEModel(model_xml= "examples/micronet_m0_224x224.xml",
                model_bin= "examples/micronet_m0_224x224.bin",
                device_name= "CPU")

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
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis = 0)

    # inference
    output = model.infer(img)

    # select the right output
    output = output["1796"]

    # do the necessary postprocessing
    sm = softmax(output)
    result1 = np.argmax(sm)
    result5 = np.argsort(sm)

    # measure top-1
    total_sum1 += result1 == gts[i]
    total_sum5 += gts[i] in result5

print(f"Top-1: {total_sum1 / len(gts)}")
print(f"Top-5: {total_sum5 / len(gts)}")