#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import blobconverter
import cv2
import depthai as dai
import numpy as np

import time

# Get Argument First
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--shaves', type=int, default=6, help="Number of shaves to use for blob")
parser.add_argument('-nn', '--model_name', type=str, default=None, help="Name of the model in the zoo")
parser.add_argument('-zoo', '--zoo_type', type=str, default=None, help="Zoo type")
parser.add_argument('-shape', '--input_shape', type=int, nargs='+', help="List of ints", required=True)
args = parser.parse_args()

print("Parsed arguments")
print(args)


# Start defining a pipeline
pipeline = dai.Pipeline()

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(blobconverter.from_zoo(name=args.model_name, zoo_type=args.zoo_type, shaves=args.shaves)))
detection_nn.setNumInferenceThreads(1)

nn_in = pipeline.createXLinkIn()
nn_in.setStreamName("in_nn")
nn_in.out.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    detection_in = device.getInputQueue("in_nn")
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)

    fps_storage = []

    while len(fps_storage) <= 30:

        frame = np.random.randint(256, size=args.input_shape, dtype=int)

        nn_data = dai.NNData()
        nn_data.setLayer("input", frame)

        start = time.time()
        detection_in.send(nn_data)

        in_nn = q_nn.get().getFirstLayerFp16()
        diff = time.time() - start

        fps_storage.append(1 / diff)
        print(1 / diff)

    print(np.mean(fps_storage[5:]), np.std(fps_storage[5:]))
