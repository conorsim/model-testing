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
parser.add_argument('-fp16', '--fp16', action="store_true", help="Input must be FP16")
parser.add_argument('-c', '--cache', action="store_true", help="Use cache with blobconverter.")

args = parser.parse_args()

print("Parsed arguments")
print(args)


# Start defining a pipeline
pipeline = dai.Pipeline()

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(blobconverter.from_zoo(name=args.model_name, zoo_type=args.zoo_type,
                                                    shaves=args.shaves, use_cache = args.cache)))
detection_nn.setNumInferenceThreads(1)
detection_nn.input.setBlocking(True)
#detection_nn.out.setBlocking(True)

nn_in = pipeline.createXLinkIn()
nn_in.setStreamName("in_nn")
nn_in.out.link(detection_nn.input)
#nn_in.out.setBlocking(True)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    detection_in = device.getInputQueue("in_nn", maxSize=25, blocking=True)
    q_nn = device.getOutputQueue(name="nn", maxSize=5, blocking=True)

    fps_storage = []

    for repetition in range(5):

        print(f"Repetition: {repetition}")

        # feed 55 messages
        for i in range(25):
            if not args.fp16:
                frame = np.random.randint(256, size=args.input_shape, dtype=int)
                nn_data = dai.NNData()
                nn_data.setLayer("input", frame)
            else:
                frame = np.random.rand(*args.input_shape)
                frame = frame.astype(np.float16).flatten().tolist()
                nn_data = dai.Buffer()
                nn_data.setData(frame)

            detection_in.send(nn_data)

        for i in range(5):
            q_nn.get().getFirstLayerFp16()

        start = time.time()
        for i in range(20):
            q_nn.get().getFirstLayerFp16()
        diff = time.time() - start

        fps_storage.append(20/diff)

    """
    while len(fps_storage) <= 30:

        if not args.fp16:
            frame = np.random.randint(256, size=args.input_shape, dtype=int)
            nn_data = dai.NNData()
            nn_data.setLayer("input", frame)
        else:
            frame = np.random.rand(*args.input_shape)
            frame = frame.astype(np.float16).flatten().tolist()
            nn_data = dai.Buffer()
            nn_data.setData(frame)

        start = time.time()
        detection_in.send(nn_data)

        in_nn = q_nn.get().getFirstLayerFp16()
        diff = time.time() - start

        fps_storage.append(1 / diff)
        print(1 / diff)
    """

    print(np.mean(fps_storage), np.std(fps_storage), fps_storage)
