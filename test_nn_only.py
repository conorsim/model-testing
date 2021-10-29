#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import blobconverter
import cv2
import depthai as dai
import numpy as np

import time
import os

import zipfile

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

# Downloading model
model_path = blobconverter.from_zoo(name=args.model_name,
                                    zoo_type=args.zoo_type,
                                    shaves=args.shaves,
                                    use_cache = args.cache)

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(model_path))
detection_nn.setNumInferenceThreads(1)
detection_nn.input.setBlocking(True)

nn_in = pipeline.createXLinkIn()
nn_in.setMaxDataSize(6609600)
nn_in.setStreamName("in_nn")
nn_in.out.link(detection_nn.input)

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
            frame = np.random.randint(256, size=args.input_shape, dtype=np.uint8)
            nn_data = dai.NNData()

            if "super-resolution" in args.model_name:
                nn_data.setLayer("0", frame)
                frame = np.transpose(frame, (1, 2, 0))
                frame = cv2.resize(frame, (args.input_shape[2] * 4, args.input_shape[1] * 4), cv2.INTER_LINEAR)
                frame = np.transpose(frame, (2, 0, 1))
                nn_data.setLayer("1", frame)
            else:
                nn_data.setLayer("input", frame)

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