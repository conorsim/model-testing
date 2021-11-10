#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import blobconverter
import cv2
import depthai as dai
import numpy as np
from classes import class_names
from depthai_sdk import FPSHandler

import time

# Get Argument First
parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-s', '--shaves', type=int, default=6, help="Number of shaves to use for blob")
args = parser.parse_args()

# Link video in with the detection network

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

camera = not args.video
labels = class_names()


# Start defining a pipeline
pipeline = dai.Pipeline()

# NeuralNetwork
print("Creating Neural Network...")
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(str(blobconverter.from_zoo(name="single-image-super-resolution-1032", shaves=args.shaves)))
detection_nn.setNumInferenceThreads(2)

if camera:
    print("Creating Color Camera...")
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(480, 270)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(1)

    nn_xout = pipeline.createXLinkOut()
    nn_xout.setStreamName("out_nn")

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)

    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(1920, 1080)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(6220800)

    cam_rgb.preview.link(detection_nn.inputs["0"])
    cam_rgb.preview.link(manip.inputImage)

    manip.out.link(detection_nn.inputs["1"])

    detection_nn.out.link(nn_xout.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

frame = None
bboxes = []

# detection_nn.input.setBlocking(True)
# nn_xout.input.setBlocking(True)
# cam_xout.input.setBlocking(True)

# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        fps = FPSHandler(maxTicks=2)

    q_nn = device.getOutputQueue(name="out_nn", maxSize=10, blocking=True) # changed "nn" to "in_nn"


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame(result):
        new_frame = np.array(result).reshape((3, 1080, 1920)).transpose(1, 2, 0)
        new_frame = np.where(new_frame > 1.0, 1.0, new_frame)
        new_frame = np.where(new_frame < 0.0, 0.0, new_frame)
        new_frame = (new_frame * 255).astype(np.uint8)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)

        # in_rgb = q_rgb.get()
        # original_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
        # original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        return np.ascontiguousarray(new_frame)#, np.ascontiguousarray(original_frame)

    result = None

    while should_run():

        fps.tick("test")

        in_nn = q_nn.get()
        in_rgb = q_rgb.get()

        result = np.array(in_nn.getFirstLayerFp16())

        frame = get_frame(result)

        print(fps.tickFps("test"))
        cv2.imshow("hi-res", frame)

        if cv2.waitKey(1) == ord('q'):
            break
