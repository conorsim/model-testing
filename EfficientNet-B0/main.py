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



# NOTE: video must be of size 224 x 224. We will resize this on the
# host, but you could also use ImageManip node to do it on device

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
detection_nn.setBlobPath(str(blobconverter.from_zoo(name="efficientnet-b0", shaves=args.shaves)))
detection_nn.setNumInferenceThreads(2)

if camera:
    print("Creating Color Camera...")
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(224, 224)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    cam_rgb.setFps(30)

    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("rgb")
    cam_rgb.preview.link(cam_xout.input)
    cam_rgb.preview.link(detection_nn.input)
else:
    face_in = pipeline.createXLinkIn()
    face_in.setStreamName("in_nn")
    face_in.out.link(detection_nn.input)

# Create outputs
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

frame = None
bboxes = []


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        resized = cv2.resize(arr, shape)
        return resized.transpose(2, 0, 1)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        fps = FPSHandler(maxTicks=2)
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        fps = FPSHandler(cap, maxTicks=2)

        detection_in = device.getInputQueue("in_nn")
    q_nn = device.getOutputQueue(name="nn", maxSize=1, blocking=True)


    def should_run():
        return cap.isOpened() if args.video else True


    def get_frame():
        if camera:
            in_rgb = q_rgb.get()
            new_frame = np.array(in_rgb.getData()).reshape((3, in_rgb.getHeight(), in_rgb.getWidth())).transpose(1, 2, 0).astype(np.uint8)
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            return True, np.ascontiguousarray(new_frame)
        else:
            return cap.read()



    result = None

    while should_run():

        read_correctly, frame = get_frame()

        if not read_correctly:
            break

        fps.tick("test")

        if not camera:
            nn_data = dai.NNData()
            nn_data.setLayer("input", to_planar(frame, (224, 224)))
            detection_in.send(nn_data)

        in_nn = q_nn.get()

        output = np.array(in_nn.getFirstLayerFp16())
        result = softmax(output)
        result_conf_1 = np.max(result)
        result_1 = {
            "name" : labels[np.argmax(result)],
            "conf" : round(100 * result_conf_1, 2)
        }
        result = np.delete(result, np.argmax(result))
        result_conf_2 = np.max(result)
        result_2 = {
            "name" : labels[np.argmax(result)],
            "conf" : round(100 * result_conf_2, 2)
        }
        result = np.delete(result, np.argmax(result))
        result_conf_3 = np.max(result)
        result_3 = {
            "name" : labels[np.argmax(result)],
            "conf" : round(100 * result_conf_3, 2)
        }
        top_3 = [result_1, result_2, result_3]

        frame_main = frame.copy()
        cv2.putText(frame_main, "{}".format(result_1["name"]), (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))
        cv2.putText(frame_main, "Confidence: {}%".format(result_1["conf"]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))
        cv2.putText(frame_main, "{}".format(result_2["name"]), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))
        cv2.putText(frame_main, "Confidence: {}%".format(result_2["conf"]), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))
        cv2.putText(frame_main, "{}".format(result_3["name"]), (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))
        cv2.putText(frame_main, "Confidence: {}%".format(result_3["conf"]), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color=(255, 255, 255))

        print(fps.tickFps("test"))
        #cv2.putText(frame_main, "Fps: {:.2f}".format(fps.tickFps("test")), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
        cv2.imshow("rgb", cv2.resize(frame_main, (400, 400)))

        if cv2.waitKey(1) == ord('q'):
            break
