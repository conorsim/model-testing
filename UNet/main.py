#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import blobconverter
import cv2
import depthai as dai
import numpy as np
from classes import class_names

import time

class FPSHandler:
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None

        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

        self.storage = []

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        fps = self.frame_cnt / (self.timestamp - self.start)
        self.storage.append(fps)
        if len(self.storage) > 25:
            self.storage.pop(0)
        return fps

    def fps_average(self):
        if len(self.storage) < 25:
            return 0, 0

        return np.mean(self.storage), np.std(self.storage)



# Get Argument First
parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
parser.add_argument('-s', '--shaves', type=int, default=6, help="Number of shaves to use for blob")
parser.add_argument('-f', '--fps', action="store_true", help="Compute FPS only without post-processing")
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
detection_nn.setBlobPath(str(blobconverter.from_zoo(name="unet-camvid-onnx-0001", shaves=args.shaves)))
detection_nn.setNumInferenceThreads(1)

if camera:
    print("Creating Color Camera...")
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(480, 368)
    cam_rgb.setInterleaved(False)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

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


def get_mask(x, total_classes):
    x = np.argmax(x, axis = 0)
    x = x * 255 / total_classes
    x = x.astype(np.uint8)
    output_colors = cv2.applyColorMap(x, cv2.COLORMAP_JET)

    # reset the color of 0 class
    output_colors[x == total_classes - 1] = [0, 0, 0]
    return output_colors

def show_overlay(frame, output_colors):
    return cv2.addWeighted(frame,0.65, output_colors,0.35,0)


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        resized = cv2.resize(arr, shape)
        return resized.transpose(2, 0, 1)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    if camera:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=True)
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        fps = FPSHandler(cap)

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

        fps.next_iter()

        if not camera:
            nn_data = dai.NNData()
            nn_data.setLayer("input", to_planar(frame, (480, 368)))
            detection_in.send(nn_data)

        in_nn = q_nn.get()

        if args.fps:
            # skip inference if FPS only
            print(f"FPS: {fps.fps()}")
            print(f"{fps.fps_average()}")
            continue


        output = np.array(in_nn.getFirstLayerFp16()).reshape(12, 368, 480)
        result = get_mask(output, 12)

        frame_main = frame.copy()
        frame_main = show_overlay(frame_main, result)

        cv2.putText(frame_main, "Fps: {:.2f}".format(fps.fps()), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))
        print(fps.fps_average())
        cv2.imshow("rgb", cv2.resize(frame_main, (480, 368)))

        if cv2.waitKey(1) == ord('q'):
            break
