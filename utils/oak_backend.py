#!/usr/bin/env python3

import depthai as dai
import numpy as np
import logging
from queue import SimpleQueue
import threading

'''
This code contains a simple API for setting up the OAK
'''

class OakModel:

    def __init__(self,
                 model_path : str,
                 inference_threads : int = 2,
                 openvino_version : dai.OpenVINO = dai.OpenVINO.VERSION_2021_4):
        self.model_path = model_path
        self.openvino_version = openvino_version
        self.inference_threads = inference_threads

        self.input_queue = SimpleQueue()
        self.output_queue = SimpleQueue()

        # start thread
        pipeline = self.__create_pipeline()
        self.thread = threading.Thread(target=self.__run_pipeline, args = (pipeline,), daemon=True)
        self.thread.start()

    def infer(self, image):
        self.input_queue.put(image)
        output = self.output_queue.get()
        return output

    def __create_pipeline(self):

        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=self.openvino_version)

        frame_xin = pipeline.createXLinkIn()
        frame_xin.setStreamName("frame_in")

        detection_nn = pipeline.createNeuralNetwork()
        detection_nn.setBlobPath(self.model_path)
        detection_nn.setNumPoolFrames(4)
        detection_nn.setNumInferenceThreads(self.inference_threads)

        #frame_xout = pipeline.createXLinkOut()
        #frame_xout.setStreamName("frame_out")

        nn_xout = pipeline.createXLinkOut()
        nn_xout.setStreamName("nn_out")

        frame_xin.out.link(detection_nn.input)
        detection_nn.out.link(nn_xout.input)
        #detection_nn.passthrough.link(frame_xout.input)

        return pipeline


    def __run_pipeline(self, pipeline):

        logging.info("Starting the pipeline")
        with dai.Device(pipeline) as device:
            #q_rgb = device.getOutputQueue(name="frame_out", maxSize=1, blocking=True)
            q_in = device.getInputQueue("frame_in")
            q_nn = device.getOutputQueue(name="nn_out", maxSize=1, blocking=True)

            while True:

                # read frame
                frame = self.input_queue.get()
                if frame is None:
                    continue

                logging.info("Got frame")

                # preprocess frame
                w, h = frame.shape[1], frame.shape[0]
                img_data = dai.ImgFrame()
                img_data.setData(frame.transpose(2, 0, 1).flatten())
                img_data.setWidth(w)
                img_data.setHeight(h)

                q_in.send(img_data)

                q_out = q_nn.get()

                # read output
                output = np.array(q_out.getFirstLayerFp16())

                self.output_queue.put(output)
