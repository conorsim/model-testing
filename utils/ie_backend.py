#!/usr/bin/env python3

from openvino.inference_engine import IECore

'''
This code contains a simple API for setting up the OAK
'''

class IEModel:

    def __init__(self,
                 model_xml : str,
                 model_bin : str,
                 device_name : str
                 ):
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device_name = device_name

        ie = IECore()
        net = ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(net.input_info))
        self.exec_net = ie.load_network(network=net, device_name=self.device_name)

        print(f"OUTPUTS: {net.outputs}")
        print(f"Please use the correct one in the post-processing!")

    def infer(self, image):
        output = self.exec_net.infer(inputs={self.input_blob: image})
        return output
