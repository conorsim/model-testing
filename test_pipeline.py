#!/usr/bin/env python3
import os
import sys
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help="The name of the neural network folder in this repository for testing FPS")
args = parser.parse_args()

script_path = os.path.join(os.getcwd(), f'{args.nn}/main.py')
p = Popen([sys.executable, '-u', script_path, '-cam'],
          stdout=PIPE, stderr=STDOUT, bufsize=1)
with p.stdout:
    fps_measurements = []
    for line in iter(p.stdout.readline, b''):
        out = line.decode('utf-8')
        try:
            fps = float(out)
            fps_measurements.append(fps)
        except:
            pass
        if len(fps_measurements) == 25:
            break

valid_measurements = fps_measurements[5:]

print(np.mean(valid_measurements), np.std(valid_measurements), valid_measurements)
