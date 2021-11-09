#!/usr/bin/env python3
import os
import sys
from subprocess import Popen, PIPE, STDOUT
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn', type=str, help="The name of the neural network folder in this repository for testing FPS")
args = parser.parse_args()

script_path = os.path.join(os.getcwd(), f'{args.nn}/main.py')
p = Popen([sys.executable, '-u', script_path, '-cam'],
          stdout=PIPE, stderr=STDOUT, bufsize=1)
with p.stdout:
    fps_measurements = []
    count = 0
    measure_count = 0
    for line in iter(p.stdout.readline, b''):
        out = line.decode('utf-8')
        try:
            fps = float(out)
            count += 1
        except:
            pass
        if count == 6:
            start = time.time()
        if count == 25:
            diff = time.time() - start
            fps_measurements.append(20/diff)
            count = 0
            measure_count += 1
        if measure_count == 5:
            break

valid_measurements = fps_measurements[5:]

print(np.mean(fps_measurements), np.std(fps_measurements), fps_measurements)
