# Python + Arduino

# import pyfirmata as pyf
import time
import sys
import serial

def get_measurements(device,baudrate = 9600,print_val = False):
    measurements = []

    arduino = serial.Serial(device,baudrate)
    x = arduino.readline()

    if print_val == True:
        print(str(x.decode().strip()))

    measurements.append(x)
    # return measurements

device = '/dev/tty.usbmodem14201'
while True:
    get_measurements(device)
