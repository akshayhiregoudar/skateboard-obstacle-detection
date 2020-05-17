#!/usr/bin/env python

import socket
from struct import *
# from time import sleep

# usb0 ip address (obtained via 'ipconfig' command after connecting the phone to raspi and turning on the usb tethering)
UDP_IP = "192.168.42.16"  # set as static ip address on raspi and hence can be used even after reconnecting

# Port number
UDP_PORT = 50000

# Print IP and Port number
'''
print "Receiver IP: ", UDP_IP
print "Port: ", UDP_PORT
'''

# Define sock and bind it to the given port
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Create a new csv file to store the data
file = open("data.csv","w")

while True:
    data, addr = sock.recvfrom(1024)

    # Unpack data
    a = unpack_from('!f', data, 0)
    b = unpack_from('!f', data, 4)
    c = unpack_from('!f', data, 8)
    d = unpack_from('!f', data, 12)
    e = unpack_from('!f', data, 16)
    f = unpack_from('!f', data, 20)

    # Write to file
    file.write(str(a))
    file.write(str(b))
    file.write(str(c))
    file.write(str(d))
    file.write(str(e))
    file.write(str(f))
    file.write("\n")

    # For custom frequency
    # sleep(0.1)

# Close the file (useful if loop is timed rather than being an infinite loop as in our case above)
file.close()

# To print the data in console
'''
while True:
    data, addr = sock.recvfrom(1024)
    print "acc_x: ",\
        "%1.4f" %unpack_from ('!f', data, 0),\
        "acc_y: ",\
        "%1.4f" %unpack_from ('!f', data, 4),\
        "acc_z: ",\
        "%1.4f" %unpack_from ('!f', data, 8),\
        "g_x: ",\
        "%1.4f" %unpack_from ('!f', data, 12),\
        "g_y: ",\
        "%1.4f" %unpack_from ('!f', data, 16),\
        "g_z: ",\
        "%1.4f" %unpack_from ('!f', data, 20),\
        "r_x: ",\
        "%1.4f" %unpack_from ('!f', data, 24),\
        "r_y: ",\
        "%1.4f" %unpack_from ('!f', data, 28),\
        "r_z: ",\
        "%1.4f" %unpack_from ('!f', data, 32)
'''