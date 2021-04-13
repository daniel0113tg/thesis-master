# USAGE
# python client.py --server-ip SERVER_IP

# import the necessary packages
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import cv2 as cv
import copy

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5558".format(
	args["server_ip"]))

# get the host name, initialize the video stream, and allow the
# camera sensor to warmup
print("paso")
rpiName =socket.gethostname()
print(rpiName)
vs = cv.VideoCapture(0)
#vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	ret, frame = vs.read()
	if not ret:
		print("break")
		break
	if frame is not None:
		##frame = cv.resize(frame,(224,224), interpolation = cv.INTER_AREA)
		frame = cv.flip(frame,1)
		sender.send_image(rpiName, frame)
