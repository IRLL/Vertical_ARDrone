#!/usr/bin/python

import rospy
import cv2
import cv2.cv as cv
import numpy as np
from copy import deepcopy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from image_converter import ToOpenCV
from std_msgs.msg import Float32MultiArray

#from threading import Lock, Thread

class Viewer ():
	#class constants
	UPDATE_SPEED = 10 #process at 10 hz
	AREA_DIST_CONST = 1

	def __init__(self):
		self.vloc = 0
		self.hloc = 0
		self.height, self.width = (-1, -1)

		self.video_sub = rospy.Subscriber('/ardrone/image_raw', Image, self.receive_image_callback)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.rx_state_callback)
		self.controller_sub = rospy.Subscriber('v_controller/control_state', Float32MultiArray, self.rx_controller_state_callback)
		rospy.init_node('viewer', anonymous=False)

			
	
	def rx_state_callback(self, data):
		self.vloc = data.data;
	def rx_controller_state_callback(self, data):
		self.hloc = data.data[0]

	def receive_image_callback(self, data):
		image = np.asarray(ToOpenCV(data)) #convert image

		self.get_image_size(image)
		
		hloc, vloc = self.rescale(self.hloc, self.vloc)
		#draw the lines line marking the y pos
		cv2.line(image, (0, vloc), (639, vloc), 0)	#draw horizontal line	
		cv2.line(image, (hloc, 0), (hloc, 479), 0)	#draw vertical line	

		#show the image
		cv2.imshow('image', image)
		cv2.waitKey(1)
			
	def get_image_size(self, image):
		self.height, self.width, _ = image.shape

	def rescale(self, x, y):
		width_div = float(self.width)/2
		height_div = float(self.height)/2
		newx = float(x * width_div) + width_div			
		newy = float(y * height_div) + height_div			
		return int(newx),int(newy)

if __name__ == "__main__":
	viewer = Viewer()
	rospy.spin()
