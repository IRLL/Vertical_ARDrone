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
from ardrone_autonomy.msg import Navdata

#from threading import Lock, Thread

class Viewer ():
	#class constants
	UPDATE_SPEED = 15 #process at 10 hz
	AREA_DIST_CONST = 1

	def __init__(self):
		self.vloc = 0
		self.hloc = 0
		self.height, self.width = (-1, -1)
		self.battery = -1.0
		self.navdata = Navdata()
		
		self.threshold = rospy.get_param("v_controller/threshold")
		self.video_sub = rospy.Subscriber('/ardrone/image_raw', Image, self.receive_image_callback)
		self.nav_sub = rospy.Subscriber('/ardrone/navdata', Navdata, self.receive_nav_callback)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.rx_state_callback)
		rospy.init_node('viewer', anonymous=False)

			
	
	def rx_state_callback(self, data):
		self.hloc = data.data[0]/2
		self.vloc = data.data[1]/1.5

	def receive_image_callback(self, data):
		image = np.asarray(ToOpenCV(data)) #convert image

		self.get_image_size(image)
		
		hloc, vloc = self.rescale(self.hloc, self.vloc)
		#draw the lines line marking the y pos
		line_thickness = 3
		r = 255
		g = 255
		b = 255
		#cv2.line(image, (0, vloc), (639, vloc), (r, g, b), line_thickness)	#draw horizontal line	
		#cv2.line(image, (hloc, 0), (hloc, 479), (r, g, b), line_thickness)	#draw vertical line

		#draw lines for the vertical thresholds
		#high_t = int(self.height/2 + self.threshold*self.height/2)
		#low_t = int (self.height/2-self.threshold*self.height/2)
		#cv2.line(image, (0, high_t), (639, high_t), (255,0,0))
		#cv2.line(image, (0, low_t), (639, low_t), (255,0,0))

		#draw a center line
		#cv2.line(image, (0, self.height/2), (639, self.height/2), (0,0,255))	
		
		#write current battery level on the image
		cv2.putText(image, "battery: {:.2f}%".format(self.battery), (int(self.width*.01
),int(self.height*0.95)), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0))

		#print the height
		cv2.putText(image, "altd: {:.2f}m".format(float(self.navdata.altd)/1000), (int(self.width*.01
),int(self.height*0.88)), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0))

		#velocity
		cv2.putText(image, "vx: {:.2f}m/s".format(self.navdata.vx/1000), (int(self.width*.01
),int(self.height*0.80)), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0))
		cv2.putText(image, "vy: {:.2f}m/s".format(self.navdata.vy/1000), (int(self.width*.01
),int(self.height*0.72)), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0))
		cv2.putText(image, "vz: {:.2f}m/s".format(self.navdata.vz/1000), (int(self.width*.01
),int(self.height*0.64)), cv2.FONT_HERSHEY_COMPLEX, .8, (0, 255, 0))

		#show the image
		cv2.imshow('image', image)
		cv2.waitKey(1)

	def receive_nav_callback(self, data):
		self.battery = data.batteryPercent
		self.navdata = data
			
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
