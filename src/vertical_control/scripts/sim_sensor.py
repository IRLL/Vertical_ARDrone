#!/usr/bin/python
from __future__ import print_function
import cv2
import rospy
import numpy as np
from copy import deepcopy
from sensor_msgs.msg import Image
from threading import Lock
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from image_converter import ToOpenCV
from math import sqrt
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from sensor_msgs.msg import Range

class Sim_sensor():
	#class constants
	UPDATE_SPEED = 200 #process at 10 hz
	AREA_DIST_CONST = 35.623

	def __init__(self):

		self.latest_image = None

		self.image_lock = Lock()
		self.lastx = 0
		self.lasty = 0

		self.images = dict()
		self.height, self.width = (-1, -1)
		self.elevation = 0.0
		self.hover_distance = rospy.get_param("v_controller/hover_distance")


		rospy.init_node('simulated_sensors', anonymous=False)
		self.learner_pub = rospy.Publisher('v_controller/state', Float32MultiArray)
		self.not_visible_pub = rospy.Publisher('v_controller/visible', Bool)
		self.controller_pub = rospy.Publisher('v_controller/control_state', Float32MultiArray)
		self.video_sub = rospy.Subscriber('/ardrone/image_raw', Image, self.receive_image_callback)
		self.height_sub = rospy.Subscriber('/sonar_height', Range, self.receive_height_callback)
		self.rate = rospy.Rate(self.UPDATE_SPEED)



	def receive_image_callback(self, data):
		self.image_lock.acquire()
		try:
			self.latest_image = data
		finally:
			self.image_lock.release()
	def receive_height_callback(self, Range_data):
		height = Range_data.range
		self.elevation = height
		self.controller_pub.publish(None, [height])

	def sign(self, value):
		if (value > 0.0):
			return 1
		if (value < 0.0):
			return -1

		return 0


	def processing_function(self):
		print("waiting for images to come in...")
		while (not rospy.is_shutdown()) and self.latest_image is None:
			rospy.sleep(.5)
		print("done!")

		self.get_image_size()

		while not rospy.is_shutdown(): #do image processing here
			#get the latest image
			self.image_lock.acquire()
			try:
				image = self.latest_image
			finally:
				self.image_lock.release()

			image = np.asarray(ToOpenCV(image))

			x, y, distance = self.find_orange(image)

			self.learner_pub.publish(None,[x,y,self.elevation])
			#self.controller_pub.publish(None, [distance])
			#need to add publisher for other info
			self.rate.sleep()

	def find_orange (self, image):
		#parameters for locating orange
		lower_target_range = (0, 170, 0)
		upper_target_range = (200, 255, 255)

		x, y, distance = self.process_image (image, lower_target_range, upper_target_range)
		return x, y, distance

	def get_image_size(self):
		self.image_lock.acquire()
		try:
			image = self.latest_image
		finally:
			self.image_lock.release()
		image = np.asarray(ToOpenCV(image))

		self.height, self.width, _ = image.shape

	def rescale(self, x, y):
		width_div = float(self.width)/4.0
		height_div = float(self.height)/3.0
		newx = float(x - self.width/2)/width_div
		newy = float(y - self.height/2)/height_div
		return newx,newy

	def process_image (self, image, lower_range, upper_range):
		#outputs = dict()

		#create copy of the image
		self.images["src"] = deepcopy(image)

		#give it a nice blur
		#images["MFilter"] = cv2.medianBlur (images["Src"], 9)

		#convert to HSV
		self.images["hsv"] = cv2.cvtColor (self.images["src"], cv2.COLOR_BGR2HSV)

		#filter for color
		self.images["color_filter"] = cv2.inRange (self.images["hsv"], lower_range, upper_range)

		#locate contours
		contours, heirarchy = cv2.findContours (deepcopy (self.images["color_filter"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		#self.images['contours'] = deepcopy(self.images['src'])
		#cv2.drawContours(self.images['contours'], contours, -1, (0,0,0), thickness=2)

		#self.images['center'] = deepcopy(self.images['src'])
		#cv2.circle(self.images['center'], (self.width/2, self.height/2), 1, 0, 5)

		if contours: #check if we have any contours first
			moment = cv2.moments(contours[0])
			area = moment["m00"] #cv2.contourArea(contours[0])
			if area != 0:
				xpos = moment["m10"] / area
				ypos = moment["m01"] / area
				distance = self.AREA_DIST_CONST * 1/sqrt(area)
				self.not_visible_pub.publish(1)

				xpos, ypos = self.rescale(xpos, ypos)
				self.lasty = 3 * self.sign(ypos)
				self.lastx = 4 * self.sign(xpos)
				#print("a", area)
				#print("d", distance)
			else:
				xpos = self.lastx
				ypos = self.lasty
				distance = self.hover_distance
				self.not_visible_pub.publish(0)
		else:
			xpos = self.lastx
			ypos = self.lasty
			distance = self.hover_distance
			self.not_visible_pub.publish(0)


		return xpos, ypos, distance

if __name__ == '__main__':
	sim_sensor = Sim_sensor()
	sim_sensor.processing_function()
