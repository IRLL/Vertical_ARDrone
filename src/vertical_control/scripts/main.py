#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Image       # for recieving video feed
from geometry_msgs.msg import Twist     # controlling the movements
from std_msgs.msg import Empty
import numpy as np
import cv2
from image_converter import ToOpenCV
from threading import Lock
from vision_processor import Vision_Processor
from controller import stabilizer

class Lab5():
	def __init__(self):
		self.video_sub = rospy.Subscriber('/ardrone/image_raw',Image,self.Receive_image)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)
		self.drone_land_pub = rospy.Publisher('/ardrone/land', Empty)
		self.drone_takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
		rospy.init_node('lab5', anonymous=True)
		self.vp = Vision_Processor()
		self.controller = pi_controller_2d(.002, .0001, .05)
		self.go_forward = 0


	def Receive_image(self, data):
		self.image_lock.acquire()
		try:
			self.image = data
		finally:
			self.image_lock.release()
	def center_distance(self, position):
		screen_center= (self.vp.height/2, self.vp.width/2)
		distance_x = (position[0] - screen_center[1])
		distance_y = -1*(position[1] - screen_center[0])

		return distance_x, distance_y

	def main_process(self):
		key = None
		command = Twist()
		class error:
			x=0
			y=0

		self.image_lock.acquire()
		try:
			temp_image = self.image
		finally:
			self.image_lock.release()
		if temp_image is None:
			pass
		else:
			cv_image = np.asarray(ToOpenCV(temp_image))
			ball_center = self.vp.process_image(cv_image, (0, 170, 0), (200, 255, 255))
			if (ball_center == (-1,-1)):
				print 'looking for ball!'
				command.angular.z = -.5	

			else:
				#print "center: %f,%f" % (self.vp.height/2, self.vp.width/2)
				#print "ball: %f,%f" % (ball_center[0], ball_center[1])
				error.x, error.y = self.center_distance(ball_center)
				correction = self.controller.run(error)
				print correction.x, correction.y
				command.angular.z = -correction.x
				command.linear.z = correction.y
				if (abs(correction.x) < .01):
					self.go_forward = 1
				if (self.go_forward == 1):
					command.linear.x = 1
					print "going forward!"
			self.drone_pub.publish(command)
			#cv2.imshow("raw image", self.vp.images['src'])
			#cv2.imshow("HSV", self.vp.images['hsv'])
			#cv2.imshow("color filtered", self.vp.images['color_filter'])
			#cv2.imshow("contours", self.vp.images['contours'])
			cv2.imshow("center", self.vp.images['center'])
			#print ball_center
			key = cv2.waitKey(50) & 0xFF
		return key

def main():
	exit = 0
	instance = Lab5()

	#start flying!
	rospy.sleep(.5)
	instance.drone_takeoff_pub.publish(Empty())

	while(exit == 0):

		key = instance.main_process()
		if key == ord('q'):
			exit = 1
	cv2.destroyAllWindows()
	instance.drone_land_pub.publish(Empty())

if __name__ == '__main__':
	main()
		
