import cv2
import cv2.cv as cv
import numpy as np
from copy import deepcopy
from sensor_msgs.msg import Image
from threading import Lock, Thread


UPDATE_SPEED = 10 #process at 10 hz

class Vision_Processor ():
	def __init__(self):
		self.video_sub = rospy.Subscriber('/ardrone/image_raw', Image, receive_image_callback)
		self.learner_pub = rospy.Publisher('v_controller/state', Float32)
		self.controller_pub = rospy.Publisher('v_controller/control_state', Float32MultiArray)
		rospy.init_node('simulated_sensors', anonymous=False)

		self.rate = rospy.Rate(UPDATE_SPEED)
		self.current_image = None

		self.image_lock = Lock()

		

		self.processing_thread = Thread(target=processing_function)
		self.images = dict()
		self.images['src'] = None
		self.height, self.width = (-1, -1)



	def receive_image_callback(self, data):
		self.image_lock.acquire()
		try:
			self.current_image = data
		finally:
			self.image_lock.release()


	def processing_function(self):
		#do image processing here
		while 1:
			
				
			self.rate.sleep()
		
	def find_orange (self, image):
		self.image = image
		lower_target_range = (0, 170, 0)
		upper_target_range = (200, 255, 255)	
		self.center_target = self.process_image (image, lower_target_range, upper_target_range)	 
		return self.center_target

	def get_image_size(self):
		#wait till we get an image
		if self.images['src'] is None:
			self.height, self.width = (-1, -1)
		else:
			self.height, self.width, _ = self.images['src'].shape

	def process_image (self, image, lower_range, upper_range):
		outputs = dict()
		if self.height == -1:
			self.get_image_size()
			
		#create copy of the image
		self.images["src"] = image	 #deepcopy (image) 
	
		#give it a nice blur
		#images["MFilter"] = cv2.medianBlur (images["Src"], 9)

		#convert to HSV
		self.images["hsv"] = cv2.cvtColor (self.images["src"], cv2.COLOR_BGR2HSV)

		#filter for color
		self.images["color_filter"] = cv2.inRange (self.images["hsv"], lower_range, upper_range)

		#write contours
		contours, heirarchy = cv2.findContours (deepcopy (self.images["color_filter"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


		self.images['contours'] = deepcopy(self.images['src'])
		cv2.drawContours(self.images['contours'], contours, -1, (0,0,0), thickness=2)
		
		self.images['center'] = deepcopy(self.images['src']) 
		cv2.circle(self.images['center'], (self.width/2, self.height/2), 1, 0, 5)		

		for i in contours:
			moment = cv2.moments(i)
			center = (moment["m10"] / moment["m00"], moment["m01"] / moment["m00"])
			int_center = ( int(center[0]), int(center[1]) )
			cv2.circle(self.images['center'], int_center, 1, 0, 5)
			return center
		
		return (-1,-1)

