import cv2
import cv2.cv as cv
import numpy as np
from copy import deepcopy
from sensor_msgs.msg import Image
from threading import Lock, Thread

class Vision_Processor ():
	#class constants
	UPDATE_SPEED = 10 #process at 10 hz
	AREA_DIST_CONST = 1

	def __init__(self):
		self.video_sub = rospy.Subscriber('/ardrone/image_raw', Image, receive_image_callback)
		self.learner_pub = rospy.Publisher('v_controller/state', Float32)
		self.controller_pub = rospy.Publisher('v_controller/control_state', Float32MultiArray)
		rospy.init_node('simulated_sensors', anonymous=False)

		self.rate = rospy.Rate(UPDATE_SPEED)
		self.latest_image = None

		self.image_lock = Lock()

		self.images = dict()
		self.height, self.width = (-1, -1)

		self.processing_thread = Thread(target=processing_function)
		

	def receive_image_callback(self, data):
		self.image_lock.acquire()
		try:
			self.latest_image = data 
		finally:
			self.image_lock.release()


	def processing_function(self):
		print "waiting for images to come in..."
		while (not rospi.is_shutdown()) and self.latest_image is None:
			pass		
		print "done!"

		self.get_image_size()

		while not rospy.is_shutdown(): #do image processing here
			#get the latest image
			self.image_lock_acquire()
			try:
				image = self.latest_image
			finally:
				self.image_lock_release()	

			image = np.asarray(ToOpenCV(image))

			x, y, distance = self.locate_orange(image)
			x, y = self.rescale(x, y)

			self.learner_pub.publish(y)		
			#need to add publisher for other info
			self.rate.sleep()
		
	def find_orange (self, image):
		#parameters for locating orange
		lower_target_range = (0, 170, 0)
		upper_target_range = (200, 255, 255)	

		x, y, distance = self.process_image (image, lower_target_range, upper_target_range)	 
		return x, y, distance

	def get_image_size(self):
		self.height, self.width, _ = self.latest_image.shape

	def rescale(x, y):
		width_div = float(self.width)/2
		height_div = float(self.height)/2
		newx = float(x - width_div)/width_div			
		newy = float(y - height_div)/height_div			
		return newx,newy

	def process_image (self, image, lower_range, upper_range):
		outputs = dict()
			
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

		#assume only one contour
		moment = cv2.moments(contours[0])
		area = moment["m00"] #cv2.contourArea(contours[0])
		
		xpos = moment["m10"] / moment["m00"]
		ypos = moment["m01"] / moment["m00"]
		distance = area/AREA_DIST_CONST

		return xpos, ypos, distance 
