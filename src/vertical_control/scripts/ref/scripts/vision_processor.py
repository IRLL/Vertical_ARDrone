import cv2
import cv2.cv as cv
import numpy as np
from copy import deepcopy

class Vision_Processor ():
	def __init__(self):
		self.images = dict()
		self.images['src'] = None
		self.height, self.width = (-1, -1)


	def process_orange (self, image):
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



		"""
		if lower_range != (100, 0, 0):
			cv2.imshow ("orange", self.images["Color_Filter"])
			cv2.waitKey(25)
		#cv2.drawContours (images["Src"], outputs["Contours"], -1, (0, 0, 0), thickness=5)

		outputs["Big_Contours"] = []

		if lower_range == (100, 0, 0):
			self.home_image = images["Color_Filter"]

		for i in outputs["Contours"]:
			area = cv.ContourArea (cv.fromarray (i))
			moment = cv2.moments (i)
			if area > 3330:
				images["Contours"] = images["Src"]
				cv2.drawContours(images["Contours"], outputs["Contours"], -1, (0,255,0), 3)
				cv2.imshow("contours", images["Contours"])
				outputs["Big_Contours"].append (i)
				center = (moment["m10"] / moment["m00"], moment["m01"] / moment["m00"])
				images["Center"] = images["Color_Filter"]
				int_center = ( int(center[0]), int(center[1]) ) 
				cv2.circle(images["Center"], int_center, 1, 0, 5)
				cv2.imshow("center", images["Center"])
				cv2.waitKey(25)
				return center
		return (-1,-1)
	"""
