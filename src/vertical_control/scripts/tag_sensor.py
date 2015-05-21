#!/usr/bin/python
import rospy
from sensor_msgs.msg import Image
from threading import Lock
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from ardrone_autonomy.msg import Navdata

class Tag_sensor():
	#class constants
	UPDATE_SPEED = 10 #process at 10 hz

	def __init__(self):
		self.nav_lock = Lock()
		self.latest_data = None
 
		rospy.init_node('tag_sensors', anonymous=False)
		
		self.learner_pub = rospy.Publisher('v_controller/state', Float32)
		self.controller_pub = rospy.Publisher('v_controller/control_state', Float32MultiArray)
		self.nav_sub = rospy.Subscriber('/ardrone/navdata', Navdata, self.receive_nav_callback)
		
		self.rate = rospy.Rate(self.UPDATE_SPEED)

		

	def receive_nav_callback(self, data):
		self.nav_lock.acquire()
		try:
			self.latest_data = data 
		finally:
			self.nav_lock.release()


	def processing_function(self):
		print "waiting for nav data to come in..."
		while (not rospy.is_shutdown()) and self.latest_data is None:
			rospy.sleep(.5)		
		print "done!"


		while not rospy.is_shutdown(): #do image processing here
			#get the latest image
			self.nav_lock.acquire()
			try:
				data = self.latest_data
			finally:
				self.nav_lock.release()	

			if data.tags_count > 0: #if tags are visible
				x, y, distance = self.read_tag(data)
				x, y = self.rescale(x, y)
				print x, y, distance
			else:
				print "can't see any tags"
				x, y, distance = 0.0, 0.0, 1.0 #default_values
				

			self.learner_pub.publish(y)		
			self.controller_pub.publish(None, [x, distance])
			self.rate.sleep()

	def read_tag(self, navdata):
		return navdata.tags_xc[0], navdata.tags_yc[0], navdata.tags_distance[0]/100

	def rescale(self, x, y):
		width_div = 1000/2
		height_div = 1000/2
		newx = float(x - width_div)/width_div			
		newy = float(y - height_div)/height_div			
		return newx,newy

	
if __name__ == '__main__':
	tag_sensor = Tag_sensor()	
	tag_sensor.processing_function()
