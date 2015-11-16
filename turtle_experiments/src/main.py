#!/usr/bin/python

import rospy
import time
import std_srvs.srv as services
from std_msgs.msg import Bool
from std_msgs.msg import Empty
from geometry_msgs.msg import Twist

class Main():
	def __init__(self):
		rospy.init_node('v_control_main')
		s = rospy.Service('/v_control/reset_world', services.Empty, self.reset_world)
		
		self.gazebo_reset = rospy.ServiceProxy('/gazebo/reset_world', services.Empty)

		self.takeoff_pub = rospy.Publisher('ardrone/takeoff', Empty)
		self.land_pub = rospy.Publisher('ardrone/land', Empty)
		self.soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
		self.enable_pub = rospy.Publisher('v_controller/move_enable', Bool)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)

		#wait a bit before publishing
		time.sleep(.5)
		self.soft_reset_pub.publish(Empty())
		time.sleep(.1)
		self.enable_pub.publish(Bool(1)) #enable the drone controller

		#register closing function
		rospy.on_shutdown(self.exit_handler)

		rospy.spin()

	def exit_handler(self):
		self.enable_pub.publish(Bool(0)) #disable the drone controller
		self.land_pub.publish(Empty())

	def reset_world(self, req):
		print "got reset request"
		#do all resetting here
		
		#disable the drone controller, publish an empty message
		self.enable_pub.publish(Bool(0)) #disable the drone controller
		rospy.sleep(.1)
		self.drone_pub.publish(Twist()) #publish command to stop drone
		
		#reset the world
		self.gazebo_reset()

		#takeoff the drone
		self.takeoff_pub.publish(Empty())

		#wait for the drone to stabilize
		rospy.sleep(5)
		self.soft_reset_pub.publish(Empty()) #reset all modules requiring a soft reset
		print "done!"
		return services.EmptyResponse()


if __name__ == "__main__":
	h = Main()
