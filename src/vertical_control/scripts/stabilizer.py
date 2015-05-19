#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from pi_controller import pi_controller_2d, xy

class Stabilizer():
	def __init__(self, proportional=.1, integral=0.001):
		self.controller = pi_controller_2d(proportional, integral) 
		rospy.init_node('stabilizer', anonymous=False)
		self.error_sub = rospy.Subscriber('v_controller/control_state', Float32MultiArray, self.run)
		self.stabilizer_pub = rospy.Publisher('v_controller/stabilizer_cmd', Twist)

	def run(self, data):
		command = Twist()		

		x = data.data[0]
		y = (7000 - data.data[1]) / 1000
		error = xy(x, y)
		correction = self.controller.run(error)
		
		command.angular.z = -correction.x

		command.linear.x = correction.y
		self.stabilizer_pub.publish(command)
		


if __name__ == "__main__":
	stabilizer = Stabilizer()
	rospy.spin() 
