#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from pi_controller import pi_controller_2d, xy
from std_msgs.msg import Empty

class Stabilizer():
	def __init__(self):
		self.hover_distance = rospy.get_param("v_controller/hover_distance")
		h_gains = rospy.get_param("v_controller/gains/horizontal")
		d_gains = rospy.get_param("v_controller/gains/distance")
		proportional = [h_gains['p'], d_gains['p']]
		integral = [h_gains['i'], d_gains['i']]
		self.controller = pi_controller_2d(proportional, integral) 
		rospy.init_node('stabilizer', anonymous=False)
		self.error_sub = rospy.Subscriber('v_controller/control_state', Float32MultiArray, self.run)
		self.reset_sub = rospy.Subscriber('v_controller/soft_reset', Empty, self.reset)
		self.stabilizer_pub = rospy.Publisher('v_controller/stabilizer_cmd', Twist)

	def run(self, data):
		command = Twist()		

		x = data.data[0]
		y = data.data[1] - self.hover_distance
		error = xy(x, y)
		correction = self.controller.run(error)
		
		command.angular.z = -correction.x

		command.linear.x = correction.y
		self.stabilizer_pub.publish(command)
	def reset(self,data):
		self.controller.reset()
		


if __name__ == "__main__":
	stabilizer = Stabilizer()
	rospy.spin() 
