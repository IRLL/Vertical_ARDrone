#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from pi_controller import pi_controller

class Agent():
	def __init__(self, proportional=.1, integral=0.001):
		self.controller = pi_controller(proportional, integral) 
		rospy.init_node('agent', anonymous=False)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.run)

	def run(self, data):
		command = Twist()		

		state = data.data
		correction = self.controller.run(state)
		
		command.linear.z = -correction

		self.action_pub.publish(command)
		


if __name__ == "__main__":
	agent = Agent()
	rospy.spin() 
