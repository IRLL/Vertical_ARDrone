#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from pi_controller import pi_controller
from std_msgs.msg import Empty

class Agent():
	def __init__(self):
		gains = rospy.get_param("v_controller/gains/vertical")
		self.controller = pi_controller(gains['p'], gains['i']) 
		rospy.init_node('agent', anonymous=False)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.run)
		self.reset_sub = rospy.Subscriber('v_controller/soft_reset', Empty, self.reset)

	def run(self, data):
		command = Twist()		

		state = data.data
		correction = self.controller.run(state)
		
		command.linear.z = -correction

		self.action_pub.publish(command)

	def reset(self,data):
		self.controller.reset()
		


if __name__ == "__main__":
	agent = Agent()
	rospy.spin() 
