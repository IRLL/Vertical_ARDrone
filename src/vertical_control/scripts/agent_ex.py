#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from pi_controller import pi_controller
from std_msgs.msg import Empty
from std_msgs.msg import Bool
import std_srvs.srv as services

class Agent():
	def __init__(self):
		gains = rospy.get_param("v_controller/gains/vertical")
		self.controller = pi_controller(gains['p'], gains['i']) 
		rospy.init_node('agent', anonymous=False)
		print "waiting for service"
		rospy.wait_for_service('/v_control/reset_world')
		print "done"
		self.reset_sim = rospy.ServiceProxy('/v_control/reset_world', services.Empty)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.run)
		self.visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_calback)
		self.visible = 1

	def run(self, data):

		if not self.visible:
			print "reseting world"
			self.reset_sim()
			return
		
		command = Twist()		

		state = data.data
		correction = self.controller.run(state)
		
		command.linear.z = -correction

		self.action_pub.publish(command)
	def visible_calback(self, visible):
		self.visible = visible.data

	def reset(self):
		self.controller.reset()
		


if __name__ == "__main__":
	agent = Agent()
	time.sleep(.1)
	agent.reset_sim()
	agent.reset()
	rospy.spin() 