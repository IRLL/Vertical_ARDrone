#!/usr/bin/env python

import rospy
import time
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from pi_controller import pi_controller
from std_msgs.msg import Empty
from std_msgs.msg import Bool
import std_srvs.srv as services
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

class Agent():
	def __init__(self):
		vgains = rospy.get_param("v_controller/gains/vertical")
		hgains = rospy.get_param("v_controller/gains/horizontal")
		self.hcontroller = pi_controller(hgains['p'] , hgains['i'] ) 
		self.vcontroller = pi_controller(vgains['p'] , vgains['i'] ) 
		rospy.init_node('agent', anonymous=False)
		print "waiting for service"
		rospy.wait_for_service('/v_control/reset_world')
		rospy.wait_for_service('/gazebo/set_model_state')
		print "done"
		#self.reset_sim = rospy.ServiceProxy('/v_control/reset_world', services.Empty)
		self.reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
		self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
		self.enable_controller = rospy.Publisher('v_controller/move_enable', Bool)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.run)
		self.visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_calback)
		self.visible = 1

	def run(self, data):
		"""
		if not self.visible:
			print "reseting world"
			self.reset_sim()
			return
		"""
		
		command = Twist()		

		herror = data.data[0]
		verror = data.data[1]

		hcorrection = self.hcontroller.run(herror)
		vcorrection = self.vcontroller.run(verror)
		
		command.linear.x = -vcorrection
		command.linear.y = -hcorrection

		self.action_pub.publish(command)
	def visible_calback(self, visible):
		self.visible = visible.data

	def reset(self):
		self.enable_controller.publish(Bool(0))
		self.takeoff_pub.publish(Empty())
		rospy.sleep(1)
		self.soft_reset_pub.publish(Empty())
		


if __name__ == "__main__":

	agent = Agent()
	time.sleep(.5)
	agent.takeoff_pub.publish(Empty())
	a = SetModelStateRequest() 
	a.model_state.model_name = 'quadrotor'
	a.model_state.pose.position.z = 3
	a.model_state.pose.position.x = 3
	agent.reset_pos(a)

	time.sleep(.5)
	agent.reset()
	rospy.spin() 
