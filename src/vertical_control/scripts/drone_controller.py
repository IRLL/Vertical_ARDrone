#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from std_msgs.msg import Bool
import std_srvs.srv as services
import time

class Controller():
	UPDATE_SPEED = 40
	def __init__(self):
		self.agent_twist = Twist()
		self.stabilizer_twist = Twist()
		self.enabled = 1
		self.gazebo_reset = rospy.ServiceProxy('/gazebo/reset_world', services.Empty)

		rospy.init_node('drone_controller', anonymous=False)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)
		self.reset_sub = rospy.Subscriber('v_controller/soft_reset', Empty, self.reset)
		self.enable_sub = rospy.Subscriber('v_controller/move_enable', Bool, self.move_en_callback)
		self.agent_sub = rospy.Subscriber('v_controller/agent_cmd', Twist, self.rx_agent_callback)
		self.stabilizer_sub = rospy.Subscriber('v_controller/stabilizer_cmd', Twist, self.rx_stabilizer_callback)

		self.rate = rospy.Rate(self.UPDATE_SPEED)

	def rx_agent_callback(self, data):
		self.agent_twist = data

	def rx_stabilizer_callback(self, data):
		self.stabilizer_twist = data

	def run(self):
		if self.enabled:
			cmd = self.agent_twist
			cmd.linear.z = self.stabilizer_twist.linear.z
			self.drone_pub.publish(cmd)

	def move_en_callback(self, data):
		self.enabled = data.data
		if not self.enabled:
			self.drone_pub.publish(Twist())

	def reset(self, empty):
		self.drone_pub.publish(Twist())
		self.enabled = 1	

def land_drone():
	controller.land_pub.publish(Empty())

if __name__ == "__main__":
	controller = Controller()

	while not rospy.is_shutdown():
		controller.run()
		controller.rate.sleep()
