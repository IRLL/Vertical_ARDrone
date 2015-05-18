#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
class Controller():
	def __init__(self):
		self.agent_twist = Twist()
		self.stabilizer_twist = Twist()

		self.rate = rospy.Rate(UPDATE_SPEED)

		self.agent_sub = rospy.Subscriber('v_controller/agent_cmd', Twist, self.rx_agent_callback)
		self.stabilizer_sub = rospy.Subscriber('v_controller/stabilizer_cmd', Twist, self.rx_stabilizer_callback)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)
		self.takeoff_pub = rospy.Publisher('ardrone/takeoff', Empty) 
		self.land_pub = rospy.Publisher('ardrone/land', Empty) 
		rospy.init_node('viewer', anonymous=False)

	def rx_agent_callback(self, data):
		self.agent_twist = data

	def rx_stabilizer_callback(self, data):
		self.stabilizer_twist = data
	
	def run():
		cmd = stabilizer_twist
		cmd.linear.z = self.agent_twist.linear.z
		self.drone_pub = self.drone_pub.publish(cmd)

if __name__ == "__main__":
	controller = Controller()

	controller.takeoff_pub.publish(Empty())
	while not rospy.is_shutdown():
		controller.run()
		controller.rate.sleep()
	
	controller.land_pub.publish(Empty())
