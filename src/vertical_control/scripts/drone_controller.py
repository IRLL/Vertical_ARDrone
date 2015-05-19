#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
class Controller():
	UPDATE_SPEED = 10
	def __init__(self):
		self.agent_twist = Twist()
		self.stabilizer_twist = Twist()


		rospy.init_node('drone_controller', anonymous=False)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)
		self.takeoff_pub = rospy.Publisher('ardrone/takeoff', Empty) 
		self.land_pub = rospy.Publisher('ardrone/land', Empty) 
		self.agent_sub = rospy.Subscriber('v_controller/agent_cmd', Twist, self.rx_agent_callback)
		self.stabilizer_sub = rospy.Subscriber('v_controller/stabilizer_cmd', Twist, self.rx_stabilizer_callback)

		self.rate = rospy.Rate(self.UPDATE_SPEED)
	def rx_agent_callback(self, data):
		self.agent_twist = data

	def rx_stabilizer_callback(self, data):
		self.stabilizer_twist = data
	
	def run(self):
		cmd = self.stabilizer_twist
		cmd.linear.z = self.agent_twist.linear.z
		self.drone_pub.publish(cmd)

if __name__ == "__main__":
	controller = Controller()
	rospy.sleep(5)
	controller.takeoff_pub.publish(Empty())
	while not rospy.is_shutdown():
		controller.run()
		controller.rate.sleep()
	
	controller.land_pub.publish(Empty())
