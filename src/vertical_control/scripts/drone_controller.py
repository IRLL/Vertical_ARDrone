#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import time
class Controller():
	UPDATE_SPEED = 5 
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
		self.agent_twist = self.smooth(data, self.agent_twist)

	def rx_stabilizer_callback(self, data):
		self.stabilizer_twist = self.smooth(data, self.stabilizer_twist)

	def smooth(self, new_cmd, prev_cmd):
		new_cmd.linear.x += prev_cmd.linear.x 
		new_cmd.linear.x /= 2

		new_cmd.linear.y += prev_cmd.linear.y 
		new_cmd.linear.y /= 2

		new_cmd.linear.z += prev_cmd.linear.z 
		new_cmd.linear.z /= 2

		new_cmd.angular.x += prev_cmd.angular.x 
		new_cmd.angular.x /= 2

		new_cmd.angular.y += prev_cmd.angular.y 
		new_cmd.angular.y /= 2

		new_cmd.angular.z += prev_cmd.angular.z 
		new_cmd.angular.z /= 2

		return new_cmd
	
	def run(self):
		cmd = self.stabilizer_twist
		cmd.linear.z = self.agent_twist.linear.z
		self.drone_pub.publish(cmd)

def land_drone():
	controller.land_pub.publish(Empty())

if __name__ == "__main__":
	controller = Controller()
	rospy.on_shutdown(land_drone)
	time.sleep(.5)
	controller.takeoff_pub.publish(Empty())
	while not rospy.is_shutdown():
		controller.run()
		controller.rate.sleep()
