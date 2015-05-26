#!/usr/bin/python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import std_srvs.srv as services
import time
class Controller():
	UPDATE_SPEED = 5 
	def __init__(self):
		self.agent_twist = Twist()
		self.stabilizer_twist = Twist()
		self.gazebo_reset = rospy.ServiceProxy('/gazebo/reset_world', services.Empty)

		rospy.init_node('drone_controller', anonymous=False)
		self.drone_pub = rospy.Publisher('cmd_vel', Twist)
		self.takeoff_pub = rospy.Publisher('ardrone/takeoff', Empty) 
		self.reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
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

	def reset(self):
		#command the drone to stop
		controller.drone_pub.publish(Twist())
		#reset all objects in gazebo
		self.gazebo_reset()
		#make sure the drone is flying
		controller.takeoff_pub.publish(Empty())
		#send the stop command again
		controller.drone_pub.publish(Twist())

		#sleep for a bit
		rospy.sleep(1)
		
		#command all other modules to reset themselves
		controller.reset_pub.publish(Empty())
		

def land_drone():
	controller.land_pub.publish(Empty())

if __name__ == "__main__":
	controller = Controller()
	rospy.on_shutdown(land_drone)
	time.sleep(.5)
	controller.reset()
	rospy.sleep(0.5)

	while not rospy.is_shutdown():
		controller.run()
		controller.rate.sleep()
