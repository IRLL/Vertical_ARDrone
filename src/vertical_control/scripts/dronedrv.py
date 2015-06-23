from geometry_msgs.msg import Twist
import rospy
from std_msgs.msg import Empty

class dronedrv():
	def __init__(self):
		self.cmd_pub = rospy.Publisher('cmd_vel', Twist)
		self.takeoff_pub = rospy.Publisher('ardrone/takeoff', Empty)
		self.land_pub = rospy.Publisher('ardrone/land', Empty)
		rospy.init_node('dronedrv', anonymous=True)
		self.kill_pub = rospy.Publisher('ardrone/reset', Empty)

	def land(self):
		self.land_pub.publish(Empty())	
	def takeoff(self):
		self.takeoff_pub.publish(Empty())	
	def kill(self):
		self.kill_pub.publish(Empty())	

	def cmd(self, twist):
		self.cmd_pub.publish(twist)

