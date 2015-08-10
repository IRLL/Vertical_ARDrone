#!/usr/bin/env python

# QUESTIONS:
# Are we giving enough delay for an action to complete?
# How do you control the range of values for actions?
# [NEW] Why reward is computer before action is taken from old code?

import rospy
import time
import sys
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

from xbox_controller import xboxcontroller

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt



np.set_printoptions(7, threshold=np.nan)

class Data():
    def __init__(self, n, m, traj_length):
        self.x = np.empty(shape=(n,traj_length+1))
        #self.u = np.empty(shape=(m,traj_length))
        self.u = []
        for i in range(m):
            self.u.append(np.empty(shape=(1,traj_length)))
        self.r = np.empty(shape=(1,traj_length))

class Agent():

	def __init__(self):

		#gains = rospy.get_param("v_controller/gains/vertical")
		#self.controller = pi_controller(gains['p'], gains['i'])
		rospy.init_node('agent', anonymous=False)
		print "waiting for service"
		rospy.wait_for_service('/v_control/reset_world')
		print "done"
		self.reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		self.enable_controller = rospy.Publisher('v_controller/move_enable', Bool)
		self.soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
		self.takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.getState)
		self.visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_callback)
		self.threshold = rospy.get_param("v_controller/threshold")
		self.visible = 0

		# Parameter definition
		self._n = 2 # Number of states
		self._m = 2 # Number of inputs

		self._rate = .01 # Learning rate for gradient descent Original+0.75

		self._traj_length = 100 # Number of time steps to simulate in the cart-pole system 100
		self._rollouts = 50 # Number of trajectories for testing 50
		self._num_iterations = 50 # Number of learning episodes/iterations	30

		time.sleep(.1)
		#self._my0 = pi/6-2*pi/6*np.random.random((N,1)) # Initial state of the cart pole (between -60 and 50 deg)
		#self._my0 = np.array([[self._state]])
		#self._s0 = 0.0001*np.eye(self._n)

		# Parameters for Gaussian policies
		#self._theta = np.random.rand(self._n*self._m,1) # Remember that the mean of the normal dis. is theta'*x
		#self._sigma = np.random.rand(1,self._m) # Variance of the Gaussian dist.
		self._theta = []
		self._sigma = []
		for i in range(self._m):
			th = np.random.random((self._n,1))
			for j in range(self._n):
				#th[:,0][j] = th[:,0][j] if np.random.randint(2, size=1) == 1 else -th[:,0][j]
				th[:,0][j] = 0.0
			self._theta.append(th)
			self._sigma.append(np.random.random((1,1)))
		self._theta = [np.array([[ 0.0024157],[ 0.7418576]]), np.array([[ 0.8569861],[-0.003895 ]])]
		self._sigma = [np.array([[ 0.6094814]]), np.array([[ 0.8811964]])]

		# theta [array([[-0.0009055],
	     #[ 0.2194222]]), array([[ 0.0800124],
		#[-0.0012494]])]
		# sigma Sigma:  [array([[ 0.2622844]]), array([[ 0.1004269]])]



		self._data = [Data(self._n, self._m, self._traj_length) for i in range(self._rollouts)]

		self._mat = []
		self._vec = []
		for i in range(self._m):
			self._mat.append(np.empty(shape=(100,self._n+1)))
			self._vec.append(np.empty(shape=(100,1)))
		self._r = np.empty(shape=(1,100))
		self._reward_per_rates_per_iteration = np.empty(shape=(1,200))


	def reset_sim(self, x, y, angle):
		self.enable_controller.publish(Bool(0))
		self.takeoff_pub.publish(Empty())
		rospy.sleep(.1)
		a = SetModelStateRequest()
		a.model_state.model_name = 'quadrotor'
		a.model_state.pose.position.z = 2
		a.model_state.pose.position.x = 3 + x
		a.model_state.pose.position.y = 0 + y
		self.reset_pos(a)
		a.model_state.model_name = 'unit_sphere_4'
		a.model_state.pose.position.z = 0.052
		a.model_state.pose.position.x = 3
		a.model_state.pose.position.y = 0
		self.reset_pos(a)
		rospy.sleep(.5)
		self.soft_reset_pub.publish(Empty())


	def startEpisode(self):
		#x = random.uniform(-.23, .23)
		#y = random.uniform(-.23, .23)
		x = random.uniform(-.8, .8)
		y = random.uniform(-.23, .23)
		self.reset_sim(x,y,0)
		rospy.sleep(.5)


	def getState(self, data):
		self._state_x = data.data[0]
		self._state_y = data.data[1]

	def visible_callback(self, visible):
		self.visible = visible.data


	def test(self, theta=None, traj_length=100000):

		self._traj_length = traj_length

		if theta != None:
			self._theta = theta

		print "Initial Theta: ", self._theta

		self._data = [Data(self._n, self._m, self._traj_length) for i in range(self._rollouts)]
		"""
		#print "Quadrotor hovers!!!"
		print "starting human control"
		time.sleep(1)
		self.enable_controller.publish(Bool(0)) #disable modules like stabilizer
		"""
		self.takeoff_pub.publish(Empty())
		rospy.sleep(5)

		while self.visible == 0:
			if rospy.is_shutdown():
				sys.exit()
			print "Object not detected"

		self.soft_reset_pub.publish(Empty())
		print "Object detected"
		print "Test starting"
		"""
		gamecontroller = xboxcontroller()
		gamecontroller.run() #run xbox controller for human to get drone into position (blocking call)
		print "human control over"
		"""

		#self.soft_reset_pub.publish(Empty()) #re-enable modules like stabilizer

		# initial state
		self._data[0].x[:,0] = np.array([[-self._state_x/2.0, -self._state_y/2.0]])

		# Perform a trial of length L
		for steps in range(self._traj_length):
			if rospy.is_shutdown():
				sys.exit()

			# Draw an action from the policy
			action = [None] * self._m
			xx = self._data[0].x[:,steps]
			for j in range(self._m):
				th_p = self._theta[j].conj().T
				# Use action learned
				action[j] = np.dot(th_p, xx).conj().T[0]

				action[j] = action[j] * 0.05

				action[j] = round(action[j], 5)
				if self.visible == 0:
					action[j] = 0.0

				self._data[0].u[j][:,steps] = action[j]

			print "Action: ", action

			command = Twist()
			command.linear.x = action[0]
			command.linear.y = action[1]
			self.action_pub.publish(command)
			rospy.sleep(.2)

			# Get next state
			state = np.array([[-self._state_x/2.0, -self._state_y/2.0]])
			self._data[0].x[:,steps+1] = state

			# Calculating the reward (Remember: First term is the reward for accuracy, second is for control cost)
			u = np.array([action])
			u_p = u.conj().T
			reward = -sqrt(state[0][0]**2 + state[0][1]**2) - sqrt(np.dot(np.dot(u, np.eye(self._m) * 0.0998), u_p))

			self._data[0].r[:,steps] = [reward]

			#print "State: ", state
			#print "Action: %.2f Reward: %f" %(action, reward)

		# end for steps...


	def train(self):
		print "Initial Theta: ", self._theta
		print "Sigma: ", self._sigma
		print "Learning Rate: ", self._rate
		plt.ion()
		plt.show()

		for k in range(self._num_iterations): # Loop for learning
			print "______@ Iteration: ", k

			# In this section we obtain our data by simulating the cart-pole system
			for trials in range(self._rollouts):
				# reset
				#print "____________@ Trial #", (trials+1)
				self.startEpisode()

				# Draw the initial state
				#init_state = np.random.multivariate_normal(self._my0[:,0], self._s0, 1)
				#self._data[trials].x[:,0] = init_state[0,:]
				self._data[trials].x[:,0] = np.array([[-self._state_x/2.0, -self._state_y/2.0]])

				# Perform a trial of length L
				for steps in range(self._traj_length):
					if rospy.is_shutdown():
						sys.exit()

					# Draw an action from the policy
					action = [None] * self._m
					xx = self._data[trials].x[:,steps]
					for j in range(self._m):
						th_p = self._theta[j].conj().T
						sig = self._sigma[j]
						action[j] = np.random.multivariate_normal(np.dot(th_p, xx).conj().T, sig)

						action[j] = round(action[j], 5)
						#print "action ", j, ":", action[j]
						if self.visible == 0:# or (sqrt(xx[0][0]**2) <= .5 and sqrt(xx[0][1]**2) <= .5):
							action[j] = 0.0
						elif action[j] > 1.0: # saturate
							action[j] = 1.0
						elif action[j] < -1.0:
							action[j] = -1.0

						self._data[trials].u[j][:,steps] = action[j]

					#print "Action: ", action[0], " ", action[1]

					command = Twist()
					command.linear.x = action[0] + 0.05
					command.linear.y = action[1]
					self.action_pub.publish(command)
					rospy.sleep(.2)



					# Get next state
					state = np.array([[-self._state_x/2.0, -self._state_y/2.0]])
					self._data[trials].x[:,steps+1] = state

					# Calculating the reward (Remember: First term is the reward for accuracy, second is for control cost)
					#u = self._data[trials].u[:,steps]
					#u_p = self._data[trials].u[:,steps].conj().T
					u = np.array([action])
					u_p = u.conj().T
					#reward = -sqrt(state[0][0]**2 + state[0][1]**2) - sqrt(np.dot(np.dot(u, np.eye(self._m) * 0.01), u_p))
					reward = -sqrt(state[0][0]**2 + state[0][1]**2) - sqrt(np.dot(np.dot(u, np.eye(self._m) * 0.0998), u_p))
					#print "Action: ", action
					#print "Reward 1: ", -sqrt(state[0][0]**2 + state[0][1]**2)
					#print "Reward 2: ", -sqrt(np.dot(np.dot(u, np.eye(self._m) * 0.01), u_p))
					#print "The Reward: ", reward

					self._data[trials].r[:,steps] = [reward]


					#print "State: ", state
					#print "Action: %.2f Reward: %f" %(action, reward)



				# end for steps...
			# end for trials

			self._gamma = 0.9


			# This section calculates the derivative of the lower bound of the expected average return
			# using Natural Actor Critic
			j = 0

			if self._gamma == 1:
				for Trials in range(np.max(np.shape(self._data))):
				    j = j + np.sum(self._data[Trials].r)/np.max(np.shape(self._data[Trials].r))
				#end for Trials...
				j = j / np.max(np.shape(self._data))
			# end if gamma...

			# OBTAIN GRADIENTS
			for j in range(self._m):
				i = 0
				for Trials in range(np.max(np.shape(self._data))):
					self._mat[j][i,:] = np.append(np.zeros((1,self._n)),np.array([[1]]),axis=1)
					self._vec[j][i,0] = 0
					for Steps in range(np.max(np.shape(self._data[Trials].u[j]))):
						xx = self._data[Trials].x[:,Steps]
						u  = self._data[Trials].u[j][:,Steps]
						th_p = self._theta[j].conj().T
						sig = self._sigma[j]
						DlogPiDThetaNAC = np.dot((u-np.dot(th_p,xx)),np.array([xx]))/(sig**2) # TODO:check
						decayGamma = self._gamma**Steps
						self._mat[j][i,:] = self._mat[j][i,:] + decayGamma*np.append(DlogPiDThetaNAC, np.array([[0]]), axis=1) # TODO:check
						self._vec[j][i,0] = self._vec[j][i,0] + decayGamma*(self._data[Trials].r[:,Steps][0]-j) # TODO:check
					# end for Steps
					i = i+1
				#end for Trials

				# cond(Mat)
				mat = self._mat[j]
				mat_p = self._mat[j].conj().T
				vec = self._vec[j]
				nrm = np.diag(np.append(np.std(self._mat[j][:,0:self._n],ddof=1,axis=0),[1],axis=0))
				w = np.dot(nrm,np.dot(np.linalg.inv(np.dot(nrm,np.dot(mat_p,np.dot(mat,nrm)))),np.dot(nrm,np.dot(mat_p,vec)))) #TODO:check
				dJdtheta = w[0:(np.max(np.shape(w))-1)]; #TODO:check
				# the expected average return

				# Update of the parameter vector theta using gradient descent
				self._theta[j] = self._theta[j] + self._rate*dJdtheta;
			print "Theta: ", self._theta

			# Calculation of the average reward
			for z in range(np.shape(self._data)[0]): #TODO:check
				self._r[0,z] = np.sum(self._data[z].r) #TODO:check
			# end for z...

			if np.isnan(self._r.any()): #TODO:check
				print "System has NAN:", i
				print "..@ learning rate: ", self._rate #FIXME: DOUBLECHECK WITH MATLAB
				print "breaking this iteration ..."
				break
			# end if isnan...

			self._reward_per_rates_per_iteration[0,k] = np.mean(self._r) #TODO:check


			#############
			# Plotting the average reward to verify that the parameters of the
			# regulating controller are being learned

			print "Mean: ", np.mean(self._r)
			plt.scatter(k, np.mean(self._r), marker=u'x', c=np.random.random((2,3)), cmap=cm.jet)
			plt.grid()
			plt.draw()
			time.sleep(0.05)
		# end for k...
		plt.show(block=True)


	def reset(self,data):
		self.controller.reset()


if __name__ == "__main__":
	agent = Agent()
	time.sleep(.5)
	#agent.train()
	# test learned policy


	'''agent.test(
			theta = [np.array([[ 0.4004521],[ 1.6403012]]),
                         np.array([[ 1.0060473],[ 0.9353602]])],
			traj_length = 100000
	)'''

	''' action cost 0.1 very small actions
	agent.test(
			theta = [np.array([[-0.0010017],[ 0.1943709]]), np.array([[ 0.0700936],[-0.0014179]])],
			traj_length = 100000
	)
	'''

	'''# action cost 0.03 normal actions
	agent.test(
			theta = [np.array([[ 0.0012831],[ 1.2272073]]), np.array([[  1.1457246e+00],[  9.0374841e-04]])],
			traj_length = 100000
	)
	'''

	# action cost 0.08 normal actions
	'''agent.test(
			theta = [np.array([[  9.3849396e-04],[  1.3003893e+00]]), np.array([[ 0.2314742],[ 0.000232 ]])],
			traj_length = 100000
	)'''

	# action cost 0.0998 with states NOT normalized
	agent.test(
			theta = [np.array([[ 0.0019655],[ 0.9171811]]), np.array([[ 1.041202 ],[-0.0031267]])],
			traj_length = 100000
	)

	rospy.spin()
