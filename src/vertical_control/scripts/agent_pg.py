#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from pi_controller import pi_controller
from std_msgs.msg import Empty

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
import time

plt.ion()
plt.show()

np.set_printoptions(7, threshold=np.nan)

class Data():
    def __init__(self, n, traj_length):
        self.x = np.empty(shape=(n,traj_length+1))
        self.u = np.empty(shape=(1,traj_length))
        self.r = np.empty(shape=(1,traj_length))

class Agent():

	def __init__(self):

		gains = rospy.get_param("v_controller/gains/vertical")
		#self.controller = pi_controller(gains['p'], gains['i']) 
		rospy.init_node('agent', anonymous=False)
		self.action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.getState)
		self.reset_sub = rospy.Subscriber('v_controller/soft_reset', Empty, self.reset)

		# Parameter definition
		self._n = 1 # Number of states
		self._m = 1 # Number of inputs

		self._rate = .75 # Learning rate for gradient descent

		self._traj_length = 50 # Number of time steps to simulate in the cart-pole system
		self._rollouts = 100 # Number of trajectories for testing
		self._num_iterations = 100 # Number of learning episodes/iterations	

	def startEpisode(self):
		#self._my0 = pi/6-2*pi/6*np.random.random((N,1)) # Initial state of the cart pole (between -60 and 50 deg)
		self._my0 = np.array([[self._state]])
		self._s0 = 0.0001*np.eye(self._n)

		# Parameters for Gaussian policies
		self._theta = np.random.random((self._n*self._m,1)) # Remember that the mean of the normal dis. is theta'*x
		self._sigma = np.random.random((1,self._m)) # Variance of the Gaussian dist.

		self._data = [Data(self._n, self._traj_length) for i in range(self._rollouts)]

		self._mat = np.empty(shape=(100,self._n+1))
		self._vec = np.empty(shape=(100,1))
		self._r = np.empty(shape=(1,100))
		self._reward_per_rates_per_iteration = np.empty(shape=(1,200))

	def run(self, data):
		command = Twist()		

		state = data.data
		#correction = self.controller.run(state)
		
		command.linear.z = -correction

		self.action_pub.publish(command)

	def getState(self, data):
		self._state = data.data

	def train(self):
		for k in range(self._num_iterations): # Loop for learning
			print "______@ Iteration: ", k

			# In this section we obtain our data by simulating the cart-pole system
			for trials in range(self._rollouts):
				# reset

				# Draw the initial state
				init_state = np.random.multivariate_normal(self._my0[:,0], self._s0, 1)
				self._data[trials].x[:,0] = init_state[0,:]

				# Perform a trial of length L
				for steps in range(self._traj_length):

					# Draw an action from the policy
					action = np.random.multivariate_normal(np.dot(self._theta.conj().T, self._data[trials].x[:,steps]).conj().T, self._sigma)
					print action
					rospy.sleep(.1)
					self._data[trials].u[:,steps] = action


					# Calculating the reward (Remember: First term is the reward for accuracy, second is for control cost)
					reward = -sqrt(np.dot(self._data[trials].x[:,steps].conj().T, self._data[trials].x[:,steps])) - \
							  sqrt(np.dot(self._data[trials].u[:,steps].conj().T, self._data[trials].u[:,steps]))
					self._data[trials].r[:,steps] = [reward]

					# Draw next state from envrionment
					# This is the solution of the typical linear model dx/dt = Ax + bu
					#commonD = I*(mc + mp) + mc*mp*l**2
					#A = np.array([ [0,                          1,                          0, 0],
					#               [0, -((I + mp*l**2)*d)/commonD,               mp**2*g*l**2, 0],
					#               [0,                          0,                          0, 1],
					#               [0,          -(mp*l*d)/commonD, (mp*g*l*(mc + mp))/commonD, 0] ])

					#b = np.array([ [0], [(I + mp*l**2)/commonD], [0], [(mp*l)/commonD] ])

					#xnDum = np.dot(A,data[trials].x[:,steps]) + np.dot(b,data[trials].u[:,steps])

					#state = data[trials].x[:,steps] + dt*xnDum
					state = np.array([[self._state]])

					self._data[trials].x[:,steps+1] = state

				# end for steps...
			# end for trials
			

	def reset(self,data):
		self.controller.reset()
		

if __name__ == "__main__":
	agent = Agent()
	rospy.sleep(.1)
	agent.startEpisode()
	agent.train()
	rospy.spin() 
