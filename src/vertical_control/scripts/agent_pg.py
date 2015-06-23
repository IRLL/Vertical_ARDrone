#!/usr/bin/env python

# QUESTIONS:
# Why sigma is random?
# Should theta be set to zero?
# Do we need to complete the trajectories when ball is out of sight?
# Are we giving enough delay for an action to complete?
# How do you control the range of values for actions?

import rospy
import time
import sys
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelStateRequest

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from math import sqrt



np.set_printoptions(7, threshold=np.nan)

class Data():
    def __init__(self, n, traj_length):
        self.x = np.empty(shape=(n,traj_length+1))
        self.u = np.empty(shape=(1,traj_length))
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
		self.state_sub = rospy.Subscriber('v_controller/state', Float32, self.getState)
		self.visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_callback)
		self.threshold = rospy.get_param("v_controller/threshold")
		self.visible = 0
	
		# Parameter definition
		self._n = 1 # Number of states
		self._m = 1 # Number of inputs

		self._rate = .03 # Learning rate for gradient descent Original+0.75

		self._traj_length = 100 # Number of time steps to simulate in the cart-pole system
		self._rollouts = 50 # Number of trajectories for testing 50
		self._num_iterations = 30 # Number of learning episodes/iterations	30

		time.sleep(.1)
		#self._my0 = pi/6-2*pi/6*np.random.random((N,1)) # Initial state of the cart pole (between -60 and 50 deg)
		#self._my0 = np.array([[self._state]])
		#self._s0 = 0.0001*np.eye(self._n)

		# Parameters for Gaussian policies
		self._theta = np.random.rand(self._n*self._m,1) # Remember that the mean of the normal dis. is theta'*x
		self._sigma = np.random.rand(1,self._m) # Variance of the Gaussian dist.
		self._theta = np.array([[ 4.5509869]])
		self._sigma = np.array([[ 0.7623104]])

		self._data = [Data(self._n, self._traj_length) for i in range(self._rollouts)]

		self._mat = np.empty(shape=(100,self._n+1))
		self._vec = np.empty(shape=(100,1))
		self._r = np.empty(shape=(1,100))
		self._reward_per_rates_per_iteration = np.empty(shape=(1,200))
		
		print "Initial Theta: ", self._theta
		print "Sigma: ", self._sigma
		print "Learning Rate: ", self._rate

	def reset_sim(self, z):
		self.enable_controller.publish(Bool(0))
		self.takeoff_pub.publish(Empty())
		rospy.sleep(.1)
		a = SetModelStateRequest() 
		a.model_state.model_name = 'quadrotor'
		a.model_state.pose.position.z = z
		self.reset_pos(a)
		rospy.sleep(.5)
		self.soft_reset_pub.publish(Empty())


	def startEpisode(self):
		z = random.uniform(1.8, 4.2)
		self.reset_sim(z)
		
		
	def getState(self, data):
		self._state = data.data

	def visible_callback(self, visible):
		self.visible = visible.data

	def test(self, theta=None, sigma=None, traj_length=100000):
		self._traj_length = traj_length
		
		if theta != None and sigma != None:
			self._theta = theta
			self._sigma = sigma

		self._data = [Data(self._n, self._traj_length) for i in range(self._rollouts)]		
		
		'''
		print "Quadrotor hovers!!!"
		time.sleep(1)
		self.enable_controller.publish(Bool(0))
		self.takeoff_pub.publish(Empty())
		rospy.sleep(15)
		
		while self.visible == 0:
			if rospy.is_shutdown():
				sys.exit()
			print "Object not detected"
			
		self.soft_reset_pub.publish(Empty())
		print "Object detected"
		print "Test starting"
		'''
		self.startEpisode()

		# initial state
		self._data[0].x[:,0] = np.array([[-self._state/2.0]])	
		
		# Perform a trial of length L
		for steps in range(self._traj_length):
			if rospy.is_shutdown():
				sys.exit()

			# Use action learned
			action = np.dot(self._theta.conj().T, self._data[0].x[:,steps]).conj().T[0]
				
			action = round(action, 3)
			if self.visible == 0:
				action = 0.0

			command = Twist()
			command.linear.z = action
			self.action_pub.publish(command)
			rospy.sleep(.2)

			# Calculating the reward
			current_state = -self._state # negation to get correct states
			state = np.array([[current_state/2.0]])
			print "State: ", state
			print "Action: %.2f" %(action)

			self._data[0].x[:,steps+1] = state

		# end for steps...


	def train(self):
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
				self._data[trials].x[:,0] = np.array([[-self._state/2.0]])

				# Perform a trial of length L
				for steps in range(self._traj_length):
					if rospy.is_shutdown():
						sys.exit()

					# Draw an action from the policy
					action = np.random.multivariate_normal(np.dot(self._theta.conj().T, self._data[trials].x[:,steps]).conj().T, self._sigma)
						
					action = round(action, 3)
					if self.visible == 0:
						action = 0.0

					command = Twist()
					command.linear.z = action
					self.action_pub.publish(command)
					rospy.sleep(.2)

					self._data[trials].u[:,steps] = action

					# Calculating the reward (Remember: First term is the reward for accuracy, second is for control cost)
					#reward = -sqrt(np.dot(self._data[trials].x[:,steps].conj().T, self._data[trials].x[:,steps])) - \
					#		  sqrt(np.dot(self._data[trials].u[:,steps].conj().T, self._data[trials].u[:,steps]))
					current_state = -self._state # negation to get correct states
					#reward = current_state
					
					# Calculating the reward (Remember: First term is the reward for accuracy, second is for control cost)
					reward = -sqrt(current_state**2) - sqrt(0.0001*action**2)
					if reward == -0.0:
						reward = 0.0
					
					#if self.visible == 0:
					#	reward += -100
					
					#print "Reward: ", reward

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
					state = np.array([[current_state/2.0]])
					#print "State: ", state
					#print "Action: %.2f Reward: %f" %(action, reward)

					self._data[trials].x[:,steps+1] = state

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
			i = 0
			for Trials in range(np.max(np.shape(self._data))):
				self._mat[i,:] = np.append(np.zeros((1,self._n)),np.array([[1]]),axis=1)
				self._vec[i,0] = 0
				for Steps in range(np.max(np.shape(self._data[Trials].u))):
				    xx = self._data[Trials].x[:,Steps]
				    u  = self._data[Trials].u[:,Steps]

				    DlogPiDThetaNAC = np.dot((u-np.dot(self._theta.conj().T,xx)),np.array([xx]))/(self._sigma**2) # TODO:check
				    decayGamma = self._gamma**Steps
				    self._mat[i,:] = self._mat[i,:] + decayGamma*np.append(DlogPiDThetaNAC, np.array([[0]]), axis=1) # TODO:check
				    self._vec[i,0] = self._vec[i,0] + decayGamma*(self._data[Trials].r[:,Steps][0]-j) # TODO:check
				# end for Steps
				i = i+1
			#end for Trials

			# cond(Mat)
			nrm = np.diag(np.append(np.std(self._mat[:,0:self._n*self._m],ddof=1,axis=0),[1],axis=0))
			w = np.dot(nrm,np.dot(np.linalg.inv(np.dot(nrm,np.dot(self._mat.conj().T,np.dot(self._mat,nrm)))),np.dot(nrm,np.dot(self._mat.conj().T,self._vec)))) #TODO:check
			dJdtheta = w[0:(np.max(np.shape(w))-1)]; #TODO:check
			# the expected average return

			# Update of the parameter vector theta using gradient descent
			self._theta = self._theta + self._rate*dJdtheta;
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
			plt.draw()
			time.sleep(0.05)
		# end for k...
		plt.show(block=True)


	def reset(self,data):
		self.controller.reset()
		

if __name__ == "__main__":
	agent = Agent()
	time.sleep(.1)
	agent.train()
	# test learned policy
	'''
	agent.test(
			theta = np.array([[0.1572069], [3.5357239]]), 
			sigma = np.array([[0.3553431]]),
			traj_length = 100000
	)
	agent.test(
			theta = np.array([[ 0.9644904]]), 
			sigma = np.array([[ 0.0288127]]),
			traj_length = 100000
	)
	'''
	rospy.spin() 
