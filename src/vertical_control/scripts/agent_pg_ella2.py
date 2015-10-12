#!/usr/bin/env python

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
from math import sqrt, exp, isinf

from createSys import createSys
from constructPolicies import constructPolicies
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
from drawAction import drawAction
from initPGELLA import initPGELLA
from learnPGELLA import learnPGELLA
from testPGELLA import testPGELLA
from structs import Data

import cPickle as pickle


class Agent():

    def __init__(self):
        rospy.init_node('agent_pg_ella', anonymous=False)

        self._reset_pos = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._enable_controller = rospy.Publisher('v_controller/move_enable', Bool)
        self._soft_reset_pub = rospy.Publisher('v_controller/soft_reset', Empty)
        self._takeoff_pub = rospy.Publisher('/ardrone/takeoff', Empty)
        self._action_pub = rospy.Publisher('v_controller/agent_cmd', Twist)
        self._state_sub = rospy.Subscriber('v_controller/state', Float32MultiArray, self.getState)
        self._visible_sub = rospy.Subscriber('v_controller/visible', Bool, self.visible_callback)
        self._threshold = rospy.get_param("v_controller/threshold")


    def start_pg(self, nSystems, poliType, baseLearner, gamma,
                 trajLength, numRollouts, numIterations):
        # Creating Tasks
        self.Tasks = createSys(nSystems, poliType, baseLearner, gamma)

        # Constructing policies
        self.Policies = constructPolicies(self.Tasks)

        # Calculating theta
        self.Policies = self.calcThetaStar(self.Tasks, self.Policies,
                                           learningRate, trajLength,
                                           numRollouts, numIterations)


    def getState(self, data):
        self._state_x = data.data[0]
        self._state_y = data.data[1]
        self._state_z = data.data[2] - 1.5 #dc bias so that it floats at 1.5 meters
        #print "x:%s y:%s z:%s" %(_state_x, _state_y, _state_z)

    def visible_callback(self, visible):
        self._visible = visible.data

    def reset_sim(self, x, y, z=0, angle=0):
        self._enable_controller.publish(Bool(0))
        self._takeoff_pub.publish(Empty())
        rospy.sleep(.5)
        a = SetModelStateRequest()
        a.model_state.model_name = 'quadrotor'
        a.model_state.pose.position.z = 2 + z
        a.model_state.pose.position.x = 3 + x
        a.model_state.pose.position.y = 0 + y
        self._reset_pos(a)
        a.model_state.model_name = 'unit_sphere_4'
        a.model_state.pose.position.z = 0.052
        a.model_state.pose.position.x = 3
        a.model_state.pose.position.y = 0
        self._reset_pos(a)
        rospy.sleep(.5)
        self._soft_reset_pub.publish(Empty())

    def startEpisode(self):
        #x = random.uniform(-.23, .23)
        #y = random.uniform(-.23, .23)
        x = random.uniform(-.8, .8)
        y = random.uniform(-.23, .23)
        self.reset_sim(x,y,0)
        rospy.sleep(1)


    def obtainData(self, policy, L, H, param):
        print "waiting for service"
        rospy.wait_for_service('/v_control/reset_world')
        print "done"

        N = param.param.N
        M = param.param.M
        data = [Data(N, M, L) for i in range(H)]

        time.sleep(.5)


        # Based on Jan Peters; codes.
        # Perform H

        for trials in range(H):
            self.startEpisode()

            # Save associated policy
            data[trials].policy = policy

            # Draw the first state
            data[trials].x[:,0] = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/1.5]])

            # Perform a trial of length L
            for steps in range(L):
                if rospy.is_shutdown():
                    sys.exit()
                # Draw an action from the policy
                xx = data[trials].x[:,steps]
                print "current state: ", xx
                data[trials].u[:,steps] = drawAction(policy, xx, param)
                print "action: ", data[trials].u[:,steps]


                if not self._visible:
                    data[trials].u[:,steps] *= 0.0
                else:
                    for j in range(M):
                        data[trials].u[:,steps][j] += param.param.disturbance[j]
                        data[trials].u[:,steps][j] = round(data[trials].u[:,steps][j], 5)

                        if data[trials].u[:,steps][j] > 1.0: # saturate
                            data[trials].u[:,steps][j] = 1.0
                        elif data[trials].u[:,steps][j] < -1.0:
                            data[trials].u[:,steps][j] = -1.0

                if self._visible and self._state_z > 2.8:
                    data[trials].u[:,steps][2] = -0.1
                '''
                if self._visible and xx[2] >= .9:
                    data[trials].u[:,steps][2] = -0.1
                elif self._visible and xx[2] <= -.7:
                    data[trials].u[:,steps][2] = 0.1
                '''


                command = Twist()
                command.linear.x = data[trials].u[:,steps][0]
                command.linear.y = data[trials].u[:,steps][1]
                command.linear.z = data[trials].u[:,steps][2]
                self._action_pub.publish(command)
                rospy.sleep(.2)

                print "action: ", data[trials].u[:,steps]

                # Draw next state from environment
                #data[trials].x[:,steps+1] = drawNextState(data[trials].x[:,steps], data[trials].u[:,steps], param, i)
                state = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/1.5]])
                #state = np.array([[-_state_x/4.0, -_state_y/3.0]])
                data[trials].x[:,steps+1] = state

                # Obtain the reward from the
                #data[trials].r[0][steps] = rewardFnc(data[trials].x[:,steps], data[trials].u[:,steps]) # FIXME: Not similar to original

                u = data[trials].u[:,steps]
                u_p = u.conj().T
                #imp_eye_state = np.zeros((3,3))
                #imp_eye_act = np.zeros((3,3))
                #np.fill_diagonal(imp_eye_state, [10, 10, 5])
                #np.fill_diagonal(imp_eye_act, [5, 5, 2.5])
                #reward = -sqrt(np.dot(np.dot(state, imp_eye_state), state.conj().T)) - \
                #          sqrt(np.dot(np.dot(u, imp_eye_act), u_p))
                reward = -sqrt(np.dot(np.dot(state, np.eye(N) * 10), state.conj().T)) - \
                          sqrt(np.dot(np.dot(u, np.eye(M) * 5), u_p))
                print "reward: ", reward

                if isinf(reward):
                    print "Error: INFINITY"
                    sys.exit(1)

                data[trials].r[0][steps] = reward

        return data


    def calcThetaStar(self, Params, Policies, rates,
                      trajlength, rollouts, numIterations):

        plt.ion()

        nSystems = np.shape(Params)[0]
        r = np.empty(shape=(1, rollouts))

        for i in range(nSystems):
            # TODO: Clear screen
            print "@ Task: ", i
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.grid()

            policy = Policies[i].policy # Resetting policy IMP

            for k in range(numIterations):
                print "@ Iteration: ", k
                print "    Initial Theta: ", policy.theta
                print "    Sigma: ", policy.sigma
                print "    Learning Rate: ", rates
                data = self.obtainData(policy, trajlength, rollouts, Params[i])
                dJdtheta = None
                if Params[i].param.baseLearner == "REINFORCE":
                    dJdtheta = episodicREINFORCE(policy, data, Params[i])
                else:
                    dJdtheta = episodicNaturalActorCritic(policy, data, Params[i]) # TODO: won't use but should finish

                policy.theta = policy.theta + rates*dJdtheta.reshape(9, 1)

                for z in range(rollouts):
                    r[0, z] = np.sum(data[z].r)

                if np.isnan(r.any()):
                    print "System has NAN"
                    print "exiting..."
                    import sys
                    sys.exit(1)

                print "Mean: ", np.mean(r)
                time.sleep(1)
                ax.scatter(k, np.mean(r), marker=u'x', c=np.random.random((2,3)), cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)

            Policies[i].policy = policy # Calculating theta
        plt.show(block=True)

        return Policies


if __name__ == "__main__":
    nSystems = 2  # Integer number of tasks
    learningRate = .01  # Learning rate for stochastic gradient descent


    # Parameters for policy
    poliType = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                        # 'Gauss' => Gaussian Policy
    baseLearner = 'NAC'  # Base Learner
                               # 'REINFORCE' => Episodic REINFORCE
                               # 'NAC' => Episodic Natural Actor Critic
    gamma = 0.9  # Discount factor gamma

    trajLength = 100 # Number of time steps to simulate in the cart-pole system
    numRollouts = 10 # Number of trajectories for testing
    numIterations = 200 # Number of learning episodes/iterations # 200

    agent = Agent()
    time.sleep(.5)

    agent.start_pg(nSystems, poliType, baseLearner, gamma,
                   trajLength, numRollouts, numIterations)
    rospy.spin()