#!/usr/bin/env python

import rospy
import time
import sys
from datetime import datetime
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

    def startPg(self, nSystems=1 , learningRate=0.35, poliType='Gauss', baseLearner='NAC', gamma=0.9,
                 trajLength=100, numRollouts=40, numIterations=150,
                 task_file='task.p', policy_file='policy.p', isLoadPolicy=False):
        if not isLoadPolicy:
            # Creating Tasks
            self.Tasks = createSys(nSystems, poliType, baseLearner, gamma)

            # Constructing policies
            self.Policies = constructPolicies(self.Tasks)

            # Calculating theta
            self.Policies = self.calcThetaStar(self.Tasks, self.Policies,
                                               learningRate, trajLength,
                                               numRollouts, numIterations)

            self.savePolicies(task_file, policy_file)
        else:
            self.loadPolicies(task_file, policy_file)


    def savePolicies(self, task_file, policy_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.Tasks, open(task_file, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Policies, open(policy_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadPolicies(self, task_file, policy_file):
        # Load PG policies and tasks to a file
        self.Tasks = pickle.load(open(task_file, 'rb'))
        self.Policies = pickle.load(open(policy_file, 'rb'))

    def getState(self, data):
        self._state_x = data.data[0]
        self._state_y = data.data[1]
        self._state_z = data.data[2] - 1.5 #dc bias so that it floats at 1.5 meters
        #print "x:%s y:%s z:%s" %(_state_x, _state_y, _state_z)

    def visible_callback(self, visible):
        self._visible = visible.data

    def resetSim(self, x, y, z=0, angle=0):
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
        self.resetSim(x,y,0)
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
            data[trials].x[:,0] = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/3.0]])

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
                state = np.array([[-self._state_x/4.0, -self._state_y/3.0, self._state_z/3.0]])
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

        tasks_time = [0] * nSystems

        for i in range(nSystems):
            # TODO: Clear screen
            print "@ Task: ", i
            fig = plt.figure(i)
            ax = fig.add_subplot(111)
            ax.grid()

            policy = Policies[i].policy # Resetting policy IMP
            start_time = datetime.now()

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
                ax.scatter(k, np.mean(r), marker=u'x', c='blue', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)

            stop_time = datetime.now()
            tasks_time[i] = str(stop_time - start_time)

            Policies[i].policy = policy # Calculating theta
        print "Task completion times: ", tasks_time
        plt.show(block=True)

        return Policies


if __name__ == "__main__":
    _n_systems = 1  # Integer number of tasks
    _learning_rate = .35  # Learning rate for stochastic gradient descent


    # Parameters for policy
    _poli_type = 'Gauss'  # Policy Type (Only supports Gaussian Policies)
                        # 'Gauss' => Gaussian Policy
    _base_learner = 'NAC'  # Base Learner
                               # 'REINFORCE' => Episodic REINFORCE
                               # 'NAC' => Episodic Natural Actor Critic
    _gamma = 0.9  # Discount factor gamma

    _traj_length = 150 # Number of time steps to simulate in the cart-pole system
    _num_rollouts = 40 # Number of trajectories for testing
    _num_iterations = 120 # Number of learning episodes/iterations # 200

    agent = Agent()
    time.sleep(.5)

    agent.startPg(nSystems = _n_systems,
                  learningRate = _learning_rate,
                  poliType = _poli_type,
                  baseLearner = _base_learner,
                  gamma = _gamma,
                  trajLength = _traj_length,
                  numRollouts = _num_rollouts,
                  numIterations = _num_iterations,
                  task_file = 'task.p',
                  policy_file = 'policy.p',
                  isLoadPolicy = False)

    #agent.startPg(task_file='task.p', policy_file='policy.p', isLoadPolicy=True)
    rospy.spin()