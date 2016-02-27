#!/usr/bin/env python
from __future__ import print_function
import rospy
import time
import sys
import types
from datetime import datetime
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Empty, Bool
from gazebo_msgs.srv import SetModelState, SetModelStateRequest

from xbox_controller import xboxcontroller

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from matplotlib import cm
from math import sqrt, exp, isinf
from sys import stdout

from createSys import createSys
from constructPolicies import constructPolicies
from episodicREINFORCE import episodicREINFORCE
from episodicNaturalActorCritic import episodicNaturalActorCritic
from finiteDifferences import finiteDifferences
from drawAction import drawAction
from initPGELLA import initPGELLA
from computeHessian import computeHessian
from updatePGELLA import updatePGELLA
from structs import Data, Hessianarray, Parameterarray, PGPolicy, Policy

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Agent():

    def __init__(self, n_systems, learning_rate, gamma):
        rospy.init_node('agent_pg_ella', anonymous=False)

        print("waiting for service")
        rospy.wait_for_service('/v_control/reset_world')
        print("done")

        self._n_systems = n_systems
        self._learning_rate = learning_rate
        self._gamma = gamma

        self._reset_pos = \
            rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._enable_controller = \
            rospy.Publisher('v_controller/move_enable', Bool, queue_size=1)
        self._soft_reset_pub = \
            rospy.Publisher('v_controller/soft_reset', Empty, queue_size=10)
        self._takeoff_pub = \
            rospy.Publisher('/ardrone/takeoff', Empty, queue_size=10)
        self._land_pub = \
            rospy.Publisher('/ardrone/land', Empty, queue_size=10)
        self._action_pub = \
            rospy.Publisher('v_controller/agent_cmd', Twist, queue_size=1)
        self._state_sub = \
            rospy.Subscriber('v_controller/state', Float32MultiArray,
                             self.getState)
        self._visible_sub = \
            rospy.Subscriber('v_controller/visible', Bool,
                             self.visible_callback)
        self._threshold = \
            rospy.get_param("v_controller/threshold")

        self._hover_height = .4

    def startPg(self, poli_type='Gauss', base_learner='NAC',
                traj_length=100, num_rollouts=40, num_iterations=150,
                task_file='task.p', policy_file='policy.p',
                avg_file='average.p', is_continue=False, is_load=False,
                source=None, tasks=None):
        self.Policies_ = None
        if is_continue:
            # Continue learning from a loaded policy
            # num_iterations treated as additional iterations

            # Load policy from file
            self.loadPolicies(task_file, policy_file, avg_file)

            # Ensure number of systems are equal
            if (self.Tasks[0].nSystems != self._n_systems):
                import sys
                print("Error: File has different number of systems")
                sys.exit(1)

            # Continue Learning
            self.Policies, self.Avg_rPG = \
                self.calcThetaStar(self.Tasks, self.Policies,
                                   self._learning_rate, traj_length,
                                   num_rollouts, num_iterations,
                                   avgRPG=self.Avg_rPG)

            self.savePolicies(task_file, policy_file, avg_file)
        elif not is_load:
            # Creating Tasks
            if not isinstance(tasks, types.NoneType):
                self.Tasks = copy.deepcopy(tasks)
            else:
                self.Tasks = createSys(self._n_systems, poli_type,
                                       base_learner, self._gamma)

            ave = None
            if not isinstance(source, types.NoneType):
                sum_ = np.zeros((self.Tasks[0].param.N * self.Tasks[0].param.M, 1))
                for p in source.Policies:
                    sum_ += p.policy.theta
                ave = sum_ / source._n_systems
                print("Ave Policy: ", ave.T)

            # Constructing policies
            self.Policies = constructPolicies(self.Tasks, init_policy=ave)

            # Calculating theta
            self.Policies, self.Avg_rPG = \
                self.calcThetaStar(self.Tasks, self.Policies,
                                   self._learning_rate, traj_length,
                                   num_rollouts, num_iterations)

            self.savePolicies(task_file, policy_file, avg_file)
        else:
            self.loadPolicies(task_file, policy_file, avg_file)

    def savePolicies(self, task_file, policy_file, avg_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.Tasks, open(task_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Policies, open(policy_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.Avg_rPG, open(avg_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def loadPolicies(self, task_file, policy_file, avg_file):
        # Load PG policies and tasks to a file
        self.Tasks = pickle.load(open(task_file, 'rb'))
        self.Policies = pickle.load(open(policy_file, 'rb'))
        self.Avg_rPG = pickle.load(open(avg_file, 'rb'))

    def getState(self, data):
        self._state_x = data.data[0]
        self._state_y = data.data[1]
        self._state_z = data.data[2] - self._hover_height
        # bias so that it floats at .240 meters
        # .235 ultrasonic read still works, using .240
        # -.240 is too low

    def visible_callback(self, visible):
        self._visible = visible.data

    def resetSim(self, x=0, y=0, z=0, angle=0):
        # self._enable_controller.publish(Bool(0))
        # self._land_pub.publish(Empty())
        # rospy.sleep(10)
        self._takeoff_pub.publish(Empty())
        rospy.sleep(1)
        a = SetModelStateRequest()
        a.model_state.model_name = 'quadrotor'
        a.model_state.pose.position.z = 2 + z
        a.model_state.pose.position.x = 3 + x
        a.model_state.pose.position.y = 0 + y

        a.model_state.pose.orientation.x = 0
        a.model_state.pose.orientation.y = 0
        a.model_state.pose.orientation.z = 0
        a.model_state.pose.orientation.w = 0

        b = SetModelStateRequest()
        b.model_state.model_name = 'unit_sphere_4'
        b.model_state.pose.position.z = 0.052
        b.model_state.pose.position.x = 3
        b.model_state.pose.position.y = 0

        self._reset_pos(a)
        self._reset_pos(b)
        rospy.sleep(.1)

    def startEpisode(self):
        _x = _y = _angle = 0.0
        self.resetSim(x=_x, y=_y, angle=_angle)
        rospy.sleep(1)

        # while True:
        #    _x = _y = _angle = 0.0
        #    # _x = random.uniform(-.3, .3)
        #    # _y = random.uniform(-.2, .2)
        #    # _angle = random.unifor,m(-1.0, 1.0)
        #    self.resetSim(x=_x, y=_y, angle=_angle)
        #    rospy.sleep(1)
        #    if self._state_x <= .2 and self._state_x >= -.2 and \
        #       self._state_y <= .2 and self._state_y >= -.2:
        #        self.last_start_x = _x
        #        self.last_start_y = _y
        #        self.last_start_angle = _angle
        #        return

    def land(self):
        self.land_pub.publish(Empty())

    def obtainData(self, policy, L, H, param):
        N = param.param.N
        M = param.param.M
        data = [Data(N, M, L) for i in range(H)]

        time.sleep(.5)

        # Based on Jan Peters; codes.
        # Perform H
        noiseval = []
        landed = 0
        hasLanded = False

        msg = "\r    Trial {t} of {H} | Drone Landed {l} of {H}"
        stdout.write(msg.format(t=0, H=H, l=0))
        stdout.flush()
        for trials in range(H):
            hasLanded = False
            self.startEpisode()

            # Save associated policy
            data[trials].policy = copy.deepcopy(policy)

            # Finite Differences
            if param.param.baseLearner == "FD":
                if trials != 0:
                    if np.mod(trials, 2) == 1:
                        noiseval = \
                            np.random.uniform(-0.1, 0.1, (N*M, 1))
                    else:
                        noiseval = -noiseval
                    data[trials].policy.theta = data[0].policy.theta + noiseval
                # print("\n      T:", data[trials].policy.theta.T)  # theta
                # print("      P:", policy.theta.T)  # policy

            command = Twist()
            command.linear.x = 0
            command.linear.y = 0
            command.linear.z = 0
            command.angular.z = 0
            self._action_pub.publish(command)
            rospy.sleep(.1)

            # self.resetSim(0, 0, 0, 0)
            # rospy.sleep(1)

            # Draw the first state
            data[trials].x[:, 0] = np.array([[-self._state_x/4.0,
                                             -self._state_y/3.0,
                                             self._state_z/3.0]])

            # Perform a trial of length L
            for steps in range(L):
                if rospy.is_shutdown():
                    sys.exit()
                # Draw an action from the policy
                xx = data[trials].x[:, steps]
                if param.param.baseLearner == "FD":
                    data[trials].u[:, steps] = drawAction(data[trials].policy,
                                                          xx, param)
                else:
                    data[trials].u[:, steps] = drawAction(policy, xx, param)

                if not self._visible or hasLanded:
                    data[trials].u[:, steps] *= 0.0
                else:
                    for j in range(M):
                        data[trials].u[:, steps][j] += \
                            param.param.disturbance[j]
                        data[trials].u[:, steps][j] = \
                            round(data[trials].u[:, steps][j], 5)

                        # saturate
                        if data[trials].u[:, steps][j] > 1.0:
                            data[trials].u[:, steps][j] = 1.0
                        elif data[trials].u[:, steps][j] < -1.0:
                            data[trials].u[:, steps][j] = -1.0

                # ensures quadrotor doesn't go off
                # the max altitude of 3.0
                if self._visible and \
                   self._state_z >= (2.8 - self._hover_height):
                    data[trials].u[:, steps][2] = -0.1

                if not hasLanded:
                    if self._state_z <= 0 and self._visible and \
                       self._state_x < .7 and self._state_x > -.7 and \
                       self._state_y < .6 and self._state_y > -.8:
                        data[trials].u[:, steps][0] = 0.0
                        data[trials].u[:, steps][1] = 0.0
                        data[trials].u[:, steps][2] = 0.0
                        hasLanded = True
                        self._land_pub.publish(Empty())
                        landed += 1
                        rospy.sleep(3)
                    else:
                        command = Twist()
                        command.linear.x = data[trials].u[:, steps][0]
                        command.linear.y = data[trials].u[:, steps][1]
                        command.linear.z = data[trials].u[:, steps][2]
                        command.angular.z = 0.0
                        self._action_pub.publish(command)
                        rospy.sleep(.4)

                # Draw next state from environment
                if hasLanded:
                    x = np.array([[0.0, 0.0, 0.0]])
                else:
                    x = np.array([[-self._state_x/4.0,
                                   -self._state_y/3.0,
                                   self._state_z/3.0]])

                data[trials].x[:, steps+1] = x

                # Obtain the reward
                u = data[trials].u[:, steps]
                r_u = sqrt(np.dot(u, u.T))
                x[0, 2] = x[0, 2] * 3.0
                r_x = sqrt(np.dot(x, x.T))

                # reward = -sqrt(np.dot(np.dot(x, np.eye(N) * 10), x.T)) - \
                #    sqrt(np.dot(np.dot(u, np.eye(M) * 5), u.T))
                reward = -10*r_x - 1*r_u

                if isinf(reward):
                    print("Error: INFINITY")
                    sys.exit(1)

                data[trials].r[0][steps] = reward
            stdout.write(msg.format(t=trials+1, H=H, l=landed))
            stdout.flush()

        print()
        return data

    def calcThetaStar(self, Params, Policies, rates,
                      trajlength, rollouts, numIterations,
                      avgRPG=None):

        plt.ion()

        nSystems = self._n_systems
        print("Number of Systems", nSystems)
        r = np.zeros(shape=(1, rollouts))

        tasks_time = [0] * nSystems
        start_it = [0] * nSystems

        if isinstance(avgRPG, types.NoneType):
            Avg_rPG = [np.zeros((numIterations, 1)) for i in range(nSystems)]
        elif isinstance(avgRPG, np.ndarray):
            # Convert numpy array to list
            tmp_avg_r = []
            n_iterations = np.shape(avgRPG)[0]
            for j in range(nSystems):
                start_it[j] = n_iterations
                tmp = np.append(avgRPG[:, j].reshape(n_iterations, 1),
                                np.zeros((numIterations, 1)), axis=0)
                tmp_avg_r.append(tmp)
            Avg_rPG = tmp_avg_r
        elif type(avgRPG) == list:
            for j in range(nSystems):
                start_it[j] = np.shape(avgRPG[j])[0]
                avgRPG[j] = np.append(avgRPG[j],
                                      np.zeros((numIterations, 1)), axis=0)
            Avg_rPG = avgRPG

        for i in range(nSystems):
            # TODO: Clear screen
            print("@ Task: ", i + 1)
            print("    Perturbation: ", Params[i].param.disturbance)
            fig = plt.figure("PG Task " + str(i+1))
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

            if not isinstance(avgRPG, types.NoneType):
                # plot the rewards from previous learning session
                for k in range(start_it[i]):
                    ax.scatter(k, Avg_rPG[i][k], marker=u'x',
                               c='green', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            # policy = copy.deepcopy(Policies[i].policy) # Resetting policy IMP
            policy = Policies[i].policy
            start_time = datetime.now()

            for k in range(start_it[i], numIterations+start_it[i]):
                print("@ Iteration: ", k)
                print("    Theta: ", policy.theta.T)
                print("    Sigma: ", policy.sigma)
                print("    Learning Rate: ", rates)
                print("    Perturbation: ", Params[i].param.disturbance)
                data = self.obtainData(policy, trajlength, rollouts, Params[i])
                dJdtheta = None
                if Params[i].param.baseLearner == "REINFORCE":
                    dJdtheta = episodicREINFORCE(policy, data, Params[i])
                elif Params[i].param.baseLearner == "NAC":
                    dJdtheta = \
                        episodicNaturalActorCritic(policy, data, Params[i])
                elif Params[i].param.baseLearner == "FD":
                    dJdtheta = finiteDifferences(data, rollouts)
                    _check = rates * dJdtheta

                t_size = Params[i].param.N * Params[i].param.M
                policy.theta = policy.theta + rates*dJdtheta.reshape(t_size, 1)

                for z in range(rollouts):
                    r[0, z] = np.sum(data[z].r)
                    # np.mean(data[z].r) # np.sum(data[z].r)

                if np.isnan(r.any()):
                    print("System has NAN")
                    print("exiting...")
                    import sys
                    sys.exit(1)

                avg_r = np.mean(r)
                print("    Mean: ", avg_r)
                Avg_rPG[i][k] = np.mean(avg_r)

                time.sleep(1)
                ax.scatter(k, avg_r, marker=u'x', c='blue', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.05)

            stop_time = datetime.now()
            tasks_time[i] = str(stop_time - start_time)

            Policies[i].policy = policy  # Calculating theta
            print("    Learned Theta: ", Policies[i].policy.theta.T)
            print("")
            file_name = "PG Task {n}.png".format(n=i+1)
            plt.savefig(file_name, bbox_inches='tight')
            plt.close(fig)
        print("Task completion times: ", tasks_time)
        # plt.show(block=True)

        return Policies, Avg_rPG

    def startElla(self, traj_length=100, num_rollouts=40,
                  learning_rate=0.1, mu1=0.1, mu2=0.1, k=1,
                  model_file='model.p', is_load=False, source=None):

        if not isinstance(source, types.NoneType):
            self.targetStartIndex = len(source.Tasks)
            # Append target tasks to source tasks list
            for task in self.Tasks:
                source.Tasks.append(task)
            self.Tasks = source.Tasks
            self.Tasks[0].nSystems = len(self.Tasks)
            self._n_systems = self.Tasks[0].nSystems
            # Append target policies to source policies list
            for policy in self.Policies:
                source.Policies.append(policy)
            self.Policies = source.Policies

        if not is_load:
            # self._learning_rate = learning_rate
            self.modelPGELLA = initPGELLA(self.Tasks, k, mu1, mu2,
                                          learning_rate,
                                          self.targetStartIndex)

            print("Learn PGELLA")
            self.modelPGELLA = self.learnPGELLA(self.Tasks, self.Policies,
                                                self._learning_rate,
                                                traj_length, num_rollouts,
                                                self.modelPGELLA)

            self.saveModelPGELLA(model_file)
        else:
            self.loadModelPGELLA(model_file)

    def saveModelPGELLA(self, model_file):
        # Save PG policies and tasks to a file
        pickle.dump(self.modelPGELLA, open(model_file, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def loadModelPGELLA(self, model_file):
        # Load PG policies and tasks to a file
        self.modelPGELLA = pickle.load(open(model_file, 'rb'))

    def getSeedRandom(self, size, n_sets):
        self.randNumbers = []
        for i in range(n_sets):
            temp = range(size)
            random.shuffle(temp)
            self.randNumbers += temp

    def getNextRandom(self):
        if len(self.randNumbers) > 0:
            return self.randNumbers.pop(0)
        return None

    def learnPGELLA(self, Tasks, Policies, learningRate,
                    trajLength, numRollouts, modelPGELLA):
        counter = 1
        tasks_size = np.shape(Tasks)[0]
        ObservedTasks = np.zeros((tasks_size, 1))
        limitOne = 0
        limitTwo = tasks_size
        print("Task size: ", tasks_size)

        HessianArray = [Hessianarray() for i in range(tasks_size)]
        ParameterArray = [Parameterarray() for i in range(tasks_size)]

        # self.getSeedRandom(tasks_size, 1)
        # taskId = 0 # FIXME to get rid of random choice of task
        # while len(self.randNumbers) > 0:
        # while not np.all(ObservedTasks):  # Repeat until all tasks are observed
        for taskId in range(tasks_size):
            # Pick a random task
            # taskId = np.random.randint(limitOne, limitTwo, 1)
            # taskId = self.getNextRandom()
            # if counter == 1:
            #    self.randNumbers.append(taskId)
            print("Task ID: ", taskId)

            if ObservedTasks[taskId] == 0:
                # Entry is set to 1 when corresponding task is observed
                ObservedTasks[taskId] = 1

            # Policy Gradientts on taskId
            # print("    Learn Theta")
            # print("    Theta: ", Policies[taskId].policy.theta.T)
            # print("    Sigma: ", Policies[taskId].policy.sigma)
            # print("    Perturbation: ", Tasks[taskId].param.disturbance)
            # data = self.obtainData(Policies[taskId].policy, trajLength,
            #                        numRollouts, Tasks[taskId])
            #
            # if Tasks[taskId].param.baseLearner == 'REINFORCE':
            #     dJdTheta = episodicREINFORCE(Policies[taskId].policy,
            #                                  data, Tasks[taskId])
            # elif Tasks[taskId].param.baseLearner == 'NAC':
            #     dJdTheta = episodicNaturalActorCritic(Policies[taskId].policy,
            #                                           data, Tasks[taskId])
            # elif Tasks[taskId].param.baseLearner == 'FD':
            #     dJdTheta = finiteDifferences(data, numRollouts)

            # t_size = Tasks[taskId].param.N * Tasks[taskId].param.M
            # Updating theta*
            # Policies[taskId].policy.theta = Policies[taskId].policy.theta + \
            #    learningRate * dJdTheta.reshape(t_size, 1)

            # if Tasks[taskId].param.baseLearner != 'FD':
            if False:
                # Computing Hessian
                print("    Data for Hessian")
                data = self.obtainData(Policies[taskId].policy, trajLength,
                                       numRollouts, Tasks[taskId])

                # D = computeHessian(data, Policies[taskId].policy.sigma)
                D = computeHessian(data, np.array([[0.1, 0.1, 0.1]]))
            else:
                # Use the identity matrix
                D = np.identity(np.shape(Policies[taskId].policy.theta)[0])

            HessianArray[taskId].D = D

            ParameterArray[taskId].alpha = \
                copy.deepcopy(Policies[taskId].policy.theta)

            # Perform Updating L and S
            # Perform PGELLA for that Group
            modelPGELLA = updatePGELLA(modelPGELLA, taskId, ObservedTasks,
                                       HessianArray, ParameterArray)

            print("Iterating @: ", counter)
            counter = counter + 1

            # taskId += 1 # FIXME to get rid of random choice of task

        print("All Tasks observed @: ", counter-1)

        return modelPGELLA

    def startTest(self, traj_length=100, num_rollouts=40,
                  num_iterations=150, learning_rate=0.1,
                  test_file='test.p', is_load=False,
                  is_continue=False, baseline=None):
        if is_continue:
            # Continue learning from a loaded policy
            # num_iterations treated as additional iterations

            # Load policy from file
            self.loadModelTest(test_file)
            avg_RPGELLA = self.Test_Avg_rPGELLA
            avg_RPG = self.Test_Avg_rPG

            # Ensure number of systems are equal
            # if (self.Tasks[0].nSystems != self._n_systems):
            #    import sys
            #    print("Error: File has different number of systems")
            #    sys.exit(1)

        elif not is_load:
            self._n_systems = np.shape(self.modelPGELLA.S)[1]
            print("number of systems: ", self._n_systems)
            self._learning_rate = learning_rate
            # Creating new PG policies
            # self.PGPol = constructPolicies(self.Tasks)
            self.PGPol = copy.deepcopy(self.Policies)  # NEW
            if not isinstance(baseline, types.NoneType):
                # print("PGPOL1: ", baseline.Policies[0].policy.theta.T)
                self.PGPol[self.modelPGELLA.targetIdx:] = copy.deepcopy(baseline.Policies[:])
                # print("PGPOL2: ", self.PGPol[self.modelPGELLA.targetIdx].policy.theta.T)

            self.PolicyPGELLAGroup = \
                [PGPolicy() for i in range(self._n_systems)]

            k = self.modelPGELLA.k
            for i in range(self._n_systems):  # loop over all tasks
                theta_PG_ELLA = np.dot(self.modelPGELLA.L,
                                       self.modelPGELLA.S[:, i].reshape(k, 1))
                policyPGELLA = Policy()
                policyPGELLA.theta = theta_PG_ELLA
                policyPGELLA.sigma = self.PGPol[0].policy.sigma
                self.PolicyPGELLAGroup[i].policy = policyPGELLA
                print("Task Id#", i+1, " ", theta_PG_ELLA.T)

            avg_RPGELLA = self.Test_Avg_rPGELLA = None
            avg_RPG = self.Test_Avg_rPG = None
            self.saveModelTest(test_file)

        else:
            self.loadModelTest(test_file)
            avg_RPGELLA = self.Test_Avg_rPGELLA
            avg_RPG = self.Test_Avg_rPG

        print("Test Phase")
        # Testing and comparing PG and PG-ELLA
        self.testPGELLA(self.Tasks, self.PGPol,
                        self._learning_rate, traj_length, num_rollouts,
                        num_iterations, self.PolicyPGELLAGroup,
                        avgRPGELLA=avg_RPGELLA, avgRPG=avg_RPG)

        self.saveModelTest(test_file)

    def saveModelTest(self, test_file):
        test = {}
        test['alpha'] = self._learning_rate
        test['pg_pol'] = self.PGPol
        test['pgella_pol'] = self.PolicyPGELLAGroup
        test['avg_RPG'] = self.Test_Avg_rPG
        test['avg_RPGELLA'] = self.Test_Avg_rPGELLA
        # Save PG policies and tasks to a file
        pickle.dump(test, open(test_file, 'wb'), pickle.HIGHEST_PROTOCOL)

    def loadModelTest(self, test_file):
        # Load PG policies and tasks to a file
        test = pickle.load(open(test_file, 'rb'))
        self._learning_rate = test['alpha']
        self.PGPol = test['pg_pol']
        self.PolicyPGELLAGroup = test['pgella_pol']
        self.Test_Avg_rPG = test['avg_RPG']
        self.Test_Avg_rPGELLA = test['avg_RPGELLA']

    def testPGELLA(self, Tasks, PGPol, learningRate, trajLength,
                   numRollouts, numIterations, PolicyPGELLAGroup,
                   avgRPGELLA=None, avgRPG=None):

        plt.ion()

        tasks_size = self._n_systems

        tasks_time = [0] * tasks_size
        start_it = [0] * tasks_size

        if isinstance(avgRPGELLA, types.NoneType) or \
           isinstance(avgRPG, types.NoneType):
            Test_Avg_rPGELLA = \
                [np.zeros((numIterations, 1)) for i in range(tasks_size)]
            Test_Avg_rPG = \
                [np.zeros((numIterations, 1)) for i in range(tasks_size)]
        elif type(avgRPGELLA) == list or type(avgRPG) == list:
            for j in range(tasks_size):
                start_it[j] = np.shape(avgRPG[j])[0]
                avgRPG[j] = np.append(avgRPG[j],
                                      np.zeros((numIterations, 1)), axis=0)
                avgRPGELLA[j] = np.append(avgRPGELLA[j],
                                          np.zeros((numIterations, 1)), axis=0)
            Test_Avg_rPG = avgRPG
            Test_Avg_rPGELLA = avgRPGELLA
        # Avg_rPGELLA = np.zeros((numIterations, tasks_size))
        # Avg_rPG = np.zeros((numIterations, tasks_size))
        for k in range(self.modelPGELLA.targetIdx, tasks_size):  # Test over all tasks
            print("@ Task: ", k + 1)
            fig = plt.figure("PG-ELLA Task " + str(k+1))
            ax = fig.add_subplot(111)
            ax.grid()
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Reward')

            if not isinstance(avgRPG, types.NoneType):
                # plot the rewards from previous learning session
                for m in range(start_it[k]):
                    ax.scatter(m, Test_Avg_rPGELLA[k][m],
                               marker=u'*', color='r', cmap=cm.jet)
                    ax.scatter(m, Test_Avg_rPG[k][m],
                               marker=u'*', color='b', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            start_time = datetime.now()
            print("Init PG policy: ", PGPol[k].policy.theta.T)
            # Loop over Iterations
            for m in range(start_it[k], numIterations+start_it[k]):
                print("    @ Iteration: ", m+1)
                print("    Perturbation: ", Tasks[i].param.disturbance)
                # PG
                data = self.obtainData(PGPol[k].policy, trajLength,
                                       numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdTheta = episodicREINFORCE(PGPol[k].policy,
                                                 data, Tasks[k])
                elif Tasks[k].param.baseLearner == 'NAC':
                    dJdTheta = episodicNaturalActorCritic(PGPol[k].policy,
                                                          data, Tasks[k])
                elif Tasks[k].param.baseLearner == 'FD':
                    dJdTheta = finiteDifferences(data, numRollouts)

                # Update Policy Parameters
                PGPol[k].policy.theta = PGPol[k].policy.theta + \
                    learningRate * dJdTheta.reshape(9, 1)
                print("PG policy    : ", PGPol[k].policy.theta.T)

                # PG-ELLA
                dataPGELLA = self.obtainData(PolicyPGELLAGroup[k].policy,
                                             trajLength, numRollouts, Tasks[k])

                if Tasks[k].param.baseLearner == 'REINFORCE':
                    dJdThetaPGELLA = \
                        episodicREINFORCE(PolicyPGELLAGroup[k].policy,
                                          dataPGELLA, Tasks[k])
                elif Tasks[k].param.baseLearner == 'NAC':
                    dJdThetaPGELLA = \
                        episodicNaturalActorCritic(PolicyPGELLAGroup[k].policy,
                                                   dataPGELLA, Tasks[k])
                elif Tasks[k].param.baseLearner == 'FD':
                    dJdThetaPGELLA = finiteDifferences(dataPGELLA, numRollouts)

                # Update Policy Parameters
                PolicyPGELLAGroup[k].policy.theta = \
                    PolicyPGELLAGroup[k].policy.theta + \
                    learningRate * dJdThetaPGELLA.reshape(9, 1)
                print("PGELLA policy: ", PolicyPGELLAGroup[k].policy.theta.T)

                # Computing Average in one System per iteration
                data_size = np.shape(data)[0]
                Sum_rPG = np.zeros((data_size, 1))
                for z in range(data_size):
                    Sum_rPG[z, :] = np.sum(data[z].r)

                dataPG_size = np.shape(dataPGELLA)[0]
                Sum_rPGELLA = np.zeros((dataPG_size, 1))
                for z in range(dataPG_size):
                    Sum_rPGELLA[z, :] = np.sum(dataPGELLA[z].r)

                Test_Avg_rPGELLA[k][m] = np.mean(Sum_rPGELLA)
                Test_Avg_rPG[k][m] = np.mean(Sum_rPG)

                # Plot graph
                ax.scatter(m, Test_Avg_rPGELLA[k][m],
                           marker=u'*', color='r', cmap=cm.jet)
                ax.scatter(m, Test_Avg_rPG[k][m],
                           marker=u'*', color='b', cmap=cm.jet)
                ax.figure.canvas.draw()
                fig.canvas.draw()
                fig.canvas.flush_events()

            stop_time = datetime.now()
            tasks_time[k] = str(stop_time - start_time)

            file_name = "PG-ELLA Task {n}.png".format(n=k+1)
            plt.savefig(file_name, bbox_inches='tight')
            plt.close(fig)

        print("Task completion times: ", tasks_time)
        self.PGPol = PGPol
        self.Test_Avg_rPG = Test_Avg_rPG
        self.PolicyPGELLAGroup = PolicyPGELLAGroup
        self.Test_Avg_rPGELLA = Test_Avg_rPGELLA

if __name__ == "__main__":
    #np.random.seed(10)
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    #random.seed(10)
    n_systems = 19  # tasks [SOURCE]
    n_systems_ = 1  # tasks [TARGET]
    learning_rate = 0.000001  # Learning rate for stochastic gradient descent
    gamma = 0.9999  # Discount factor gamma

    # Parameters for policy
    # Policy Type (Only supports Gaussian Policies)
    # 'Gauss' => Gaussian Policy
    poli_type = 'Gauss'
    # Base Learner
    # 'REINFORCE' => Episodic REINFORCE
    # 'NAC' => Episodic Natural Actor Critic
    # 'FD' => Finite Differences
    base_learner = 'FD'

    traj_length = 150  # Number of time steps
    num_rollouts = 15  # Number of trajectories/trials for testing
    num_iterations = 0  # Number of learning episodes/iterations [SOURCE]
    num_iterations_ = 10  # Number of learning episodes/iterations [TARGET]

    agent = Agent(n_systems, learning_rate, gamma)  # [SOURCE]
    agent_ = Agent(n_systems_, learning_rate, gamma)  # [TARGET]
    agent_b = Agent(n_systems_, learning_rate, gamma) # [BASELINE]
    time.sleep(.5)

    # Learning PG
    # agent.startPg(poli_type, base_learner, traj_length,
    #              num_rollouts, num_iterations,
    #              task_file='task_4_fd_source.p',
    #              policy_file='policy_4_fd_source.p',
    #              avg_file='average_4_fd_source.p',
    #              is_load=False)
    # Continue Learning PG
    # NOTE: Make a Backup of the files before running to ensure
    #       you have a copy of the original policy
    # agent.startPg(poli_type, base_learner, traj_length,
    #               num_rollouts, num_iterations,
    #               task_file='task_19_fd_source.p',
    #               policy_file='policy_19_fd_source.p',
    #               avg_file='average_19_fd_source.p',
    #               is_continue=True)
    # Loading PG policies from file
    agent.startPg(task_file='task_19_fd_source.p',
                  policy_file='policy_19_fd_source.p',
                  avg_file='average_19_fd_source.p',
                  is_load=True)

    # Learning PG
    # agent_.startPg(poli_type, base_learner, traj_length,
    #                num_rollouts, num_iterations_,
    #                task_file='task_1_fd_target.p',
    #                policy_file='policy_1_fd_target.p',
    #                avg_file='average_1_fd_target.p',
    #                is_load=False, source=agent)
    # agent_b.startPg(poli_type, base_learner, traj_length,
    #                 num_rollouts, num_iterations_,
    #                 task_file='task_1_fd_target_base.p',
    #                 policy_file='policy_1_fd_target_base.p',
    #                 avg_file='average_1_fd_target_base.p',
    #                 is_load=False, tasks=agent_.Tasks)

    # Continue Learning PG
    # NOTE: Make a Backup of the files before running to ensure
    #       you have a copy of the original policy
    # agent_.startPg(poli_type, base_learner, traj_length,
    #                num_rollouts, num_iterations_,
    #                task_file='task_1_fd_target.p',
    #                policy_file='policy_1_fd_target.p',
    #                avg_file='average_1_fd_target.p',
    #                is_continue=True)
    # Loading PG policies from file
    agent_.startPg(task_file='task_1_fd_target.p',
                   policy_file='policy_1_fd_target.p',
                   avg_file='average_1_fd_target.p',
                   is_load=True)
    agent_b.startPg(task_file='task_1_fd_target_base.p',
                    policy_file='policy_1_fd_target_base.p',
                    avg_file='average_1_fd_target_base.p',
                    is_load=True)

    # Learning ELLA
    # traj_length = 150
    num_rollouts = 0  # 200
    learning_rate_ella = learning_rate
    mu1 = 0.00001  # 0.0000001 #0.0000001 # 0.01  # exp(-5)  # Sparsity coefficient
    mu2 = 0.001  # 0.00000001 #0.001 #0.00000001  # exp(-5)  # Regularization coefficient
    k = 4  # Number of inner layers

    # Learning PGELLA from SOURCE to TARGET
    # agent_.startElla(traj_length, num_rollouts,
    #                  learning_rate_ella, mu1, mu2, k,
    #                  model_file='model_1_fd_target.p',
    #                  is_load=False, source=agent)
    # Load PGELLA Model from file
    agent_.startElla(model_file='model_1_fd_target.p',
                    is_load=True, source=agent)

    # Testing Phase
    # traj_length = 150
    num_rollouts = 15  # 100
    num_iterations = 60  # 200
    # learning_rate = .1

    # agent_.startTest(traj_length, num_rollouts, num_iterations,
    #                  learning_rate, test_file='test_1_fd_target.p',
    #                  is_load=False, baseline=agent_b)

    agent_.startTest(traj_length, num_rollouts, num_iterations, learning_rate,
                   test_file='test_1_fd_target.p', is_continue=True)

    rospy.spin()
