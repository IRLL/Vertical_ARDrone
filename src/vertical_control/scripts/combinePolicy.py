from __future__ import print_function
import numpy as np
import cPickle as pickle

PATH = '../../../final_converged/'

def savePolicies(tasks, policies, avg_rPG,
                 task_file, policy_file, avg_file):
    # Save PG policies and tasks to a file
    pickle.dump(tasks, open(task_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(policies, open(policy_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(avg_rPG, open(avg_file, 'wb'), pickle.HIGHEST_PROTOCOL)

def loadPolicies(task_file, policy_file, avg_file):
    # Load PG policies and tasks to a file
    tasks = pickle.load(open(task_file, 'rb'))
    policies = pickle.load(open(policy_file, 'rb'))
    avg_rPG = pickle.load(open(avg_file, 'rb'))
    return tasks, policies, avg_rPG


if __name__ == '__main__':
    #==================CONFIG VARIABLES====================
    rm_idx = [] # identify indices to remove/prune out

    files = {'tsk_files':[PATH+'task_new.p', PATH+'task_700.p'],
             'pol_files':[PATH+'policy_new.p', PATH+'policy_700.p'],
             'avg_files':[PATH+'average_new.p', PATH+'average_700.p']}

    num_files = len(files['tsk_files'])

    t_out_file = PATH + 'task_new_10.p'
    p_out_file = PATH + 'policy_new_10.p'
    a_out_file = PATH + 'average_new_10.p'
    #======================================================

    task_all = []
    policy_all = []
    avg_r_all = []

    total_tasks = 0

    # Retrieve data from files
    for i in range(num_files):
        tasks, policies, avg_r = loadPolicies(task_file = files['tsk_files'][i],
                                              policy_file = files['pol_files'][i],
                                              avg_file = files['avg_files'][i])
        n_tasks = len(tasks)
        print("{} tasks in set {}".format(n_tasks, i+1))
        if type(avg_r) == np.ndarray:
            n_iterations = np.shape(avg_r)[0]
        total_tasks += n_tasks
        # Append one-by-one each item (task, policy) in their respective list
        for j in range(n_tasks):
            task_all.append(tasks[j])
            policy_all.append(policies[j])
            # numpy array
            if type(avg_r) == np.ndarray:
                avg_r_all.append(avg_r[:,j].reshape(n_iterations, 1))
            # list
            else:
                avg_r_all.append(avg_r[j])

    print("Total # of Tasks: ", total_tasks)
    task_all[0].nSystems = total_tasks

    savePolicies(task_all, policy_all, avg_r_all,
                 task_file = t_out_file,
                 policy_file = p_out_file,
                 avg_file = a_out_file)
