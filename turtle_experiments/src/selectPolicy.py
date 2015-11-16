import numpy as np
import cPickle as pickle

PATH = '../../../'

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


if __name__ == "__main__":
    #==================CONFIG VARIABLES====================
    rm_idx = [] # identify indices to remove/prune out

    t_file = PATH + 'task_new.p'
    p_file = PATH + 'policy_new.p'
    a_file = PATH + 'average_new.p'

    t_new_file = PATH + 'task_new.p'
    p_new_file = PATH + 'policy_new.p'
    a_new_file = PATH + 'average_new.p'
    #======================================================

    tasks, policies, avg_r = loadPolicies(task_file = t_file,
                                          policy_file = p_file,
                                          avg_file = a_file)

    print "# of Tasks: ", len(tasks)
    print "avg_r: ", avg_r
    n_sys = len(tasks)
    if len(rm_idx) > 0:
        prune_tasks = [ tasks[i] for i in range(n_sys) if i not in rm_idx ]
        prune_policies = [policies[i] for i in range(n_sys) if i not in rm_idx]
        prune_avg_r = np.array([])
        n_iterations = np.shape(avg_r)[0]
        for i in range(n_sys):
            if i not in rm_idx:
                prune_avg_r = np.append(prune_avg_r, avg_r[:,i], axis=0)
        prune_avg_r = prune_avg_r.reshape(n_sys-len(rm_idx), n_iterations).conj().T
    else:
        prune_tasks = tasks
        prune_policies = policies
        prune_avg_r = avg_r

    prune_tasks[0].nSystems = n_sys

    savePolicies(prune_tasks, prune_policies, prune_avg_r,
                 task_file = t_new_file,
                 policy_file = p_new_file,
                 avg_file = a_new_file)
