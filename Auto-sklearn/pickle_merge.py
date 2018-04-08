from __future__ import division
import os
import pickle

import collections
Experiment = collections.namedtuple('Experiment', 'train test')

consolidated = {}
subjects = ['ant', 'camel', 'ivy', 'jedit', 'log4j', 'lucene', 'poi', 'synapse', 'velocity', 'xerces']
folders = [ x for x in os.listdir(".") if 'PickleLocker_' in x]
svm_folder = [f+'/' for f in folders if 'svm' in f]
consolidated['svm'] = {}
dt_folder = [f+'/' for f in folders if 'dt' in f]
consolidated['dt'] = {}
rf_folder = [f+'/' for f in folders if 'rf' in f]
consolidated['rf'] = {}
knn_folder = [f+'/' for f in folders if 'knn' in f]
consolidated['knn'] = {}


assert(len(svm_folder) + len(dt_folder) + len(rf_folder) + len(knn_folder) == len(folders)), "Something is wrong"

learners = [svm_folder, dt_folder, rf_folder, knn_folder]
for folders in learners:
    for folder in folders:
        _, learner, perf, evals = folder.replace('/', '').split('_')
        files = [folder + f for f in os.listdir(folder) if '.p' in f]
        for subject in subjects:
            print(learner, perf, evals, subject)
            if perf not in consolidated[learner].keys():
                consolidated[learner][perf] = {}
            if evals not in consolidated[learner][perf].keys():
                consolidated[learner][perf][evals] = {}
            consolidated[learner][perf][evals][subject] = {}

            pickle_files = [file for file in files if subject in file]
            if len(pickle_files) == 0:
                consolidated[learner][perf][subject] = []
                continue
            else:
                # Find the pickle file with largest number. Result of a bug in the code
                pickle_file = sorted(pickle_files, key=lambda x: int(x.split('/')[-1].replace('.p', '').split('_')[1]), reverse=True)[0]
                all_data = pickle.load(open(pickle_file, 'rb'))
                keys = all_data.keys()
                for key in keys:
                    perform_temp_store = []
                    time_temp_store = []
                    data = all_data[key]['automl']
                    for d in data:
                        confusion_matrix = d[0]
                        time = d[1]
                        A = confusion_matrix[0][0]
                        B = confusion_matrix[0][1]
                        C = confusion_matrix[1][0]
                        D = confusion_matrix[1][1]
                        pd = D/(B+D)
                        prec = D/(D+C)
                        if perf == 'F1':
                            perform = 2 * pd * prec/(pd + prec)
                        elif perf == 'Prec':
                            perform = prec
                        else:
                            print("Somethign is wrong")

                        perform_temp_store.append(perform)
                        time_temp_store.append(time)

                    consolidated[learner][perf][evals][subject][key] = {}
                    consolidated[learner][perf][evals][subject][key]['measure'] = perform_temp_store
                    consolidated[learner][perf][evals][subject][key]['time'] = time_temp_store

pickle.dump(consolidated, open('merge_pickle.p', 'wb'))