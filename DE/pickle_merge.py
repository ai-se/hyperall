import os
import pickle
import numpy

cwd = os.getcwd()
data_path = os.path.join(cwd, "../Data/DefectPrediction/")
print(data_path)

repeats = ["25", "50", "100"]
perf_measures = ["f1", "precision"]
learners = ["SVM", "KNN", "DTC", "RF"]

procedure = []
pickle_files = {}
results = {}
datasets_proc = {}
time_records = pickle.load(open("time.pickle", 'rb'))


data = {"ant": ["ant-1.3.csv", "ant-1.4.csv", "ant-1.5.csv", "ant-1.6.csv", "ant-1.7.csv"],
        "ivy": ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],
        "lucene": ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],
        "poi": ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],
        "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],
        "velocity": ["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"],
        "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"],
        "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"],
        "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"],
        "xerces": ["xerces-1.1.csv", "xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }

for dataset, datasets in data.items():
    for i in range(len(datasets) - 1):
        if dataset not in datasets_proc.keys():
            datasets_proc[dataset] = []
        datasets_proc[dataset].append((datasets[i], datasets[i + 1]))

for l in learners:
    results[l] = {}
    for m in perf_measures:
        results[l][m] = {}
        for r in repeats:
            results[l][m][r] = {}
            for k, v in datasets_proc.items():
                results[l][m][r][k] = {}
                for i_v in v:
                    print(l, m, r, k, i_v[0])
                    results[l][m][r][k][i_v[0]] = {"test": i_v[1], "measure": [], "time": []}

for m in perf_measures:
    pickle_files[m] = {}
    for r in repeats:
        pickle_files[m][r] = []
        file_path = "dump_1/%s_%s_de_results/" % (r, m)
        files = [x[2] for x in os.walk(file_path)][0]
        pickle_files[m][r] = files
        for f_name in files:
            release = f_name.split("_")
            software = release[2].split("-")
            if release[2] != data[software[0]][-1]:
                f = open(file_path + f_name, 'rb')
                stored_result = pickle.load(f)
                stored_result = stored_result.values()[-1]
                for l, v in stored_result.items():
                    measure = []
                    for i in range(0, len(v[0]), 3):
                        print(i, i + 3)
                        a = v[0][i:i + 3]
                        measure.append(numpy.mean(a))
                    temp = [v[2] / float(len(measure))] * len(measure)
                    for i in range(len(temp)):
                        temp[i] -= time_records[l][m][software[0]][release[2]][i]/3.0
                        temp[i] *= 3
                    results[l][m][r][software[0]][release[2]]['time'] = temp
                    results[l][m][r][software[0]][release[2]]['measure'] = measure
                    print(temp, measure)


with open('de_merged_dict_v3.p', 'wb') as handle:
    pickle.dump(results, handle)

