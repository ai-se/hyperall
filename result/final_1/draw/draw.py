import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np


width = 0.35

f = open("collect.p", "rb")
data = pickle.load(f)

for e in data.keys():
    for m in data[e].keys():
        fig, ax = plt.subplots()
        temp = []
        f = open("csv/" + e + "_" + m + ".csv", "w")
        writer = csv.writer(f)
        writer.writerow(temp)
        learners = sorted(data[e][m].keys())
        optimizers = sorted(data[e][m][learners[0]].keys())
        ind = np.array([1.5, 3, 4.5, 6])
        colors = ['#ff3366', '#3294fc', '#8a7a1c', '#acc12f']
        t = []
        for i,o in enumerate(optimizers):
            means = [data[e][m][l][o][1] for l in learners]
            stds = [data[e][m][l][o][2] for l in learners]
            t.append(ax.bar(ind+(i*width), means, width, color=colors[i], bottom=0, yerr=stds))

        print e, m, "test"
        ax.set_ylabel('Ranks')
        ax.set_title('Evaluations: ' + str(e) + " Measure: " + str(m))
        ax.set_xticks(ind + 1.5 / 2)
        ax.set_xticklabels(learners)
        ax.legend(t, optimizers)
        plt.tight_layout()
        plt.savefig('./Chart/'+e+'_'+m+'.png')
        # plt.cla()

        # for l in data[e][m].keys():
        #     if len(temp) == 0: temp.append(['Learner'] + sorted(data[e][m][l].keys()))
        #     t = [l]
        #     for o in sorted(data[e][m][l].keys()):
        #         t.append(data[e][m][l][o][0])
        #     temp.append(t)
        # writer.writerows(temp)
        # f.close()
