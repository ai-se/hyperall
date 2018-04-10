import pickle
import csv

f = open("collect.p", "rb")
data = pickle.load(f)

for e in data.keys():
    for m in data[e].keys():
        temp = []
        f = open("csv/" + e + "_" + m + "_test.csv", "w")
        writer = csv.writer(f)
        for l in data[e][m].keys():
            t = [l]

            for o in sorted(data[e][m][l].keys()):
                t.append(data[e][m][l][o][0])
            temp.append(t)
        writer.writerows(temp)
        f.close()
