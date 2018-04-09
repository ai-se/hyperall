import pickle
import csv

f = open("collect.p", "rb")
data = pickle.load(f)

for e in data.keys():
    for m in data[e].keys():
        temp = []
        f = open("csv/" + e + "_" + m + ".csv", "w")
        writer = csv.writer(f)
        writer.writerow(temp)
        for l in data[e][m].keys():
            for o in data[e][m][l].keys():
                t = [l, o]
                temp.append(t + list(data[e][m][l][o]))
        writer.writerows(temp)
        f.close()
