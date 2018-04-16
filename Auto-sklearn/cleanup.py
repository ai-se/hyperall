import os

folders = [d + "/" for d in os.listdir(".") if "PickleLocker_" in d]

for folder in folders:
    files = [folder + f for f in os.listdir(folder) if ".p" in f]
    for file in files:
        os.system("rm " + file)

import pdb
pdb.set_trace()