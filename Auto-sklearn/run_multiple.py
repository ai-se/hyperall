import os


files = ['run_ant.py', 'run_camel.py', 'run_ivy.py', 'run_jedit.py', 'run_log4j.py', 'run_lucene.py',  'run_poi.py',
         'run_synapse.py', 'run_velocity.py', 'run_xerces.py']

for file in files:
    print(file)
    os.system('python3 ' + file + '&')