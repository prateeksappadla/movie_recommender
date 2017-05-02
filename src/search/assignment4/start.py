import os 
from assignment2 import inventory



#The indexer is a command-line Python program. It runs the following three MapReduce programs by invoking your coordinator 
#from assignment 3. The number of reducers for the inverted index builder and document store builder should be set to the 
#number of partitions you're using for the respective component in your assignment 2 search engine. Only one reducer should 
#be used by the IDF map builder (unless you partitioned it in assignment 2).


ni = inventory.num_ind
nd = inventory.num_doc
os.system('python3.6 -m assignment3.coordinator --mapper_path=assignment4/mapper.py --reducer_path=assignment4/reducer.py --job_path=assignment4/df_jobs --num_reducers=' + str(ni))
os.system('python3.6 -m assignment3.coordinator --mapper_path=assignment4/mapdocstr.py --reducer_path=assignment4/reddocstr.py --job_path=assignment4/df_jobs --num_reducers=' + str(nd))
os.system('python3.6 -m assignment3.coordinator --mapper_path=assignment4/mapidf.py --reducer_path=assignment4/redidf.py --job_path=assignment4/df_jobs --num_reducers=1')



