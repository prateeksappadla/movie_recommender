from tornado.ioloop import IOLoop
import tornado.web
import hashlib
from tornado import gen, httpclient, process, httpserver, netutil
import json, urllib, subprocess
from assignment3 import inventory, map_task, reduce_task
import getpass
import socket
import os, sys
import string
import glob
from optparse import OptionParser


@gen.coroutine
def main():
     
     #The coordinator takes as input (via command-line arguments) a mapper program, a reducer program, 
     #the job's working directory (for both inputs and outputs), and the number of reducers (N) to use.
     parser = OptionParser()
     parser.add_option("--mapper_path", dest="mp")#, type="string")
     parser.add_option("--reducer_path", dest="rp")#, type="string")
     parser.add_option("--job_path", dest="jp")#, type="string")
     parser.add_option("--num_reducers", dest="nr")#, type="int")

     (options, args) = parser.parse_args()
     mapper_path = options.mp #wordcount/mapper.py 
     reducer_path = options.rp #wordcount/reducer.py 
     job_path = options.jp #fish_jobs 
     num_reducers = int(options.nr) #1

     #First, the coordinator searches the working directory for files that match the pattern "*.in", 
     #such as 0.in, 1.in, etc. These files are inputs to the MapReduce application. For each of the M input 
     #files, a mapper task is run. Mapper tasks are assigned to workers in a round-robin fashion.

     input_files = glob.glob(str(job_path) +  '/*.in')
     i = 0
     futures = []
     http = httpclient.AsyncHTTPClient()
     servers = inventory.URL#["linserv2.cims.nyu.edu:34514", "linserv2.cims.nyu.edu:34515"]
     for i_file in input_files:
          server = servers[i % len(servers)]
          i += 1
          params = urllib.parse.urlencode({'mapper_path': mapper_path,'input_file': i_file, 'num_reducers': num_reducers}) 
          url = "http://%s/map?%s" % (server, params)
          print("Fetching", url)
          futures.append(http.fetch(url))         #need to extract map_id.
     responses = yield futures
     ids = []
     for r in responses:
          resp = json.loads(r.body.decode())
          ids.append(resp['map_task_id'])
     #When the mapper tasks finish, the reducer tasks are run. Reducer tasks are assigned to workers in a 
     #round-robin fashion as well. Each reducer task writes its output to a file (such as job_path/0.out, 
     #where 0 is the index of the reducer task).
     #have a string map_ids
     map_ids = ''
     for myID in ids:
          map_ids += ',' + myID
     map_ids = map_ids[1:]
     #map_ids = 'nihijlcjlf618tmzl3p5v7tq8uququgz'
     i = 0
     print ('map complete')
     futures2 = []
     for j in range(num_reducers):
          server = servers[i % len(servers)]
          i += 1
          params = urllib.parse.urlencode({'map_task_ids': map_ids,'reducer_path': reducer_path, 'job_path': job_path, 'reducer_ix': j}) 
          url = 'http://%s/reduce?%s' % (server, params)
          print("Fetching", url)
          futures2.append(http.fetch(url))
     responses = yield futures2
     
     print ('complete')

if __name__ == "__main__":
    
     IOLoop.current().run_sync(main)
     
     
