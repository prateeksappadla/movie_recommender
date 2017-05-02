from tornado.ioloop import IOLoop
import tornado.web
import hashlib
from tornado import gen, httpclient
import json, urllib, subprocess
from assignment3 import inventory
import getpass
import socket
import sys
import pickle
import os
import string
import random


def generateMapID():
      return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32))

def hashstr(a):
     sum = 0
     mixer = [2, 3, 7, 11, 13]
     i = 0
     for ltr in a:     
          sum += ord(ltr)*mixer[i%5]
          i += 1
     return sum

def partition (kv_pairs, num_part):
     res = []
     for i in range(num_part):
          res.append([])
     for myPair in kv_pairs:
          if len(myPair) == 2:

               key = myPair[0]
               part = int(hashlib.md5(myPair[0].encode()).hexdigest()[:8], 16)
               #print part
               res[part%num_part].append(myPair)
     myPart = []
     for i in range(num_part):
         myPart.append([]) 

     for myPair in kv_pairs:
          if len(myPair) == 2:
               try:
                    myPart[int(myPair[0]) % num_part].append(myPair)           #docID modulo nPartitions
               except:
                    part = int(hashlib.md5(myPair[0].encode()).hexdigest()[:8], 16)
                    myPart[part%num_part].append(myPair)

     return myPart

class MapperHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
          N = int(self.get_argument('num_reducers'))
          mapper_path = str(self.get_argument('mapper_path'))
          # map_task_ids = ["8a97fd755ea12827485749036e15d651", "d3486112191e4717d17d4fba189bdbf6"]
          input_file = str(self.get_argument('input_file'))
          path = str(os.path.dirname(input_file)) + '/'
          servers = inventory.URL
          http = httpclient.AsyncHTTPClient()
          f = open(input_file)
          Mybuffer = f.read()
          
          #The mapper program is run with the input file piped in through stdin. The stdout of the 
          #mapper program (a list of key-value pairs) is buffered in memory.
          p = subprocess.Popen(mapper_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

          (out, _) = p.communicate(Mybuffer.encode())#decode('utf-8').encode('utf-8'))

#When the mapper program finishes, the outputs are partitioned into N lists of key-value pairs (one for 
#each reducer). Each of the N lists is sorted by key. To find the sorted list corresponding to a key-value 
#pair, the key is hashed (modulo the number of reducers).
          out = out.decode().split('\n')
          out2 =[]
          for line in out:
               line = line.split('\t')
               out2.append(line)

          output = partition(out2, N)
          for i in range(N):
               output[i].sort(key=lambda x: x[0])
               #hash by Key

#The N output lists from the map task must not be overwritten if the same process is used to run another 
#map task before the first task's outputs are retrieved. To distinguish outputs corresponding to different
# map tasks, a unique map_task_id is generated for each task, and the N output lists are associated with 
#this map_task_id.

          map_id = generateMapID()
          print (map_id)
          for i in range(N):
               kv_pairs = []
               pickle.dump( output[i], open(map_id + '_' + str(i) + ".p", "wb" ) )

          self.write(json.JSONEncoder().encode({"map_task_id": map_id, "status": "success"}))     
          #{"map_task_id": "98384451b4c1c1009091644dce8b74eb", "status": "success"}
          

class MOutputHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
     
          reducer_ix = str(self.get_argument('reducer_ix'))
          map_task_id = str(self.get_argument('map_task_id'))


#The RetrieveMapOutput handler takes as input a map task ID and a reducer index (in the range [0 ... N-1]).
# It returns a JSON-encoded list of key-value pairs that is associated with the given map task ID and 
#reducer index.
          #print 'trying to open ' + map_task_id + '_' + reducer_ix + ".p"
          try:
               kv_pairs = pickle.load( open(  map_task_id + '_' + reducer_ix + ".p", "rb" ) )
               self.write(json.JSONEncoder().encode(kv_pairs))     
          except:
               self.write(json.JSONEncoder().encode([]))
          