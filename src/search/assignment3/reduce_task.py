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


class ReducerHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
          
          reducer_ix = str(self.get_argument('reducer_ix'))
          reducer_path = str(self.get_argument('reducer_path'))
          map_task_ids = str(self.get_argument('map_task_ids')).split(',') 
          # map_task_ids = ["8a97fd755ea12827485749036e15d651", "d3486112191e4717d17d4fba189bdbf6"]
          job_path = str(self.get_argument('job_path'))
          servers = inventory.URL#["linserv2.cims.nyu.edu:34514", "linserv2.cims.nyu.edu:34515"]

          num_mappers = len(map_task_ids)

          http = httpclient.AsyncHTTPClient()
          futures = []
          for i in range(num_mappers):
              server = servers[i % len(servers)]
              params = urllib.parse.urlencode({'reducer_ix': reducer_ix,
                                               'map_task_id': map_task_ids[i]})
              url = "http://%s/retrieve_map_output?%s" % (server, params)
              print("Fetching in reducer", url)
              futures.append(http.fetch(url))
          responses = yield futures
          kv_pairs = []
          for r in responses:
              kv_pairs.extend(json.loads(r.body.decode()))
          kv_pairs.sort(key=lambda x: x[0])
          filename = ''
          if 'idfA' in reducer_path:
              filename = './IDFMapA'
          elif 'idf' in reducer_path:
              filename = './IDFMap'
          elif 'doc' in reducer_path:
              filename = './DocPar' + str(reducer_ix) 
          else:
              filename = './IndPar' + str(reducer_ix) 

          temp = sys.stdout
          kv_string = "\n".join([pair[0] + "\t" + pair[1] for pair in kv_pairs])
          p = subprocess.Popen(reducer_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
          (out, _) = p.communicate(kv_string.encode())#decode('utf-8').encode('utf-8'))
          a = pickle.loads(out)
          #print (a)
          pickle.dump(a,  open( filename + '.p', "wb" ))
          sys.stdout = temp
          self.write(json.JSONEncoder().encode({"status": "success"}))     

class ROutputHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
          job_path = str(self.get_argument('job_path'))
          num_reducers = int(self.get_argument('num_reducers'))

          for i in range(num_reducers):
               filename = job_path + '/' + str(i) + '.out'
               self.write(filename)
               self.write('<br>')
               try:
                    f = open(filename, 'r')
                    for line in f:
                         self.write(line)
                         self.write('<br>')
               except:
                    print ('file not found')
                    self.write('<br>')
