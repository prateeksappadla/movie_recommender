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


def main():
     num_process = inventory.Num_Workers
     port = inventory.BASE_PORT 
     app = httpserver.HTTPServer(tornado.web.Application([
                 (r"/map", map_task.MapperHandler),(r"/retrieve_map_output", map_task.MOutputHandler), (r"/reduce", reduce_task.ReducerHandler),(r"/retrieve_reduce_output", reduce_task.ROutputHandler)
                   ]))
          #log.info('Front end is listening on %d', port)
     
     for i in range(num_process):
          app.add_sockets(netutil.bind_sockets(port + i))
          print ('A worker is listening on: ' + str(socket.gethostname()) + ":" + str(port + i))
     
     IOLoop.current().start()

if __name__ == "__main__":
     main()