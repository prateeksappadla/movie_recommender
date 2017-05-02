#!/usr/bin/env python3

import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado import gen, httpserver, web, netutil, process
import json
from tornado.httpclient import AsyncHTTPClient
import pickle
import xml.etree.ElementTree as ET
from optparse import OptionParser
import sys


parser = OptionParser()
parser.add_option("--job_path", dest="jp")#, type="string")
parser.add_option("--num_partitions", dest="np")#, type="string")
(options, args) = parser.parse_args()
job_path = options.jp 
num_partitions = int(options.np)
print('starting')
i = -1
Partitions = []
for i in range(num_partitions):
    empty = []
    Partitions.append(empty) 
j = 0
### Modified to read from the created dump. 
for i in range(1,164980):
    try:
        a = pickle.load( open("assignment4/NewMov/Movie" + str(i) + ".p", "rb"))
        #PageDet stores the relevant details from the dump into a new list
        pageDet = []
        title = [a['title']]
        pageDet.append(title)
        ids = [a['imdbid']]
        pageDet.append(ids)
        txt = [a['synopsis']]
        pageDet.append(txt)
        actors = a['actor_names']
        directors = a['dir_name']
        writers = [a['writer_name']]
        temp = [actors, directors, writers]
        pageDet.append(temp)
        Partitions[j%num_partitions].append(pageDet)
        j = j+1
    except:
        pass

print(j)

for i in range(num_partitions):
    f = open(job_path + '/' + str(i) + '.in', 'w')
    for page in Partitions[i]:
        f.write("%s\n" % page)



