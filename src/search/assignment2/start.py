import tornado.ioloop
import tornado.web
import tornado.httpserver
import hashlib
import getpass
import socket
from tornado import gen, httpserver, process
import json
from assignment2 import inventory, inventory2
import os
from tornado.httpclient import AsyncHTTPClient


SETTINGS = {'static_path': "webapp/"}

class FrontEndHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        name = self.get_argument('q', None)
        print ('searching for ' + str(name))
        name = name.replace(" ", "MYDELIM")
        http_client = AsyncHTTPClient()
        postings = []
        for temp_url in inventory.indUrl:
            url = "http://%s:%s%s" % (socket.gethostname(),temp_url, name)
            response = yield http_client.fetch(url)
            resp = json.loads(response.body.decode())
            postList = resp['postings']
            listText = response.body[15:-3]
            for post in postList:
                int1 = int(post[0])
                int2 = float(post[1])
                intpair = [int2,int1]
                postings.append(intpair)
        postings.sort(reverse=True)
        results = []
        num_results = 0
        loops = 0
        for posting in postings:
            if (len(results) > 9 ):
                break
            loops = loops + 1
            DocID = posting[1]
            docPartNum = DocID% inventory.num_doc
            temp_url = inventory.docUrl[docPartNum] 
            url = "http://" + str(socket.gethostname()) + ":" + temp_url + str(DocID) + "&q=" + str(name)
            print (url)
            response = yield http_client.fetch(url)
            listText = response.body[14:-3]
            resp = json.loads(response.body.decode())
            DocList = resp['results'][0]
            if DocList.pop('found'):
                results.append(DocList)

        self.finish(json.JSONEncoder().encode({"num_results": len(results), "results": results}))
        print (str(len(results)) + ' results found! in ' + str(loops) + ' loops' )
class actorHandler(tornado.web.RequestHandler):
    @gen.coroutine
    def get(self):
        name = self.get_argument('q', None)
        print ('searching for ' + str(name))
        name = name.replace(" ", "MYDELIM")
        http_client = AsyncHTTPClient()
        postings = []
        for temp_url in inventory2.indUrl:
            url = "http://%s:%s%s" % (socket.gethostname(),temp_url, name)
            response = yield http_client.fetch(url)
            resp = json.loads(response.body.decode())
            postList = resp['postings']
            listText = response.body[15:-3]
            for post in postList:
                int1 = int(post[0])
                int2 = float(post[1])
                intpair = [int2,int1]
                postings.append(intpair)
        postings.sort(reverse=True)
        results = []
        loops = 0
        for posting in postings:
            if(len(results) > 9):
                break
            loops = loops + 1
            DocID = posting[1]
            docPartNum = DocID% inventory2.num_doc
            temp_url = inventory2.docUrl[docPartNum] 
           
            url = "http://" + str(socket.gethostname()) + ":" + temp_url + str(DocID) + "&q=" + str(name)
            #print (url)
            response = yield http_client.fetch(url)
            listText = response.body[14:-3]
            resp = json.loads(response.body.decode())
            DocList = resp['results'][0]
            if (DocList['found']):
                results.append(DocList)

        self.finish(json.JSONEncoder().encode({"num_results": len(results), "results": results}))
        #print 'Results: '
        print (str(len(results)) + ' results found! in ' + str(loops) + ' loops')


if __name__ == "__main__":

    task_id = process.fork_processes(3)
    if(task_id ==0):
    
        application = tornado.web.Application([
            (r"/search", FrontEndHandler), (r"/person", actorHandler)
        ], **SETTINGS)
        MAX_PORT = 49152
        MIN_PORT = 10000
        BASE_PORT = int(hashlib.md5(getpass.getuser().encode()).hexdigest()[:8], 16) % \
        (MAX_PORT - MIN_PORT) + MIN_PORT
        print ('FrontEnd at port: ' + str(BASE_PORT))
        print ('Host name: ' + str(socket.gethostname()))
        application.listen(BASE_PORT)
    elif task_id == 1:
        os.system('python3.6 -m assignment2.indexServer')
    else:
        os.system('python3.6 -m assignment2.docServer')

    tornado.ioloop.IOLoop.current().start()