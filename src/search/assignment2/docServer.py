from assignment2 import inventory
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado import gen, httpserver, process, web, netutil
import json
from tornado.httpclient import AsyncHTTPClient
import pickle
import re

### Modified to facilitate person search

class DocHandler(tornado.web.RequestHandler):
    def initialize(self, port):
        self._port = port
    
    def get(self):
        query = str(self.get_argument('q'))
        query = query.replace("MYDELIM", " ")
        doc_id = int(self.get_argument('id'))
        #A document server is also an HTTP server. It receives requests via its GET method that consist of a 
        #document ID and a query. In response, it uses its document store to return detailed document data 
        #(title, URL, and snippet).
        http_client = AsyncHTTPClient()
        PartitionNum = self._port - D_BASE_PORT
        urltemp = 'http://www.imdb.com/title/tt'
        Docpostings = pickle.load( open( "DocPar" + str(PartitionNum) + ".p", "rb" ) )
        if doc_id in Docpostings: 
            #The snippet should be a relevant chunk of text from the document, and terms from the query should 
            #be emphasized. Don't worry too much  about cleaning up MediaWiki markup.
            result = Docpostings[doc_id][0]        
            MyDict = {}
            MyDict['imdbID'] = doc_id
            MyDict['title'] = result[0]
            titleSp = result[0].split()
            isUnd = False
            myURL = urltemp + str(doc_id) + '/'
            MyDict['url'] = myURL
            MyDict['snippet'] = ''
            MyText = result[0] + result[1]
            Qu = query.split(' ')
            snippetStart = -1
            found_one_term = False
            for term in Qu:
                positions = [m.start() for m in re.finditer(term.lower(), MyText.lower())]
                if (len(positions) >0):
                    found_one_term = True
                    for i in range(len(positions)):
                        #Search for temr in the text: plot or title
                        newPos = [m.start() for m in re.finditer(term.lower(), MyText.lower())]
                        pos = newPos[i]
                        begin = pos-1 if pos else 0
                        MyText = MyText[:begin] + ' <strong>' + MyText[pos:pos +len(term)] + '</strong> ' + MyText[pos +len(term)+1:]
                        if snippetStart < 0:
                            snippetStart = begin-50 if begin>50 else 0
                        else: 
                            if pos<snippetStart:
                                snippetStart += 19
                    snippet = ''
                    if snippetStart >0:
                        while (MyText[snippetStart] != ' '):
                            snippetStart +=1
                        snippet = '... '
                    if(len(MyText) > snippetStart+400):
                        snipend = snippetStart + 400
                        try:
                            while (MyText[snipend] != ' '):
                                snipend +=1
                            snippet += MyText[snippetStart:snipend]
                        except:
                            snippet +=MyText[snippetStart:]
                        if(len(MyText) > snippetStart+400):
                            snippet += ' ...'
                    else: 
                        snippet += MyText[snippetStart:]
                    MyDict['snippet'] += snippet
            if found_one_term == False:
                    #Most likely, the term was a person
                    self.write('exception: term not found in document')
                    print ('could not find term ' + str(doc_id) + ' ' + str(term))
                    self.write(MyText)
                    #print (MyText)
                    MyDict['found'] = False
            else:
                    MyDict['found'] = True
            RS = [MyDict]
            self.write(json.JSONEncoder().encode({"results": RS}))            ### print as a dictionary
        else:
            ### Raise exception
            self.write('exception: document not found')            ### print as a dictionary
            print ('could not find ' + str(doc_id))
            
#The output should be JSON-encoded
class DocActHandler(tornado.web.RequestHandler):
    def initialize(self, port):
        self._port = port
    #Mostly the same, minor but different search
    def get(self):
        query = str(self.get_argument('q'))
        query = query.replace("MYDELIM", " ")
        doc_id = int(self.get_argument('id'))
        #A document server is also an HTTP server. It receives requests via its GET method that consist of a 
        #document ID and a query. In response, it uses its document store to return detailed document data 
        #(title, URL, and snippet).
        http_client = AsyncHTTPClient()
        PartitionNum = self._port - D_BASE_PORT
        urltemp = 'http://www.imdb.com/title/tt'
        Docpostings = pickle.load( open( "DocPar" + str(PartitionNum) + ".p", "rb" ) )
        if doc_id in Docpostings: #Docpostings.has_key(doc_id):
            #The snippet should be a relevant chunk of text from the document, and terms from the query should 
            #be emphasized. Don't worry too much  about cleaning up MediaWiki markup.
            result = Docpostings[doc_id][0]        
            MyDict = {}
            MyDict['imdbID'] = doc_id
            MyDict['title'] = result[0]
            titleSp = result[0].split()
            isUnd = False
            myURL = urltemp + str(doc_id) + '/'
            MyDict['url'] = myURL
            MyDict['snippet'] = ''
            MyDict['Actors'] = ''
            MyDict['Directors'] = ''
            MyDict['Writers'] = ''
            MyText = result[0] + result[1]
            Actors = result[2]
            Directors = result[3]
            Writers = result[4]
            #uniqiue delimiter for person
            PersonText =  Actors + ' PERDELIM ' + Directors + ' PERDELIM ' + Writers
            Qu = query.split(' ')
            snippetStart = -1
            found_one_term = False
            for term in Qu:
                #Search for temr in the Person: actor, director or writer
                positions = [m.start() for m in re.finditer(term.lower(), PersonText.lower())]
                if (len(positions) >0):
                    found_one_term = True
                    for i in range(len(positions)):
                        #print positions[i]
                        newPos = [m.start() for m in re.finditer(term.lower(), PersonText.lower())]
                        pos = newPos[i]
                        begin = pos-1 if pos else 0
                        PersonText = PersonText[:begin] + ' <strong>' + PersonText[pos:pos +len(term)] + '</strong> ' + PersonText[pos +len(term)+1:]
                    PersonList = PersonText.split(' PERDELIM ')
                    MyDict['Actors'] += PersonList[0]
                    MyDict['Directors'] += PersonList[1]
                    MyDict['Writers'] += PersonList[2]
                    snippet = ''
                    if(len(MyText) <400):
                        MyDict['snippet'] += MyText
                    else:
                        snipend = 400
                        try:
                            while (MyText[snipend] != ' '):
                                snipend +=1
                            snippet += MyText[:snipend]
                        except:
                            snippet +=MyText
                        if(len(MyText) > 400):
                            snippet += ' ...'
            if found_one_term == False:
                    #Most likely, the term was in text
                    #self.write('exception: term was not a person in document')
                    print ('term was not find a person in ' + str(doc_id) + ' ' + str(term))
                    #self.write(MyText)
                    #print (MyText)
                    MyDict['found'] = False
            else:
                MyDict['found'] = True
            RS = [MyDict]
            self.write(json.JSONEncoder().encode({"results": RS}))            ### print as a dictionary
        else:
            ### Raise exception
            self.write('exception: document not found')            ### print as a dictionary
            print ('could not find ' + str(doc_id))

if __name__ == "__main__":

    task_id = process.fork_processes(inventory.num_doc)
    
    D_BASE_PORT = inventory.port2 
    port = D_BASE_PORT + task_id 

    app = httpserver.HTTPServer(web.Application([web.url(r"/doc", DocHandler, dict(port=port)), web.url(r"/docA", DocActHandler, dict(port=port))]))
    app.add_sockets(netutil.bind_sockets(port))
    print ('docServer' + str(task_id) + ' at port: ' + str(port))
    tornado.ioloop.IOLoop.current().start()