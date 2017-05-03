from assignment2 import inventory
import tornado.ioloop
import tornado.web
import tornado.httpserver
from tornado import gen, httpserver, web, netutil, process
import json
from tornado.httpclient import AsyncHTTPClient
import pickle

def dotProd(a, b):
    if len(a) == len(b):
        score = 0
        for i in range(len(a)):
            score += a[i] *b[i]
        return score
    else:
        return 0


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, port):
        self._port = port

    def get(self):
        query = str(self.get_argument('q'))
        query = query.replace("MYDELIM", " ")
        http_client = AsyncHTTPClient()
        PartitionNum = self._port - I_BASE_PORT
        #print ('Partition: ')
        #print (PartitionNum)
        #print (query)
        postings = pickle.load( open( "IndPar" + str(PartitionNum) + ".p", "rb" ) )
        IDF = pickle.load( open( "IDFMap.p", "rb" ) )
        #print ('able to finish')
        #it first uses the document frequency table to create a vector-space representation of the query. 
        #Each dimension of the vector should be set to the corresponding term's TF-IDF value. For the query, 
        #you can set the TF of each term to 1.
        Qu = query.split(' ')
        QueryVector = []
        #print ('starting loop')
        for term in Qu:
            if term in IDF: #IDF.has_key(term):
                QueryVector.append(IDF[term])
            else:
                QueryVector.append(0)           
        
        #QueryVector = [term_1_tfidf, term_2_idf, term3_idf .... ]

         #It then looks up the postings list for each term in the query. Each document gets converted to its 
         #vector space representation. Each document vector should have the same number of dimensions as the 
         #query vector. Again, each dimension should be set to the corresponding term's TF-IDF value       
        #print 'QueryVector'
        #print QueryVector
        term_no = -1
        MyMap = {}
        for term in Qu:
            term_no +=1
            #print term_no
            if term in postings:
                DocList = postings[term]
            #print 'List start'
            #print DocList
            #print 'List over'
            #print len(DocList)
                for DocItem in DocList:
                    DocID = DocItem[0]
                    DocIDF = DocItem[1]
                    if term in IDF: #IDF.has_key(term):
                        DocIDF = float(DocIDF)*IDF[term]
                    else:
                        DocIDF = 0  
                    #print 'DocID: ' + str(DocID) 
                    #print 'IDF: ' + str(DocIDF)
                    if DocID in MyMap: #MyMap.has_key(DocID):
                         MyMap[DocID][term_no] = DocIDF
                    else:
                        MyMap[DocID] = [Q * 0 for Q in QueryVector]
                        MyMap[DocID][term_no] = DocIDF

        #The documents are then scored. Each document's score is the inner product (a.k.a. dot product, 
        #effectively correlation here) of its vector and the query vector. In addition, scores should be 
        #biased so that documents with the query terms in their title receive especially high scores.
        ScoredListing = []
        for docID in MyMap:
            score = dotProd(MyMap[docID], QueryVector)
            ScoredListing.append([score, docID])        ## Bias comes from tfs itself
        
        #Finally, the K highest-scoring documents are written out as a JSON-encoded list of (docID, score) pairs.
        ScoredListing.sort(reverse=True)                            ## check so sorted by score
        topPostings = ScoredListing[:30]
        TP = []
        for posting in topPostings:
            TP.append([posting[1], posting[0]])
        #print 'Length from ' + str(self._port)
        #print len(ScoredListing)
        self.write(json.JSONEncoder().encode({"postings": TP}))    

if __name__ == "__main__":

    task_id = process.fork_processes(inventory.num_ind)
    I_BASE_PORT = inventory.port1   
    #application.listen(I_BASE_PORT)
    port = I_BASE_PORT + task_id
    app = httpserver.HTTPServer(web.Application([web.url(r"/index", IndexHandler, dict(port=port))]))
    app.add_sockets(netutil.bind_sockets(port))
    print ('indexServer' + str(task_id) + ' at port: ' + str(port))
    tornado.ioloop.IOLoop.current().start()