#!/usr/bin/env python3

from itertools import groupby
from operator import itemgetter
import sys
import pickle

DocStore = {}

data = map(lambda x: x.strip().split('\t'), sys.stdin)
for doc_id, group in groupby(data, itemgetter(0)):
    for _,MyP in group:
        #print (doc_id, MyP)
        MyL = MyP.split('DEL__IM')
        if len(MyL) == 5:
            try:
                d_id = int(doc_id)
                DocStore[d_id] = [(MyL[0], MyL[1], MyL[2], MyL[3], MyL[4])]
            except:
                DocStore[doc_id] = [(MyL[0], MyL[1], MyL[2], MyL[3], MyL[4])]
        #else:
            #print('Error with partition')
            #print(doc_id)
            #print('DELIM1')
            #print(MyP) 

#for term in DocStore:
	#print(term, DocStore[term])

pickle.dump(DocStore,sys.stdout.buffer)

#print(pickle.dumps(DocStore))

    #print('%s\t%d' % (word, total))