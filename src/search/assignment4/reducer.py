#!/usr/bin/env python3

from itertools import groupby
from operator import itemgetter
import sys
import pickle

InvtIndex = {}

data = map(lambda x: x.strip().split('\t'), sys.stdin)
for doc_id, group in groupby(data, itemgetter(0)):
    for _,MyP in group:
        MyL = MyP.split(',')
        #print (doc_id, MyL)
        if len(MyL) == 2:
            #print ('success')
            if MyL[0] in InvtIndex:
                InvtIndex[MyL[0]].append((doc_id, MyL[1]))
            else:
                InvtIndex[MyL[0]] = [(doc_id, MyL[1])]
        else:
            print('Error with partition')
            print(doc_id
            	)
            print('DELIM1')
            print(MyP) 

#for term in InvtIndex:
#	print(term, InvtIndex[term])

pickle.dump(InvtIndex,sys.stdout.buffer)

    #print('%s\t%d' % (word, total))