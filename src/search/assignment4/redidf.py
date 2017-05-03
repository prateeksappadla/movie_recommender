#!/usr/bin/env python3

from itertools import groupby
from operator import itemgetter
import sys
import pickle
import math
#from sets import set

doc_ids = set()
document_count = {}

data = map(lambda x: x.strip().split('\t'), sys.stdin)
for term, group in groupby(data, itemgetter(0)):
    total = 0
    for _, doc_id in group:
        total += 1
        doc_ids.add(doc_id)

    #print('%s\t%d' % (term, total))
    if term in document_count:
        document_count[term].append(total)
    else:
        document_count[term] = total

IDFmap = {}
for term in document_count:
    IDFmap[term] = math.log(len(doc_ids) / float(document_count[term]))
    #print (term, document_count[term])

#for term in IDFmap:
    #print (term, IDFmap[term])

pickle.dump(IDFmap,sys.stdout.buffer)
#print(pickle.dumps(IDFmap))

    #print('%s\t%d' % (word, total))