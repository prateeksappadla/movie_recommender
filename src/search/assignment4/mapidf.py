#!/usr/bin/env python3

import sys, nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter


tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('* [[Apache Wave]]')


Titles = []
Texts = []
Person = []
diction = []
CollapsedTexts = []
CollapsedTitles = []
i = -1
#### TODO temporary
docIDs=[0,1,2]
num_ind  = 3

for ln in sys.stdin:
    i+=1
    lin = ln[2:-3]
    lin = lin.replace('\"', '\'')
    line = lin.split('], [')
    title = line[0][1:-1].lower()
    Titles.append(tokenizer.tokenize(title))
    countTitle = Counter(Titles[i])
    #CollapsedTitles.append(count)
    ids = line[1][1:-1]
    txt = line[2][1:-1].lower()
    Texts.append(tokenizer.tokenize(txt))
    countText = Counter(Texts[i])
    
    people = ''
    temp = line[3][2:-1]
    actlist = temp.split('\', \'')
    for act in actlist:
        people += act + ' '
    temp = line[4][1:-1]
    dirlist = temp.split('\', \'')
    for direc in dirlist:
        people += direc + ' '
    people += line[5][1:-1]
    Person.append(tokenizer.tokenize(people))
    countPerson = Counter(Person[i])
    #CollapsedTexts.append(count)
    #diction.append(title + ' ' + title + ' ' + title + ' ' + title + ' ' + txt)
    Count = countText + countTitle 
    for term in Count:
         print('%s\t%s' % (term, ids))

