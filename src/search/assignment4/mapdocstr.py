#!/usr/bin/env python3

import sys, nltk
from nltk.tokenize import RegexpTokenizer
from collections import Counter

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('* [[Apache Wave]]')

### Modified to facilitate person search

Titles = []
Texts = []
diction = []
Person = []
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
    actors = ''
    directors = ''
    writer = ''
    actlist = temp.split('\', \'')
    for act in actlist:
        actors += act + ' ; '
    temp = line[4][1:-1]
    dirlist = temp.split('\', \'')
    for direc in dirlist:
        directors += direc + ' ; '
    writer += line[5][1:-1]
    people = actors + directors + writer
    Person.append(tokenizer.tokenize(people))
    countPerson = Counter(Person[i])
    #CollapsedTexts.append(count)
    #diction.append(title + ' ' + title + ' ' + title + ' ' + title + ' ' + txt)
    Count = countText + countTitle
    print('%s\t%sDEL__IM%sDEL__IM%sDEL__IM%sDEL__IM%s' % (ids, title, txt, actors, directors, writer))

