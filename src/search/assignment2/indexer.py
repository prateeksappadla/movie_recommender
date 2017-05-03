import nltk
import string
import xml.etree.ElementTree as ET
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import inventory
import pickle
from math import log

tokenizer = RegexpTokenizer(r'\w+')
tokenizer.tokenize('* [[Apache Wave]]')
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()

tree = ET.parse('info_ret.xml')
root = tree.getroot()

urltemp = 'http://www.imdb.com/title/tt'

def tokenize(text):
	return tokenizer.tokenize(txt)

i = -1
Titles = []
Texts = []
CollapsedTexts = []
CollapsedTitles = []
OrigText = []
OrigTitles = []
docIDs = []
diction = {}
for pg in root.findall('{http://www.mediawiki.org/xml/export-0.10/}page'):
	i+=1
  	title = pg.find('{http://www.mediawiki.org/xml/export-0.10/}title').text
  	OrigTitles.append(title)
  	title = title.lower()
  	#no_punctuation = lowers.translate(None, string.punctuation)
  	Titles.append(tokenizer.tokenize(title))
  	count = Counter(Titles[i])
  	CollapsedTitles.append(count)
  	#Titles[i] = [lancaster.stem(t) for t in Titles[i]]
	ids = int(pg.find('{http://www.mediawiki.org/xml/export-0.10/}id').text)
	docIDs.append(ids)
  	for rvs in pg.findall('{http://www.mediawiki.org/xml/export-0.10/}revision'):
  		txt = rvs.find('{http://www.mediawiki.org/xml/export-0.10/}text').text
  		OrigText.append(txt)
  		txt = txt.lower()
  		#no_punctuation = lowers.translate(None, string.punctuation)
  		Texts.append(tokenizer.tokenize(txt))
  		count = Counter(Texts[i])
  		CollapsedTexts.append(count)
  		#Texts[i] = [lancaster.stem(t) for t in Texts[i]]
  		diction[i] = title + ' ' + title + ' ' + title + ' ' + title + ' ' + txt 	
  		#### USE title multiple times to increase term weightage in title

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
tfs = tfidf.fit_transform(diction.values())
feature_names = tfidf.get_feature_names()
## Implement later?

NumIndPart = inventory.num_ind
NumDocPart = inventory.num_doc
IndPostings = []
DocPostings = []

DocFreq = {}

for i in range(NumIndPart):
	Mappering = {}
	IndPostings.append(Mappering)		### type of IndPostings

for i in range(NumDocPart):
	Mappering = {}
	DocPostings.append(Mappering)		### type of DocPostings

DocWords = []
for i in range(len(docIDs)):
	myIndPart = docIDs[i] % NumIndPart			#docID modulo nPartitions
	DocWords.append(CollapsedTexts[i] + CollapsedTitles[i])
	for term in DocWords[i]:
		termList = [ docIDs[i], DocWords[i][term]]#tfs[docIDs[i], tfidf.vocabulary_[term]] ] 			
		if IndPostings[myIndPart].has_key(term):
 			IndPostings[myIndPart][term].append(termList)
		else:
			IndPostings[myIndPart][term] = []
			IndPostings[myIndPart][term].append(termList)
		DFList = [docIDs[i], DocWords[i][term]]
		if DocFreq.has_key(term):
			DocFreq[term].append(DFList)
		else:
			DocFreq[term] = []
			DocFreq[term].append(DFList)
	myDocPart = docIDs[i] % NumDocPart			#docID modulo nPartitions
	mytitle = OrigTitles[i]
	titleSp = mytitle.split()
	myURL = urltemp
	myURL += str(docIDs[i]) + '/'
	termList = [docIDs[i], mytitle, myURL, OrigText[i]]
	if DocPostings[myDocPart].has_key(docIDs[i]):
 		DocPostings[myDocPart][docIDs[i]].append(termList)
	else:
	   	DocPostings[myDocPart][docIDs[i]] = [termList]	

for i in range(NumIndPart):
	posting = IndPostings[i]
	filename = 'IndPar' + str(i) + '.p'
	pickle.dump( posting, open( filename, "wb" ) )

for i in range(NumDocPart):
	results = DocPostings[i]
	filename = 'DocPar' + str(i) + '.p'
	pickle.dump( results, open( filename, "wb" ) )

IDFMap = {}
for term in DocFreq:
	IDFMap[term] = log(float(len(docIDs) +1)/len(DocFreq[term])) 

filename = 'DocFreq.p'
pickle.dump( DocFreq, open( filename, "wb" ) )

filename = 'IDFMap.p'
pickle.dump( IDFMap, open( filename, "wb" ) )

