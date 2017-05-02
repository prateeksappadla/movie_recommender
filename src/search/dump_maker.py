import csv
from bs4 import BeautifulSoup
import requests
import pickle
from nltk.corpus import stopwords

IMDBmap = {}
#spamReader = csv.reader(open('links.csv'), delimiter=',', quotechar='|')
count = 0
filename = 'links.csv'
with open(filename) as f:
	reader = csv.reader(f)
	#read from links file to create a mapping of movie id to IMDB_ID
	for row in reader:
		if count:
			IMDBmap[int(row[0])] = row[1]
		count += 1
	print (count)
#For each movie
for i in range(1,164980):
	print(i)
	myMap = {}
	#Check if movie was in 'links.csv'
	try:
		imdbID = IMDBmap[i]
	except:
		continue
	#Store IMDB_ID
	myMap['imdbid'] = imdbID
	r = requests.get("http://www.imdb.com/title/tt" + imdbID + "/")
	data = r.text  
	soup = BeautifulSoup(data)
	#Search for movie title
	movtitle = soup.find('title').string
	title = movtitle.split('(')[0]
	myMap['title'] = title
	#Search for movie genre
	genre_list = soup.find_all('span', itemprop="genre")
	mylist = []
	for list in genre_list:
		mylist.append (str(list.string))
	myMap['genres'] = mylist
	#Search for movie directors
	try:
		#For some movies, directors may not be listed
		d_obj = soup.find('span', itemprop="director")
		t = d_obj.find('a')
		dir_name = [str(t.string)]
	#full_link = t['href']
		#Stores the director's IMBD id
		dir_id = [str(t['href'][6:15])]
		try:
			d2_obj = d_obj.find_next("span", itemprop="director")
			t2 = d2_obj.find('a')
			dir_name.append(str(t2.string))
			dir_id.append(str(t2['href'][6:15]))
		except:
			pass
	except:
		dir_name = []
		dir_id = []
	myMap['directors'] = dir_id
	myMap['dir_name'] = dir_name
	#Search for movie writer
	try:
		w_obj = soup.find('span', itemprop="creator")
		t = w_obj.find('a')
		# Stores the writers's IMBD id
		creat_id = str(t['href'][6:15])
		creator = str(t.string)
	except:
		creat_id = ''
		creator = ''
	myMap['writer'] = creat_id
	myMap['writer_name'] = creator
	#Search for movie rating
	try:
		rat = soup.find('span', attrs={"class":"rating"})
		rating = str(rat.next.string)
	except:
		rating = 'NA'
	myMap['rating'] = rating
	#Search for the movie's actors
	act  = soup.find_all('td',attrs={ "itemprop":"actor"})
	actr = act[:10]
	links = []
	names = []
	for actor in actr:
		temp = actor.find('a')
		# Stores the actor's IMBD id
		links.append(str(temp['href'][6:15]))
		names.append(str(temp.find('span').string))
	myMap['actors'] = links
	myMap['actor_names'] = names
	#Search for additional movie information, if available
	try:
		Coun = str(soup.find('h4', string = 'Country:').find_next('a').string)
	except:
		Coun = ''
	try:
		Rdate = str(soup.find('h4', string = 'Release Date:').next.next.string).split('(')[0][1:-1]
	except:
		Rdate	= ''
	#Rdate = str(soup.find('h4', string = 'Gross:').string).split('(')[0][1:-1]
	try:
		Gross = str(soup.find('h4', string = 'Gross:').next.next.string)[8:-16]
	except:
		Gross= ''
	try:
		Color = str(soup.find('h4', string = 'Color:').find_next('a').string)
	except:
		Color = ''
	myMap['country'] = Coun 
	myMap['release_date'] = Rdate
	myMap['gross'] = Gross
	myMap['color'] = Color
	#New page, to access more information
	r = requests.get("http://www.imdb.com/title/tt" + imdbID + "/trivia?tab=mc&ref_=tt_trv_cnn")
	data = r.text  
	soup = BeautifulSoup(data)
	Follow = []
	#Search for related movies
	try:
		b = soup.find('a',attrs = { "id":"follows"}).find_next('a')
		NextAvail = True
	except:
		NextAvail = False
	while (NextAvail):
		try:
			b2 = b.next.next.string[1:]
		except:
			NextAvail = False
			break
		breaks = str(b2).split('(')
		if(len(breaks) == 2):
			#Store the movie's IMDB_ID
			Follow.append(str(b['href'][7:]))
		b = b.find_next('a')
		try:
			c = b['href']
		except:
			NextAvail = False
	try:
		b = soup.find('a',attrs = { "id":"followed_by"}).find_next('a')
		NextAvail = True
	except:
		NextAvail = False
	while (NextAvail):
		try:
			b2 = b.next.next.string[1:]
		except:
			NextAvail = False
			break
		breaks = str(b2).split('(')
		if(len(breaks) == 2):
			Follow.append(str(b['href'][7:]))
		b = b.find_next('a')
		try:
			c = b['href']
		except:
			NextAvail = False
	myMap['related_movies'] = Follow
	#Search for movie synopsis
	### Temporarily removed
	#r  = requests.get("http://www.imdb.com/title/tt" + imdbID + "/synopsis?ref_=ttpl_pl_syn")
	#data = r.text
	#soup = BeautifulSoup(data)
	#a = soup.find('div', id='swiki.2.1')
	#pl = str(a)
	#p = pl.split('<')	
	#fin = []
	#for para in p:
	#	te = para.split('>')
	#	if(len(te)>1):
	#		fin.append(te)
	#synop = ''
	#for para in fin:
	#	if len(para[1]) > 2:
	#		synop += str(para[1])
	#synop = synop.replace('\n', ' \n ')
	synop = ''
	#Search for movie plot summary, if synopsis not found
	if len(synop) < 10:
		r  = requests.get("http://www.imdb.com/title/tt" + imdbID + "/plotsummary?ref_=tt_stry_pl/")
		data = r.text
		soup = BeautifulSoup(data)
		try:
			a = soup.find('p', attrs = {'class':'plotSummary'})
			synop = str(a.string)
			synop = synop.replace('\n', ' \n ')
		except:
			pass
	syn = ' '.join([word for word in synop.split() if word not in stopwords.words("english")])
	myMap['synopsis'] = syn
	
	#Dump all the movie information
	filename = 'assignment4/NewMov/Movie' + str(i) + '.p'
	pickle.dump( myMap, open( filename, "wb" ) )


