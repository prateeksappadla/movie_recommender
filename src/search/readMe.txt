IMDB: 
Transfer file 'links.csv' from the download file to your directory. 
Then, run command: 

python dump_maker.py 

This should create a pickle file for each movie with the following attributes:
imdbid, title, genres, directors, dir_name, writer, writer_name, rating, actors, actor_names, country, release_date, gross, color, related_movies, synopsis. 

Here, we've stored upto 2 directors, 1 writer, and 10 actors, as provided in the IMDB page. The attribute 'related_movies' contains all the movies which were listed under the tags 'follows', or 'followed_by', usually indicating other movies which are in the same series as the movie we are viewing. The 'synopsis' currently contains the first provided 'Plot summary' for the movie. 

We have provided support for 2 searches, plot search and person search. Plot search searches for a query in the movie's plot, and is accessed with the extension '/search?'. Person search searches for the query in the names of the actors, directors and writers, and is accessed with the extension '/person?'

----------------------
In order to run the system, navigate to the search_movies folder, then execute the following commands in a shell: 

./assignment4/reformat_all.sh
python3.6 -m assignment3.workers
python3.6 -m assignment4.start
python3.6 -m assignment2.start 


On doing this, you'll be provided with the host name and the port on which to run the search. Some example search queries are:

http://172-17-36-176.dynapool.nyu.edu:22433/search?q=avenge
http://172-17-36-176.dynapool.nyu.edu:22433/person?q=Cage
http://172-17-36-176.dynapool.nyu.edu:22433/search?q=hill

Note that searching for person will only display results for personnel, not movie plots, and visa-versa.
