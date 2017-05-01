import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import logging
import argparse

logger = logging.getLogger('Latent Factor Model')
logger.setLevel(logging.INFO)

def split_data(df):
	""" Split the data into training, validation and test partitions by random sampling.

	    80% of the data is randomly sampled to be the training partition.
	    10% is held out as a validation dataset to tune the hyperparameters.
	    10% is held out as a test partition to test the final performance of the model.

	    Args
	    	df: pandas dataframe object containing the dataset

	    Returns
			df_train: Dataframe corresponding to training partition
			df_valid: Dataframe corresponding to validation partition
			df_test: Dataframe corresponding to test partition
	"""
	random_seed = 1
	df_train = df.sample(frac=0.8, random_state=random_seed)
	df_rem = df.loc[~df.index.isin(df_train.index)]
	df_valid = df_rem.sample(frac=0.5, random_state=random_seed)
	df_test = df_rem.loc[~df_rem.index.isin(df_valid.index)]
	logger.info("Shape of training dataframe: " + str(df_train.shape))
	logger.info("Shape of validation dataframe: " + str(df_valid.shape))
	logger.info("Sahpe of test dataframe: " + str(df_test.shape))

	return df_train, df_valid, df_test


def create_sparse_coo_matrix(df, n_users, n_items, movie_dict):
	""" Create a scipy sparse coo matrix from the given dataframe 

		Args
			df: Dataframe object to be converted to sparse matrix
			n_users: Number of rows in the sparse matrix
			n_items: Number of columns in the sparse matrix
			movie_dict: Dictionary mapping the movies in the dataset to a movie id

		Returns
			sparse_matrix_coo (scipy.sparse.coo_matrix): Sparse matrix in COO form	
	"""

	# Map the movie_ids in the data to the new movie_ids given by the dictionary movie_dict
	movie_id_list = list(map(lambda x: movie_dict[x], df['movieId'].tolist()))
	# Map the user_id in the dataframe to userid - 1 [to account for zero based indexing]
	user_id_list = list(map(lambda x: x - 1, df['userId'].tolist()))
	sparse_matrix_coo = sparse.coo_matrix((df['rating'].tolist(),(user_id_list, movie_id_list)),shape=(n_users,n_items))
	logger.debug("Shape of created sparse matrix: " + str(sparse_matrix_coo.shape))
	logger.debug("Number of non_zero elements in the sparse matrix: " + str(sparse_matrix_coo.nnz))
	logger.debug("Number of entries in the input dataframe:[should match the number of non zero entries in sparse matrix] " + str(df.shape[0]))
	return sparse_matrix_coo


def create_moviedict(df):
	# README file for the dataset: http://files.grouplens.org/datasets/movielens/ml-20m-README.html
	# User-ids are in the range (1, 138493). We just subract 1 from each userId to convert the range to (0,138492)
	# Total number of movies are 27278 but the the range of movieIds is bigger than (1,27278)
	# We need to map the movieIds to the range (0,27277)
	# Only movies with at least one rating or tag are included in the dataset. As we see above, the number of unique movies
	# for which we have atleast one rating is 26744 
	ind = 0
	movie_list = [] # List which is reverse of movie_dict, contains original movieId at index 'new id'
	movie_dict = {}   # Dictionary from original movieId to new id
	for movieId in df['movieId'].unique():
	    movie_list.append(movieId)
	    movie_dict[movieId] = ind
	    ind += 1
	return movie_list, movie_dict    
