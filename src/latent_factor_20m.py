import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import logging

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
	df_train = df.sample(frac=0.8)
	df_rem = df.loc[~df.index.isin(df_train.index)]
	df_valid = df_rem.sample(frac=0.5)
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


def latent_factor_train(sparse_ratings_coo, k=500, lambda1=1, lambda2=1, epochs=50, lr=0.1):
	""" Learn a latent factor decomposition for the given ratings matrix

		sparse_ratings_coo = Q * P.transpose() 
		where
		Q: dense matrix of dimensions (n_users x k)
		P: dense matrix of dimensions (n_items x k)  
		Randomly initialize the matrices Q and P and then learn their values using Stochastic Gradient Descent

		Args
			sparse_ratings_coo : Sparse matrix with dimensions (n_users x n_items)
			k: Number of latent factors
			lambda1: Regularization parameter for Q
			lambda2: Regularization parameter for P
			epochs: Number of iterations for stochastic gradient descent
			lr: Learning rate for updates during Stochastic Gradient Descent

		Returns
			Q: dense matric of dimensions (n_users x k)
			P: dense matrix of dimensions (n_items x k)	
	"""   
    # Initialize the latent factor matrices randomly
	Q = np.random.uniform(0,0.1,size=(sparse_ratings_coo.shape[0],k))
	P = np.random.uniform(0,0.1,size=(sparse_ratings_coo.shape[1],k))   
	for ep in range(1,epochs+1):
		print("Epoch " + str(ep))
		for i,j,v in zip(sparse_ratings_coo.row, sparse_ratings_coo.col, sparse_ratings_coo.data):
			err = v - Q[i,:].dot(P[j,:])
            
			grad_Q = 2 * (lambda1 * Q[i,:] - (err)*P[j,:])  
			grad_P = 2 * (lambda2 * P[j,:] - (err)*Q[i,:])
            
			Q[i,:] -= lr * grad_Q
			P[j,:] -= lr * grad_P                                           
                                                       
	return Q,P    


def latent_factor_predict(sparse_test_coo, Q, P):
    """ Get the ratings for the test data points using the learnt Q and P matrices
    """
    pred = []
    
    for i,j in zip(sparse_test_coo.row, sparse_test_coo.col):
        pred.append(Q[i,:].dot(P[j,:]))
    
    return pred


# Read the ratings csv file into a pandas Dataframe
filename = '../data/ml-20m/ratings.csv'
df = pd.read_csv(filename)

n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]
print("Number of unique users: %d" % n_users)
print("Number of unique movies: %d" % n_items)

df_train, df_valid, df_test = split_data(df)

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


# Create sparse matrix for the training data
sparse_train_coo = create_sparse_coo_matrix(df_train, n_users, n_items, movie_dict)
sparse_valid_coo = create_sparse_coo_matrix(df_valid, n_users, n_items, movie_dict)
sparse_test_coo = create_sparse_coo_matrix(df_test, n_users, n_items, movie_dict)


# Hyperparameter search
latent_factors = [100, 300, 500, 700, 1000, 5000]
lrs = [0.01, 0.1, 1, 10]
lambdas = [0.1, 1, 10 ]

for k in latent_factors:
	for lr in lrs:
		for lambdar in lambdas:
			Q,P = latent_factor_train(sparse_train_coo, k, lambdar, lambdar, 20, lr)
			pred = latent_factor_predict(sparse_valid_coo, Q, P)
			actual = sparse_valid_coo.data
			mse = mean_squared_error(pred,actual)
			print("k: " + str(k) + ", lr: " + str(lr) + " lambda: "+str(lambdar) + " MSE: " + str(mse))

# Q,P = latent_factor_train(sparse_train_coo, 100, 1, 1, 1, 0.01)
# pred = latent_factor_predict(sparse_valid_coo, Q, P)
# actual = sparse_valid_coo.data
# mse = mean_squared_error(pred,actual)
# print(" MSE: " + str(mse))

