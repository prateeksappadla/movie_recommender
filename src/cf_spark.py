from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse

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
	# logger.info("Shape of training dataframe: " + str(df_train.shape))
	# logger.info("Shape of validation dataframe: " + str(df_valid.shape))
	# logger.info("Sahpe of test dataframe: " + str(df_test.shape))

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
	# logger.debug("Shape of created sparse matrix: " + str(sparse_matrix_coo.shape))
	# logger.debug("Number of non_zero elements in the sparse matrix: " + str(sparse_matrix_coo.nnz))
	# logger.debug("Number of entries in the input dataframe:[should match the number of non zero entries in sparse matrix] " + str(df.shape[0]))
	return sparse_matrix_coo


# Configure Spark
appName = "Recommender"
master = "local[4]"

conf = SparkConf().setAppName(appName).setMaster(master)
conf.set("spark.executor.heartbeatInterval","3600s")
sc = SparkContext(conf=conf)

# Read the input file
filename = '../data/ml-latest-small/ratings.csv'
df = pd.read_csv(filename)

n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]
print("Number of unique users: %d" % n_users)
print("Number of unique movies: %d" % n_items)

ind = 0
movie_list = [] # List which is reverse of movie_dict, contains original movieId at index 'new id'
movie_dict = {}   # Dictionary from original movieId to new id
for movieId in df['movieId'].unique():
    movie_list.append(movieId)
    movie_dict[movieId] = ind
    ind += 1

df_train, df_valid, df_test = split_data(df)
sparse_train_coo = create_sparse_coo_matrix(df_train, n_users, n_items, movie_dict)
sparse_valid_coo = create_sparse_coo_matrix(df_valid, n_users, n_items, movie_dict)
sparse_test_coo = create_sparse_coo_matrix(df_test, n_users, n_items, movie_dict)

k = 300
lambda1 = 1
lambda2 = 1
lr = 0.01
epochs = 20


# Number of non zero ratings in the sparse ratings matrix
train_nnz = sparse_train_coo.nnz

# Initialize Q and P matrices randomly
Q = np.random.uniform(0,0.1,size=(sparse_train_coo.shape[0],k))
P = np.random.uniform(0,0.1,size=(sparse_train_coo.shape[1],k)) 

sparse_train_broadcast = sc.broadcast(sparse_train_coo)

rating_entries = zip(sparse_train_coo.row, sparse_train_coo.col, sparse_train_coo.data)
rating_entries_p = sc.parallelize(rating_entries)

def compute_gradient(rating_entry):
	err = rating_entry[2] - Q[rating_entry[0],:].dot(P[rating_entry[1],:])
	grad_Q = 2 * (lambda1 * Q[rating_entry[0],:] - (err)*P[rating_entry[1],:])  
	grad_P = 2 * (lambda2 * P[rating_entry[1],:] - (err)*Q[rating_entry[0],:])
	return (grad_Q,grad_P)


for ep in range(1, epochs + 1): 
	print("Epoch " + str(ep))
	Q_broad = sc.broadcast(Q)
	P_broad = sc.broadcast(P)

	grad = rating_entries_p.map(compute_gradient)
	grad_Q = grad.map(lambda x: x[0])
	grad_P = grad.map(lambda x: x[1])

	grad_Q_sum = sc.accumulator(0.0)
	grad_P_sum = sc.accumulator(0.0) 

	grad_Q.foreach(lambda x: grad_Q_sum.add(x))
	grad_P.foreach(lambda x: grad_P_sum.add(x))

	print("Gradient Q sum: " + str(grad_Q_sum))
	print("Gradient P sum: " + str(grad_P_sum))

    # Adding all entries and then divinding may cause overflow?? 
	Q -= (lr * grad_Q_sum.value / train_nnz) 
	P -= (lr * grad_P_sum.value / train_nnz)

	Q_broad.unpersist(True)
	P_broad.unpersist(True)


def latent_factor_predict(sparse_test_coo, Q, P):
    """ Get the ratings for the test data points using the learnt Q and P matrices
    """
    pred = []
    
    for i,j in zip(sparse_test_coo.row, sparse_test_coo.col):
        pred.append(Q[i,:].dot(P[j,:]))
    
    return pred

pred = latent_factor_predict(sparse_valid_coo, Q, P)
actual = sparse_valid_coo.data
mse = mean_squared_error(pred,actual)
print("Mean Squared Error: " + str(mse))
