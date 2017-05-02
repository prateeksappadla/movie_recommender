from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import argparse
import utils

def configure_spark(master):
	# Configure Spark
	appName = "Recommender"
	conf = SparkConf().setAppName(appName).setMaster(master)
	conf.set("spark.executor.heartbeatInterval","3600s")
	sc = SparkContext(conf=conf)
	return sc

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="User based Collaborative Filtering on Spark")
	parser.add_argument("--k", type=int, help="Number of top similar users to use for making predictions", default=100)
	parser.add_argument("--filename", type=str, default="../data/ml-20m/ratings.csv", help="Path to input file")
	parser.add_argument("--master", type=str, default="local[4]", help="URL of spark master node")
	args = parser.parse_args()

	sc = configure_spark(args.master)

	# Read the input file
	df = pd.read_csv(args.filename)

	n_users = df['userId'].unique().shape[0]
	n_items = df['movieId'].unique().shape[0]
	print("Number of unique users: %d" % n_users)
	print("Number of unique movies: %d" % n_items)

	movie_list, movie_dict = utils.create_moviedict(df)
	df_train, df_valid, df_test = utils.split_data(df)
	sparse_train_coo = utils.create_sparse_coo_matrix(df_train, n_users, n_items, movie_dict)
	sparse_valid_coo = utils.create_sparse_coo_matrix(df_valid, n_users, n_items, movie_dict)
	sparse_test_coo = utils.create_sparse_coo_matrix(df_test, n_users, n_items, movie_dict)

	# Get the average user rating for each user
	user_sum = sparse_train_coo.sum(axis=1)
	print(user_sum.shape)

	# Convert Sparse COO matrix to CSR matrix
	sparse_train_csr = sparse_train_coo.tocsr()
	data = sparse_train_csr.data
	indices = sparse_train_csr.indices
	indptr = sparse_train_csr.indptr

	useravg = np.empty(shape=(indptr.shape[0] - 1,),dtype=np.float64)
	for user_num in range(indptr.shape[0] - 1):
		useravg[user_num] = user_sum[user_num,0] / (indptr[user_num + 1] - indptr[user_num] + 1e-9)


	data_useravg = np.empty(shape=data.shape,dtype=np.float64)
	for user_num in range(indptr.shape[0] - 1):
	    data_useravg[indptr[user_num]: indptr[user_num + 1]] = user_sum[user_num,0] / (indptr[user_num + 1] - indptr[user_num])

	# Subtract user average rating from all ratings
	data_centered = data - data_useravg

	# Create sparse CSR matrix of centered user data
	sparse_train_ucentered = sparse.csr_matrix((data_centered,indices,indptr),shape=(n_users,n_items))

	sparse_train_ucentered_b = sc.broadcast(sparse_train_ucentered)


	def compute_norms(userid):
		row = sparse_train_ucentered_b.value.getrow(userid)
		return np.sqrt(row.dot(row.T))[0,0] 

	users_p = sc.parallelize(range(n_users))

	norms = users_p.map(compute_norms)
	norms_b = sc.broadcast(norms.collect())

	useravg_b = sc.broadcast(useravg)

	sparse_test_csr_b = sc.broadcast(sparse_test_coo.tocsr())

	def usercf(userid):
		actual = sparse_test_csr_b.value.getrow(userid).toarray().squeeze(axis=0)
		
		if np.count_nonzero(actual) > 0:
			sparse_train = sparse_train_ucentered_b.value
			row = sparse_train.getrow(userid)
			
			# Compute similarity with all other users
			sim = row.dot(sparse_train.transpose())
			sim /= (norms_b.value[userid] + 1e-9)
			sim /= (np.array(norms_b.value) + 1e-9)
			sim = np.array(sim).reshape(-1)

			# Get top k similar users
			top_k_users = sim.argsort()[-args.k-1:-1]
			top_k_sim = sim[top_k_users]

		    # Initialize all predictions to the average rating given by that user
			pred = np.ones(sparse_train.shape[1])
			pred *= useravg_b.value[userid]
		    
			top_k_ratings = np.empty((args.k,sparse_train.shape[1]),dtype=np.float64)
			for i in range(args.k):
				top_k_ratings[i] = sparse_train.getrow(top_k_users[i]).toarray()
		                
		    # Map of non-zero rating entries
			ratings_map = (top_k_ratings != 0).astype(int)
		    #print(ratings_map)
		    
			normalizer = top_k_sim.reshape((1,-1)).dot(ratings_map)
		    #print(normalizer.shape)
		    
		    # pred is of shape (n_items,) and the other operand is of shape (1,n_items), hence the indexing [0,:]
			pred += (top_k_sim.reshape((1,-1)).dot(top_k_ratings) / (normalizer + 1e-9))[0,:]
		
			test_mask = (actual != 0).astype(int)	        
			pred = pred * test_mask

			pred_nz = pred[pred.nonzero()]
			actual_nz = actual[actual.nonzero()]
			mse_user = mean_squared_error(pred_nz,actual_nz) * np.count_nonzero(actual)
			return mse_user
		else:
			return 0

	users_mse = users_p.map(usercf)

	total_mse = sc.accumulator(0.0)

	users_mse.foreach(lambda x: total_mse.add(x))

	print("MSE: " + str(total_mse.value / sparse_test_coo.nnz))
