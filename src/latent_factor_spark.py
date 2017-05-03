from pyspark import SparkContext, SparkConf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import utils
import argparse

def configure_spark(master):
	# Configure Spark
	appName = "Recommender"
	conf = SparkConf().setAppName(appName).setMaster(master)
	conf.set("spark.executor.heartbeatInterval","3600s")
	sc = SparkContext(conf=conf)
	return sc


parser = argparse.ArgumentParser(description="Latent Factor Model")
parser.add_argument("--k", type=int, default=100, help="Number of latent factors")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for Stochastic Gradient Descent")
parser.add_argument("--lambdar" ,type=float, default=1.0, help="Regularization strength")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for Stochastic Gradient Descent")
parser.add_argument("--filename", type=str, default="../data/ml-20m/ratings.csv", help="Path to input file")
parser.add_argument("--master", type=str, default="local[4]", help="URL of spark master node")	
args = parser.parse_args()


# Configure Spark
sc = configure_spark(args.master)

# Read the ratings file into a pandas Dataframe object
df = pd.read_csv(args.filename)

n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]
print("Number of unique users: %d" % n_users)
print("Number of unique movies: %d" % n_items)

df_train, df_valid, df_test = utils.split_data(df)
movie_list, movie_dict = utils.create_moviedict(df)
sparse_train_coo = utils.create_sparse_coo_matrix(df_train, n_users, n_items, movie_dict)
sparse_valid_coo = utils.create_sparse_coo_matrix(df_valid, n_users, n_items, movie_dict)
sparse_test_coo = utils.create_sparse_coo_matrix(df_test, n_users, n_items, movie_dict)

# Number of non zero ratings in the sparse ratings matrix
train_nnz = sparse_train_coo.nnz

# Initialize Q and P matrices randomly
np.random.seed(1)
Q = np.random.uniform(0,0.1,size=(sparse_train_coo.shape[0],args.k))
P = np.random.uniform(0,0.1,size=(sparse_train_coo.shape[1],args.k)) 

rating_entries = zip(sparse_train_coo.row, sparse_train_coo.col, sparse_train_coo.data)
rating_entries_p = sc.parallelize(rating_entries)

def compute_gradient(rating_entry):
	err = rating_entry[2] - Q[rating_entry[0],:].dot(P[rating_entry[1],:])
	grad_Q = 2 * (args.lambdar * Q[rating_entry[0],:] - (err)*P[rating_entry[1],:])  
	grad_P = 2 * (args.lambdar * P[rating_entry[1],:] - (err)*Q[rating_entry[0],:])
	return (grad_Q,grad_P)


for ep in range(1, args.epochs + 1): 
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

	# print("Gradient Q sum: " + str(grad_Q_sum))
	# print("Gradient P sum: " + str(grad_P_sum))

    # Adding all entries and then divinding may cause overflow?? 
	Q -= (args.lr * grad_Q_sum.value / train_nnz) 
	P -= (args.lr * grad_P_sum.value / train_nnz)

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
