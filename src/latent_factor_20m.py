import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import logging
import argparse
import utils


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
    # Set a seed manually to be able to reproduce results
	np.random.seed(1)
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


parser = argparse.ArgumentParser(description="Latent Factor Model")
parser.add_argument("--k", type=int, default=100, help="Number of latent factors")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for Stochastic Gradient Descent")
parser.add_argument("--lambdar" ,type=float, default=1.0, help="Regularization strength")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for Stochastic Gradient Descent")
parser.add_argument("--filename", type=str, default="../data/ml-20m/ratings.csv", help="Path to input file")
args = parser.parse_args()

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

Q,P = latent_factor_train(sparse_train_coo, args.k, args.lambdar, args.lambdar, args.epochs, args.lr)
pred_valid = latent_factor_predict(sparse_valid_coo, Q, P)
actual_valid = sparse_valid_coo.data
mse_valid = mean_squared_error(pred_valid, actual_valid)
print("Validation MSE: " + str(mse_valid))

pred_test = latent_factor_predict(sparse_test_coo, Q, P)
actual_test = sparse_test_coo.data
mse_test = mean_squared_error(pred_test, actual_test)
print("Test MSE: " + str(mse_test))