import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse
import logging

logger = logging.getLogger('Latent Factor Model')
logger.setLevel(logging.INFO)

# Read the ratings csv file into a pandas Dataframe
filename = '../data/ml-latest-small/ratings.csv'
df = pd.read_csv(filename)

n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]
print("Number of unique users: %d" % n_users)
print("Number of unique movies: %d" % n_items)


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


def compute_norms(sparse_train_csr):
    """ Compute the norm of the rating vector for each user.
        These norm values will be used for normalizing during computation of cosine similarity

        Args: 
            sparse_train_csr: Sparse Ratings matrix

        Returns:
            norms: numpy array of shape (n_users, ) containing the L2 norm of each user vector in the ratings matrix    
    """
    n_users = sparse_train_csr.shape[0]
    norms = np.empty(shape=(n_users,),dtype=np.float64)
    
    for i in range(n_users):
        row = sparse_train_csr.getrow(i)
        #getrow returns a 1 x n vector. Dot product gives a 1 x 1 numpy array and then we index using [0,0] to get a scalar
        norms[i] = np.sqrt(row.dot(row.T))[0,0]
    return norms    


def user_sim(userid, sparse_train_csr, norms):
    """ Compute the similarity between a particular user and all other users
        
        Return an numpy array containing the similarity with all other users

        Args
            userid: User id of the particular user for whom we want recommendations
            sparse_train_csr: Sparse ratings matrix in CSR form
            norms: The norms of the user vectors in the rating matrix

        Returns
            user_sim: Numpy array of size (n_users,) having similarity of this user with all other users     
    """
    row = sparse_train_csr.getrow(userid)
    sim = row.dot(sparse_train_csr.transpose())
    
    sim /= (norms[userid] + 1e-9)
    sim /= (norms + 1e-9)
    
    #Convert from np.matrix datatype to a numpy ndarray, reshape to a vector and then return
    return np.array(sim).reshape(-1)    


def user_cf(userid, user_avg, user_sim, sparse_train_csr, k=100):
    """ Predict the user based collaborative filtering ratings for a particular user.

        Args:
            userid: User id of the particular user for whom we want recommendations
            user_avg: Numpy array of size (n_users,) containing the average rating for each user
            user_sim: Numpy array of shape (n_users,) 
            sparse_train_csr: Sparse ratings matrix in CSR form
            k: Number of nearest neighbors to consider for making recommendations

        Returns:
            pred: Numpy array of size (n_items,) containing the predicted ratings for this particular user    
    """    
    top_k_users = user_sim.argsort()[-k-1:-1]
    top_k_sim = user_sim[top_k_users]
    
    # Initialize all predictions to the average rating given by that user
    pred = np.ones(sparse_train_csr.shape[1])
    pred *= user_avg
    
    #print(pred)
    
    top_k_ratings = np.empty((k,sparse_train_csr.shape[1]),dtype=np.float64)
    for i in range(k):
        top_k_ratings[i] = sparse_train_csr.getrow(top_k_users[i]).toarray()
        
    #print(top_k_ratings)    
        
    # Map of non-zero rating entries
    ratings_map = (top_k_ratings !=0).astype(int)
    #print(ratings_map)
    
    normalizer = top_k_sim.reshape((1,-1)).dot(ratings_map)
    #print(normalizer.shape)
    
    # pred is of shape (n_items,) and the other operand is of shape (1,n_items), hence the indexing [0,:]
    pred += (top_k_sim.reshape((1,-1)).dot(top_k_ratings) / (normalizer + 1e-9))[0,:]
    print(pred[:10])
    return pred


def compute_usercf_MSE(sparse_train_csr, sparse_test_csr, k=100):    
    """ Compute MSE on the held out test data
    """
    # Compute the average rating for each user
    data = sparse_train_csr.data
    indptr = sparse_train_csr.indptr
    useravg = np.empty(shape=(indptr.shape[0] - 1,),dtype=np.float64)
    for user_num in range(indptr.shape[0] - 1):
        useravg[user_num] = user_sum[user_num,0] / (indptr[user_num + 1] - indptr[user_num] + 1e-9)
    
    norms = compute_norms(sparse_train_csr)    
    n_users = sparse_train_csr.shape[0]
    
    total_mse = 0
    
    for user in range(n_users):
        usim = user_sim(user, sparse_train_csr, norms)
        pred = user_cf(user, useravg[user], usim, sparse_train_csr,k)
        
        actual = sparse_test_csr.getrow(user).toarray().squeeze(axis=0)
        
        if np.count_nonzero(actual) > 0:
            test_mask = (actual != 0).astype(int)
            
            pred = pred * test_mask

            pred_nz = pred[pred.nonzero()]
            actual_nz = actual[actual.nonzero()]
            mse_user = mean_squared_error(pred_nz,actual_nz)

            print(mse_user)        
            total_mse += (mse_user * np.count_nonzero(actual))
        
    return total_mse/sparse_test_csr.nnz             


# Test above three methods on a toy example
# trial_mat_coo = sparse.random(10,7,density=0.25)
# trial_mat_csr = trial_mat_coo.tocsr()

# trial_test_coo = sparse.random(10,7, density=0.15)
# trial_test_csr = trial_test_coo.tocsr()

# trial_mse = compute_usercf_MSE(trial_mat_csr, trial_test_csr,5)
# print(trial_mse)

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


# Get the average user rating for each user
user_sum = sparse_train_coo.sum(axis=1)
print(user_sum.shape)

# Convert Sparse COO matrix to CSR matrix
sparse_train_csr = sparse_train_coo.tocsr()
data = sparse_train_csr.data
indices = sparse_train_csr.indices
indptr = sparse_train_csr.indptr

data_useravg = np.empty(shape=data.shape,dtype=np.float64)
for user_num in range(indptr.shape[0] - 1):
    data_useravg[indptr[user_num]: indptr[user_num + 1]] = user_sum[user_num,0] / (indptr[user_num + 1] - indptr[user_num])

# Subtract user average rating from all ratings
data_centered = data - data_useravg

# Create sparse CSR matrix of centered user data
sparse_train_ucentered = sparse.csr_matrix((data_centered,indices,indptr),shape=(n_users,n_items))
logger.info("Number of non zero entries in centered matrix: " + str(sparse_train_ucentered.nnz))

sparse_valid_csr = sparse_valid_coo.tocsr()
ks = [50] #, 100, 200, 500
for k in ks:
    print("k: " + str(k))
    ucf_mse = compute_usercf_MSE(sparse_train_ucentered, sparse_valid_csr, k)
    print("k: " + str(k) + " MSE: " + str(ucf_mse))

# Compute MSE on test data with best performing k



