
# coding: utf-8

# In[1]:

# Create Train and test splits of the data
# Create Evaluation framework

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


# In[2]:

# Read the data
#filename = './data/ml-small/ratings.csv'
filename = './data/ml-20m/ratings.csv'
df = pd.read_csv(filename)

n_users = df['userId'].unique().shape[0]
n_items = df['movieId'].unique().shape[0]
print("Number of unique users: %d" % n_users)
print("Number of unique movies: %d" % n_items)


# In[3]:

# Create a dictionary from movieId to index
ind = 0
movie_dict = {}
movie_list = []

for item in df['movieId'].unique():
    movie_list.append(item)
    movie_dict[item] = ind
    ind += 1   
    
assert(len(movie_list) == n_items)


# In[4]:

# Create user-item ratings matrix from the csv file data
ratings = np.zeros((n_users, n_items))

# df.itertuples() returns a Pandas Frame object
for row in df.itertuples():
    ratings[row[1] - 1, movie_dict[row[2]]] = row[3]


# In[5]:

# Split data into training and test sets by removing 10 ratings per user from the training set and adding to test set 
# All selected users had rated at least 20 movies. There are a total of 100004 ratings in this version of the dataset
# 10 ratings per user means a test set comprising of 6710 ratings which is around 6.7% of the total data
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

train, test = train_test_split(ratings)


# In[6]:

# Method to compute the mean squared error between the predictions and the test data
def get_mse(pred, actual):
    # Ignore zero terms in the matrix
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


# In[11]:

def global_average_baseline(ratings):
    pred = np.zeros(ratings.shape)
    global_avg = np.sum(ratings) / np.count_nonzero(ratings)
    pred += global_avg
    return pred

global_avg_predictions = global_average_baseline(train)
print("Global average rating: " + str(global_avg_predictions[0,0]))
global_avg_mse = get_mse(global_avg_predictions, test)
print("Global Average Baseline MSE: " + str(global_avg_mse))


# In[14]:

# def user_average(ratings):
#     # Small constant used for numerical stability
#     epsilon = 1e-9
#     user_avg = np.sum(ratings,axis=1) / (np.count_nonzero(ratings,axis=1) + epsilon)
    
#     pred = np.zeros(ratings.shape)
#     pred += np.expand_dims(user_avg,axis=1)
#     return pred

# user_avg_predictions = user_average(train)
# print(user_avg_predictions[:5, :5])
# user_avg_mse = get_mse(user_avg_predictions, test)
# print("User Average MSE: " + str(user_avg_mse))    


# # In[15]:

# def movie_average(ratings):
#     epsilon = 1e-9
#     movie_avg = np.sum(ratings,axis=0) / (np.count_nonzero(ratings, axis=0) + epsilon)
    
#     pred = np.zeros(ratings.shape)
#     pred += np.expand_dims(movie_avg,axis=0)
#     return pred

# movie_avg_predictions = movie_average(train)
# print(movie_avg_predictions[:5,:5])
# movie_avg_mse = get_mse(movie_avg_predictions,test)
# print("Movie Average MSE: " + str(movie_avg_mse))


# # In[17]:

# # Global Baseline method
# def global_baseline_method(ratings):
#     """Global Baseline Method
#         r_ij = b + u_ib + m_jb 
#         r_ij: predicted rating for a user i for movie j
#         b: global average rating
#         u_ib: difference of avergae rating given by user and the global average (Captures bias of user)
#         m_jb: difference of average rating given to movie and the global average    
#     """
#     epsilon=1e-9
#     global_avg = np.sum(ratings) / np.count_nonzero(ratings) 
    
#     user_avg = np.sum(ratings,axis=1) / (np.count_nonzero(ratings, axis=1) + epsilon)
#     user_bias = user_avg - global_avg
    
#     movie_avg = np.sum(ratings, axis=0) / (np.count_nonzero(ratings,axis=0) + epsilon)
#     movie_bias = movie_avg - global_avg
    
#     pred = np.zeros(ratings.shape)
#     pred += global_avg
#     pred += np.expand_dims(user_bias,axis=1)
#     pred += np.expand_dims(movie_bias,axis=0)
#     return pred

# gbm_predictions = global_baseline_method(train)
# print(gbm_predictions[:5,:5])
# gbm_mse = get_mse(gbm_predictions, test)
# print("Global Baseline Method MSE: " + str(gbm_mse))
    


# # In[27]:

# # User - User Collaborative Filtering

# def user_centered_cosine(ratings):
#     """Method to compute user-user similarity matrix using centered cosine similarity
#     """   
#     epsilon=1e-9
#     centered_ratings = np.zeros(ratings.shape)
#     non_zero = (ratings != 0).astype(int)
    
#     user_avg = np.sum(ratings,axis=1) / (np.count_nonzero(ratings, axis=1) + epsilon)
#     # Subtract average user rating from each rating
#     centered_ratings = ratings - np.expand_dims(user_avg,axis=1)
#     centered_ratings *= non_zero
#     sim = centered_ratings.dot(centered_ratings.T)
#     norms = np.array([np.sqrt(np.diagonal(sim))])
#     return (sim / norms / norms.T)

# def user_cf_predictions(ratings, user_similarity, k=20):
#     """Method to make user-user CF predictions
#         k: Number of similar users to use to make predictions
#     """
#     epsilon=1e-9
#     pred = np.zeros(ratings.shape)
    
#     #Select top k similar users for each user
#     top_k = np.argsort(user_similarity,axis=1)[:,-k-1:-1]
    
#     for user in range(ratings.shape[0]):
#         top_k_sim = user_similarity[user,top_k[user]]
#         pred[user] = np.expand_dims(top_k_sim,axis=0).dot(ratings[top_k[user],:])
        
#         # Normalize predictions by dividing by sum of similarities corresponding to similar users who have non-zero ratings
#         # for this movie
#         non_zero_ratings = (ratings[top_k[user],:] != 0).astype(int)
#         normalizer = np.sum(np.expand_dims(top_k_sim,axis=1) * non_zero_ratings, axis=0)
#         pred[user] /= (normalizer + epsilon) 
    
#     return pred


# user_sim = user_centered_cosine(train)
# user_cf_predictions = user_cf_predictions(train, user_sim, 200)
# user_cf_mse = get_mse(user_cf_predictions, test)
# print("User-User CF MSE: "+ str(user_cf_mse))
                


# # In[49]:

# def user_cf_predictions_bias(ratings, user_similarity, k=20):
#     """Method to make user-user CF predictions by adjusting for user bias
#         k: Number of similar users to use to make predictions
#     """
#     epsilon=1e-9
#     pred = np.zeros(ratings.shape)
    
#     centered_ratings = np.zeros(ratings.shape)
#     non_zero = (ratings != 0).astype(int)
    
#     user_avg = np.sum(ratings,axis=1) / (np.count_nonzero(ratings, axis=1) + epsilon)
#     pred += np.expand_dims(user_avg, axis=1)
#     # Subtract average user rating from each rating
#     centered_ratings = ratings - np.expand_dims(user_avg,axis=1)
#     centered_ratings *= non_zero

#     # Select top k similar users for each user (Shape of top_k = (num_users,k))
#     top_k = np.argsort(user_similarity,axis=1)[:,-k-1:-1]
#     print(top_k.shape)
    
#     for user in range(ratings.shape[0]):
#         top_k_sim = user_similarity[user,top_k[user]]
        
#         # Normalize predictions by dividing by sum of similarities corresponding to similar users who have non-zero ratings
#         # for this movie
#         non_zero_ratings = (centered_ratings[top_k[user],:] != 0).astype(int)
#         normalizer = np.sum(np.expand_dims(top_k_sim,axis=1) * non_zero_ratings, axis=0)
            
#         pred[user] += (np.expand_dims(top_k_sim,axis=0).dot(centered_ratings[top_k[user],:]) / (np.expand_dims(normalizer,axis=0) + epsilon)).flatten()
    
#     return pred

# bias_user_cf_predictions = user_cf_predictions_bias(train, user_sim, 200)
# bias_user_cf_mse = get_mse(bias_user_cf_predictions, test)
# print("Bias Adjusted User-User CF MSE: "+ str(bias_user_cf_mse))    


# # In[43]:

# # Item_Item Collaborative Filtering

# def item_centered_cosine(ratings):
#     epsilon=1e-9
#     centered_ratings = np.zeros(ratings.shape)
    
#     item_avg = np.sum(ratings,axis=0,dtype=np.float64) / (np.count_nonzero(ratings, axis=0) + epsilon)
#     non_zero = (ratings != 0).astype(int)
#     centered_ratings = ratings - item_avg.T
#     centered_ratings *= non_zero
    
#     sim = centered_ratings.T.dot(centered_ratings) + epsilon
#     norms = np.array([np.sqrt(np.diagonal(sim))])
#     return (sim / norms / norms.T)

# item_sim = item_centered_cosine(train)    


# # In[60]:

# def item_cf_predictions_bias(ratings, item_sim, k=20):
#     """Method to make item-item CF predictions by adjusting for user bias
#         k: Number of similar users to use to make predictions
#     """
#     epsilon=1e-9
#     pred = np.zeros(ratings.shape)
#     non_zero = (ratings != 0).astype(int)
        
#     centered_ratings = np.zeros(ratings.shape)

#     movie_avg = np.sum(ratings,axis=0) / (np.count_nonzero(ratings, axis=0) + epsilon)
#     pred += np.expand_dims(movie_avg, axis=0)
    
#     centered_ratings = ratings - np.expand_dims(movie_avg,axis=0)
#     centered_ratings *= non_zero
    
    
#     # Select top-k similar items for each item
#     top_k = np.argsort(item_sim,axis=1)[:,-k-1:-1]
#     print(top_k.shape)
    
    
#     for movie in range(ratings.shape[1]):
#         top_k_sim = item_sim[movie,top_k[movie]]
#         #print(top_k_sim.shape)
        
#         non_zero_ratings = (centered_ratings[:,top_k[movie]] != 0).astype(int)
#         #print(non_zero_ratings.shape)
#         normalizer = np.sum(np.expand_dims(top_k_sim,axis=0) * non_zero_ratings, axis=1)
#         #print(normalizer.shape)
        
#         #print((centered_ratings[:,top_k[movie]].dot(top_k_sim) / (normalizer + epsilon)).shape)
#         pred[:,movie] += (centered_ratings[:,top_k[movie]].dot(top_k_sim) / (normalizer + epsilon))
#     return pred     


# bias_item_cf_predictions = item_cf_predictions_bias(train, item_sim, 1000)
# bias_item_cf_mse = get_mse(bias_item_cf_predictions, test)
# print("Bias Adjusted Item-Item CF MSE: "+ str(bias_item_cf_mse))    


# # In[9]:

# # Latent Factor Model

# # Two possibilities: 1. Initialize P and Q using SVD, then learn using SGD
# # 2. Choose k(number of latent factors as a hyperparameter, initialize P and Q randomly and then learn using SGD) 
# from numpy.linalg import svd

# Q , s, P = svd(train,full_matrices=True)
# print(Q.shape, s.shape, P.shape)


# # In[64]:

# print(s[:30])
# print(s[-30:])


# # In[32]:

# # Latent Factor Model

# def learn_latent_factor_random(ratings, k=50, lambda1=1, lambda2=1, epochs=50, lr=0.1):
#     """Learn a latent factor model for the ratings matrix
#         Randomly initialize Q and P matrices of dimensions
#         Q: users * k
#         P: items * k
        
#         lambda1: regularization strength for Q
#         lambda2: regularization strength for P
#         and then learn by SGD
#     """
    
# #     Q = np.random.randn(ratings.shape[0], k)
# #     P = np.random.randn(ratings.shape[1], k)

#     Q = np.random.uniform(0,0.1,size=(ratings.shape[0], k))
#     P = np.random.uniform(0,0.1,size=(ratings.shape[1], k))
    
#     non_zero_r = np.transpose(np.nonzero(ratings))
#     print(non_zero_r[0])
    
#     for ep in range(1,epochs+1):
#         print("Epoch: " + str(ep))
#         for rating in non_zero_r:
#             err = ratings[rating[0],rating[1]] - Q[rating[0],:].dot(P[rating[1],:])
#             #print(err)
#             grad_Q = 2 * (lambda2 * Q[rating[0],:] - (err)*P[rating[1],:])  
#             grad_P = 2 * (lambda1 * P[rating[1],:] - (err)*Q[rating[0],:])
            
#             Q[rating[0],:] -= lr * grad_Q
#             P[rating[1],:] -= lr * grad_P                                           
                                                       
#     return Q,P                                                       


# # In[27]:

# def learn_latent_factor_svd(ratings, k=50, lambda1=1, lambda2=1, epochs=50, lr=0.1):
#     """ Initialize the matrices Q and P using SVD
#     """
    
#     U, s, V = svd(ratings,full_matrices=True)
    
#     Q = U[:,:k]
#     P = V[:,:k].dot(np.diag(s[:k]))
#     non_zero_r = np.transpose(np.nonzero(ratings))
#     print(non_zero_r[0])
    
#     for ep in range(1,epochs+1):
#         print("Epoch: " + str(ep))
#         for rating in non_zero_r:
#             err = ratings[rating[0],rating[1]] - Q[rating[0],:].dot(P[rating[1],:])
#             print(err)
#             grad_Q = 2 * (lambda2 * Q[rating[0],:] - (err)*P[rating[1],:])  
#             #print(grad_Q)
#             grad_P = 2 * (lambda1 * P[rating[1],:] - (err)*Q[rating[0],:])
#             #print(grad_P)
                        
#             Q[rating[0],:] -= (lr * grad_Q)
#             P[rating[1],:] -= (lr * grad_P)    
#             #break
#         break                                               
#     return Q,P    


# # In[34]:

# #Q,P = learn_latent_factor_svd(train,650)
# Q,P = learn_latent_factor_random(train,500)

# def latent_factor_predictions(Q, P):
#     pred = Q.dot(P.T)
#     return pred

# lf_predictions = latent_factor_predictions(Q,P)
# lf_mse = get_mse(lf_predictions,test)
# print("Latent Factor Predictions MSE: " + str(lf_mse))


# # In[ ]:



