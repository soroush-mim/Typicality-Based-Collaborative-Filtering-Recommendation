import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

def create_item_groups(tag_relevance , tag_id_key , movie_id_key , relevance_key):
    """a function for creating item groups

    Args:
        tag_relevance ([pandas dataframe]): [this dataframe in each row has tag id and move id and relevance between theses 2
        this dataframe has to be sorted first based on movieid and next based on tagid and for each pair of movies and tags we 
        this dataframe has a relevance]
        tag_id_key ([str]): [tag id key in tag_relevance datafram]
        movie_id_key ([str]): [movie id key in tag_relevance datafram]
        relevance_key ([str]): [relevance id key in tag_relevance datafram]

    Returns:
        [numpy matrix]: [its a tags_num * movies_num matrix whcih in each cell ij has relvence between movie j and tag i]
    """    
    #item group is a matrix which row i is Ki
    tags = tag_relevance[tag_id_key].unique()
    tags_num = len(tags)
    movies_num = len(tag_relevance[movie_id_key].unique())
    item_groups = np.zeros((tags_num , movies_num))
    for i in range(tags_num):
        item_groups[i] = tag_relevance[tag_relevance[tag_id_key]==tags[i]][relevance_key].to_numpy()
        #tag relevance has to be sorted based on movie id
    return item_groups

def create_rates(ratings , user_id_key , movie_id_key , rating_key , movies):
    """create a numpy matrix which in cell ij we have reta of user i on movie j
        if a user havent rate a movie we put zero in corresponding cell 

    Args:
        ratings ([dataframe]): [in each row has userid and movieid and rate , all rates should be greater than zero
        ratings should be sorted first on userid and next on movieid]
        user_id_key ([str]): [user id key in ratings dataframe]
        movie_id_key ([str]): [movie id key in ratings dataframe]
        rating_key ([str]): [rating key in ratings dataframe]
        movies ([numpy array]): [an array with movie ids in order]

    Returns:
        [numpy matrix]: [in cell ij we have reta of user i on movie j]
    """    
    #data set must have not zero rate and we set zero for rate of items 
    #that a user has not rated
    users = ratings[user_id_key].unique()
    users_num = len(users)
    movies_num = len(movies)
    rates = np.zeros((users_num , movies_num))
    for i in range(users_num):
        user_ratings = ratings[ratings[user_id_key]==users[i]]
        user_ratings = user_ratings[[movie_id_key , rating_key]]
        user_ratings.set_index(movie_id_key, drop=True, inplace=True)
        user_ratings = user_ratings.to_dict()[rating_key]
        for j in user_ratings.keys():
            rates[i][np.where(movies==j)]=user_ratings[j]
    return rates

def create_user_typicality_matrix(rates , item_groups , users, users_all_rates_num , Rmax):
    """create user typicality matrix

    Args:
        rates ([numpy matrix]): [output of create_rates funcs on training dataset]
        item_groups ([numpy matrix]): [output of create_rates funcs on tag_relevance]
        users ([numpy array]): [sorted user ids]
        users_all_rates_num ([numpy array]): [for each user have number of rates that user has given]
        Rmax ([int]): [max value of ratings]

    Returns:
        [numpy matrix]: [typicality matrix which in cell ij we have typicality of user i in item group j]
    """    
    users_num = rates.shape[0]
    tags_num = item_groups.shape[0]
    user_rates_num = np.zeros((users_num , tags_num))
    for i in range(users_num):
        for j in range(tags_num):
            user_rates_num[i][j] = np.count_nonzero(rates[i]*item_groups[j])
    user_rates_num[user_rates_num==0] = 1
    S_r = rates @ item_groups.T
    S_r = S_r / (user_rates_num * Rmax)
    S_f = user_rates_num / users_all_rates_num
    M = (S_r + S_f)/2
    return M

def sim_matrix(M , sim_type , gamma):
    """create similarity matrix for rows of matrix M
        and set gamma as trashhold

    Args:
        M ([numpy matrix]): [typicality matrix]
        sim_type ([str]): [type of similarity (Distance based or Cosine based or Correlation based)]
        gamma ([int]): [trash hold for similarities]

    Returns:
        [numpy matrix]: [in cell ij it has similarity between user i and j]
    """    
    if sim_type == 'Distance based':
        sim = np.exp(-pairwise_distances(M))
    if sim_type == 'Cosine based':
        sim = cosine_similarity(M)
    if sim_type == 'Correlation based':
        sim = np.corrcoef(M)
    sim[sim<gamma] = 0
    return sim

def predict(test_ratings , sim_matrix ,train_rates , movies , users):
    """predict ratings based on rates matrix and similarity matrix if it can not predict 
       any rate it put rating to zero

    Args:
        test_ratings ([numpy array]): [in each row has user id and movie id and rate]
        sim_matrix ([numpy matrix]): [similarity matrix]
        train_rates ([numpy matrix]): [rates matrix]
        movies ([numpy array]): [movie ids in order]
        users ([numpy array]): [user ids in order]

    Returns:
        [numpy array ]: [predicted rates]
    """    
    predict = np.zeros((test_ratings.shape[0],1))
    k = 0
    for i in test_ratings:
        R = train_rates[: , np.where(movies==i[1])]
        R = R.reshape(R.shape[0],R.shape[1])
        Sim = sim_matrix[np.where(users==i[0])]
        Sim[R.T == 0]=0
        if np.sum(Sim)>0:
            predict[k] = (Sim @ R) / np.sum(Sim)
        else:
            predict[k] = 0
        k+=1

    return predict

def MAE(predicts , actual):
    """calculate MAE

    Args:
        predicts ([numpy array]): [predicted rates]
        actual ([numpy array]): [actual rates]

    Returns:
        [int]: [MAE value]
    """    
    diff = predicts - actual
    return np.sum(np.abs(diff))/np.count_nonzero(predicts!=0)

def coverage(predicts):
    """calculate coverage of recommender

    Args:
        predicts ([numpy array]): [predicted rates]

    Returns:
        [int]: [value of coverage]
    """    
    return np.count_nonzero(predicts) / len(predicts)