import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from funcs import *

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from funcs import *

cov = []
tag_popularity_trashhold = 80
tag_relevance_trashhold = 0.08
for test_size in [.1,.3,.5,.7]:

    #data preproccesing
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    ratings = ratings.drop(columns = ['timestamp'])

    tags = pd.read_csv('tag-genome/tags.dat' , header = None , sep = '\t' , names = ['TagID', 'Tag' , 'TagPopularity'])
    #only keep tags for item groups with tag popularity greater than trashhold
    tags = tags[tags['TagPopularity']>tag_popularity_trashhold]

    tag_relevance = pd.read_csv('tag-genome/tag_relevance.dat' , header = None , sep = '\t' , names = [ 'MovieID' ,'TagID', 'Relevance'])
    tag_relevance = tag_relevance[tag_relevance['TagID'].isin(tags['TagID'])]
    #set tag relevence smaller than trashhold to 0
    tag_relevance.loc[tag_relevance['Relevance']<tag_relevance_trashhold , ('Relevance')] = 0
    tag_relevance = tag_relevance[tag_relevance['MovieID'].isin(ratings['movieId'].unique())]

    ratings = ratings[ratings['movieId'].isin(tag_relevance['MovieID'].unique())]

    good_split = False

    while not good_split:
        test_ratings=ratings.sample(frac=test_size)
        train_ratings=ratings.drop(test_ratings.index)
        if set(train_ratings['userId'].unique()) >= set(test_ratings['userId'].unique()):
            good_split = True
        

    #user ids and movie ids
    users = train_ratings['userId'].unique()
    movies = tag_relevance['MovieID'].unique()

    #number of rates of each user
    users_all_rates_num = train_ratings['userId'].value_counts().sort_index().to_numpy()
    users_all_rates_num = users_all_rates_num.reshape(users_all_rates_num.shape[0],1)

    #creating itemgroups and rates matrix
    item_groups = create_item_groups(tag_relevance , 'TagID' , 'MovieID' , 'Relevance')
    train_rates = create_rates(train_ratings , 'userId' , 'movieId' , 'rating' , movies)
    #test ratings
    test_ratings = test_ratings.to_numpy().astype('int64')
    #typicality matrix
    M = create_user_typicality_matrix(train_rates , item_groups , users , users_all_rates_num , 5.0)
    #actual retes from test rating array
    actual =np.ravel(test_ratings[:,2]).reshape(test_ratings.shape[0],1)

    sim  = sim_matrix(M , 'Distance based' , .01)
    predicts = predict(test_ratings , sim , train_rates , movies , users)
    cov.append(coverage(predicts))

plt.plot([0.9,.7,.5,.3] , cov  )
plt.xlabel('train test ratio')
plt.ylabel('coverage')
plt.savefig('coverage')
   
