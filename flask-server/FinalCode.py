#!/usr/bin/env python
# coding: utf-8

# # **Recommendation of points of interest**

# # 1. Introduction
# 
# Following code uses Gowalla data set ("https://snap.stanford.edu/data/loc-gowalla.html"). The dataset includes 36,001,959 visits made by 407,533 users in 2,724,891 POIs. This data covers in particular the locations of users' visits as well as their social networks (or "friends"). This data is used to recommend places to travel next to the user. 
# 
# The recommendations are generated in following 4 ways - 
# 1) Highest rated by all the users (popular places to visit)
# 2) Highest rated by friends
# 3) Highest rated by user themselves in past
# 4) A collaborative approach, originally presented in: "iGSLR: Personalized Geo-Social Location Recommendation: A Kernel Density Estimation Approach", Zhang and Chow, SIGSPATIAL'13 ("https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.814&rep=rep1&type=pdf")
# 
# The 4th method uses a combination of social networks of the user and the relation of the geographical location of places visited in the past and recommended place to calculate ratings. The code for the 4th method is based on the linked paper, and from this blog post ("https://towardsdatascience.com/where-to-travel-next-a-guide-to-building-a-recommender-system-for-pois-5116adde6db")
# 
# 
# The code is implemented using **Surprise** library (http://surpriselib.com/).
# 

# In[ ]:
# from IPython import get_ipython


# installing all the dependencies
import sys
# pip install geopy


# In[ ]:


# conda install -c conda-forge scikit-surprise


# In[ ]:

# importing all the important libraries
import pandas as pd
import os
import sys
import numpy as np
from geopy.distance import geodesic 
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset
import matplotlib.pyplot as plt
from surprise import AlgoBase
from surprise import PredictionImpossible
from itertools import combinations
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.accuracy import rmse
from surprise import accuracy


# # 2. Data loading and processing
# 
# In order to process and load the data, perform the following steps:
# * Extract the dataset and load it into a *pandas* *dataframe*.
# * The data is extracted from 2 csv files ("Gowalla_totalCheckins.csv" and "Gowalla_edges.csv") and loaded into 2 dataframes (df_totalCheckins and df_edges) from which 4 data frames are made (df_checkins, df_friendship, df_locations, df_userinfo) which are further filtered.

# In[ ]:


df_totalCheckins = pd.read_csv("./Gowalla_totalCheckins.csv", names=["userid", "dateTime", "lat", "lng", "placeid"])
df_edges = pd.read_csv("./Gowalla_edges.csv", names=["userid1", "userid2"])
# print(df_totalCheckins[0:3])
# print(df_edges[0:3])


# In[ ]:


# df_checkins = df_totalCheckins[["userid", "placeid"]].copy(deep=True).drop_duplicates(inplace=False,ignore_index=True) #to remove duplicate rows
df_checkins = df_totalCheckins[["userid", "placeid"]].copy(deep=True) 
#deep= True so that changes in the df_checkins won't be done in df_totalCheckins
# print(df_checkins[0:3])


# In[ ]:


df_friendship = df_edges.copy(deep=True)
# print(df_friendship[0:3]) 


# In[ ]:


df_locations = df_totalCheckins[["placeid", "lng", "lat"]].copy(deep=True).drop_duplicates(inplace=False,ignore_index=True) #to remove duplicate rows
# df_locations = df_totalCheckins[["placeid", "lng", "lat"]].copy(deep=True)
df_locations.columns=["id", "lng", "lat"]
# print(df_locations[0:3])


# In[ ]:


df_userinfo = df_totalCheckins[["userid"]].copy(deep=True).drop_duplicates(inplace=False, ignore_index=True)
# df_userinfo = df_totalCheckins[["userid"]].copy(deep=True)
df_userinfo.columns=["id"]
# print(df_userinfo[0:3])


# Only a small dataset is taken because of computing reasons

# In[ ]:


df_checkins = df_checkins.head(20000)


# *   Remove users who have made less than 5 visits or more than 50 visits.

# In[ ]:


# calculate the number of checkins for each user (for all places)
df_grouped_checkins  = df_checkins.groupby(['userid'] , as_index=False).count()


# In[ ]:


# recover only the userid with more than 5 places visited and it is also necessary to reduce to less than 50 places because of the complexity
filtered_user_ids = df_grouped_checkins[(df_grouped_checkins.placeid >= 5) & (df_grouped_checkins.placeid <= 50)].userid.values 
# filtered_user_ids = df_grouped_checkins[(df_grouped_checkins.placeid > 0)].userid.values 
df_filtered_checkins = df_checkins[df_checkins.userid.isin(filtered_user_ids)]
# print(df_filtered_checkins[0:20])


# In[ ]:


# Filter dtaframe df_friendship too
df_filtered_friendship = df_friendship[df_friendship.userid1.isin(filtered_user_ids)]


# In[ ]:


# Filter dtaframe userinfo too
df_filtered_userinfo = df_userinfo[df_userinfo.id.isin(filtered_user_ids)]


# *   Associate each user with their list of friends and place the result in a *dataframe*: *df_user_friends*.
# 

# In[ ]:


df_user_friends = df_friendship.groupby('userid1').userid2.apply(list).reset_index(name='friends_list') 


# * Calculate the frequency of each pair *(user, POI)* and put the result in a *dataframe* *df_frequencies*.
# 

# In[ ]:


# Join df_filtered_checkins with df_locations using place_id and id fields
#following is a left outer join=> use keys (not cols) from left table only=> only places that r in left table make to final table 
df_checkins_locations = pd.merge(df_filtered_checkins, df_locations,left_on="placeid",right_on="id",how="left") 
# print(df_filtered_checkins[0:20])
# print(df_checkins_locations[0:10])
df_checkins_locations = df_checkins_locations.dropna() 
# print(df_checkins_locations[0:20])


# In[ ]:


# df_checkins_locations.head()


# In[ ]:


df_frequencies = df_checkins_locations.groupby(['userid', 'placeid'])["id"].count().reset_index(name="frequency")
# df_frequencies


# In[ ]:


# Join df_frequencies with df_locations using place_id and id fields
df1 = pd.merge(df_frequencies, df_locations,left_on="placeid",right_on="id",how="inner") 
df_frequencies = df1
# df_frequencies


# * Update the frequencies of *df_frequencies* to bring them back to the interval [0, 10] by applying normalization using tanx function.
# 
# where $f_{min}$ and $f_{max}$ are respectively the minimum and maximum number of all the visit frequencies of any POI in the dataset.

# In[ ]:


# calculate fmin
f_min = df_frequencies['frequency'].min()
# calculate f max
f_max = df_frequencies['frequency'].max()
# print(f_min, f_max)


# In[ ]:


#np.tanh = numpy tanh function
df_frequencies["ratings"] = df_frequencies["frequency"].apply(lambda x: 10*np.tanh(10*(x-f_min)/(f_max-f_min)))


# In[ ]:


# df_frequencies.head()
# df_frequencies[0:40]


# * Load *df_frequencies* into the *Suprise* framework using the *load_from_df()* function.
# 

# In[ ]:


reader = Reader(rating_scale=(0, 10))
data = Dataset.load_from_df(df_frequencies[['userid', 'placeid', 'ratings']], reader)


# *Use the *train_test_split()* function to split *df_frequencies* into a training set (*training set*, 75% of dataset) and a test set (*test set*, 25% of dataset) data).

# In[ ]:


trainset, testset = train_test_split(data, test_size=.25)


# *Associate each user with his list of POIs visited and place the result in a *dataframe* *df_user_POI*.

# In[ ]:


df_user_POI = df_frequencies.groupby('userid').placeid.apply(list).reset_index(name='POI_list') 


# In[ ]:


# print(df_user_POI[0:20])


# # 3. Geographical Influence
# 
# The *dataframe* *df_user_POI* associates each user $u$ with the list $L_u$ of POIs he has visited.
# 
# 
# * Use *df_user_POI* to calculate for each user $u$ the distances $d_{ij}$ between each pair of POIs visited: 
# 
# $\forall p_i, p_j \in L_u \times L_u, d_{ij} = distance(p_i, p_j)$. 
# 
# We will denote this list of distances by $D_u$.
# 

# In[ ]:


# def distance(pi, pj):
    
#     # retrieve first place latitude and longitude
#     lat0 = df_locations[df_locations.id == pi].lat.values
#     lng0 = df_locations[df_locations.id == pi].lng.values

#     # retrieve second place latitude and longitude
#     lat1 = df_locations[df_locations.id == pj].lat.values
#     lng1 = df_locations[df_locations.id == pj].lng.values

#     # calculate distance in km using the geopy library

#     return geodesic((lat0,lng0), (lat1,lng1)).km

# def distance_pair_list(Lu):
#     # collect all the combinations of pairs of places already visited
#     if len(Lu) <= 1 :
#         pairs = []
#     else:
#         pairs = list(combinations(Lu,2))
        
    
#     dist_list = []
    
#     # calculate the distance between each pair of squares
#     if pairs != [] :
#         for pair in pairs :
#             dist_list.append(distance(pair[0], pair[1]))
#     return dist_list


# # * Use $D_u$ to calculate the density $f_u$ of any distance

# # The smoothing parameter $h = 1.06\hat{\sigma}n^{-1/5}$, where $n$ is the number of POIs present in $L_u$ and $\hat{\sigma}$ is the standard deviation of $D_u$. Implement the expression $\hat{f}_u$ in a *density()* function.

# # In[ ]:


# def k_gaussian(x):
#     return (1/np.sqrt(2*np.pi))*np.exp(-x**2/2)

# def density(Du, dij, h):
#     D2 = list(map(lambda x : (dij-x)/h,Du))
#     density = 1/(len(Du)*h)*sum(list(map(k_gaussian,D2)))
#     return density


# # * The density $\hat{f}_u$ is used to estimate the probability that a POI $p_i$ not visited by a user $u$ matches the geographic preferences of $u$ given his visit history. In order to obtain this probability, we calculate the distance between $p_i$ and each of the POIs of the list $L_u$ and we then estimate the probability of each of these distances passing through $\hat{f}_u$.

# # * The final probability that a user $u$ visits a POI $p_i$ is obtained as follows:
# # 
# # **P**(p_i \| L_u) represents the probability that the user $u$ visits the POI $p_i$ given the geographical criterion. Implement the above equation in a *geo_proba()* function which takes as input the list $L_u$ and a POI, and which returns the probability of visiting this POI.

# # In[ ]:


# def geo_proba(Lu, pi, Du):
#     # pi= POI not visited by the user
#     # Lu = list of placeid of places visited by the user
#     # d is the list of distances btw pi and the places already visited by the user
#     # Du is the distance between each pair of places already visited
#     d = []
    
#     # retrieve the longitude and latitude of the candidate place to recommend
#     lat_i = df_locations[df_locations.id==pi].lat.values[0]
#     long_i = df_locations[df_locations.id == pi].lng.values[0]
    
#     # calculate all the distances between the candidate place to recommend and each place already visited in the Lu list
#     for l in Lu :
#         long_j = df_locations[df_locations.id == l].lng.values[0]
#         lat_j = df_locations[df_locations.id == l].lat.values[0]
#         d.append(geodesic((lat_j,long_j), (lat_i,long_i)).km)
    
    
#     # calculate h
#     n = len(Lu)
#     sigma = np.std(Du) #used to compute the standard deviation along the specified axis. This function returns the standard deviation of the array elements. The square root of the average square deviation (computed from the mean), is known as the standard deviation.
#     h = 1.06*sigma*n**(-1/5)
    
#     # calculate density
#     density_list = list(map(lambda x : density(Du, x, h),d))

#     return np.mean(density_list)


# 
# 
# ---
# 
# 

# # 4. Social influence
# 
# The *dataframe* *df_user_friends* associates each $u$ user with their $F(u)$ friends.
# 
# * For each pair of users $(u, v)$, we calculate their *social similarity* using the Jaccard coefficient.
# 
# Implement this coefficient in the *social_similarity()* function.
# 

# In[ ]:


def social_similarity(u, v):
    #friends of u
    list1 = df_user_friends[df_user_friends.userid1==u].friends_list.values[0]
    
    # friends of v
    list2 = df_user_friends[df_user_friends.userid1==v].friends_list.values[0]
    sim = len(list(set(list1) & set(list2))) / len(list(set(list1) | set(list2)))
    
    #similarity is based on same friendhips rather than if they went to the same places


# * This coefficient can be used in a collaborative filtering model.
# 
# $r_{ui}$ indicates the visit frequency of $u$ in $i$, extracted from *df_frequencies*.

# In[ ]:


def get_rate(u,i,trainset):
    #i= location id, u= user id
    if(user_is_in_trainset(u) and item_is_in_trainset(i)):
        inner_iid = trainset.to_inner_iid(i) # get the inner id of the location
        inner_uid = trainset.to_inner_uid(u) # get the inner id of the user
    else:
        #not present in trainset
        r = 0
        return r
    

    res_ir = trainset.ir[inner_iid] # get rates given for the location, ir method returns item's ratings (ie list of ratings of an item fiven by different users. list is of tuples of form (user_inner_id, rating).
    uid_ir = list(map(lambda x:x[0],res_ir)) #list of userid who have given rating to that location
    rate_ir = list(map(lambda x:x[1],res_ir)) #list of ratings

    if uid_ir.count(inner_uid) == 1: #ie if user u has rated item i
        r = rate_ir[uid_ir.index(inner_uid)] 
    
    # if the place has not been visited/rated by u
    else:
        r = 0
    return r


# In[ ]:


def r_hat(user_i,j,trainset,F_i) :
    #user_i = user id, j = location id, F_i = list of ids of friends of user with user_i
    rate_j = np.zeros(len(F_i)) #used to generate an array containing zeros.
    sim_list = np.zeros(len(F_i))
    
    F_i = enumerate(F_i)
    
    for i,user in F_i : #enumerate func converts a list into list of tuples where each tuple = (index, element)

        rate_j[i] = get_rate(user,j,trainset)
        sim_list[i] = social_similarity(user_i,user) #list of similarity scores of a user n his/her frnds => similarity btw 2 ppl depend on no. of their mutual frnds

    return np.dot(sim_list,rate_j) / max(np.sum(sim_list),0.01) #returns the dot product of two arrays, used to compute the sum of all elements, the sum of each row, and the sum of each column of a given array. 


# In[ ]:


def user_is_in_trainset(u) :

    try :
        trainset.to_inner_uid(u)
        res = 1
    except :
        res = 0

    return res


# In[ ]:


def item_is_in_trainset(i) :

    try :
        trainset.to_inner_iid(i)
        res = 1
    except :
        res = 0

    return res


# In[ ]:


def social_proba(u, i, Lu):
    # Lu = list of placeid of places visited by the user
    #retrieve the list of all places
    L = list(np.unique(df_locations.id))
    # retrieve the list of places visited by u
    # retrieve the list of places that have not yet been visited by user u
    L_diff = list(set(L)-set(Lu))
    # only consider locations that are in the training set
    L_diff = [i for i in L_diff if item_is_in_trainset(i)]
#     for i in L_diff:
#         if (item_is_in_trainset(i)==0):
#             L_diff.remove(i)
    
    # reduce the list of places to recommend to 50 due to computational complexity
    L_diff = L_diff[:200]
    # get u's friends list
    F_i = df_user_friends[df_user_friends.userid1==u].friends_list.values[0] #F_i = list of ids friends of user with u
    # consider only the list of friends who are in the training set
    
    for i in F_i:
        if (user_is_in_trainset(i)==0):
            F_i.remove(i)
        
    

    if F_i == []:
        return np.nan

    else:

        numerator = r_hat(u,i,trainset,F_i)
        denominator = max(max(list(map(lambda x : r_hat(u,x,trainset,F_i),L_diff))),0.01)
        
        return numerator/denominator

    


# 
# 
# ---
# 
# 

# # 5. Generation and evaluation of recommendations

# In[ ]:


#generates top 8 recommendations based on highest rated places by a user's friends
def frndRecom(u):
    listFrnd = df_user_friends[df_user_friends.userid1==u].friends_list.values[0]
    df_topRatedFrnd = df_frequencies[df_frequencies['userid'].isin(listFrnd)].sort_values('ratings',ascending=False)
    return df_topRatedFrnd[0:8]["placeid"].tolist()


# In[ ]:


#generates top 8 recommendations based on highest rated places by all users (popular places)
def topRatedRecom(u):
    df_topRated = df_frequencies.sort_values('ratings',ascending=False)
    return df_topRated[0:8]["placeid"].tolist()


# In[ ]:


#generates top 8 recommendations based on highest rated places by the user themselves in the past
def topVisitedRecom(u):
    df_topVisited = df_frequencies[df_frequencies.userid==u].sort_values('ratings',ascending=False)
    return df_topVisited[0:8]["placeid"].tolist()


# In[ ]:


#generates 6 friends of user
def Frnds(u):
    listFrnd = df_user_friends[df_user_friends.userid1==u].friends_list.values[0]
    return listFrnd[0:6]


# In[ ]:


#generates top 8 recommendations based on collaborative approach, using information about the geographical location of the place and social networks of the user.
def genRecom(u):
    
    Lu = list(set(df_user_POI[df_user_POI.userid == u].POI_list.values[0])) #list of all the places visited by the user
    L = list(np.unique(df_filtered_checkins.placeid)) # list of all the unique locations
    L_diff = list(set(L)-set(Lu)) # list of all the places not visited by user
    #making the list smaller because of computing reasons
    L_diff=L_diff[0:200]
    
    data = {'score':[],
        'placeid':[]}
    scores = pd.DataFrame(data)
    Du = distance_pair_list(Lu)
    
    for i in L_diff:
        if np.isnan(social_proba(u, i, Lu)):
                score = 0   #return value of social_proba is nan when user has no frnds
                # score = geo_proba(Lu, i, Du)   #return value of social_proba is nan when user has no frnds

        else :
            score = (social_proba(u, i, Lu)) 
#         print(score)
        df = pd.DataFrame({'score': [score], 'placeid': [i]})
        scores = pd.concat([scores, df])


    scores = scores.sort_values('score',ascending=False)
    scores = scores[0:8] #return top 8 recommendations for user
    return scores["placeid"].tolist()



   


# In[ ]:




def recoms(u):
    data={
    "frndRecom" : frndRecom(u),
    "topRatedRecom" : topRatedRecom(u),
    "topVisitedRecom": topVisitedRecom(u),
    "genRecom" : genRecom(u),
    "Frnds":Frnds(u),
    }
    return data



