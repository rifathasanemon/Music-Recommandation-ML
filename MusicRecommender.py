#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import Recommenders as Recommenders


# In[92]:


df1 = pd.read_csv('song_data.csv')
df1.head()


# In[93]:


df2 = pd.read_csv('data_2.csv')
df2.head()


# In[94]:


#Adding Both Data to a Single data
song_df = pd.merge(df2, df1.drop_duplicates(['song_id']), on='song_id', how='left')
song_df.head()


# In[95]:


#checking the columns of song & title with artist name
song_df['song'] = song_df['title']+' - '+song_df['artist_name']
song_df.head()


# In[96]:


# using first 20,000 data for recommandation
song_df = song_df.head(20000)


# In[97]:


#group these songs by listen count
song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
song_grouped.head()


# In[98]:


#seeing the percentage of song listening
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum ) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])


# In[99]:


#using the recommender class

pr = Recommenders.popularity_recommender_py()


# In[100]:


pr.create(song_df, 'user_id', 'song')


# In[101]:


# display the top popular songs
pr.recommend(song_df['user_id'][1])


# In[102]:


#checking sameitems


# In[103]:


sm = Recommenders.item_similarity_recommender_py()
sm.create(song_df, 'user_id', 'song')


# In[104]:


user_items = sm.get_user_items(song_df['user_id'][1])


# In[105]:


# display user songs history
for user_item in user_items:
    print(user_item)


# In[106]:


# give song recommendation for that user
ir.recommend(song_df['user_id'][10])


# In[107]:


# give related songs based on the words
ir.get_similar_items(['Meadowlarks - Fleet Foxes', 'Come Back To Bed - John Mayer'])


# In[ ]:




