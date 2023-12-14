#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will show you how to create an interactive visualization of caption embeddings using a pretrained Sentence-Bert model and TSNE. I will also display this visualization using Streamlit for future sharing. Use kmeans after tsne

# In[1]:


import pandas as pd
import mysql.connector
from mysql.connector import Error
pd.set_option('display.max_colwidth', None)
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Streamlit app
st.title("Caption Embeddings Visualization")

# Dropdown menu for selecting contest_numbers
contest_numbers = st.sidebar.selectbox("Select Contest Numbers", [863])

try:
    # Connect to MySQL database
    connection = mysql.connector.connect(host='dbnewyorkcartoon.cgyqzvdc98df.us-east-2.rds.amazonaws.com',
                                         database='new_york_cartoon',
                                         user='dbuser',
                                         password='Sql123456')
    if connection.is_connected():
        db_Info = connection.get_server_info()
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()

        # pulling down data from SQL database via search
        sql_select_Query = f"SELECT caption, mean, ranking, contest_num FROM result WHERE contest_num in (863);"

        cursor.execute(sql_select_Query)

        # show attributes names of target data
        num_attr = len(cursor.description)
        attr_names = [i[0] for i in cursor.description]

        # get all records
        records = cursor.fetchall()
        df = pd.DataFrame(records, columns=attr_names)

        sentences = df['caption'].tolist()

        model = SentenceTransformer('all-MiniLM-L12-v2')

        caption_embeddings = model.encode(sentences)  # embeddings

        silhouette_scores = []
        K_range = range(2, 7)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(caption_embeddings)
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(caption_embeddings, labels)
            silhouette_scores.append(silhouette_avg)

        optimal_k = K_range[np.argmax(silhouette_scores)]

        X = list(caption_embeddings)
        X = np.array(X)

        X_embedded = TSNE(n_components=2).fit_transform(X)

        df_embeddings = pd.DataFrame(X_embedded)
        df_embeddings = df_embeddings.rename(columns={0: 'x', 1: 'y'})
        df_embeddings = df_embeddings.assign(text=df.caption.values)
        df_embeddings['mean'] = df['mean']
        df_embeddings['contest_num'] = df['contest_num'].astype(str)
        df_embeddings['ranking'] = df['ranking']

        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans.fit(X_embedded)
        cluster_labels = kmeans.labels_
        df_embeddings['cluster_label'] = cluster_labels.astype(str)

        # Display the scatter plot using Plotly Express
        color_scale = px.colors.qualitative.Set1
        fig = px.scatter(
            df_embeddings, x='x', y='y',
            color='cluster_label',
            color_discrete_sequence=color_scale,
            labels={'cluster_label': 'Cluster Group'},
            hover_data=['text', 'mean', 'ranking', 'contest_num'],
            title=f'Caption Embedding Visualization for Contest Number(s) {contest_numbers}'
        )

        # Display the Plotly figure using Streamlit
        st.plotly_chart(fig)

except Error as e:
    st.error(f"Error while connecting to MySQL: {e}")

if not contest_numbers or contest_numbers == "All":
    st.markdown("**Please select a specific contest number from the dropdown menu to view details for that contest.**")

