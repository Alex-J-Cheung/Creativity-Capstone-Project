#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will show you how to create an interactive visualization of caption embeddings using a pretrained Sentence-Bert model and TSNE. I will also display this visualization using Streamlit for future sharing. 
# 
# I used 1 contest in this code primarily as a demonstration of how the plot looks like and how the code functions overall.
# 
# The first step is to load in the data from our SQL database. Specifically, I load in the caption text, their mean, ranking, and their contest number because we want to show these data points on the plot later on. Next, I loaded the pretrained SBert model and created caption embeddings. 
# 
# The next step is use kmeans clustering to create clusters for the captions. I chose to silhouette scores because they provided a clearer picture of the optimal amount of clusters compared to the elbow method. I chose a high range of clusters because it would a lot of them that represent different topics.
# 
# In order to create the plot, we first fit the embeddings onto a t-SNE 2d algorithm and create a new dataframe since t-SNE only accepts information from a dataframe. I insert the data that I pulled down from the SQL database into individual columns since we want each piece of data is going to be displayed to be unique. I then fit the t-SNE embeddings into kmeans and insert that data into the new database.
# 
# All that's left to do is to visualize the plot. I used Streamlit to share this visualization with others. I made a dashboard consisting of a dropdown menu which first views the entire plot as a whole and then the user can select which contest to view by selecting the dropdown menu. To view this plot in Streamlit, first download this file as a Python file since Streamlit does not natively support Jupyter Notebook. Next, go to the directory that contains this file or your own file that contains the code for the plot. Next, activate your conda environment since the code needs packages to run. Lastly, type into your console/terminal "streamlit run caption_embeddings_plot.py" and it will load a Streamlit tab in your web browser that has your interactive plot. 

# In[ ]:


# libraries
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
        K_range = range(49, 50)

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

