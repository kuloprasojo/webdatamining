import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib

st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Calvin Rifansyah")
st.write("##### Nim   : 200411100072 ")
st.write("##### Kelas : Penambangan Data C ")
data_set_description, upload_data, preporcessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Prepocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Brain Tumor (Tumor Otak) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/jillanisofttech/brain-tumor")
    
with upload_data:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preporcessing:
    st.write("""# Preprocessing""")
    
    df.y.value_counts()
    df = df.drop(columns=["Unnamed: 0"])

    X = df.drop(columns="y")
    Y = df.y

    #ganti numerik
    colum_cat = ['GENDER','LUNG_CANCER']
    
    df_clean = df.copy()
    for i in colum_cat:
    catlist = df_clean[i].unique()
    for j, val in enumerate(catlist):
         dftobjfinal = df_clean[i].replace({val:j},inplace=True)
         #print(dftobjfinal)
         st.write(j,val)

    df_clean
