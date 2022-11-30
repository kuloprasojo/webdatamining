import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    
    colum_cat = ['GENDER','LUNG_CANCER']
    for i in colum_cat:
        st.write(df[i].value_counts())
    
    df_clean = df.copy()
    for i in colum_cat:
        st.write(f'Catagory of {i}')
        catlist = df_clean[i].unique()
        for j, val in enumerate(catlist):
             dftobjfinal = df_clean[i].replace({val:j},inplace=True)
             st.write(j,val)
    
    df_clean

    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor()
    y_pred = clf.fit_predict(df_clean)

    x_score = clf.negative_outlier_factor_
    outlier_score = pd.DataFrame()
    outlier_score["score"] = x_score

    #threshold
    threshold2 = -1.5
    filtre2 = outlier_score["score"] < threshold2
    outlier_index = outlier_score[filtre2].index.tolist()

    len(outlier_index)

    df_clean = df.copy()

    g = df_clean.groupby('LUNG_CANCER')
    df_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
    df_balanced = df_balanced.reset_index(drop=True)
    x = df_balanced[['SMOKING','PEER_PRESSURE','ALCOHOL CONSUMING']]
    y = df_balanced['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
