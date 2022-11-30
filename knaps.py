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

with modeling:
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state = 42, shuffle=True)

    st.write("""# Random Forest Classifier""")
    from sklearn.ensemble import RandomForestClassifier
    r_forest = RandomForestClassifier(criterion = 'entropy', max_depth = 20, n_estimators = 10000)
    r_forest.fit(x_train,y_train)
    predicted = r_forest.predict(x_test)
    score = r_forest.score(x_test, y_test)
    rf_score_ = np.mean(score)

    st.write('Accuracy : %.3f' % (rf_score_))

    st.write("""# Random Forest Classifier""")
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    logmodel= LogisticRegression()
    logmodel.fit(x_train,y_train)

    predictions=logmodel.predict(x_test)
    st.write(classification_report(y_test,predictions))
