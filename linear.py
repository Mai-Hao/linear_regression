# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st

# Split data
df = pd.read_csv("credit access.csv", encoding='latin-1')

st.title("Linear regression")
st.write("## Predict credit value")

uploaded_file = st.file_uploader("Import a file", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1')
    df.to_csv("data.csv", index = False)

X = df.drop(columns=['giatri'])
y = df['giatri']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 12)

# Build model:
model = LinearRegression()

model.fit(X_train, y_train)

yhat_test = model.predict(X_test)


score_train=model.score(X_train, y_train)
score_test=model.score(X_test, y_test)

# Evaluate model:
mse=mean_squared_error(y_test, yhat_test)
rmse=mean_squared_error(y_test, yhat_test, squared=False)
mae=mean_absolute_error(y_test, yhat_test)


menu = ["Aim of model", "Model development", "Forecasting"]
choice = st.sidebar.selectbox('Category', menu)

if choice == 'Aim of model':    
    st.subheader("Aim of model")
    st.write("""
    ###### Model to forecast credit value of households
    """)  
    st.write("""###### Model using LinearRegression""")
    st.image("LSM.png")
    st.image("LSM_1.png")

elif choice == 'Model development':
    st.subheader("Model development")
    st.write("##### 1. Data frame")
    st.dataframe(df.head(3))
    st.dataframe(df.tail(3))  
    
    st.write("##### 2. Data visualisation")
    u=st.text_input('Inser var')
    fig1 = sns.regplot(data=df, x=u, y='giatri')    
    st.pyplot(fig1.figure)

    st.write("##### 3. Build model...")
    
    st.write("##### 4. Evaluation")
    st.code("Score train:"+ str(round(score_train,2)) + " vs Score test:" + str(round(score_test,2)))
    st.code("MSE:"+str(round(mse,2)))
    st.code("RMSE:"+str(round(rmse,2)))
    st.code("MAE:"+str(round(mae,2)))

    
elif choice == ' ':
    st.subheader("Forecasting")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1)
            st.dataframe(lines)
            # st.write(lines.columns)
            flag = True       
    if type=="Input":        
        git = st.number_input('Insert credit value')
        DT = st.number_input('Insert squares of house')
        TN = st.number_input('Insert income')
        SPT = st.number_input('Insert dependent people')
        GTC = st.number_input('Insert collateral value')
        TCH = st.number_input('Insert age of household head')
        VPCT = st.number_input('Insert borrowing in OTC market')
        LS = st.number_input('Insert credit history')
        lines={'credit_value':[git],'DT':[DT],'TN':[TN],'SPT':[SPT],'GTC':[GTC],'TCH':[TCH],'VPCT':[VPCT],'LS':[LS]}
        lines=pd.DataFrame(lines)
        st.dataframe(lines)
        flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)
            X_1 = lines.drop(columns=['giatri'])   
            y_pred_new = model.predict(X_1)       
            st.code("predicted value: " + str(y_pred_new))
