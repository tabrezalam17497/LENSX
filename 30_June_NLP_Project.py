import streamlit as st
import joblib
import pandas as pd
import gmail
spam_model=joblib.load("spam_classifier.pkl")
language_model=joblib.load("lang_det.pkl")
news_model=joblib.load("news_cat.pkl")
review_model=joblib.load("review.pkl")

st.set_page_config(layout="wide")

#st.title("LENS eXpert(NLP suits)")
import streamlit as st

def custom_title(text, color="white", size=42, bg="teal"):
    st.markdown(f"""
        <h1 style='
            background-color: {bg};
            color: {color};
            font-size: {size}px;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        '>{text}</h1>
    """, unsafe_allow_html=True)

custom_title("LENS eXpert (NLP suits)", color="yellow", size=42, bg="black")




tab1,tab2,tab3,tab4=st.tabs(["🤖Spam Classifier","🌐Language Detection","🍔Food Review Sentiment","📰News Classification"])
with tab1:
    Msg=st.text_input("Enter Msg")
    if st.button("Prediction"):
        pred=spam_model.predict([Msg])
        if pred[0]==0:
            st.image("my_spam_logo.png")
        else:
            st.image("my_not_spam_logo.jpg")
    uploaded_file=st.file_uploader("Choose a file",type=["csv","txt"])
    
    
    if uploaded_file:
            
        df_spam=pd.read_csv(uploaded_file,header=None,names=['Msg'])
       
        pred=spam_model.predict(df_spam.Msg)
        df_spam.index=range(1,df_spam.shape[0]+1)
        df_spam["Prediction"]=pred
        df_spam["Prediction"]=df_spam["Prediction"].map({0:'Spam',1:'Not Spam'})
        st.dataframe(df_spam) 

    
       





with tab2:
    Msg2=st.text_input("Enter Msg ")
    if st.button("Prediction "):
        pred=language_model.predict([Msg2])
        st.success(pred[0])

    uploaded_file=st.file_uploader("Choose a file ",type=["csv","txt"])
    
    
    if uploaded_file:
            
        df_lang=pd.read_csv(uploaded_file,header=None,names=['Msg2'])
       
        pred=language_model.predict(df_lang.Msg2)
        df_lang.index=range(1,df_lang.shape[0]+1)
        df_lang["Prediction"]=pred
        st.dataframe(df_lang) 

 


with tab3:
    Msg3=st.text_input("Enter Message")
    if st.button(" Prediction"):
        pred=review_model.predict([Msg3])
        if pred[0]==1:
            st.image("not_spam.png")
        else:
            st.image("my_not_like_logo.jpg")

    uploaded_file=st.file_uploader("Select a file ",type=["csv","txt"])
    
    
    if uploaded_file:
            
        df_review=pd.read_csv(uploaded_file,header=None,names=['Msg3'])
       
        pred=review_model.predict(df_review.Msg3)
        df_review.index=range(1,df_review.shape[0]+1)
        df_review["Prediction"]=pred
        df_review["Prediction"]=df_review["Prediction"].map({0:"Dislike",1:"Like"})
        st.dataframe(df_review)  

with tab4:
    Msg4=st.text_input("Enter your  Message here ")
    if st.button(" view Prediction "):
        pred=news_model.predict([Msg4])
        st.success(pred[0])
        

    uploaded_file=st.file_uploader("Select a file here ",type=["csv","txt"])
    
    
    if uploaded_file:
            
        df_news=pd.read_csv(uploaded_file,header=None,names=['Msg4'])
       
        pred=news_model.predict(df_news.Msg4)
        df_news.index=range(1,df_news.shape[0]+1)
        df_news["Prediction"]=pred
        st.dataframe(df_news)

st.sidebar.image("photo1.jpg")
with st.sidebar.expander("👥ABOUT US"):
    st.write("We are group of students trying to understand the concept of NLP")
with st.sidebar.expander("📞CONTACT US"):
    st.write("9774685521")
    st.write("tabrezalam17497@gmail.com")
with st.sidebar.expander("Help"):
    st.write("We have use sklearn and nltk library here")   
    


