import streamlit as st
import time

st.title("***SENTIMENTAL ANALYSIS ON PRODUCT REVIEWS***")
st.divider()
st.header("***About Me***")
st.write("""Sentiment analysis of product reviews is crucial for understanding customer opinions and improving business strategies. This project utilizes a Random Forest classifier to predict the sentiment of textual reviews. TF-IDF, CountVectorizer, and word embeddings are employed for feature extraction to capture meaningful text representations. The model is trained on labeled review data and evaluated using accuracy, precision, recall, and F1-score. By leveraging ensemble learning, the Random Forest classifier enhances prediction robustness, making it a reliable approach for sentiment classification.
    """)
st.divider()

st.subheader("***Insights from the product***")
st.caption("**Hey there, Initially loading of model would take atleast a minute, meanwhile go through the video below and get to know about our product**")
st.video(r'archive\Akash_project_1.mp4')
st.divider()

with st.spinner("Getting the model ready..",show_time=True):
    import files.mainmod as md
    st.success("The model is now trained")       
st.divider()

     
st.caption("***feed some text to analyze the sentiments***")


text=st.text_input("_shoot the text_","this is a default text")
predval=md.predictor(text)
st.write(f"***Sentiment***: :blue[{predval}]")

with st.sidebar:
    st.sidebar.write("***Jaane se pehle***")
    g2g=st.sidebar.checkbox("*About my Guru*")
    if g2g:
        st.sidebar.caption("***developed by Team HAK***")
        st.sidebar.caption("***Contact us - hak181318@gmail.com***")
        
st.divider()
    


