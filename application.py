import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,wordpunct_tokenize
import re
nltk.download('stopwords')
with open('model.pkl','rb') as f:
  model=pickle.load(f)

words=stopwords.words('english')
def cleaner(col):
  col=re.sub('<.*?>','',col)
  col=re.sub('https?://\S+|www\.\S+','',col)
  return re.sub('[^\w\s]','',col)
def stopword_removal(col):
  L=[]
  tokens=wordpunct_tokenize(col)
  for word in tokens:
    if word not in words:
      L.append(word)
  return " ".join(L)

def predict(text):
  text=cleaner(text)
  text=stopword_removal(text)
  prediction= model.predict([text])[0]
  if prediction==0:
    return "Real News"
  else:
    return "Fake News"

st.header('Fake News Classifier')
text=st.text_input('Enter the News Data')
if st.button('Classify News'):
  st.success(f'This is {predict(text)}')
st.info('This model predicts the news based on the frequency of the words so it can not always be right if it predicts news to be fake.')