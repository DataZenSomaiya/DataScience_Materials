#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords
import re
def Noiseremoval(text):
    text=text.lower()
    text=re.sub(r'\W',' ',text)#this will remove all non word characters like#,$
    text=re.sub(r'\d',' ',text)#will remove digits
    text=re.sub(r'\s+',' ',text)#will remove extra spaces
    new=[]
    words=nltk.word_tokenize(text)
    for word in words:
        if word not in stopwords.words('english'):
            new.append(word)
    text=' '.join(new)
    # print(text)
    return text


# In[2]:


from nltk.stem.wordnet import WordNetLemmatizer
lem=WordNetLemmatizer()
def lemmetize(sentence):
        words=nltk.word_tokenize(sentence)
        words=[lem.lemmatize(word,pos='v') for word in words]
        sentence=' '.join(words)
        return sentence


# In[ ]:




