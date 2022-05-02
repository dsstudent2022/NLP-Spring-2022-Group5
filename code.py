#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Start


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import string
import math
import spacy
nlp = spacy.load('en')
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
stopwords = stopwords.words('english')


# In[10]:


GFC_data = pd.read_csv("GFC.csv")


# In[11]:


GFC_data


# In[3]:


#Source: https://gist.github.com/tolgayan/a309c5655efc2b40704d86f83ab3d792
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
        
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
        
    return results

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key = lambda x: (x[1], x[0]),reverse = True)


# In[14]:


GFC_data_2 = " ".join(GFC_data['Article'])


# In[15]:


GFC_data_2


# In[16]:


#Prepare a cleaning algorithm that will make our text easily translate into numbers for analysis
sentences = []
def cleanData(doc, stemming = False):
    doc = doc.lower()
    doc = nlp(doc)
    #pos_tokens = nltk.pos_tag(text)
    tokens = [tokens.lower_ for tokens in doc]
    tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
    tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
    #tokens = [tokens for tokens in tokens if (tokens.pos != 'PPN')]
    #tokens = [tokens for tokens in tokens if (nltk.pos_tag(tokens[i]) != 'NNP')]
    tokens = [tokens for tokens in tokens if (tokens.is_alpha == True)]
    final_token = [token.lemma_ for token in tokens]
    together = " ".join(final_token)
    tagged_sentence = nltk.tag.pos_tag(together.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    together = " ".join(edited_sentence)
    sentences.append(together)
    #print(sentences)


# In[17]:


cleanData(GFC_data_2)


# In[18]:


len(sentences)


# In[19]:


cv = CountVectorizer(min_df = 0.5, stop_words=stopwords, max_features=5500)
word_count_vector = cv.fit_transform(sentences)


# In[20]:


tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf=True)
tfidf_transformer.fit(word_count_vector)


# In[21]:


feature_names = cv.get_feature_names()
tf_idf_vector = tfidf_transformer.transform(cv.transform(sentences[:]))


# In[22]:


sorted_items = sort_coo(tf_idf_vector.tocoo())


# In[23]:


keywords = extract_topn_from_vector(feature_names, sorted_items, 100)


# In[24]:


print("\n===Keywords===")
keywords


# In[25]:


keywords_list = []
score_list = []
for k in keywords:
    keywords_list.append(k)
    score_list.append(keywords[k])


# In[26]:


keywords_df = pd.DataFrame(list(keywords))
keywords_df['Importance'] = score_list


# In[27]:


keywords_df.columns = ['GFC-Keyword', 'GFC-Keyword Value of Importance']


# In[28]:


keywords_df.to_csv('GFC_keywords.csv')


# In[ ]:


#Topic Modeling


# In[ ]:


len(sentences)


# In[ ]:


tf_idf_vectorizer = TfidfVectorizer(min_df=0.5)
cv_vectorizer = CountVectorizer()


# In[140]:


#Bring in sentences
tf_idf_arr = tf_idf_vectorizer.fit_transform(sentences)


# In[141]:


cv_arr = cv_vectorizer.fit_transform(sentences)


# In[142]:


vocab_tf_idf = tf_idf_vectorizer.get_feature_names()
len(vocab_tf_idf)


# In[143]:


from sklearn.decomposition import LatentDirichletAllocation


# In[144]:


#Once we have vocabulary, we build the LDA model by creating the LDA class:
#Implementation of LDA:
lda_model = LatentDirichletAllocation(n_components=5, max_iter=20, random_state=1234)


# In[145]:


#fit transform our model on count vectorizer: running this will return our topics


# In[146]:


X_topics = lda_model.fit_transform(tf_idf_arr)


# In[147]:


#.components gives us our topic distribution


# In[148]:


topic_words = lda_model.components_


# In[149]:


#Retrieve the Topics
n_top_words = 5
for i, topic_dist in enumerate(topic_words):
    sorted_topic_dist = np.argsort(topic_dist)
    topic_words = np.array(vocab_tf_idf)[sorted_topic_dist]
    topic_words = topic_words[:-n_top_words:-1]
    #print("Topic", str(0), topic_words)
    print("Topic", str(i+1), topic_words)


# In[150]:


#To view what topics are assigned to the documents:
doc_topic = lda_model.transform(tf_idf_arr)
for n in range(doc_topic.shape[0]):
    topic_doc = doc_topic[n].argmax()
    print("Document", n+1, "--Topic:", topic_doc)


# # Models

# In[ ]:


#Testing Data


# In[34]:


PC19_data = pd.read_csv("pc19.csv")


# In[35]:


PC19_data


# In[36]:


PC19_data_2 = " ".join(PC19_data['Article'])


# In[37]:


PC19_data_2


# In[38]:


#Prepare a cleaning algorithm that will make our text easily translate into numbers for analysis
sentences = []
def cleanData(doc, stemming = False):
    doc = doc.lower()
    doc = nlp(doc)
    #pos_tokens = nltk.pos_tag(text)
    tokens = [tokens.lower_ for tokens in doc]
    tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
    tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
    #tokens = [tokens for tokens in tokens if (tokens.pos != 'PPN')]
    #tokens = [tokens for tokens in tokens if (nltk.pos_tag(tokens[i]) != 'NNP')]
    tokens = [tokens for tokens in tokens if (tokens.is_alpha == True)]
    final_token = [token.lemma_ for token in tokens]
    together = " ".join(final_token)
    tagged_sentence = nltk.tag.pos_tag(together.split())
    edited_sentence = [word for word,tag in tagged_sentence if tag != 'NNP' and tag != 'NNPS']
    together = " ".join(edited_sentence)
    sentences.append(together)
    #print(sentences)


# In[40]:


cleanData(PC19_data_2)


# In[41]:


len(sentences)


# In[42]:


cv = CountVectorizer(min_df = 0.5, stop_words=stopwords, max_features=5500)
word_count_vector = cv.fit_transform(sentences)


# In[43]:


tfidf_transformer = TfidfTransformer(smooth_idf = True, use_idf=True)
tfidf_transformer.fit(word_count_vector)


# In[44]:


feature_names = cv.get_feature_names()
tf_idf_vector = tfidf_transformer.transform(cv.transform(sentences[:]))


# In[45]:


sorted_items = sort_coo(tf_idf_vector.tocoo())


# In[46]:


keywords = extract_topn_from_vector(feature_names, sorted_items, 100)


# In[47]:


print("\n===Keywords===")
keywords


# In[48]:


keywords_list = []
score_list = []
for k in keywords:
    keywords_list.append(k)
    score_list.append(keywords[k])


# In[49]:


keywords_df = pd.DataFrame(list(keywords))
keywords_df['Importance'] = score_list


# In[50]:


keywords_df.columns = ['PC19-Keyword', 'PC19-Keyword Value of Importance']


# In[51]:


keywords_df.to_csv('PC19.csv')


# In[ ]:


#Organizing our Training Data and Testing Data


# In[ ]:


#Naive Bayes


# In[ ]:


#source: previous project & https://www.analyticsvidhya.com/blog/2021/07/performing-sentiment-analysis-with-naive-bayes-classifier/


# In[ ]:


# Split into training and testing data
x = GFC_data_2['text']
y = PC19_data_2['text'] 
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.30, random_state=42)


# In[ ]:


# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words='english')
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

y = vec.fit_transform(y).toarray()
y_test = vec.transform(y_test).toarray()


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(x, y)


# In[ ]:


model.score(x_test, y_test)


# In[ ]:


#LSTM


# In[ ]:


#Source: work & https://www.analyticsvidhya.com/blog/2021/06/natural-language-processing-sentiment-analysis-using-lstm/


# In[ ]:


# Importing required libraries
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt


# In[ ]:


# Encoded the target column
lb=LabelEncoder()
GFC_data_2['sentiment'] = lb.fit_transform(GFC_data_2['sentiment'])


# In[ ]:


model = Sequential()
model.add(Embedding(500, 120, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(176, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())


# In[ ]:


#SVM


# In[ ]:


from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_linear.fit(train_vectors, GFC_data_2['sentiment'])
prediction_linear = classifier_linear.predict(test_vectors)
# results
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
report = classification_report(testData['Label'], prediction_linear, output_dict=True)
print('positive: ', report['pos'])
print('negative: ', report['neg'])


# In[ ]:


####END

