import pandas as pd
import numpy as np 


import time, os
import pickle
from collections import Counter
from tqdm import tqdm


import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from spacy.lang.en import English
from spacy import displacy
#nlp = spacy.load("en_core_web_sm")


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import seaborn as sns


def load_clean_nlp(path,data,text_column):
    data = open(path+data,"rb")
    listings = pickle.load(data)
    df = pd.DataFrame(listings)
    df[text_column]  =[' '.join(i) for i in df[text_column]]
    df.drop_duplicates(subset = [text_column], keep ='first',inplace=True)
    df.reset_index(drop=True,inplace=True)

    return df

def spacy_tagging_demo (text):
    mytokens = nlp(text)

    tokens_list = []
    for token in mytokens:
        mytoken = {}
        mytoken['text'] = token.text
        mytoken['lemma'] = token.lemma_
        mytoken['POS'] = token.pos_
        mytoken['tag'] = token.tag_
        mytoken['dep'] = token.dep_
        mytoken['shape'] = token.shape_
        mytoken['Is_Alpha?'] = token.is_alpha
        mytoken['Is_stop?'] = token.is_stop
        mytoken['Is_punct?'] = token.is_punct
        mytoken['Is_digit?'] = token.is_digit
        tokens_list.append(mytoken)

    tokens_df = pd.DataFrame(tokens_list)
    return tokens_df


new_stop_words = ['experience','years','work','learning','data','science','race',
'gender','religion','origin','sexual','orientation','color','status',
'national','statistics','team','new','customer','help','opportunity',
'business','must','position','education','employment','please',
'working','applicant','best','disability','job','using','may',
'strong','etc','building','related','support','including'
 ,'skills','technical','across','time','tools','analytics','development'
,'understanding','fortune','workplace','company','report','required','requirement',
'health','year','veteran','provide','e.g.','ability','amazon','united','states','degree','bachelors',
'qualifications', 'preferred', 'scientist', 'bachelor', 'responsibilities', 'engineer', 'group', 'equal', '000', 'minimum',
'san','san francisco', 'francisco', 'insurance', 'dental', 'vision', 'program', 'medical', 'management', 'dental', 'vision'
     ,'information','masters','applied','information','knowledge']


def text_process(text,extra_stopwords,punctuation,pos):
    punctuation = string.punctuation
    final_stopwords =  list(STOP_WORDS)+new_stop_words
    if text !=None:
        mytokens = [word for word in text if word.is_digit != True and word.is_punct != True]
        mytokens = [word for word in mytokens if word.pos_ not in pos]
        mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
        mytokens = [word for word in mytokens if word not in final_stopwords and word not in punctuation and '@' not in word and 'http' not in word]
        mytokens = " ".join([i for i in mytokens])
        return mytokens
    else:
        return ""

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


def count_vectorizer_display(processed_text,topics=5, max_o = .8, min_o = 5):
    vectorizer = CountVectorizer(ngram_range=(1,2),stop_words = 'english',max_df = max_o,min_df = min_o)
    doc_word_matrix = vectorizer.fit_transform(processed_text)
    nmf_model= NMF(topics)
    document_topic_matrix = nmf_model.fit_transform(doc_word_matrix)
    display_topics(nmf_model,vectorizer.get_feature_names(),10)

    return document_topic_matrix

def tfidf_vectorizer_display(processed_text,topics=5,max_features = 2000, max_o = .8, min_o = 5):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3),binary =False,stop_words = 'english',max_features=2000,max_df = max_o,min_df = min_o)
    doc_word_matrix = tfidf_vectorizer.fit_transform(processed_text)
    nmf_model= NMF(topics)
    document_topic_matrix = nmf_model.fit_transform(doc_word_matrix)
    display_topics(nmf_model,tfidf_vectorizer.get_feature_names(),10)

    return document_topic_matrix

def tsne_viz(doc_topic_matrix,topic_df):
    tsne = TSNE(n_components = 2,random_state=0)
    top_2d_tsne = tsne.fit_transform(doc_topic_matrix)

    plt.figure(figsize=(15,8))
    ax = sns.scatterplot(x=top_2d_tsne[:,0],y=top_2d_tsne[:,1],hue = topic_df.Topic)
    plt.setp(ax.get_legend().get_texts(), fontsize='14') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='24') # for legend title
    plt.title("TSNE 2-Dimension Topic Visualization",fontsize = '20')
    plt.savefig("tsne.svg",format='svg')
    plt.show()

def pca_viz(doc_topic_matrix,topic_df):
    pca = PCA(n_components=2)
    pca.fit(doc_topic_matrix)
    top_2d_pca = pca.transform(doc_topic_matrix)

    plt.figure(figsize=(15,8))
    ax = sns.scatterplot(x=top_2d_pca[:,0],y=top_2d_pca[:,1],hue = topic_df.Topic)
    plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize='26') # for legend title
    plt.title('PCA 2-Dimension Topic Visualization', fontsize = '20')
    plt.savefig("pca.svg",format='svg')
    plt.show()



