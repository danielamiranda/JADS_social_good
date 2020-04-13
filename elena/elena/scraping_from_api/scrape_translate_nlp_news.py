"""
@author: Elena Stamatelou
"""
#!pip install newsapi-python
#!pip install googletrans
#!python -m spacy download en_core_web_md
# From NEWS API

from newsapi import NewsApiClient
import pandas as pd
from googletrans import Translator
import spacy
nlp = spacy.load('en_core_web_md')

#read lexicons
indexes_pos = {}    
positive = pd.read_csv('positive-words.txt')
for index, row in positive.iterrows():
    positive.at[index, "words"] = row["words"].split(' ')[0]
    indexes_pos[positive.at[index, "words"]] = index

indexes_neg = {}    
negative = pd.read_csv('negative-words.txt')
for index, row in negative.iterrows():
    negative.at[index, "words"] = row["words"].split(' ')[0]
    indexes_neg[negative.at[index, "words"]] = index

# scrape news
newsapi = NewsApiClient(api_key='a5c184912f3c454d98143a47ea6109e9')
#top-headlines
top_headlines = newsapi.get_top_headlines(country='gr')

for i in range(5):#len(top_headlines['articles'])):

    title = top_headlines['articles'][i]['title']
    translation = Translator().translate(title, dest='en')
    print(translation.origin, ' -> ', translation.text)

    doc = nlp(translation.text)
    polarity_scores = 0 
    for token in doc:
        lemmatized_token = token.lemma_
#    print(lemmatized_token)
        if (lemmatized_token in indexes_pos) :
            polarity_scores = polarity_scores + 1
            print('positive', lemmatized_token)
        if (lemmatized_token in indexes_neg) :
            polarity_scores = polarity_scores - 1
            print('negative',lemmatized_token)
    print("polarity",polarity_scores)
