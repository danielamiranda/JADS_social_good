import preprocessing
import simple_text_classification
import FastText
import polarity_analysis
import pandas as pd

### READ ###
# read the train data 
usa = preprocessing.train_data(13)
usa.to_csv("usa.csv", index=False)
# read the training data 
# from until 22 - 29 april
dutch_news = preprocessing.read_dutchnews_translated_data()

### CLASSIFICATION: healthcare, science, economy, travel
## Logistic Regression 
dutch_news_topics = pd.DataFrame
predictions_headlines = simple_text_classification.logistic_regression_classification(usa, dutch_news['headlines_en'])
predictions_headlines.columns = ['label_headline','probability_headline']
predictions_content = simple_text_classification.logistic_regression_classification(usa, dutch_news['content_en'])
predictions_content.columns = ['label_content','probability_content']
dutch_news_topics = pd.DataFrame(pd.concat([dutch_news, predictions_headlines, predictions_content], axis = 1))

# FastText
predictions_headlines_fasttext = FastText.fasttext_classification(usa, dutch_news['headlines_en'])
predictions_headlines_fasttext.columns = ['f_label_headline','f_probability_headline']
predictions_content_fasttext = FastText.fasttext_classification(usa, dutch_news['content_en'])
predictions_content_fasttext.columns = ['f_label_content','f_probability_content']
dutch_news_topics = pd.DataFrame(pd.concat([dutch_news_topics, predictions_headlines_fasttext, predictions_content_fasttext], axis = 1))
di= {0: "economy", 1: "healthcare", 2: "science", 3: "travel"}
dutch_news = dutch_news.replace({"label_headline": di}).replace({"label_content": di})

dutch_news_topics.to_csv("topic_classification_predictions.csv", index = False)

### POLARITY ##
dutch_news_polarity = pd.DataFrame()
polarity_headlines = polarity_analysis.polarity(dutch_news['headlines_en']).reset_index(drop = True)
polarity_content = polarity_analysis.polarity(dutch_news['content_en']).reset_index(drop = True)
dutch_news_polarity = pd.DataFrame(pd.concat([dutch_news, polarity_headlines, polarity_content], axis = 1))

dutch_news_polarity.to_csv("polarity_predictions.csv", index = False)


### VISUALIZATIONS ### 