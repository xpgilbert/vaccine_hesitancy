# TEXT ANALYSIS
### Imports

## Data Processing, Basic Visualizations, and Linear Algebra
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
import os
os.chdir('..')

## Text cleaning
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

## Vectorizers
from sklearn.feature_extraction import text
#from nltk.util import ngrams
from nltk import FreqDist

## Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

## KMeans Exploration
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

## Sentiment Analysis Exploration
import textblob

## Feature Selection
from sklearn.feature_selection import RFECV
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.preprocessing import FunctionTransformer

## Classification, Evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix

## Load the data
df = pd.read_csv('data/full_ucsd_data.csv')

## Segment out the interested data, including constructs and comments
interested = ['id','state', 'cov_band', 'comments', 'month', 'hour', 'day', 'wave'
            , 'anxiety', 'depressed','life_avg', 'anx_avg', 'cov_avg', 'vax_avg'
            ,'life_med', 'anx_med', 'cov_med', 'vax_med', 'anx_band','perc_med','m5','flu1','flu2']
df = df[interested]
df.head()

## Extract data with only non-null comments
comdf = df[~df['comments'].isnull()]
comdf['length'] = comdf['comments'].apply(len)
print(f'Number of comments: {comdf.shape[0]}')

## What are our popular comments
comms = comdf['comments']
print(comms.value_counts()[:10])

## Plot histogram of comment lengths
g=sns.boxplot(x='anx_med', y='length', data=comdf)
g.set_yscale('log')
g.set_title('Length by Anxiety Medians')
g.set_ylabel('Log of Length')
fig = g.get_figure()
fig.savefig('plots/length_anxiety.png')

## Histograms by hesitancy
g=sns.FacetGrid(comdf, col='cov_band',size=4)
g.map(sns.histplot, 'length', log_scale=True)
g.fig.suptitle('By Vaccine Hesitancy', fontsize=10)
g.fig.savefig('plots/length_hesitancy.png')

## Histograms by anxiety
g=sns.FacetGrid(comdf, col='anx_band',size=4)
g.map(sns.histplot, 'length', log_scale=True)
g.fig.suptitle('By General Anxiety', fontsize=10)
g.fig.savefig('plots/length_anxiety_class.png')

## Check class balance
print(f'Hesitant count:     {len(comdf[comdf.cov_band==0].comments)}')
print(f'Not hesitant count: {len(comdf[comdf.cov_band==1].comments)}')
print(f'More anxious count: {len(comdf[comdf.anx_band==0].comments)}')
print(f'Less anxious count: {len(comdf[comdf.anx_band==1].comments)}')

#### 3.5.2 Data Preparation
## Add comment to stop words since it makes up the most popular comments
## Add 'none' and 'im' to stopwords list from the wordcount above
stop_words = stopwords.words('english')
stop_words = set(stop_words + ['comment','comments','none','im','nope'])

## Text processing functions
def lower_nopunc(samp_string):

    ## Lower string
    low_string = samp_string.lower()
    
    ## Remove punctuation
    nopunc = [char for char in low_string if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return nopunc

def no_stops(samp_string):

    temp_list = samp_string.split()
    fin_list = []
    for word in temp_list:
        if word not in stop_words:
            fin_list.append(word)
    return fin_list

def lemmatizer(word_list):

    lemma_list = []
    lemmatizer = WordNetLemmatizer()
    for word in word_list:
        lemma_list.append(lemmatizer.lemmatize(word))
    return lemma_list

def string_cleaner(samp_string):
    
    word_list = lemmatizer(no_stops(lower_nopunc(samp_string)))
    output = " ".join(word_list)
    return output

## Create new clean_comment column
comdf['clean_comment'] = comdf['comments'].apply(string_cleaner)

## Whats our most common words
comment_string = ""
for i in comdf['clean_comment']:
    comment_string += i + " "

wordcount = FreqDist(comment_string.split())
wordcount.most_common(10)

##     THIS IS WHERE WE FIND THAT 'none' and 'im' SHOULD BE ADDED TO STOPWORDS     ##

## Extact variables of interest for analysis
nlp_int = ['comments', 'length', 'clean_comment'
           , 'cov_med', 'cov_band', 'anx_med', 'anx_band','life_med','vax_med','perc_med'
           ,'flu1','flu2','m5']
tdf = comdf[nlp_int]
tdf = tdf[tdf['clean_comment']!='']
tdf.dropna(inplace=True)
print(f'Temporary DataFrame Shape: {tdf.shape}')

## Create vectors
vectorizer = text.CountVectorizer()
com_vecs = vectorizer.fit_transform(tdf['clean_comment'])
print('Vectorized matrix shape:',com_vecs.shape)

## Create TfIdf matrix
com_tfidf = text.TfidfTransformer().fit_transform(com_vecs)

## Here, we will use KMeans clusterting to see how the words from the comments group together.  
## We may find some interesting results.

## KMeans 

wss = []       ## List of objective scores
sil = []          ## List of silhouette scores 

## Explore values of n from 2 to 20
for n in range(2, 20):
    if n % 4 == 0:
        print(n)
    model = KMeans(n_clusters=n, random_state=42)
    model.fit(com_tfidf)
    wss.append(-model.score(com_tfidf))
    sil.append(silhouette_score(com_tfidf, model.labels_, metric='euclidean'))

## Lets use k=5
cluster = KMeans(n_clusters=5, random_state=42)
cluster.fit(com_tfidf)

## Use wordcount to collect the words based on cluster labels
cluster_labels = cluster.predict(com_tfidf)
obs_per_cluster = np.bincount(cluster_labels)
wordcounts = []
for i in range(len(obs_per_cluster)):
    cluster_coms = np.array(tdf['clean_comment'])[cluster_labels==i]
    wc = FreqDist(' '.join(cluster_coms).split())
    wordcounts.append(wc)
    print(wordcounts[i])

## Print clusters
for i in range(len(wordcounts)):
    print(f"Most common words in cluster {i}:\n",
        wordcounts[i].most_common(10), "\n\n")

## Use textblob's NaiveBayes classifier for sentiment exploration
nb = textblob.en.sentiments.NaiveBayesAnalyzer()
sentiment_prediction = []
polarity_scores = []
counter = 0
for com in tdf['clean_comment'].dropna():
    if counter % 300 == 0:
        print(counter)
    sen = nb.analyze(com).classification
    pol = textblob.TextBlob(com).polarity
    if sen == 'pos':
        sentiment_prediction.append("Positive")
    elif sen == 'neg':
        sentiment_prediction.append("Negative")
    polarity_scores.append(pol)
    counter += 1

## Create scores and recall dictionaries for comparison later
nb_scores = {}
nb_recalls = {}

## Use NaiveBayes for predicting covid hesitancy with only term frequency vectors

## Collect our features and target
X = com_vecs
y = tdf['cov_band']

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Train model, generate predictions
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)

## Evaluate model
print('First model metrics:')
print(f'Classification Report:\n{classification_report(y_test, pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
nb_scores['nlptk_tokens'] = accuracy_score(y_test,pred)
nb_recalls['nlptk_tokens'] = recall[0]

## Create TensorFlow tokenizer vector matrix

def tokenizer(coms):

    vocab_len = 10000
    tokenizer = Tokenizer(num_words=vocab_len)

    tokenizer.fit_on_texts(coms)                                  # Fit on desc data
    tokens = tokenizer.texts_to_sequences(coms)                   # Create tokens for each text
    
    ## Pad tokens to ensure vectors are the same length
    max_len = np.max([len(token) for token in tokens])            # Find max token length
    com_tk = pd.DataFrame(pad_sequences(tokens, maxlen=max_len, padding='post')).reset_index()
    return com_tk
    
com_tf = tokenizer(tdf['clean_comment'])
com_tf.shape

## Use NaiveBayes for predicting covid hesitancy with TensorFlow tokens

## Collect our features and target

X = com_tf
y = tdf['cov_band']

## Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Train model, generate predictions
nb = MultinomialNB()
nb.fit(X_train, y_train)
pred = nb.predict(X_test)

## Evaluate model
print('With TensorFlow metrics:')
print(f'Classification Report:\n{classification_report(y_test,pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
nb_scores['tf_tokens'] = accuracy_score(y_test,pred)
nb_recalls['tf_tokens'] = recall[0]

## Commenting out the different steps showed that leaving the tfidf transformer in led to a worse model
## Leaving it in there so others can play with it

pipeline_cv = Pipeline([('vect',text.CountVectorizer())
                       #,('tfidf',text.TfidfTransformer())
                       ,('nb',MultinomialNB()) 
                       ])

params = {'vect__ngram_range':((1,1), (1,2), (2,2))         ## unigrams, bigrams, only bigrams
        ,'vect__tokenizer':(None, Tokenizer())              ## Use none or TensorFlow
        ,'vect__min_df':(1,2)
        #,'tfidf__use_idf':(True, False)
        #,'tfidf__norm': ('l1', 'l2')
        ,'nb__alpha':[1,0.1,0.01,0.001]
    }

## Get our train and test data from the start
## Target is the vaccine hesitancy class

X=tdf['clean_comment']
y=tdf['cov_band']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

## Fit gridsearch and print best params
clf = GridSearchCV(pipeline_cv, params,cv=5)
clf.fit(X_train, y_train)
print(f'Best params:\n{clf.best_params_}\n')

## Generate Predictions
pred = clf.predict(X_test)

## Evaluate model
print('Pipeline metrics:')
print(f'Classification Report:\n{classification_report(y_test,pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
nb_scores['pipeline'] = accuracy_score(y_test,pred)
nb_recalls['pipeline'] = recall[0]

## Show NaïveBayes scores and recall together
nbres = pd.DataFrame([nb_scores,nb_recalls],index=['overall accuracy','hesitant recall'])
nbres

## First lets get a baseline for our model by training it on only non-text data

## Create scores and recall dictionaries for comparison later
scores  = {}
recalls = {}

ment_matrix = tdf[['anx_med','life_med','vax_med','perc_med','m5']]#,'flu1','flu2']]
y = tdf['cov_band']

X_train, X_test, y_train, y_test = train_test_split(
    ment_matrix,y,test_size=0.3,random_state=42)

## Set RFC parameters for GridSearch Cross Validation
params = {
     'n_estimators':[50,100,200]
    ,'max_features':['auto','sqrt']
    ,'max_depth':[2,5,10]
    ,'min_samples_leaf':[1,2,4]
    ,'min_samples_split':[2,3,5]
}

## Fit gridsearch and print best params
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc,params,cv=5,verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(f'Best params:\n{clf.best_params_}\n')

## Evaluate model
print('No text data metrics:')
print(f'Classification Report:\n{classification_report(y_test, pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
scores['no_text'] = accuracy_score(y_test,pred)
recalls['no_text'] = recall[0]

## Now lets use two dataframes to predict our hesitancy target.
## First dataframe is our processed text data matrix,
## second is the survey constructs and direct measurements

## For this RFC we will use the TensorFlow Tokenizer Matrix 
## since it performed best with our NaïveBayes Model

text_matrix = com_tf
ment_matrix = tdf[['anx_med','life_med','vax_med','perc_med','m5']]#,'flu1','flu2']]

y = tdf['cov_band']

text_train, text_test, ment_train, ment_test, y_train, y_test = train_test_split(
    text_matrix,ment_matrix,y,test_size=0.3,random_state=42)

X_train = np.concatenate([text_train, ment_train], axis=1)
X_test = np.concatenate([text_test,ment_test],axis=1)

## Set RFC parameters for GridSearch Cross Validation
params = {
     'n_estimators':[50,100,200]
    ,'max_features':['auto','sqrt']
    ,'max_depth':[2,5,10]
    ,'min_samples_leaf':[1,2,4]
    ,'min_samples_split':[2,3,5]
}

## Fit gridsearch and print best params
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc,params,cv=5,verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(f'Best params:\n{clf.best_params_}\n')

## Evaluate model
print('With vectorized matrix metrics:')
print(f'Classification Report:\n{classification_report(y_test, pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
scores['only_tokens'] = accuracy_score(y_test,pred)
recalls['only_tokens'] = recall[0]

## Lets use two dataframes to predict our hesitancy target.
## First dataframe is our NLP matrix, second is the survey constructs and direct measurements

## Now lets try with the TfIdf scores

text_matrix = com_tfidf.todense()
ment_matrix = tdf[['anx_med','life_med','vax_med','perc_med','m5']]#,'flu1','flu2']]

y = tdf['cov_band']

text_train, text_test, ment_train, ment_test, y_train, y_test = train_test_split(
    text_matrix,ment_matrix,y,test_size=0.3,random_state=42)

X_train = np.concatenate([text_train, ment_train], axis=1)
X_test = np.concatenate([text_test,ment_test],axis=1)

## Set RFC parameters for GridSearch Cross Validation
params = {
     'n_estimators':[50,100,200]
    ,'max_features':['auto','sqrt']
    ,'max_depth':[2,5,10]
    ,'min_samples_leaf':[1,2,4]
    ,'min_samples_split':[2,3,5]
}

## Fit gridsearch and print best params
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc,params,cv=5,verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(f'Best params:\n{clf.best_params_}\n')

## Evaluate model
print('With TfIdf metrics:')
print(f'Classification Report:\n{classification_report(y_test, pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Record recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
scores['with_tfidf'] = accuracy_score(y_test,pred)
recalls['with_tfidf'] = recall[0]

## Show RFC scores and recall together
rfcres = pd.DataFrame([scores,recalls],index=['overall accuracy','hesitant recall'])
rfcres

## With tokens is our best bet.  
## Lets refine the hyperparameters even further to improve the model.
## Reminder that our best parameters for this model were:

{'max_depth': 5, 'max_features': 'auto', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}

## First apply featureselection to our dataset
text_matrix = com_tf
ment_matrix = tdf[['anx_med','life_med','vax_med','perc_med','m5']]#,'flu1','flu2']]

y = tdf['cov_band']

text_train, text_test, ment_train, ment_test, y_train, y_test = train_test_split(
    text_matrix,ment_matrix,y,test_size=0.2,random_state=42)

X_train = np.concatenate([text_train, ment_train], axis=1)
X_test = np.concatenate([text_test,ment_test],axis=1)

## Instantiate feature selection and classifier
rfc = RandomForestClassifier()
selector = RFECV(rfc, min_features_to_select=75, cv=5)  ## Use large number for text data matrices
selector.fit(X_train, y_train)
X_train = selector.transform(X_train)
X_test = selector.transform(X_test)
print(f'Train shape: {X_train.shape}')

## Set RFC parameters for GridSearch Cross Validation
params = {
     'n_estimators':[75,100,125]
    ,'max_features':['auto']
    ,'max_depth':[5,7,9]
    ,'min_samples_leaf':[1,2]
    ,'min_samples_split':[5,7,9]
}

## Fit gridsearch and print best params
rfc = RandomForestClassifier()
clf = GridSearchCV(rfc,params,cv=5,verbose=1,n_jobs=-1)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(f'Best params:\n{clf.best_params_}\n')

## Evaluate model
print('Final Model metrics:')
print(f'Classification Report:\n{classification_report(y_test, pred)}\n')
print(f'Confusion Matrix:\n{confusion_matrix(y_test,pred)}')

## Print recall and score
_,recall,_,_ = precision_recall_fscore_support(y_test,pred)
print(f'Overall accuracy: {accuracy_score(y_test,pred)}')
print(f'Hesitant recall: {recall[0]}')

