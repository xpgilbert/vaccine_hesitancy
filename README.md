

# Vaccine Hesitancy and General Anxiety
## Study of American perceptions during the Peak of the Pandemic

# Table of contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [1. The problem](#1-the-problem)
- [2. Purpose of the study](#2-purpose-of-the-study)
- [3. Project description](#3-project-description)
    - [3.1 Hypotheses](#31-hypotheses)
    - [3.2 Workflow](#32-workflow)
    - [3.3 Methods](#33-methods)
    - [3.4 Hypothesis 1: Analysis](#34-analysis-for-hypothesis-1)
    - [3.5 Hypothesis 2: Analysis](#35-analysis-for-hypothesis-2)
- [Conclusion](#conclusion)
- [References](#references)

##  Abstract

In this analysis, we explore relationships between vaccine hesitancy and various mental health metrics, with the bulk of the analysis around general anxieties experienced during the peak of the pandemic.  We will be using data collected by the University of California, San Diego by a survey of around 700 repeat respondents that polled various sentiments around the pandemic, politics, and mental health.  The original study showed that political affiliation is the strongest predictor of hesitancy, so we investigated whether anxiety can also be a predictor and may contribute to a more robust model.  The statistical analysis showed anxiety can be a predictor of whether someone is hesitant to get the vaccine.  Later, we will show that using machine learning with both comments left by survey respondents and our previous analysis can predict hesitancy.

[Back to top](#table-of-contents)

## Introduction

When I first started developing a project idea, the vaccine rollout in the United States was at an inflection point.  Some states and communities were participating in the program as completely as they could, while others were lacking or even shunning vaccine distribution entirely.  I was interested in whether these two opposite attitudes towards receiving a Covid-19 vaccine was related to a person’s mental health throughout the peak of the pandemic.  As I searched for data to explore, I found an extensive survey conducted during the first 6 months of the pandemic that collected a wide range of data specific to vaccine hesitancy, motivations, general anxieties and, of course, whether a participant would be willing to get the vaccine when they became eligible.  In addition, around 30% of respondents left relevant comments that could be explored with machine learning and natural language processing techniques to see if language can be a predictor of hesitancy.  After some experimentation, a Random Forest model that used both a participant’s survey data and comment data was able to predict that person’s hesitancy class with a recall score of 70%.

If we had been able to train the model strictly on text data, this could be deployed on other comment forums or even Twitter to identify hesitant individuals for marketers to target.  We may be able to train a more complicated ensemble model to predict metrics similar to the survey data, which can, as we will show in my analysis, in turn predict hesitancy.

The original study provided evidence that polarization in vaccine hesitancy is attributed to differences in media consumption, social networks, and political affiliation.  For more details on how the survey was developed, the methodology used during their analysis, and the original authors' conclusions, please read their article which can be found [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250123#pone.0250123.s004).  I also used some of the same methods for loading the data, such as removing duplicate responses, and the code, original data set, and the survey itself can be found [here](https://osf.io/kgvdy/).

[Back to top](#table-of-contents)

## 1. The problem

The Covid-19 vaccines are our greatest tool to resolving the pandemic yet hesitancy is on the rise for many communities ([1](#references)). We have found that traditional motivations, such as reaching herd immunity, are not effective at promoting the covid-related health behaviors ([2](#references)).  

[Back to top](#table-of-contents)

## 2. Purpose of the study

By investigating other motivations, specifically how vaccine hesitancy relates to anxiety behaviors, we may be able to find new avenues at promoting good health measures and increase the national vaccination rate.  Using extensive survey data and machine learning can also reveal patterns in samples of vaccine hesitant individuals for marketing targets.  We hope to learn more about the anxieties that vaccine hesitant people have and from the comments they leave when taking the survey.

[Back to top](#table-of-contents)

## 3. Project description

This study investigated two questions using the same data:
1. Did samples of hesitant and non-hesitant Americans experience different levels of anxiety during the peak of the Covid-19 pandemic?  If so, was the hesitant sample more anxious?  If this is the case, marketers may be able to leverage this data to improve the vaccine rollout and achieve the goal of herd immunity.

2. Can a respondent’s text data be a predictor of hesitancy?  If a good model can be built and trained strictly on text data, can it be deployed on other sources of comments to help marketers target hesitant people.

[Back to top](#table-of-contents)

### 3.1 Hypotheses


- Hypothesis 1: Respondents who are vaccine hesitant experienced different anxiety levels than non-hesitant respondents.


- Hypothesis 2: Comments left by survey respondents can predict whether a person is vaccine hesitant.


[Back to top](#table-of-contents)

### 3.2 Workflow

1. Exploratory Analysis   
     - Distribution of mental health metrics by hesitancy class  
     - Statewide analysis
2. Statistical Analysis
     - Non-parametric tests for 2 samples
     - Post hoc analysis with repeated measures for 6 samples (one per wave)
3. Modeling Text Data for Hesitancy Predictions
     - Exploratory Analysis and Visualizations
     - Data Preperation
     - KMeans clustering
     - Sentiment Analysis
     - Modeling using only text data
     - Modeling with both text and survey data

[Back to top](#table-of-contents)

### 3.3 Methods

- The survey used a 7-point Likert scale for the majority of questions and constructs were generated by taking the median of specific questions.  For ordinal data, it is common to use the median as an aggregator since taking the mean assumes the interval between ranks are, for the most part, the same.  Some argue that a mean does not exist for ordinal data ([3](#references)).  In addition, we will be using non-parametric testing for the bulk of our analysis ([4](#references)). 
- The dataset includes zipcode data for each observation.  We used [uszipcode](https://pypi.org/project/uszipcode/) SearchEngine to extract states and [Geopandas](https://geopandas.org/) and [Shapely](https://pypi.org/project/Shapely/) for visualizations
- To test our first hypothesis (Anxiety by Hesitancy class), we will apply the **Mann-Whitney U** test for 2 samples of ordinal, ranked data.  This test scores one sample by comparing each observation’s rank to all those in the other sample and assigning a score. 
    For two i.i.d samples ![](https://latex.codecogs.com/gif.latex?X_{1},...,X_{n}) and ![](https://latex.codecogs.com/gif.latex?Y_{1},...,Y_{m}):


<p align="center">
<img src="https://latex.codecogs.com/gif.latex?S(X,Y)=\left\{\begin{matrix}&space;1&space;&&space;if&space;&&space;Y<X&space;\\&space;\frac{1}{2}&space;&&space;if&space;&&space;Y=X&space;\\&space;0&space;&&space;if&space;&&space;Y>X&space;\end{matrix}\right." />
</p>

- The sum of these scores is used to find the ![](https://latex.codecogs.com/gif.latex?U) statistic which can be converted to a traditional ![](https://latex.codecogs.com/gif.latex?z)-score to determine whether the samples are drawn from the same distribution.


<p align="center">
<img src="https://latex.codecogs.com/gif.latex?U&space;=&space;\sum_{i=1}^{n}\sum_{j=1}^{m}&space;S(X_{i},Y_{j})" />
</p>


<p align="center">
<img src="https://latex.codecogs.com/gif.latex?z&space;=&space;\frac{U&space;-&space;m_{U}}{\sigma_{U}}" />
</p>

- Next, we will apply the Friedman test for repeated measures to see if the hesitant class’s anxiety had changed over the course of the peak of the pandemic [4](#references).  First we calculate the rank of each observation within its own block, then take the mean of all scores.  For a matrix with ![](https://latex.codecogs.com/gif.latex?n) rows and ![](https://latex.codecogs.com/gif.latex?k) columns,

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?\overline{r}_{ij}&space;=&space;\frac{1}{n}\sum_{i=1}^{k}&space;r_{ij}" />
</p>

where ![](https://latex.codecogs.com/gif.latex?r_{ij}) is the rank of the observation ![](https://latex.codecogs.com/gif.latex?x_{ij}) within block ![](https://latex.codecogs.com/gif.latex?i) and column ![](https://latex.codecogs.com/gif.latex?j).


- The test statistic, which is drawn from a ![](https://latex.codecogs.com/gif.latex?\chi^{2}) distribution, is calculated as

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Q=\frac{12n}{k(k&plus;1)}\sum_{j=1}^{k}&space;\left(\overline{r}_{ij}&space;-&space;\frac{k&plus;1}{2}\right)^{2}" />
</p>

- Finally, for our analysis of survey data, we built a linear model using hesitancy and survey wave as categorical variables to predict median anxiety.
- For our text data analysis and modeling, we will use the NLP ToolKit from Scikit Learn and the Tokenizer from TensorFlow to process the comments.  
- When exploring the data, we will use KMeans clustering to see how words from the comments cluster and will use textblob's Sentiment classifier to see how the comments split up by positive and negative classes.
- When modeling strictly the text data, we will use Naïve Bayes model for classification since it’s the go-to model for these classification problems.  When modeling in conjunction with our previous survey analysis, we will use the Random Forest Classifier from Scikit Learn.  Both a TfIdf matrix and a strict term-frequency matrix will be used for each classifier as a pseudo-parameter.
- Please refer to load_and_clean python script for the code that loaded the original data set and saved it as a more convenient .csv file.



[Back to top](#table-of-contents)


### 3.4 Analysis for Hypothesis 1


In this section, we will cover the exploratory and statistical analyses relating to the first hypothesis.  Did hesitant and non-hesitant samples of survey respondents report different levels of anxiety during the months surveyed?


We will review:
- [Exploratory Analysis](#3.4.1-Exploratory-Analysis)
- [Statewise Analysis](#3.4.2-Statewise-Analysis)
- [Data Preperation](#3.4.3-Data-Preparation)
- [Statistical Testing](#3.4.4-Testing)


[Back to top](#table-of-contents)


#### 3.4.1 Exploratory Analysis
Lets first explore the relationship between our hesitancy classes and the median anxieties using visualizations.

<p align="center">
<img src="https://github.com/xpgilbert/vaccine_hesitancy/blob/master/plots/Anxiety_by_hesitancy_class.png" />
</p>

Visually, it is hard to differentiate any class imbalance.  We would have to use more specific statistical tools to determine if there are differences between the classes.

We can also investigate how the classes are distributed geographically across the United States:

<p align="center">
<img src="https://github.com/xpgilbert/vaccine_hesitancy/blob/master/plots/hesitancy_statewise.png" />
</p>

To use the Mann-Whitney U test, we follow the steps below:
- Establish Hypotheses
     - ![](https://latex.codecogs.com/gif.latex?H_{0}): Our samples are drawn from the same distribution
     - ![](https://latex.codecogs.com/gif.latex?H_{a}): Our samples are NOT drawn from the same distributions
- State Alpha and Critical Value
     - ![](https://latex.codecogs.com/gif.latex?\alpha&space;=&space;0.05)
     - ![](https://latex.codecogs.com/gif.latex?z&space;=&space;\pm&space;1.96)
- Score samples and calculate statistic
- State Results


Our two-sided test produced the following U statistic and associated pvalue.

![](https://latex.codecogs.com/gif.latex?U&space;=&space;320702.5 )
![](https://latex.codecogs.com/gif.latex?p&space;=&space;0.003)


From the U statistic, our z statistic is calculated as:


![](https://latex.codecogs.com/gif.latex?z&space;=&space;-2.945)


At ![](https://latex.codecogs.com/gif.latex?\alpha&space;=&space;0.05) and a critical value of ![](https://latex.codecogs.com/gif.latex?\pm&space;1.96), we ***reject*** ![](https://latex.codecogs.com/gif.latex?H_{0}).  The hesitant sample is experiencing a different general anxiety in their lives than the non-hesitant sample.

![](https://latex.codecogs.com/gif.latex?Q&space;=&space;82.220)
![](https://latex.codecogs.com/gif.latex?p&space;=&space;0.0)


We ***reject*** the null that the anxieties from each wave are pulled from the same distribution.


#### 3.4.5 Results for Hypothesis 1


When comparing our two hesitancy classes, there is a clear difference in the anxieties experienced when aggregated over the course of the whole study.  We were able to reject the null hypothesis in favor of our alternative that the samples are drawn from different distributions.  The Mann-Whitney U test in python allows for one-way testing as well, which we can try below and see if the hesitant sample is experiencing less anxiety than our non-hesitant sample.  We also found that the anxieties of all our participants changed over the course of the pandemic.

### 3.5 Analysis for Hypothesis 2

Our goal here are to see if the comments left by each observation can be helpful in predicting that person's vaccine hesitancy.  If this works, then the model can be deployed to help marketers target vaccine hesitant people to get the jab.  As we tune the model, we are looking to improve our recall of the hesitant class since we care more about missing a hesitant person than missing a not hesitant person.  Reminder that our 1 class is not-hesitant.  After playing with only the comment data with NaiveBayes modeling, we will incorporate our mental health reporting data with our comment data and feed that into a RandomForest Classifier to predict the same hesitancy classes.

We will review:
- [Data Preperation](#351-data-preparation)
- [Exploratory Analysis](#352-exploratory-analysis)
- [KMeans Clustering](#353-kmeans-clusterting)
- [Sentiment Analysis](#354-sentiment-analysis)
- [Model Building](#355-model-building)
- [Results](#356-results)


[Back to top](#table-of-contents)

#### 3.5.1 Exploratory Analysis

Lets first check the comment lengths by hesitancy class:

<p align="center">
<img src="https://github.com/xpgilbert/vaccine_hesitancy/blob/master/plots/length_hesitancy.png" />
</p>

We can see that there is not much of a difference in lengths here, which is interesting on its own.  You would think those who would be more anxious about the pandemic would be more likely to leave a longer comment.  We can also inspect lengths by median anxieties as well:

<p align="center">
<img src="https://github.com/xpgilbert/vaccine_hesitancy/blob/master/plots/length_anxiety.png" />
</p>


We also investigated how the words from the comments clustered using KMeans clustering.  Our analysis produced these 5 clusters:


- Most common words in cluster 0:
 [('people', 245), ('virus', 148), ('think', 114), ('need', 88), ('get', 76), ('home', 70), ('government', 67), ('state', 66), ('coronavirus', 65), ('vaccine', 64)] 


- Most common words in cluster 1:
 [('get', 61), ('back', 57), ('normal', 36), ('hope', 31), ('soon', 28), ('thing', 24), ('people', 22), ('life', 20), ('go', 18), ('think', 14)] 


- Most common words in cluster 2:
 [('good', 20), ('survey', 3), ('easy', 1)] 


- Most common words in cluster 3:
 [('thank', 33), ('stay', 4), ('safe', 4), ('hope', 3), ('nothing', 3), ('healthy', 3), ('add', 2), ('please', 2), ('everyone', 2), ('possible', 1)] 


- Most common words in cluster 4:
 [('thanks', 22), ('want', 1), ('toilet', 1), ('paper', 1), ('asking', 1), ('feedback', 1), ('additional', 1), ('informational', 1), ('survey', 1), ('opportunity', 1)] 


Looks like our clusters could follow these patterns. Fun investigating another time:
- 0: Life with the virus, how its being addressed and by who
- 1: Looking towards the future, hopeful
- 2: About the survey, probably just the people leaving "good"
- 3: Wishing the surveyors and other people best wishes
- 4: looks like the word "thanks" and then one specific comment

#### 3.5.4 Sentiment Analysis

We used textblob's sentiment analysis tool as part of our exploratory analysis.  It did not produce any interesting results but the code is included in the scripts folder for others to review.  The table below is a preview of how the machine classified comments.


| sentiment     |clean_comment   |
| ------------- |:-------------:| 
| positive     | mostly concerned financial impact going | 
| negative     | medium making bit mess |
| positive     | concerned older generation due side effect    |
| negative | think person virus infect body please safe lea…     |
| positive     | strange scary shortage panic |


It does not look like the sentiment analysis algorithm did a lot here.  Could be worth tweaking later but that is not within the scope of this study.


#### 3.5.5 Model Building
This section covers the classification models we built to see if the comments left by a participant can predict that observation's hesitancy class.  Our workflow is:
- Use NaïveBayes model using only term-frequency vectors
- Another NaïveBayes model with TfIdf matrix
- Use pipeline for GridSearchCV to test hyperparameters
- Evaluate models with only text data
- Build Random Forest Classifier that uses text data and survey data
- Evaluate models

We created 3 models with strict tokens, 
|                 | token matrix | tfidf matrix | pipeline |
| ----------------|:------------:|:------------:|:--------:| 
| overall accuracy| 0.548        | 0.475        | 0.551    |
| hesitancy recall| 0.511        | 0.086       | 0.532    |


Ok so using the text data doesn't really help predict hesitancy on its own.  Lets build an RandomForest Classifier model that takes our text data and mental health reporting data to predict hesitancy.

For our mental health reporting data, we will use the following measurements:
- Constructs:
    - life_med: median life satisfaction
     - anx_med: median general anxiety
     - vax_med: median attitude towards vaccines in general
     - perc_med: median perception of the Coronavirus
- Direct measurement 
     - m5: 'I want to reduce my anxiety about this virus'

|                 | no text data | token matrix | tfidf matrix|
| ----------------|:------------:|:------------:|:-----------:| 
| overall accuracy| 0.680        | 0.691        | 0.699       |
| hesitancy recall| 0.619        | 0.655        | 0.590       |

After some feature selection and more specific hyperparameter tuning, the RandomForest classifier with a token matrix and some survey data was evaluated to have the metrics:

Overall accuracy: 0.687
Hesitant recall: 0.688

#### 3.5.6 Results
It looks like the classifier with tokens and survey data performed fairly well!  For a text based classifier, 70% recall is pretty impressive.  With a bit more than just text information, our model is able to predict the hesitancy class for new data.  If we are able to gather this information in addition to the text data around the internet, we may be able to better target vaccine hesitant communities to get the jab.  Further exploration could include trying different models, including xgboost, to improve our results.  Nonetheless, with around 70% recall and 70% accuracy, we have built a pretty decent classifier for the problem of identifying hesitant individuals on the internet.

[Back to top](#table-of-contents)

## Conclusion

We showed that those who were more hesitant to get the Covid-19 vaccine experienced lower levels of anxiety than their less hesitant peers during the peak of the pandemic. This probably falls directly in line with the conclusions about political affiliation that the original authors of this study demonstrated [5](#references).  Their study showed that political affiliation was a very good predictor of vaccine hesitancy.  My investigation showed that a person's anxiety during this time can also be a predictor of their hesitancy, but not in the way I was expecting.  It would seem that those who were less anxious about the pandemic are in no rush to get the vaccine, whereas those who see the pandemic as having a greater effect on their mental health would be first in line to get the vaccine when available to them.

We also played around with the comments left by the survey participants.  When preparing the data and then building various models, we showed that using a combination of term frequency vectors and some of the survey data itself can be a good predictor of hesitancy.  Unfortunately, we were not able to train a good model strictly on text data to be deployed around the internet for marketers, but if we are able to predict the same metrics about a person (life satisfaction, Covid perception, etc) and then analyze comments they leave around the internet, we should be able to predict that individual's hesitancy.

[Back to top](#table-of-contents)


## References

- (1) [Kaplan RM, Milstein A, Inluence of a Covid-19 vaccin'es effectiveness and safety profile on vaccination acceptance, Proceedings of the National Academy of Sciences Mar 2021, 118 (10) e2021726118](https://www.pnas.org/content/118/10/e2021726118)
- (2) [Rabb N, Glick D, Houston A, Bowers J, Yokum D, No Evidence that collective-good appeals best promote COVID-related health behaviors, Proceedings of the National Academy of Sciences Apr 2021, 118 (14) e2100662118](https://www.pnas.org/content/118/14/e2100662118)
- (3) [Measures of Central Tendency, Laerd Statistics](https://statistics.laerd.com/statistical-guides/measures-central-tendency-mean-mode-median-faqs.php)
- (4) [Brownlee J, How to Calculate Nonparametric Statisical Hypothesis Tests in Python,*Machine Learning Mastery*, 2018](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
- (5) [Fridman A, Gershon R, Gneezy A (2021) COVID-19 and vaccine hesitancy: A longitudinal study. PLoS ONE 16(4): e0250123. https://doi.org/10.1371/journal.pone.0250123](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250123#pone.0250123.s004)

The original data set, data dictionary, and survey itself can be found at:
[Fridman, Ariel, Rachel Gershon, and Ayelet Gneezy. “COVID-19 and Vaccine Hesitancy.” OSF, 1 Apr. 2021. Web.](https://osf.io/kgvdy/)

Shapely file for USA visualizations can be found at [https://alicia.data.socrata.com/Government/States-21basic/jhnu-yfrj](https://alicia.data.socrata.com/Government/States-21basic/jhnu-yfrj)

[Back to top](#table-of-contents)
