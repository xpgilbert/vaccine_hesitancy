# STATISTICAL ANALYSIS
### Imports

## Data Processing, Basic Visualizations, and Linear Algebra
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
import os
os.chdir('..')

## Load the data
df = pd.read_csv('data/full_ucsd_data.csv')

## Segment out the interested data, including constructs and comments
interested = ['id','state', 'cov_band', 'comments', 'month', 'hour', 'day', 'wave'
            , 'anxiety', 'depressed','life_avg', 'anx_avg', 'cov_avg', 'vax_avg'
            ,'life_med', 'anx_med', 'cov_med', 'vax_med', 'anx_band','perc_med','m5','flu1','flu2']
df = df[interested]

## Split data into two samples
## Use 6 as cut off since a response of 5 indicates hesitancy
## Group by id to satisfy independent assumption
hesi = df.loc[df['cov_med'] < 6].groupby('id').median().reset_index()
nhes = df.loc[df['cov_med'] >=6].groupby('id').median().reset_index()

print('Not hesitant count:', hesi.shape[0])
print('Hesitant count:', nhes.shape[0])

## Create our 1-d samples for testing, median
x = hesi['anx_med']
y = nhes['anx_med']

from scipy.stats import mannwhitneyu  ## Mann-Whitney U for non-parametric testing 

## Alpha = 0.05

u, p = mannwhitneyu(x, y, alternative='two-sided')
print('Mann-Whitney Results:')
print('U:',u,'\n'+'p:', round(p,5))

## Calculate z score from U statistic
## signficant value = +-1.96

nom = (u-(len(x)*len(y))/2)
den = np.sqrt(len(x)*len(y)*(len(x)+len(y)+1)/12)
z = nom/den
print(f'z: {z}')

## Friedman test
## First need ids that are shared across all waves

def find_common_ids(data):

    wave1ids = data.loc[data['wave']==1]['id']

    wave2ids = data.loc[data['wave']==2]['id']
    wave3ids = data.loc[data['wave']==3]['id']
    wave4ids = data.loc[data['wave']==4]['id']
    wave5ids = data.loc[data['wave']==5]['id']
    wave6ids = data.loc[data['wave']==6]['id']

    ## Find intersection across all waves
    ids = set(wave1ids).intersection(wave2ids, wave3ids, wave4ids, wave5ids, wave6ids)
    
    return ids
ids = find_common_ids(df)

## Pull median anxieties from each wave for the hesitant sample

anx1 = df[df['wave']==1][df['id'].isin(ids)].sort_values(by='id')['anx_med']
anx2 = df[df['wave']==2][df['id'].isin(ids)].sort_values(by='id')['anx_med']
anx3 = df[df['wave']==3][df['id'].isin(ids)].sort_values(by='id')['anx_med']
anx4 = df[df['wave']==4][df['id'].isin(ids)].sort_values(by='id')['anx_med']
anx5 = df[df['wave']==5][df['id'].isin(ids)].sort_values(by='id')['anx_med']
anx6 = df[df['wave']==6][df['id'].isin(ids)].sort_values(by='id')['anx_med']

from scipy.stats import friedmanchisquare  ## Test with repeated measures

## df = 6-1 = 5
## alpha = 0.05
## Signficant H (chi-squared) = 11.07

stat, pval = friedmanchisquare(anx1, anx2, anx3, anx4, anx5, anx6)
print('Friedman Results')
print(f'Friedman Q: {stat}\np:{round(pval,5)}')

## Test for one-sided

## Group by id again
x = df.loc[df['cov_med'] < 6].groupby('id').median().reset_index()['anx_med']
y = df.loc[df['cov_med'] >=6].groupby('id').median().reset_index()['anx_med']

u, p = mannwhitneyu(x, y, alternative='less')
print('U:',u,'\n'+'p:', round(p,5))

## Lets see if any of the waves are different at all
import scikit_posthocs as sp
sp.posthoc_mannwhitney(df, val_col='anx_med', group_col='wave', p_adjust='bonferroni')