# LOAD AND CLEAN
## Imports
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.chdir('..')

## read and load waves
wave1 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 1')
wave2 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 2')
wave3 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 3')
wave4 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 4')
wave5 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 5')
wave6 = pd.read_excel('data/All Results.xlsx', sheet_name='wave 6')


## Remove duplicates

def remove_dup(wave, wave1=wave1):
    ## Remove duplicates from wave1 from all waves
    repeats = wave1[wave1['id'].duplicated()]['id']

    ## Remove duplicates within the wave
    wave_r = wave[wave['id'].duplicated()]['id']
    repeats = repeats.append(wave_r)
    wave = wave[~wave['id'].isin(repeats)]
    return wave


wave1 = remove_dup(wave1)
wave2 = remove_dup(wave2)
wave3 = remove_dup(wave3)
wave4 = remove_dup(wave4)
wave5 = remove_dup(wave5)
wave6 = remove_dup(wave6)

print('dups removed')
## Create function to extract US state from zipcode data

from uszipcode import SearchEngine


def get_state(zip):
    search = SearchEngine()
    zipcode = search.by_zipcode(zip)
    return zipcode.state


## Apply get_state to wave1 then impute that data to the rest of the waves by shared 'id'

wave1 = wave1.loc[wave1['Progress'] == 100]
wave1['state'] = wave1.apply(lambda x: get_state(x['zip']), axis=1)
states = wave1[['id', 'state']]

waves = [wave2, wave3, wave4, wave5, wave6]
for wave in waves:
    wave['state'] = np.nan
    wave.update(states)

print('states got')
## Load the data and create constructs

def load(data, wave):
    ## Select only complete observations
    data = data.loc[data['Progress'] == 100]
    data.drop('Progress', axis=1, inplace=True)

    ## Reverse scale of certain survey questions
    data['v2'] = data['v2'].apply(lambda x: 8 - x)
    data['v3'] = data['v3'].apply(lambda x: 8 - x)
    data['v6'] = data['v6'].apply(lambda x: 8 - x)
    data['v7'] = data['v7'].apply(lambda x: 8 - x)
    data['v9'] = data['v9'].apply(lambda x: 8 - x)

    ## Create wave column
    data['wave'] = wave

    ## Create timeseries data for modeling
    data['RecordedDate'] = pd.to_datetime(data['RecordedDate'])
    data['month'] = data['RecordedDate'].apply(lambda x: x.month)
    data['hour'] = data['RecordedDate'].apply(lambda x: x.hour)
    data['day'] = data['RecordedDate'].apply(lambda x: x.day)
    data.drop('RecordedDate', axis=1, inplace=True)

    ## Construct aggregate variables
    vax_cols = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10']
    anx_cols = ['a1', 'a2', 'a3', 'a4', 'a5']
    life_cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7']
    cov_cols = ['c4', 'c5']
    perc_cols = ['c1', 'c2', 'c3']

    #### Medians
    data['vax_med'] = data[vax_cols].median(axis=1)
    data['anx_med'] = data[anx_cols].median(axis=1)
    data['life_med'] = data[life_cols].median(axis=1)
    data['cov_med'] = data[cov_cols].median(axis=1)
    data['perc_med'] = data[perc_cols].median(axis=1)

    #### Means
    data['vax_avg'] = data[vax_cols].mean(axis=1)
    data['anx_avg'] = data[anx_cols].mean(axis=1)
    data['life_avg'] = data[life_cols].mean(axis=1)
    data['cov_avg'] = data[cov_cols].mean(axis=1)
    data['perc_avg'] = data[perc_cols].mean(axis=1)

    ## Bin variables to create categorical variable
    cbins = [0, 5.75, 7]  ## See notes on sentiment cuts
    abins = [0, 4.50, 7]
    data['cov_band'] = pd.cut(data['cov_med'], bins=cbins)
    data['cov_band'] = pd.get_dummies(data['cov_band'], drop_first=True)  ## 1 ~ not hesitant
    data['anx_band'] = pd.cut(data['anx_med'], bins=abins)
    data['anx_band'] = pd.get_dummies(data['anx_band'], drop_first=True)  ## 1 ~ more anxious

    ## Drop unused columns
    to_drop = ['Duration (in seconds)', 'Latitude', 'Longitude', 'ResponseId']
    data.drop(to_drop, inplace=True, axis=1)

    return data


## Process using functions above
def load_waves(waves):
    wave = 1
    data = pd.DataFrame()

    ## Create complete dataframe from waves
    while wave <= len(waves):
        for frame in waves:
            pframe = load(data=frame, wave=wave)
            data = data.append(pframe)
            wave += 1
    return data


## Lets get some data
waves = [wave1, wave2, wave3, wave4, wave5, wave6]
df = load_waves(waves)
df.head()

## Missing values
print('nulls:\p',df.isnull().sum())

## Waves 1 and 2 didn't record anxiety or depressed. Lets try to use visualizations to help impute
## those missing values, if at all possible.
## Whats the distribution of our life-construct with binary anxiety and depressed variables
g = sns.FacetGrid(df, col='anxiety', row='depressed', size=4)
g.map(sns.countplot, 'life_med')
g.fig.suptitle('Life Construct by Anxiety and Depressed Variables')
g.axes[0, 0].set_xlabel('Not Depressed or Anxious')
g.axes[0, 1].set_xlabel('Not Depressed but Anxious')
g.axes[1, 0].set_xlabel('Depressed but not Anxious')
g.axes[1, 1].set_xlabel('Depressed AND Anxious')
g.fig.tight_layout()
g.savefig('plots/median_life_by_binary_countplots.png')

## Does anxiety and anx_band distibute the same way?
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
sns.countplot('anx_med', hue='anx_band', data=df, ax=ax[0, 0])
ax[0, 0].set_title('Reference: Anxiety Class by Construct')
ax[0, 0].set_xlabel('Median Anxiety')
ax[0, 0].legend(('Less Anxious', 'More Anxious'), loc='upper right')
sns.countplot('anx_med', hue='anxiety', data=df, ax=ax[0, 1])
ax[0, 1].set_title('Anxiety Construct by Anxiety Binary')
ax[0, 1].set_xlabel('Median Anxiety')
ax[0, 1].legend(('Less Anxious', 'More Anxious'), loc='upper right')
sns.countplot('anx_med', hue='depressed', data=df, ax=ax[1, 0])
ax[1, 0].set_title('Anxiety Construct by Depressed Binary')
ax[1, 0].set_xlabel('Median Anxiety')
ax[1, 0].legend(('Less Depressed', 'More Depressed'), loc='upper right')
sns.countplot('anx_med', hue='anx_band', data=df, ax=ax[1, 1])
ax[1, 1].set_title('Reference: Anxiety Class by Construct')
ax[1, 1].set_xlabel('Median Anxiety')
ax[1, 1].legend(('Less Anxious', 'More Anxious'), loc='upper right')
plt.tight_layout()
fig.savefig('median_anxiety_by_binary_countplot.png')

## It doesn't look like we can use our constructs or even the anxiety medians to impute the binary
## variables, therefore we should drop them from any analysis that covers waves 1 and 2.

## Impute mean to missing construct values

for col in ['life_avg', 'anx_avg', 'cov_avg', 'vax_avg']:
    df[col] = df[col].fillna(df[col].mean())

for col in ['life_med', 'anx_med', 'cov_med', 'vax_med', 'perc_med', 'm5']:
    df[col] = df[col].fillna(df[col].mean())

##  Save to csv file
# df.to_csv('data/full_ucsd_data.csv',index=False)}
print('final shape:',df.shape)