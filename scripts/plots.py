# PLOTS
## Code for visualizations saved to ‘plots’ folder

### Imports

## Data Processing, Basic Visualizations, and Linear Algebra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

## Lets check on our constructs, hesitancy
g=sns.FacetGrid(df, col='cov_band',size=4)
g.map(sns.histplot,'anx_med')
g.fig.suptitle('Anxiety Construct by Hesitancy')
g.fig.tight_layout()
g.axes[0,0].set_xlabel('Hesitant Anxiety')
g.axes[0,1].set_xlabel('Not Hesitant Anxiety')
g.savefig('plots/Anxiety_by_hesitancy_class.png')

## Correlation heatmap of hesitancy with time variable
for_corr = df[['anx_med', 'cov_med', 'anxiety', 'depressed','month']]
corr = for_corr.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,cmap='viridis',cbar=True, annot=True,mask=np.triu(corr))
plt.title('Correlation between interests and time')
plt.show()
plt.savefig('plots/correlation_heatmap_time_data.png')

## Pair plot of correlation frame
g = sns.pairplot(for_corr, palette='Set1')
g.fig.suptitle('Construct Correlations', fontsize=14)
g.fig.tight_layout()
plt.savefig('plots/pairplot_constructs.png')

## Correlation heatmap of hesitancy with other vaccine and mental health variables
for_corr = df[['cov_band','m5','flu1','flu2']]
corr = for_corr.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,cmap='viridis',cbar=True, annot=True,mask=np.triu(corr))
plt.title('Correlation between hesitancy, motivations, and flu vaccine history')
plt.show()
plt.savefig('plots/correlation_between_motivations_and_hesitancy.png')

## We are also interested in how respondants feel about reducing their anxiety about the virus
g=sns.FacetGrid(df,col='cov_band',size=4)
g.map(sns.countplot,'m5')
g.fig.suptitle('Desire to reduce anxiety about the virus by hesitancy class')
g.fig.tight_layout()
g.axes[0,0].set_xlabel('Hesitant Desire')
g.axes[0,1].set_xlabel('Not Hesitant Desire')
g.savefig('plots/m5_by_hesitancy_class.png')

## We are also interested in how respondants feel about vaccines in general
g=sns.FacetGrid(df,col='cov_band',size=4)
g.map(sns.countplot,'perc_med')
g.fig.suptitle('Vaccine Perceptions by Hesitancy Class')
g.fig.tight_layout()
g.axes[0,0].set_xlabel('Hesitant Perception')
g.axes[0,1].set_xlabel('Not Hesitant Perception')
g.savefig('plots/perc_by_hesitancy.png')