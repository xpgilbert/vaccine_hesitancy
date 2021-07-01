# STATEWISE ANALYSIS

### Imports

## Data Processing, Basic Visualizations, and Linear Algebra
import pandas as pd
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

## Code for map of states with hesitancy

## Lets take the median of each state
by_state=df.groupby(['state']).median().reset_index()

## Create new categorical variables from aggregates
cbins = [0,5.7,7]                                                    ## See notes on sentiment cuts
abins = [0,4.50,7]

data=by_state.drop(['id'],axis=1)
data['cov_band'] = pd.cut(data['cov_med'], bins=cbins)
data['cov_band'] = pd.get_dummies(data['cov_band'], drop_first=True)  ## 1 ~ not hesitant
data['anx_band'] = pd.cut(data['anx_med'], bins=abins)
data['anx_band'] = pd.get_dummies(data['anx_band'], drop_first=True)  ## 1 ~ more anxious
by_state=data
print('by_state shape:',by_state.shape)

## US State names variable
state_names = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
          "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
          "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
          "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
          "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

## What states are we missing
for item in state_names:
    if item not in list(by_state['state']):
        print(item)

## We are missing Wyoming

## Lets examine some of the statewise data with visualizations
## Import Geopandas and Shapely for mapping data to map of the US
import geopandas as gpd
#from shapely.geometry import Point, Polygon
usa = gpd.read_file('geo/States_21basic/states.shp')
#usa.head()

## Join by_state and shapely file
usa_merged = gpd.GeoDataFrame(pd.merge(usa, by_state, right_on='state', left_on='state_abbr'))

## Median anxiety by state
fig, ax = plt.subplots(figsize=(12,8))
legend_kwds = {
    'label':'Median Anxiety by State',
    'orientation':'vertical',
    'shrink':0.5
}
usa_merged.plot(column='anx_med', ax=ax, legend=True, legend_kwds=legend_kwds, cmap='viridis', figsize=(12,8))
ax.set_axis_off()
fig.savefig('plots/anxiety_statewise.png')
plt.show()

## Plot hesitancy as binary variable

fig, ax = plt.subplots(figsize=(12,8))
usa_merged.plot(column='cov_band', ax=ax, categorical=True, cmap='Set2',figsize=(12,8), legend=True)
ax.set_axis_off()
new_labels=['Hesitant','Not Hesitant']
leg = ax.get_legend()
for text, label in zip(leg.get_texts(), new_labels):
    text.set_text(label)
fig.savefig('plots/hesitancy_statewise.png')
plt.show()
