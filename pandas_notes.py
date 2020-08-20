import os
import pandas as pd
os.chdir('/home/nick/numpy')

# loading data in pandas
df = pd.read_csv('pokemon.csv')
df.head(10)
df.tail(10)
df.shape

# get column headers
print(df.columns)

# get specific column
df[['Name', 'Type 1', 'Type 2']]

# get specific rows or columns
df.iloc[0:4, 1]

# iterate through rows
for index, row in df.iterrows():
    print(index, row[['Name', '#']])

# filter the rows
df.loc[(df['Type 1'] == "Fire") & (df['Legendary'] == True)]["Name"]
df.loc[(df['Type 1'] == 'Flying') | (df['Type 2'] == 'Flying'), ['Name', 'Type 1', 'Type 2']]

# get summary
df.describe()

# sorting
df.sort_values('Name')[['Name', 'Attack']]
df.sort_values(['Attack', 'Name'], ascending = [0, 1])[['Name', 'Attack']]

# adding new column
df.columns.values
int(np.where((df.columns.values == "HP"))[0])
df['Total'] = df.iloc[:, 4:10].sum(axis=1)
df.sort_values('Total', ascending = False)[['Name', 'Total']]

# dropping a column
df = df.drop(columns = ['Total'])

# resetting index (might want to do after filtering)
new_df = df.loc[(df['Type 1'] == "Fire") & (df['Legendary'] == True)]["Name"]
new_df.reset_index(drop=True, inplace=True)
new_df

# advanced text filtering
df.loc[~(df['Name'].str.contains('Mega'))]

import re
df.loc[df['Type 1'].str.contains('fire|grass', flags = re.I, regex = True)]

# conditional changes
df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Fwooooosh'
df['Type 1']
df.loc[df['Type 1'] == 'Fwooooosh', 'Type 1'] = 'Fire'
df['Type 1']

df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = 'Test value'
df.loc[df['Total'] > 500]

# aggregate statistics
df.groupby(['Type 1']).mean().sort_values('Defense', ascending = False)['Defense']
df.groupby(['Type 1']).mean().sort_values('Attack', ascending = False)['Attack']
df.groupby(['Type 1']).count()['Name']

df.groupby(['Type 1']).agg({'Name': 'count', 'Attack': 'mean'})
df.groupby(['Type 1']).agg({'Name': 'count', 'Attack': ['mean', 'max', 'min']})
df.groupby(['Type 1', 'Type 2']).agg({'Name': 'count', 'Attack': 'mean'})

(df.groupby(['Type 1', 'Type 2']).
        agg({'Name': 'count', 'Attack': 'mean'}))

# let's emulate some dplyr-isms using pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set_style('white')

df = sns.load_dataset('diamonds')
df.head()

(df
    .filter(['carat', 'color'])
    .query('color == "E"')
    .head(10))

(df
    .filter(regex='^c')
    .query('cut in ["Ideal", "Premium"]')
    .groupby(['cut', 'color', 'clarity'])
    .agg(['mean', 'size'])
    .sort_values(by=('carat', 'mean'), ascending=False)
    .head())

(df
    .filter(regex='^c')
    .query('cut in ["Ideal", "Premium"]')
    .groupby(['cut', 'color', 'clarity'])
    .agg({'carat' : ['mean', 'count']})
    .sort_values(by=[('carat', 'mean'), ('carat', 'count')], ascending=False)
    .head())

df2 = (df
        .assign(pricecat = pd.cut(df['price'], bins = 3, labels = ['low', 'med', 'high']))
        .filter(['x', 'z', 'pricecat'])
        .rename(columns = {'x' : 'width', 'z' : 'depth'})
        .melt(id_vars = ['pricecat'], value_vars = ['width', 'depth'],
            var_name = 'dim', value_name = 'mm')
        .query('2 < mm < 10'))

g = sns.FacetGrid(data=df2, col='pricecat', hue='dim')
g.map(sns.kdeplot, 'mm', shade=True, alpha=0.5).add_legend() 
plt.show()

