
import pandas
import numpy 
import statsmodels.api as sm
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

# read in data from csv - and make the index the year ID so that we can create time series

df = pandas.io.parsers.read_csv('baseball.csv',parse_dates=True)


### CHECKING VARIABLE RELATIONSHIPS ###
# SHRINKING THE DATA TO MAKE VISUALIZATIONS EASIER #
#df['random'] = numpy.random.randn(len(df))
#df = df[df.random > 1]
#del df['random']


df = df.sort(['playerID','yearID'])
df.set_index('yearID',inplace=True)
 
# Slugging Average [SA or SLG] (with no denominator!)
# SA = Number of (Singles + [2 x Doubles] +[ 3 x Triples] + [4 x Home Runs]) divided by At Bats
df['SA'] = df.R +  2*df.X2B + 3*df.X3B+ 4*df.HR

simple_data = df[['salary','SA','playerID','weight','height']]

new_df = pandas.DataFrame()
for player in set(simple_data['playerID']):
    temp_data = simple_data[simple_data['playerID']==player]
    temp_data['cum_SA'] = temp_data['SA'].cumsum()
    new_df = new_df.append(temp_data)

# REPLACE INF VALUES WITH NAN #
new_df = new_df.replace([numpy.inf, -numpy.inf], numpy.nan)

# remove nans
new_df = new_df.dropna()

# get yearID back from index
new_df['yearID'] = new_df.index


### CHECKING VARIABLE RELATIONSHIPS ###
# SHRINKING THE DATA TO MAKE VISUALIZATIONS EASIER #
#new_df['random'] = numpy.random.rand(len(new_df))
#train_df = new_df[new_df.random > 0.4]

train_df = new_df


train_df['intercept'] = 1
X = train_df[['yearID','height','cum_SA','intercept']] 

train_df['log_salary'] = numpy.log(train_df['salary'])
y = train_df['log_salary']
model = sm.OLS(y, X)
results = model.fit()
print results.summary()


'''
# remove zero salaries
all_data = all_data[all_data.salary>0]

# scatter plot of data, too many variables to run hereb but could do this on a subset later
# pandas.tools.plotting.scatter_matrix(stats_and_salary)

# Start with a simple model using lagged runs to predict salary
simple_data = all_data[["HR","salary"]] 

# remove 0 homeruns
simple_data = simple_data[simple_data.HR != 0 ]


# log salary
simple_data['log_salary'] = all_data.salary.apply(numpy.log)

# log homeruns 
simple_data['log_HR'] = (all_data.HR+1.).apply(numpy.log)

# remove nan data
simple_data = simple_data.dropna()

# create lagged run variables from 1 to 5 years
#simple_data['R_L1'] = simple_data.HR.shift(1)
#simple_data['R_L2'] = simple_data.HR.shift(2)
#simple_data['R_L3'] = simple_data.HR.shift(3)
#simple_data['R_L4'] = simple_data.HR.shift(4)
#simple_data['R_L5'] = simple_data.HR.shift(5)


# maybe cumulative HR
#simple_data['cum_hr'] = simple_data.HR.cumsum()
 
pandas.tools.plotting.scatter_matrix(simple_data)


# instantiate class
linear_fit = LinearRegression()
# train/fit model to predict brain weights given the body weights
linear_fit.fit(simple_data.HR , simple_data.salary)
# print beta parameters (should be all positive)
print "sklearn intercept and coef (linear):", linear_fit.intercept_, linear_fit.coef_

#loop through the relevant columns and see which are impactful
for col in cols:
     ind = y_detrnd.dropna().index & all_data[col].dropna().index
     print "Regressing "+col
     reg = pandas.ols(x = all_data.loc[ind, col], y = y_detrnd[ind])
     print reg.beta
     print reg.r2_adj
'''


