
import pandas
import numpy 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression

# read in data from csv - and make the index the year ID so that we can create time series


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



