# %% imports

import pandas as pd
import numpy as np
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
     import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
# %% show top level objects
dir()

# %% load the data
Boston = load_data("Boston")
print(Boston.shape)
# %% fitting a simple linear regression model
X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
print(X.info())

# %% fitting the model and printing the results
y = Boston['medv'] # that's the response column
model = sm.OLS(y,X) # ordinary least squares model
results = model.fit()
print(summarize(results))
print(results.summary())
print(results.params)
# %% creating a design matrix with ISLP
design = MS(['lstat'])
X = design.fit_transform(Boston)
print(X)

# %% calculating predictions
new_df = pd.DataFrame({'lstat': [5, 10, 15]}) # dataset with predictions
newX = design.transform(new_df)
print(newX)

new_predictions = results.get_prediction(newX)
print(new_predictions.predicted_mean)
print(new_predictions.conf_int(alpha=0.05))
print(new_predictions.conf_int(obs=True, alpha=0.05))
# %% plotting the result with a custom function
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
ax = Boston.plot.scatter('lstat', 'medv')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)
# %%  diagnostic plots
ax = subplots(figsize=(8,8))[1]
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')
# %% leverage statistics
infl = results.get_influence()
ax = subplots(figsize=(8,8))[1]
ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
ax.set_xlabel('Index')
ax.set_ylabel('Leverage')
np.argmax(infl.hat_matrix_diag)
# %% Multiple Linear Regression
X = MS(['lstat', 'age']).fit_transform(Boston)
model1 = sm.OLS(y, X)
results1 = model1.fit()
summarize(results1)

terms = Boston.columns.drop('medv') # quickly gather all variables
terms

X = MS(terms).fit_transform(Boston)
model = sm.OLS(y, X)
results = model.fit()
summarize(results)
#%% MLP with dropped variables
minus_age = Boston.columns.drop(['medv', 'age']) 
Xma = MS(minus_age).fit_transform(Boston)
model1 = sm.OLS(y, Xma)
summarize(model1.fit())
#%% interaction variables
X = MS(['lstat',
        'age',
        ('lstat', 'age')]).fit_transform(Boston)
model2 = sm.OLS(y, X)
summarize(model2.fit())
# %% polynomial functions
X = MS([poly('lstat', degree=2), 'age']).fit_transform(Boston)
model3 = sm.OLS(y, X)
results3 = model3.fit()
summarize(results3)

# %% quantifying the extent to which the polynomial fit is superior
anova_lm(results1, results3)
#%% visualising the residuals to see if there is a pattern. There is none!
ax = subplots(figsize=(8,8))[1]
ax.scatter(results3.fittedvalues, results3.resid)
ax.set_xlabel('Fitted value')
ax.set_ylabel('Residual')
ax.axhline(0, c='k', ls='--')


# %% handling qualitative data
Carseats = load_data('Carseats')
print(Carseats)
Carseats.columns
allvars = list(Carseats.columns.drop('Sales'))
final = allvars + [('Income', 'Advertising'),
                   ('Price', 'Age')]

y = Carseats['Sales']
X = MS(final).fit_transform(Carseats) # it handles categorical values automatically!
print("X\n", X)
model = sm.OLS(y, X)
summarize(model.fit())


# %%
