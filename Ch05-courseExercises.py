# %% Imports and Loading data
import pandas as pd
import statsmodels.api as sm
from functools import partial
from ISLP.models import (ModelSpec as MS,
                         summarize)
import numpy as np
from sklearn.base import clone
import random

df = pd.read_csv("5.Py.1.csv")
print(df)
print(df.columns)
# %% Linear Regression
y = df["y"]
X = MS(["X1", "X2"]).fit_transform(df)
model = sm.OLS(y,X)
results = model.fit()
print(summarize(results))
print(results.summary())
print(results.params)

# %% Plotting the data
df.plot()
# %%

def create_random_slices():
    arr =  np.ndarray(shape=1, dtype=int)
    for i in range(10):
        x = random.randint(0, 9)
        arr = np.concatenate((arr, np.r_[slice(x*100, (x+1)*100)]))
    return arr[1:]
create_random_slices()
# %% Using bootstrap to estimate the SE for beta 1
def alpha_func(D, idx): # dataframe D with columns X and Y
   # returns an estimate for alpha by applying the minimum variance formula to the observations
   cov_ = np.cov(D[['X','Y']].loc[idx], rowvar=False)
   return ((cov_[1,1] - cov_[0,1]) /
           (cov_[0,0]+cov_[1,1]-2*cov_[0,1]))

def boot_SE(func,
            D,
            n=None,
            B=1000,
            seed=0):
    rng = np.random.default_rng(seed)
    first_, second_ = 0, 0
    n = n or D.shape[0]
    for _ in range(B): # _ as variable is often used when the value of the counter is unimportant
        idx = rng.choice(D.index,
                         n,
                         replace=True)
        value = func(D, idx)
        first_ += value
        second_ += value**2
    return np.sqrt(second_ / B - (first_ / B)**2)

def boot_OLS(model_matrix, response, D, idx):
    # generic function for bootstrapping a regression model
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params

# freezing the first two arguments as they are not changed in boot_SE
hp_func = partial(boot_OLS, MS(['X1', 'X2']), 'y') # generates a new func hp_func

SE = boot_SE(hp_func,
             df,
             B=1000,
             seed=0)
print(SE)
# %% Blocking bootstrap samples
new_rows = np.r_[slice(100,200), slice(400,500),
                 slice(100,200), slice(900,1000),
                 slice(300,400), slice(0,100),
                 slice(0,100), slice(800,900),
                 slice(200,300), slice(700,800)]
print(new_rows)
test = np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])]
print (test)


# %% plotting more (on my own)
"""# %% plotting the result with a custom function
def abline(ax, b, m, *args, **kwargs):
    "Add a line with slope m and intercept b to ax"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)
ax = df.plot.scatter('X1', 'X2')
abline(ax,
       results.params[0],
       results.params[1],
       'r--',
       linewidth=3)"""

# %%
