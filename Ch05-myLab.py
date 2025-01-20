# %% imports
import numpy as np
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize,
                         poly)
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.model_selection import \
     (cross_validate,
      KFold,
      ShuffleSplit)
from sklearn.base import clone
from ISLP.models import sklearn_sm

# %% Validation Set Approach on linear regression
Auto = load_data('Auto')
Auto_train, Auto_valid = train_test_split(Auto,
                                          test_size=196, # two equal parts
                                          random_state=0)
# fitting a linear regression
hp_mm = MS(['horsepower'])
X_train = hp_mm.fit_transform(Auto_train) # bringing the variables into the correct form
y_train = Auto_train['mpg'] # bringing the response into the correct form
model = sm.OLS(y_train, X_train) # training the model
results = model.fit()
# using the validation set
X_valid = hp_mm.transform(Auto_valid)
y_valid = Auto_valid['mpg']
valid_pred = results.predict(X_valid)
print(np.mean((y_valid - valid_pred)**2)) # MSE: 23.62
# %% Validation Set Approach on polynomial regressions
def evalMSE(terms,
            response,
            train,
            test):
      # trains a model and returns the MSE
      mm = MS(terms)
      X_train = mm.fit_transform(train)
      y_train = train[response]

      X_test = mm.transform(test)
      y_test = test[response]

      results = sm.OLS(y_train, X_train).fit()
      test_pred = results.predict(X_test)

      return np.mean((y_test - test_pred)**2)
# using evalMSE on 1st, 2nd and 3rd degree polynomial regressions
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
                       'mpg',
                       Auto_train,
                       Auto_valid)
print(MSE) # 1st: 23.62, 2nd: 18.76, 3rd: 18.80

# %% The same on a different split in the data
Auto_train, Auto_valid = train_test_split(Auto,
                                          test_size=196,
                                          random_state=3)
MSE = np.zeros(3)
for idx, degree in enumerate(range(1, 4)):
    MSE[idx] = evalMSE([poly('horsepower', degree)],
                       'mpg',
                       Auto_train,
                       Auto_valid)
print(MSE)

# %% LOOCV
# In practice, you can best do CV with sklearn
# In the ISLP package, they provide a wrapper for using
# CV tools of sklearn with models fit by statsmodels: sklearn_sm
hp_model = sklearn_sm(sm.OLS, # the model from statsmodels
                      MS(['horsepower'])) # the design matrix
X, Y = Auto.drop(columns=['mpg']), Auto['mpg']
cv_results = cross_validate(hp_model, # the result of the wrapper
                            X, # features
                            Y, # predictions
                            cv=Auto.shape[0]) # LOOCV
cv_err = np.mean(cv_results['test_score'])
print(cv_err)
print(cv_results)
# %% LOOCV on polynomial regressions
cv_error = np.zeros(5)
H = np.array(Auto['horsepower'])
M = sklearn_sm(sm.OLS)
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
                          X,
                          Y,
                          cv=Auto.shape[0])
    cv_error[i] = np.mean(M_CV['test_score'])
print(cv_error)

# %% 10fold CV on polynomial regressions
cv_error = np.zeros(5)
cv = KFold(n_splits=10,
           shuffle=True,
           random_state=0) # use same splits for each degree
for i, d in enumerate(range(1,6)):
    X = np.power.outer(H, np.arange(d+1))
    M_CV = cross_validate(M,
                          X,
                          Y,
                          cv=cv)
    cv_error[i] = np.mean(M_CV['test_score'])
print(cv_error)

# %% The same with Shuffle Split
validation = ShuffleSplit(n_splits=1,
                          test_size=196,
                          random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results['test_score']

validation = ShuffleSplit(n_splits=10,
                          test_size=196,
                          random_state=0)
results = cross_validate(hp_model,
                         Auto.drop(['mpg'], axis=1),
                         Auto['mpg'],
                         cv=validation)
results['test_score'].mean(), results['test_score'].std() # gives a sd
# %% The Bootstrap on all and on a bootstrap sample
Portfolio = load_data('Portfolio')
def alpha_func(D, idx): # dataframe D with columns X and Y
   # returns an estimate for alpha by applying the minimum variance formula to the observations
   cov_ = np.cov(D[['X','Y']].loc[idx], rowvar=False)
   return ((cov_[1,1] - cov_[0,1]) /
           (cov_[0,0]+cov_[1,1]-2*cov_[0,1]))
print(alpha_func(Portfolio, range(100))) # estimate for all 100 observations
rng = np.random.default_rng(0)
print(alpha_func(Portfolio, # estimate for 100 random selections (new bootstrap set)
           rng.choice(100,
                      100,
                      replace=True)))
# %% Bootstrap with 1000 bootstraps sets
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
alpha_SE = boot_SE(alpha_func,
                   Portfolio,
                   B=1000,
                   seed=0)
print(alpha_SE) # estimate with 1000 bootstrap sets: SE of 0.0912
# %% Using bootstrap to estimate the accuracy of coefficients
# getting the SEs for beta 0 and beta 1 and comparing them to the formulas from Ch03
# using boot_SE for that
def boot_OLS(model_matrix, response, D, idx):
    # generic function for bootstrapping a regression model
    D_ = D.loc[idx]
    Y_ = D_[response]
    X_ = clone(model_matrix).fit_transform(D_)
    return sm.OLS(Y_, X_).fit().params
# freezing the first two arguments as they are not changed in boot_SE
hp_func = partial(boot_OLS, MS(['horsepower']), 'mpg') # generates a new func hp_func
rng = np.random.default_rng(0)
# demonstration on 10 samples
print(np.array([hp_func(Auto,
          rng.choice(Auto.index,
                     392,
                     replace=True)) for _ in range(10)]))
hp_se = boot_SE(hp_func,
                Auto,
                B=1000,
                seed=10)
print(hp_se) # SE of 1000 bootstrap estimates for slope and interept
hp_model.fit(Auto, Auto['mpg'])
model_se = summarize(hp_model.results_)['std err']
print(model_se) # SE from the formula
# results are different, but bootstrap is probably more accurate
# why? the formula for SE assumes the noise variance relies on the linear model being correct (which it is not)
# also, they assume all variation comes from the errors, which bottstrap does not rely on
# %% Bootstrap SE estimates and lin reg estimates for the quadratic model
quad_model = MS([poly('horsepower', 2, raw=True)])
quad_func = partial(boot_OLS,
                    quad_model,
                    'mpg')
print(boot_SE(quad_func, Auto, B=1000)) # better fit, so better results!
M = sm.OLS(Auto['mpg'],
           quad_model.fit_transform(Auto))
print(summarize(M.fit())['std err']) # comparing to SE useing the formula
# %%
