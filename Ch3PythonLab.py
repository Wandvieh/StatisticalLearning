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
# %% fitting a simple model
X = pd.DataFrame({'intercept': np.ones(Boston.shape[0]),
                  'lstat': Boston['lstat']})
print(X.info())

# %% fit the model
y = Boston['medv'] # that's the response column
model = sm.OLS(y,X) # ordinary least squares model
results = model.fit()
