#%% imports
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
                         summarize)
from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.discriminant_analysis import \
     (LinearDiscriminantAnalysis as LDA,
      QuadraticDiscriminantAnalysis as QDA)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %% dataset
Smarket = load_data('Smarket')
print(Smarket)
print(Smarket.columns)
# %% Correlation Matrix
print(Smarket.corr(numeric_only=True))
Smarket.plot(y='Volume') # plotting the only relevant orrelation
# %% runnung a Logistic Regression
allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
design = MS(allvars)
X = design.fit_transform(Smarket)
y = Smarket.Direction == 'Up' # create Boolean array
glm = sm.GLM(y, X, family=sm.families.Binomial())
results = glm.fit()
print(summarize(results)) # lage p-values for all the coefficients
#print(results.params)
#print(results.pvalues)
#print(type(results.params)) # gives a Series object
# %% predict probabilities
probs = results.predict()
# if no further argument is given, predicts on the training data
# exog argument can be passed in the form of a design matrix
print(probs[:10]) # gives probabilities
labels = np.array(['Down']*1250)
labels[probs>0.5] = 'Up' # converng into Up and Down labels
print(confusion_table(labels, Smarket.Direction))
print((507+145)/1250, np.mean(labels == Smarket.Direction))
# training error rate is 47.8% - but the test error rate is probably higher!
# %% predict probabilities on a subset of the data and check on held out data
train = (Smarket.Year < 2005) # create boolean array
Smarket_train = Smarket.loc[train]
Smarket_test = Smarket.loc[~train]
print(Smarket_test.shape)

# train on the subset
X_train, X_test = X.loc[train], X.loc[~train]
y_train, y_test = y.loc[train], y.loc[~train]
glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)

# comparing the predictions to the previous ones
D = Smarket.Direction
L_train, L_test = D.loc[train], D.loc[~train]
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
print(confusion_table(labels, L_test))
print(np.mean(labels == L_test), np.mean(labels != L_test)) # test accuracy and error rate
# %% another logistic regression fit with only Lag1 and Lag2
model = MS(['Lag1', 'Lag2']).fit(Smarket)
X = model.transform(Smarket)
X_train, X_test = X.loc[train], X.loc[~train]
glm_train = sm.GLM(y_train,
                   X_train,
                   family=sm.families.Binomial())
results = glm_train.fit()
probs = results.predict(exog=X_test)
labels = np.array(['Down']*252)
labels[probs>0.5] = 'Up'
print(confusion_table(labels, L_test))
print((35+106)/252,106/(106+76)) # accuracy and accuracy on the increase prediction days
# accuracy is 56% - but random guessing (naive approach) also yields an accuracy of 56%!
# %% prediction directions with particular values of the variables
newdata = pd.DataFrame({'Lag1':[1.2, 1.5],
                        'Lag2':[1.1, -0.8]})
newX = model.transform(newdata)
print(results.predict(newX))
# %% Linear Discriminant Analysis
lda = LDA(store_covariance=True)
# removing the intercept, since LDA automatically adds it
X_train, X_test = [M.drop(columns=['intercept'])
                   for M in [X_train, X_test]]
lda.fit(X_train, L_train)
print(lda.means_)
print(lda.classes_) # railing _ denotes a quantitiy estimated when using fit()
print(lda.priors_) # prior prob for Down is 0.492, for Up is 0.508
print(lda.scalings_) # linear discriminant vectors
lda_pred = lda.predict(X_test)
confusion_table(lda_pred, L_test)
lda_prob = lda.predict_proba(X_test)
print(np.all(np.where(lda_prob[:,1] >= 0.5, 'Up', 'Down') == lda_pred)) # creating lda_pred on your own
print(np.all([lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred)) # creating lda_pred on your own for more than two classes
print(np.sum(lda_prob[:,0] > 0.9)) # checking how many days have a posterior prob threshold of 90% (hint: none)

# %%
