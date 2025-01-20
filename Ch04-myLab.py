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
# %% dataset: Smarket
Smarket = load_data('Smarket')
print(Smarket)
print(Smarket.columns)
# %% Correlation Matrix
print(Smarket.corr(numeric_only=True))
Smarket.plot(y='Volume') # plotting the only relevant orrelation
# %% runnung a Logistic Regression with statsmodels
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
labels[probs>0.5] = 'Up' # converging into Up and Down labels
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
print(lda.classes_) # railing _ denotes a quantity estimated when using fit()
print(lda.priors_) # prior prob for Down is 0.492, for Up is 0.508
print(lda.scalings_) # linear discriminant vectors
lda_pred = lda.predict(X_test)
confusion_table(lda_pred, L_test)
lda_prob = lda.predict_proba(X_test)
print(np.all(np.where(lda_prob[:,1] >= 0.5, 'Up', 'Down') == lda_pred)) # creating lda_pred on your own
print(np.all([lda.classes_[i] for i in np.argmax(lda_prob, 1)] == lda_pred)) # creating lda_pred on your own for more than two classes
print(np.sum(lda_prob[:,0] > 0.9)) # checking how many days have a posterior prob threshold of >90% (hint: none)

# %% Quadratic Discriminant Analysis
qda = QDA(store_covariance=True)
qda.fit(X_train, L_train)
print(qda.means_, qda.priors_) # computing means_ and priors_
print(qda.covariance_[0])
qda_pred = qda.predict(X_test)
confusion_table(qda_pred, L_test)
print(np.mean(qda_pred == L_test)) # 60% accuracy

# %% Naive Bayes
NB = GaussianNB()
NB.fit(X_train, L_train)
print("classes:", NB.classes_)
print("prior probabilities", NB.class_prior_)
print("means:", NB.theta_) # means: nr of rows = nr of classes; nr of columns = nr of features
print("variances:", NB.var_) # variances: nr of rows = nr of classes; nr of columns = nr of features
print(X_train[L_train == 'Down'].mean()) # verifying the mean
nb_labels = NB.predict(X_test) # making predictions
print(confusion_table(nb_labels, L_test))
print(NB.predict_proba(X_test)[:5]) # predicting for each observation that they belong to a class
# %% K-Nearest Neighbors
knn1 = KNeighborsClassifier(n_neighbors=1) # K = 1
X_train, X_test = [np.asarray(X) for X in [X_train, X_test]]
knn1.fit(X_train, L_train)
knn1_pred = knn1.predict(X_test)
print(confusion_table(knn1_pred, L_test))
print((83+43)/252, np.mean(knn1_pred == L_test)) # accuracy of 50%

# %% KNN with K = 3
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3_pred = knn3.fit(X_train, L_train).predict(X_test)
print(np.mean(knn3_pred == L_test)) # accuracy of 53%

# %% KNN on the Caravan dataset
Caravan = load_data('Caravan')
print(Caravan)
Purchase = Caravan.Purchase
print(Purchase.value_counts())
feature_df = Caravan.drop(columns=['Purchase'])
# %% Standardizing the data! So they are all on the same scale
# all variables are given a mean of 0 and standard deviation of 1
scaler = StandardScaler(with_mean=True, # subtracting the mean (=setting mean to 0)
                        with_std=True, # scaling to a SD of 1
                        copy=True) # always copy data (?)
scaler.fit(feature_df)
X_std = scaler.transform(feature_df)
feature_std = pd.DataFrame(X_std, columns=feature_df.columns) # new df with the standardized data
print(feature_std.std())
# %% Splitting into traing and test sets
(X_train,
 X_test,
 y_train,
 y_test) = train_test_split(np.asarray(feature_std), # turning it to a ndarray to address a bug in sklearn
                            Purchase,
                            test_size=1000,
                            random_state=0) # ensures we are getting the same split each time
# %% Fitting the model
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1_pred = knn1.fit(X_train, y_train).predict(X_test)
print(np.mean(y_test != knn1_pred), np.mean(y_test != "No")) # error rate and posterior prob of "Yes"
# error rate is 11% - seems good, but using a naive approach (always guessing no) the error rate would only be 6% (the null rate)!
print(confusion_table(knn1_pred, y_test))
# but we see that the model does better on customers that end up buying - accuracy there is 14.5%
# Double the rate from guessing!
# %% Looping through K = 1 to K = 5 to see the differences
for K in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn_pred = knn.fit(X_train, y_train).predict(X_test)
    C = confusion_table(knn_pred, y_test)
    templ = ('K={0:d}: # predicted to rent: {1:>2},' +
            '  # who did rent {2:d}, accuracy {3:.1%}')
    pred = C.loc['Yes'].sum()
    did_rent = C.loc['Yes','Yes']
    print(templ.format(
          K,
          pred,
          did_rent,
          did_rent / pred))
# %% Comparison with Logistic Regression with sklearn
logit = LogisticRegression(C=1e10, solver='liblinear')
logit.fit(X_train, y_train)
logit_pred = logit.predict_proba(X_test)
logit_labels = np.where(logit_pred[:,1] > .5, 'Yes', 'No')
confusion_table(logit_labels, y_test)
# %% Cut-off for probability at 0.25 instead of 0.5
logit_labels = np.where(logit_pred[:,1]>0.25, 'Yes', 'No')
confusion_table(logit_labels, y_test)
print(9/(20+9)) # 31% correct in its yes predictions!
# %% Linear and Poisson Regression on the Bikeshare Data
Bike = load_data('Bikeshare')
print(Bike.shape, Bike.columns)

# %% First: Linear Regression
X = MS(['mnth',
        'hr',
        'workingday',
        'temp',
        'weathersit']).fit_transform(Bike)
Y = Bike['bikers']
M_lm = sm.OLS(Y, X).fit()
print(summarize(M_lm))

# mnth[Feb] is 6.8.
# This means that relative to the JANUARY amount, in Feb there are 6.8 more riders.
# Difference with another method see below

# %% Another Linear Regression fit, categorical values are handled differently
hr_encode = contrast('hr', 'sum')
mnth_encode = contrast('mnth', 'sum')
X2 = MS([mnth_encode,
         hr_encode,
         'workingday',
         'temp',
         'weathersit']).fit_transform(Bike)
M2_lm = sm.OLS(Y, X2).fit()
S2 = summarize(M2_lm)
print(S2)

# What's the difference?
# mnth[Jan] has a coefficient estimate of -46.
# This means that relative to the YEARLY AVERAGE, in Jan there 46 fewer riders.
# What about Dec that isn't coded?
# The coeff est for Dec is the negative sum of all other 11 level estimates!
# Together the all sum to 0 and are the yearly average


# %% Despite the differences, the outcome is the same
print(np.sum((M_lm.fittedvalues - M2_lm.fittedvalues)**2)) # sum of squared differences basically zero
print(np.allclose(M_lm.fittedvalues, M2_lm.fittedvalues))

# %% Plotting a line chart over a year
# Getting the coefficients for Jan - Nov
coef_month = S2[S2.index.str.contains('mnth')]['coef']
print(coef_month)
# appending Dec
months = Bike['mnth'].dtype.categories
coef_month = pd.concat([
      coef_month,
      pd.Series([-coef_month.sum()],
                 index=['mnth[Dec]'])
])
print(coef_month)

fig_month, ax_month = subplots(figsize=(8,8))
x_month = np.arange(coef_month.shape[0])
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)

# %% Plotting a line chart over hours of the day
coef_hr = S2[S2.index.str.contains('hr')]['coef']
coef_hr = coef_hr.reindex(['hr[{0}]'.format(h) for h in range(23)])
coef_hr = pd.concat([coef_hr,
                     pd.Series([-coef_hr.sum()], index=['hr[23]'])
                    ])

fig_hr, ax_hr = subplots(figsize=(8,8))
x_hr = np.arange(coef_hr.shape[0])
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticks(x_hr[::2])
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)
# %% Second: Poisson Regression
# very little change except for a different keyword
M_pois = sm.GLM(Y, X2, family=sm.families.Poisson()).fit()
S_pois = summarize(M_pois)
# adding the Dec and 23rd hour
coef_month = S_pois[S_pois.index.str.contains('mnth')]['coef']
coef_month = pd.concat([coef_month,
                        pd.Series([-coef_month.sum()],
                                   index=['mnth[Dec]'])])
coef_hr = S_pois[S_pois.index.str.contains('hr')]['coef']
coef_hr = pd.concat([coef_hr,
                     pd.Series([-coef_hr.sum()],
                     index=['hr[23]'])])
# and again the plotting (months and hours are again to be interpreted relative to their mean)
fig_pois, (ax_month, ax_hr) = subplots(1, 2, figsize=(16,8))
ax_month.plot(x_month, coef_month, marker='o', ms=10)
ax_month.set_xticks(x_month)
ax_month.set_xticklabels([l[5] for l in coef_month.index], fontsize=20)
ax_month.set_xlabel('Month', fontsize=20)
ax_month.set_ylabel('Coefficient', fontsize=20)
ax_hr.plot(x_hr, coef_hr, marker='o', ms=10)
ax_hr.set_xticklabels(range(24)[::2], fontsize=20)
ax_hr.set_xlabel('Hour', fontsize=20)
ax_hr.set_ylabel('Coefficient', fontsize=20)
# %% Comparing the fitted values
fig, ax = subplots(figsize=(8,8))
ax.scatter(M2_lm.fittedvalues,
            M_pois.fittedvalues,
            s=20)
ax.set_xlabel('Linear Regression Fit', fontsize=20)
ax.set_ylabel('Poisson Regression Fit', fontsize=20)
ax.axline([0,0], c='black', linewidth=3, linestyle='--', slope=1)
# If the results were exactly the same, the points would only be diagonal
# Here, Poission predictions are larger for very low and very high levels of ridership
# Linear Regression predictions are larger for the middle part 

# %%
