#%% import libraries
# General
import os
import gc
import glob

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Data science
import numpy as np
import pandas as pd
import json
## Feature selection
from scipy.stats import spearmanr
from scipy.cluster import hierarchy

# ML
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn import preprocessing
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix,classification_report
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
# # Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
# Model persistence
from sklearn.externals import joblib

# mostra grafici in finestra
get_ipython().run_line_magic('matplotlib', 'qt') #'inline'

# %% check directory
os.chdir(os.path.dirname(__file__)) # change to dir name (excludes file from name) of filepath
os.getcwd()
# %% import dati
df = pd.read_csv('..\..\Dati_BB\Walden 97-19\Dati Orion\Walden 16-19 [D+domande+m_p_int_voto].csv', sep=';')#,nrwos=100)
print(df.shape)
df.head()

#%% caricamento pacchetti
#modello base
clf_base = RandomForestClassifier(n_estimators = 1000,
                            max_depth = 12,
                            random_state = 42,
                            max_features=5,
                            criterion = 'gini',
                            class_weight = "balanced")# class_weight)
# modello principale
clf = deepcopy(clf_base)

# encoders and scalers
encoder_bin = OneHotEncoder(drop = 'first') #for binary features (drops one)
encoder_cat  = OneHotEncoder() # for categorical features (no drops)

MM_scaler = MinMaxScaler()

# %% PREPARAZIONE DATI
# int voto aliases
diz_aliases_int_voto = {"Partito Democratico-PD":'PD',
                "Partito Democratico":'PD',
                "Lega con Salvini":'Destra',
                "Lega Nord":'Destra',
                "Lega":'Destra',
                "Forza Italia":'Destra',
                "Fratelli d'Italia":'Destra',
                'MoVimento 5 Stelle':'M5S',
                'Movimento 5 stelle':'M5S',
                'voterei  scheda bianca / annullerei la scheda':'bianca/nulla',
                'voterei scheda bianca / scheda nulla':'bianca/nulla',
                "piu' Europa con Emma Bonino":'+Europa',
                'Sinistra italiana (SEL + altri)':'Sinistra',
                'Potere al Popolo':'Sinistra',
                'Rifondazione Comunista':'Sinistra',
                "Fratelli d'Italia-Alleanza Nazionale&nbsp;":"Destra",
                'La Sinistra':'Sinistra'}

# pulitura dati in int voto e autocol
df['m_p_int_voto'] = df['m_p_int_voto'].replace(diz_aliases_int_voto)
df_ready = df[(df.m_p_int_voto!='preferisco non rispondere') & (df.m_p_int_voto!='sono indeciso')&
        (df.m_p_int_voto!='non andrei a votare')&(df.m_p_int_voto!='bianca/nulla')&
        (df.m_p_autocol!='preferisco non rispondere')]
df_ready.m_p_int_voto.value_counts()

# identificazione variabili di accordo
variabili_ac = []
for var_ac in df_ready.columns:
    if '_ac_' in var_ac:
        print(var_ac)
        variabili_ac.append(var_ac)

# preparazione dati pre-train-test-split
df_pre_tts = df_ready[df_ready.m_p_int_voto.map(df_ready.m_p_int_voto.value_counts()) > 100]
df_pre_tts = df_pre_tts.dropna()

# %%
####################################### 1) MODELLO di BASE
# # tieni solo variabili demografiche
# X_0 = df_pre_tts.drop(columns = variabili_ac+['m_p_autocol']+['m_p_int_voto'])
# y = df_pre_tts.m_p_int_voto

# # crea dummies
# not_dummy = ['m_p_r_eta']
# X_dummies_0 = X_0.drop(columns = not_dummy)
# encoder_bin.fit(X_dummies_0)
# X_dummies = encoder_bin.transform(X_dummies_0)

# # trasforma in dataframe
# X_dummies = pd.DataFrame(X_dummies.toarray(), columns=encoder_bin.get_feature_names())

# # scala tutte le variabili con MinMaxScaler
# # aggiungi le non_dummy
# X = pd.concat([X_0[not_dummy].reset_index(drop=True),X_dummies], axis=1)
# # scala tutto
# MM_scaler.fit(X)
# X[X.columns] = MM_scaler.transform(X)

from rachael_noodles.feature_engineering import XY_encoder
XX, yy = XY_encoder(df = df_pre_tts, target = 'm_p_int_voto',
                                    numeric_features = ['m_p_r_eta'],
                                    categorical_features = ['m_p_r_ampiezza6','m_istat_reg'],
                                    bool_features = ['m_sesso', 'm_p_pubblico_privato'])

# train test split
X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size = 1/3, random_state = 40, #y.factorize()[0]
                                                stratify = yy)

# allena modello
clf_base.fit(X_train, y_train)

# %% [markdown]
# # Hyperparameter tuning
RSEED = 42 # To keep same randomness as before

# Hyperparameter grid
param_grid = { 
    'n_estimators': [500, 1000, 1500],
    'max_features': np.arange(1, 11, 1),
    'max_depth' : np.arange(5, 15, 1),
    'criterion' :['gini', 'entropy'],
    'bootstrap': [True, False]
}

# Estimator for use in random search
estimator = RandomForestClassifier(random_state = RSEED,
                                   class_weight = 'balanced')
# Create the random search model
rfCV = RandomizedSearchCV(estimator, param_grid, n_jobs = -1, 
                        scoring = "f1_micro", cv = 4, 
                        n_iter = 20, verbose = 1, random_state=RSEED)

# Run search and store best model
rfCV.fit(X_train, y_train)
clf_base_best = rfCV.best_estimator_

# %% classification report
from sklearn.metrics import classification_report
y_pred_best = clf_base_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_best))

# %%
# cross-validation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
scores = cross_val_score(clf_base_best, X_train, y_train, cv = 4, scoring = 'f1_micro')
print("Cross-validation")
print(np.round(scores,2))

# %% [markdown]
## PLOTTING
# CONFUSION MATRIX with best model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
model = clf_base_best
y_pred = model.predict(X_test)
fig4, ax4 = plt.subplots(figsize=(10,7))
fig4 = plot_confusion_matrix(model,
    X_test,y_test, normalize='true',
    ax = ax4, values_format = '.0%', cmap = "cividis")
fig4.ax_.set_title('Intenzioni di voto', size = 24)
fig4.im_.set_clim(0, .7)
fig4.ax_.set_xlabel("Predizione", size = 20)
fig4.ax_.set_ylabel("Vero", size = 20)
fig4.im_.colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.,0)) # % in colorbar

# %% [markdown]
# ## Display feature importances
#sns.set(font_scale=1)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. Feature: %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize = (10,10))
plt.title("Feature importance", size = 20)
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation = 45, horizontalalignment='right')
plt.xlim([-1, X_train.shape[1]])
plt.subplots_adjust(bottom=.4)
plt.show()

# %%
############################### 1) MODELLO con VALORIALI SELEZIONATE
# tieni solo variabili demografiche
# X_0 = df_pre_tts.drop(columns = ['m_p_int_voto'])
# y = df_pre_tts.m_p_int_voto

# # crea dummies
# not_dummy = variabili_ac+['m_p_r_eta']
# # dummy booleane
# dummy_bool = ['m_sesso', 'm_p_pubblico_privato']
# X_dummies_bool_0 = X_0[dummy_bool]
# encoder_bin.fit(X_dummies_bool_0)
# X_dummies_bool = encoder_bin.transform(X_dummies_bool_0)
# X_dummies_bool = pd.DataFrame(X_dummies_bool.toarray(), columns=encoder_bin.get_feature_names())

# # dummy categoriche
# X_dummies_cat_0 = X_0.drop(columns = not_dummy+dummy_bool)
# encoder_cat.fit(X_dummies_cat_0)
# X_dummies_cat = encoder_cat.transform(X_dummies_cat_0)
# # trasforma in dataframe
# X_dummies_cat = pd.DataFrame(X_dummies_cat.toarray(), columns=encoder_cat.get_feature_names())

# # scala tutte le variabili con MinMaxScaler
# # aggiungi le non_dummy
# X = pd.concat([X_0[not_dummy].reset_index(drop=True),X_dummies_bool, X_dummies_cat], axis=1)

# # scala tutto
# MM_scaler = MinMaxScaler().fit(X)
# X[X.columns] = MM_scaler.transform(X)

from rachael_noodles.feature_engineering import XY_encoder
X, y = XY_encoder(df = df_pre_tts, target = 'm_p_int_voto',
                                    numeric_features = variabili_ac+['m_p_r_eta'],
                                    categorical_features = ['m_p_autocol','m_p_r_ampiezza6','m_istat_reg'],
                                    bool_features = ['m_sesso', 'm_p_pubblico_privato'])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 40, #y.factorize()[0]
                                                stratify = y)

# allena modello
clf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % clf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % clf.score(X_test, y_test))

# %% [markdown]
# # Hyperparameter tuning
## stessi parametri del modello base
# Create the random search model
rfCV = RandomizedSearchCV(estimator, param_grid, n_jobs = None, #-1 for all cores 
                        scoring = "f1_micro", cv = 4, 
                        n_iter = 10, verbose = 1, random_state=RSEED)

# Fit model
rfCV.fit(X_train, y_train)
clf_best = rfCV.best_estimator_
print("RF train accuracy: %0.3f" % clf_best.score(X_train, y_train))
print("RF test accuracy: %0.3f" % clf_best.score(X_test, y_test))
# %% classification report
y_pred_best = clf_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_best))

# %% cross-validate
scores = cross_val_score(clf_best, X_train, y_train, cv = 4, scoring='f1_micro')
print("Cross-validation")
print(np.round(scores,2))

#%% save trained model
clf_best.variables_untransformed = list(df_pre_tts.columns)
clf_best.feature_names = list(X_train.columns)
joblib.dump(clf_best, 'Models\Orion_clf_best [39 feat - no valle daosta].joblib')

# %% [markdown]
## PLOTTING
# CONFUSION MATRIX with best model
model = clf_best
fig4, ax4 = plt.subplots(figsize=(10,7))
fig4 = plot_confusion_matrix(model,
    X_test,y_test, normalize='true',
    ax = ax4, values_format = '.0%', cmap = 'cividis')
fig4.ax_.set_title('Intenzioni di voto', size = 24)
fig4.im_.set_clim(0, .7)
fig4.ax_.set_xlabel("Predizione", size = 20)
fig4.ax_.set_ylabel("Vero", size = 20)
fig4.im_.colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.,0))


# %% [markdown]
# ## Display feature importances
#sns.set(font_scale=1)
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. Feature: %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize = (10,10))
plt.title("Feature importance", size = 20)
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation = 45, horizontalalignment='right')
plt.xlim([-1, X_train.shape[1]])
plt.subplots_adjust(bottom=.4)
plt.show()


#%%
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
#%% RF MDI VS PERMUTATION IMPORTANCES
model = clf_best
result = permutation_importance(model, X_train, y_train, n_repeats=10,
                                random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(model.feature_importances_)
tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5

fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.barh(tree_indices,
         model.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(X_train.columns[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(model.feature_importances_)))
plt.tight_layout()

fig, ax2 = plt.subplots(figsize=(10,10))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=X_train.columns[perm_sorted_idx])
ax2.set_xlabel("Drop in accuracy if permuted", size = 12)
#ax2.axvline(x=0.05, c = 'red')
fig.tight_layout()
plt.show()


# %% FEATURE MULTICOLLINEARITY - DENDROGRAM
fig,ax1 = plt.subplots(figsize=(12,8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=X.columns, orientation='right',
color_threshold=1.5, ax=ax1)
dendro_idx = np.arange(0, len(dendro['ivl']))
#plt.axvline(x=1.6, c = 'orange')
#plt.axvline(x=1.7, c = 'red')
plt.tight_layout()

fig, ax2 = plt.subplots(figsize=(10,10))
ax2.imshow(np.abs(corr[dendro['leaves'], :][:, dendro['leaves']][:,::-1]),cmap='RdBu',origin='lower')
ax2.set_yticks(dendro_idx)
ax2.set_yticklabels(dendro['ivl'])
ax2.set_xticks([])
fig.tight_layout()
plt.show()

# %%
# algo comparison
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

models = []
models.append(('RF', RandomForestClassifier()))#RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'f1_micro' #'f1_micro'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=42)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Baseline Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_xlabel("Modello", size = 12)
ax.set_ylabel("Accuratezza", size = 12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.,0))
plt.show()

# %%
# hyperparameter tuning modelli di confronto
from sklearn.tree import DecisionTreeClassifier
# AdABoost
ada_param_grid = { 
    'n_estimators': [50, 100, 300, 500],
    'base_estimator':[None, DecisionTreeClassifier(),SVC(),DecisionTreeClassifier(class_weight = 'balanced',
    criterion = 'gini'),DecisionTreeClassifier(class_weight = 'balanced',
    criterion = 'entropy')],
}

# Estimator for use in random search
ada_estimator = AdaBoostClassifier(random_state = RSEED)


adaCV = GridSearchCV(ada_estimator, ada_param_grid, n_jobs = None, #-1 for all cores 
                        scoring = "f1_micro", cv = 4, verbose = 1)

# Fit model
adaCV.fit(X_train, y_train)
ada_best = adaCV.best_estimator_
print('Best hyperparameters', adaCV.best_params_)
print("Ada train accuracy: %0.3f" % ada_best.score(X_train, y_train))
print("Ada test accuracy: %0.3f" % ada_best.score(X_test, y_test))

# classification report
y_pred_ada = ada_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_ada))

# %%
# LDA
lda_param_grid = { 
    'solver': ['svd','lsqr','eigen'],
    'shrinkage':[None, 'auto']
}

# Estimator for use in random search
lda_estimator = LinearDiscriminantAnalysis()

ldaCV = GridSearchCV(lda_estimator, lda_param_grid, n_jobs = None, #-1 for all cores 
                        scoring = "f1_micro", cv = 4, verbose = 1)

# Fit model
ldaCV.fit(X_train, y_train)
lda_best = ldaCV.best_estimator_
print("LDA train accuracy: %0.3f" % lda_best.score(X_train, y_train))
print("LDA test accuracy: %0.3f" % lda_best.score(X_test, y_test))

# classification report
y_pred_lda = lda_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_lda))
# %%
# KNN
knn_param_grid = { 
    'leaf_size': list(range(1,50)),
    'n_neighbors': list(range(1,30)),
    'p' : [1,2]
}

# Estimator for use in random search
knn_estimator = KNeighborsClassifier()

scoring = "f1_macro" # works better than micro, which does not predict Sinistra
knnCV = RandomizedSearchCV(knn_estimator, knn_param_grid, n_jobs = None, #-1 for all cores 
                        scoring = scoring, cv = 4, 
                        n_iter = 40, verbose = 1, random_state=RSEED)

# Fit model
knnCV.fit(X_train, y_train)
knn_best = knnCV.best_estimator_
print("KNN train accuracy: %0.3f" % knn_best.score(X_train, y_train))
print("KNN test accuracy: %0.3f" % knn_best.score(X_test, y_test))

# classification report
y_pred_knn = knn_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_knn))
# %%
# SVC
svc_param_grid = { 
    'C': [6,7,8,9,10,11,12], 
    'kernel': ['linear','rbf']
}

# Estimator for use in random search
svc_estimator = SVC()

scoring = "f1_macro" #micro omits Sinistra
svcCV = RandomizedSearchCV(svc_estimator, svc_param_grid, n_jobs = None, #-1 for all cores 
                        scoring = scoring, cv = 4, verbose = 1,
                        n_iter = 40, random_state=RSEED)

# Fit model
svcCV.fit(X_train, y_train)
svc_best = svcCV.best_estimator_
print("Best hyperparameters: ", svcCV.best_params_)
print("SVC train accuracy: %0.3f" % svc_best.score(X_train, y_train))
print("SVC test accuracy: %0.3f" % svc_best.score(X_test, y_test))

# classification report
y_pred_svc = svc_best.predict(X_test)
print('_'+classification_report(y_test, y_pred_svc))

# %%
# best models comparison
best_models = []
best_models.append(('RF', clf_best))#RandomForestClassifier()))
best_models.append(('ADA', ada_best))
best_models.append(('LDA', lda_best))
best_models.append(('KNN', knn_best))
best_models.append(('SVM', svc_best))
# evaluate each model in turn
results = []
names = []
scoring = 'f1_micro' #'f1_micro'
for name, model in best_models:
	kfold = model_selection.KFold(n_splits=10, random_state=42)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Tuned Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.set_xlabel("Modello", size = 12)
ax.set_ylabel("Accuratezza", size = 12)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.,0))
plt.show()

# %%
