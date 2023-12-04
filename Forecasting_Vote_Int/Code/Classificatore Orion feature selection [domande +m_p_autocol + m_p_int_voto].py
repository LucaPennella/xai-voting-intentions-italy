#%% import libraries
# General
import os
import gc
import glob

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

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
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, plot_confusion_matrix,classification_report
from sklearn import metrics
from sklearn.inspection import permutation_importance


#%%
get_ipython().run_line_magic('matplotlib', 'qt') #'inline'
# %% check directory
os.chdir(os.path.dirname(__file__)) # change to dir name (excludes file from name) of filepath
os.getcwd()
# %%
df = pd.read_csv('..\..\Dati_BB\Walden 97-19\Dati Orion\Walden 16-19 [_ac_+m_p_int_voto].csv', sep=';')#,nrwos=100)
print(df.shape)
df.head()

# feature engineering

# %% int voto aliases
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

df['m_p_int_voto'] = df['m_p_int_voto'].replace(diz_aliases_int_voto)
df_ready = df[(df.m_p_int_voto!='preferisco non rispondere') & (df.m_p_int_voto!='sono indeciso')&
        (df.m_p_int_voto!='non andrei a votare')&(df.m_p_int_voto!='bianca/nulla')&
        (df.m_p_autocol!='preferisco non rispondere')]
df_ready.m_p_int_voto.value_counts()

#%% data prep
df_pre_tts = df_ready[df_ready.m_p_int_voto.map(df_ready.m_p_int_voto.value_counts()) > 100]
df_pre_tts = df_pre_tts.dropna()
X = df_pre_tts.drop(columns = ['m_p_autocol','m_p_int_voto'])
y = df_pre_tts.m_p_int_voto

# scala tutto
MM_scaler = MinMaxScaler().fit(X)
X[X.columns] = MM_scaler.transform(X)


#%% FEATURE SELECTION
# DENDROGRAM & CORR MATRIX
fig,ax1 = plt.subplots(figsize=(12,8))
corr = spearmanr(X).correlation
corr_linkage = hierarchy.ward(np.abs(corr))
with plt.rc_context({'lines.linewidth': 2}):
    dendro = hierarchy.dendrogram(corr_linkage, labels=X.columns, orientation='right',color_threshold=2.1, ax=ax1)
dendro_idx = np.arange(0, len(dendro['ivl']))
plt.box(on=None)
#plt.axvline(x=1.6, c = 'orange')
#plt.axvline(x=1.7, c = 'red')
ax1.set_xticks([])
plt.tight_layout()

fig, ax2 = plt.subplots(figsize=(10,10))
ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']][:,::-1],cmap='seismic',origin='lower')
ax2.set_yticks(dendro_idx)
ax2.set_yticklabels(dendro['ivl'])
ax2.set_xticks(np.arange(0,X.shape[1]))
empty_string_labels = ['']*X.shape[1]
ax2.set_xticklabels(empty_string_labels)
fig.tight_layout()
plt.show()



#%% select subset of features
from collections import defaultdict
threshold = 1.6
cluster_ids = hierarchy.fcluster(corr_linkage, threshold, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

X_sel = X.iloc[:,selected_features]

#%% prepare data for iteration 2 -> add autocol and int voto
# add to second iteration
# feature encoding
X_dummies_voto_0 = df_pre_tts[['m_p_int_voto']].values.reshape(-1,1)
encoder  = preprocessing.OneHotEncoder()#drop='first') #importante
encoder.fit(X_dummies_voto_0)
X_dummies_voto = encoder.transform(X_dummies_voto_0)

# trasforma in dataframe
X_dummies_voto = pd.DataFrame(X_dummies_voto.toarray(), columns=encoder.get_feature_names(['VOTO']))

# scala tutte le variabili con MinMaxScaler
# aggiungi le non_dummy
X_sel_int_voto = pd.concat([X_sel.reset_index(drop=True),X_dummies_voto], axis=1)

#%% FEATURE SELECTION 2
# DENDROGRAM & CORR MATRIX
fig,ax1 = plt.subplots(figsize=(12,8))
corr = spearmanr(X_sel_int_voto).correlation
corr_linkage = hierarchy.ward(np.abs(corr))
dendro = hierarchy.dendrogram(corr_linkage, labels=X_sel_int_voto.columns, orientation='right',
color_threshold=1.5, ax=ax1)
dendro_idx = np.arange(0, len(dendro['ivl']))
plt.tight_layout()

###################perform visual inspection

#%% prepare data for feature importance analysis for RF
# feature encoding
X_dummies_autocol_0 = df_pre_tts[['m_p_autocol']].values.reshape(-1,1)
encoder  = preprocessing.OneHotEncoder()#drop='first') #importante
encoder.fit(X_dummies_autocol_0)
X_dummies_autocol = encoder.transform(X_dummies_autocol_0)

# trasforma in dataframe
X_dummies_autocol = pd.DataFrame(X_dummies_autocol.toarray(), columns=encoder.get_feature_names(['AUTOCOL']))

# scala tutte le variabili con MinMaxScaler
# aggiungi le non_dummy
X_sel_autocol = pd.concat([X_sel.reset_index(drop=True),X_dummies_autocol], axis=1)

#%% MACHINE LEARNING
X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size = 1/3, random_state = 40, #y.factorize()[0]
                                                stratify = y)

#%%
clf = RandomForestClassifier(n_estimators = 1000,
                            max_depth = 10,
                            random_state = 42,
                            criterion = 'gini',
                            class_weight = 'balanced')

#%%
clf.fit(X_train, y_train)
print("RF train accuracy: %0.3f" % clf.score(X_train, y_train))
print("RF test accuracy: %0.3f" % clf.score(X_test, y_test))
# %% classification report
y_pred = clf.predict(X_test)
print('_'+classification_report(y_test, y_pred))

# %%
# CONFUSION MATRIX
model = clf
fig4, ax4 = plt.subplots(figsize=(10,7))
fig4 = plot_confusion_matrix(model,
    X_test,y_test, normalize='true',
    ax = ax4, values_format = '.0%', cmap = 'cividis')
fig4.ax_.set_title('Intenzioni di voto', size = 24)
fig4.im_.set_clim(0, .7)
fig4.ax_.set_xlabel("Predizione", size = 20)
fig4.ax_.set_ylabel("Vero", size = 20)


#%% cross-validation
scores = cross_val_score(model, X_train, y_train, cv = 4, n_jobs = 2)
print(scores)

#%% RF MDI VS PERMUTATION
model = clf
result = permutation_importance(model, X_train, y_train, n_repeats=5,
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
ax2.axvline(x=0.05, c = 'red')
fig.tight_layout()
plt.show()




#%% save 8 best features
X_train.columns[tree_importance_sorted_idx][-8:]











# %%
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(clf)#, threshold=0.15)
sel.fit(X_train, y_train)

sel.get_support()


selected_feat= X_train.columns[(sel.get_support())]
len(selected_feat)

print(selected_feat)


# %% [markdown]
# ## Display feature importances
sns.set(font_scale=1)
importances = sel.estimator.feature_importances_
std = np.std([tree.feature_importances_ for tree in sel.estimator],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. Feature: %s (%f)" % (f + 1, X_train.columns[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize = (10,10))
plt.title("Feature importances for int voto")
plt.bar(range(X_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.subplots_adjust(bottom=.4)
plt.show()

# %%

# %%

OH_encoder  = preprocessing.OneHotEncoder()
selected_feat2 = ['m_ac_vantaggi_globalizz_economie_mercati',
'm_ac_validita_insegnamChiesa','m_ac_legalizz_drogheLeggere',
'm_ac_italia_ipartecipazioneIn_missioniMilitariEstere',
'x0_a destra','x0_a centro destra','x0_al centro','x0_a centro sinistra','x0_a sinistra']
X_correl = X_train[selected_feat2]
#X_correl[X_correl.columns] = MM_scaler.transform(X_correl[X_correl.columns])

X_correl_dummies = OH_encoder.fit_transform(y_train.values.reshape(-1,1))
X_correl_dummies = pd.DataFrame(X_correl_dummies.toarray(), columns=OH_encoder.get_feature_names())
X_correl = pd.concat([X_correl.reset_index(drop=True),X_correl_dummies], axis=1)


# %% CORRELATION MATRIX
correl_matrix = X_correl.corr(method = 'spearman')
#fig, ax = plt.subplots(figsize=(10,7))
#ax = sns.heatmap(correl_matrix, linewidth=0.2, cmap="RdBu_r", xticklabels=True, yticklabels=True)
mask = np.zeros_like(correl_matrix)
mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 7))
    f.subplots_adjust(left=.1, bottom=.3, top = 1, right = 1.)
    correl_matrix[np.abs(correl_matrix)<.1] = 0.
    correl_matrix = np.round(correl_matrix,2).iloc[-9:,:-4]
    ax = sns.heatmap(correl_matrix,linewidth=0.2,cmap="RdBu_r",# mask=mask,#[-10:,:-10]
    vmin=-.4, vmax=.4, cbar = 0,
    square=True, xticklabels=True, yticklabels=True, annot = True, annot_kws={"size": 6})
    ax.tick_params(axis='both', which='major', labelsize=10)
#fig = plt.imshow(correl_matrix, cmap='hot', interpolation='nearest')
#plt.show()

# %%
