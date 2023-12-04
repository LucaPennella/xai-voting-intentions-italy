# %% [markdown]

#%% import libraries
import pyreadstat
import string

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

# %% [markdown]
# ## 1) lettura file .SAV e separazione labels/values e metadata
filepath = r'D:\Data & Code\BBrepo\Dati_BB\Omnibus\Project Orion rilevazione 04-03\Q_415123.sav'
ds_val, meta = pyreadstat.read_sav(filepath)
ds_lab, meta = pyreadstat.read_sav(filepath, apply_value_formats = True)

# check: stampa dimensione dati e prime righe
print(ds_lab.shape)
ds_val.head()

# %% [markdown]
# ### Funzioni disponibili per visualizzare metadati

#print(meta.column_names)
#print(pd.DataFrame({meta.column_labels:meta.column_names}))
#print(type(meta.column_names_to_labels))
#print(meta.number_rows)
#print(meta.number_columns)
#print(meta.file_label)
#print(meta.file_encoding)

#metadati = pd.DataFrame(zip(meta.column_names, meta.column_labels), columns = ['Nome', 'Label'])
#metadati.iloc[0:30]
#nomicol['TEL'],nomicol19['TEL'] = "TEL", "TEL"
#nomicol18['Q003'],nomicol19["Q003"] = "SESSO", "SESSO"  # info contenuta in Q527
#nomicol19
#meta18.value_labels
# meta.variable_to_label

# salvataggio dizionario corris QX - domande (SPSS Labels)
nomicol = meta.column_names_to_labels # dictionary w col names as labels
# salvataggio diz corrisp QX - diz corrisp valore num - stringa (SPSS value labels)
diz_qx_var_labels = meta.variable_value_labels


# %%
# cambio key TEL (originale identificativo CATI) in IDU (ID CAWI) in nomicol
nomicol['TEL'] = "IDU"

# %%
# trasformazione nomoicol da dizionario a dataframe (2 colonne)
df_corrisp_qx_domanda = pd.DataFrame.from_dict(nomicol, orient='index', columns = ['domanda']).reset_index()
# cambio nome variabile indice in QX
df_corrisp_qx_domanda.columns = ['qx', 'domanda']
# check: vedi prime righe
df_corrisp_qx_domanda.head()

# %% [markdown]
# ### Salvataggio corrispondenza Q - domanda (usare solo con nuovo Walden)
#annoWalden = '9719'
#df_corrisp_qx_domanda.to_csv("Walden "+annoWalden+"" corrispondenza q - domanda.csv", index=False, encoding='utf-8-sig', sep = ';')

# %% (archivio)
# procedura rimozione qx senza domanda corrispondente
#df_nc = pd.DataFrame([nomicol]).dropna(axis='columns')
#nomicol_c = df_nc.to_dict('r')[0]
#nomicol_c[0]

# %% (archivio)
# salvataggio df con domanda a nome colonna

# %% (archivio)
# assegnazione damande come nomi variabili a ds principali
#ds_lab_r = ds_lab.rename(columns=nomicol_c)
#ds_val_r = ds_val.rename(columns=nomicol_c)


# %% (archivio) - prime 3 righe dei dati aggiunte a mano nel csv
# # salvataggio corrispondenza Q - domanda
# #creazione df
fileNomiVar = pd.DataFrame.from_dict(nomicol,orient='index', columns=['label']).reset_index()
# # salvataggio
fileNomiVar.to_csv(r"..\..\Dati_BB\Omnibus\Omnibus giugno 18 per Orion\Omnibus 04-03 corrisp QX - domanda.csv", index=False, encoding='utf-8-sig', sep = ';')

# %% [markdown]
# ## 2) lettura tabella di conversione
#nuovi_nomi = pd.read_csv(r"..\Dati\Walden 9719 corrispondenza q - domanda - NEWLABEL.csv", sep = ";", encoding = "ISO-8859-1")
df_nuovi_nomi = pd.read_csv(r"..\..\Dati_BB\Walden 97-19\Walden 9719 corrispondenza Q-Domanda-m_Label.csv",encoding="utf-8-sig", sep = ";")
# check: prime righe
df_nuovi_nomi.head()
# nuovi label contenuti in variabile new_label_97_19

# %% [markdown]
# #### salvataggio mappatura in dizionario
# trasformazione df domanda - nuovi nomi in (serie indicizzata e poi) dizionario per seguente mappatura
diz_corrisp_domanda9719_newLabel = pd.Series(df_nuovi_nomi['new_label_97_19'].values,index=df_nuovi_nomi['domanda_97_19'].str.replace('[^\w]','')).to_dict()
diz_corrisp_domanda1219_newLabel = pd.Series(df_nuovi_nomi['new_label_97_19'].values,index=df_nuovi_nomi['domanda 12-19'].str.replace('[^\w]','')).to_dict()
# %% [markdown]
# ### 3) sostituzione variabili QXXX con nuovi labels

# %% aggiunta nuovi nomi a matrice corrispondenza q - domanda e creazione matr convers
# per ogni aggiunta in base a matching con entrata in dizionario
# il secondo 'item' in .get(item,item), opzionale, indica cosa immettere se non trova il key
df_matrice_convers_QDN = df_corrisp_qx_domanda.copy()
df_matrice_convers_QDN['domanda'] = df_matrice_convers_QDN['domanda'].str.replace('[^\w]','')
df_matrice_convers_QDN['new_label'] = [diz_corrisp_domanda9719_newLabel.get(item,item) for item in df_corrisp_qx_domanda.domanda.str.replace('[^\w]','')]
df_matrice_convers_QDN['new_label'] = [diz_corrisp_domanda1219_newLabel.get(item,item) for item in df_corrisp_qx_domanda.domanda.str.replace('[^\w]','')]

df_matrice_convers_QDN['new_label'] = df_matrice_convers_QDN['new_label'].str.replace(r'^.*Sedovessevotareoggi.*$','m_p_int_voto',1)
df_matrice_convers_QDN['new_label'] = df_matrice_convers_QDN['new_label'].str.replace(r'^.*Leipoliticamente.*$','m_p_autocol',1)
df_matrice_convers_QDN['new_label'] = df_matrice_convers_QDN['new_label'].str.replace(r'^.*Settorelavora.*$','m_p_pubblico_privato',1)
# %%
# sostituzione nuovi nomi mancanti in matr conversione con corrispettivo QX
for qn in range(len(df_matrice_convers_QDN.new_label)):
    if df_matrice_convers_QDN.new_label[qn] == None:
        df_matrice_convers_QDN.new_label[qn] = df_matrice_convers_QDN['qx'][qn]


# %%
# assegnazione nuovi nomi a ds_val e ds_lab da matr convers
# crea copie
ds_val_renamed = ds_val.copy()
ds_lab_renamed = ds_lab.copy()
# assegna nomi
ds_val_renamed.columns = df_matrice_convers_QDN['new_label'].values
ds_lab_renamed.columns = df_matrice_convers_QDN['new_label'].values
# %%
#check: prime righe
ds_val_renamed.head(3)
ds_lab_renamed.head(3)



#%% start testing experiment
trained_clf = joblib.load('Models\Orion_clf_best [39 feat - no valle daosta].joblib')
column_names = trained_clf.variables_untransformed

#%% sistemazione map _ac_
diz_aliases_ac = {"deltuttod'accordo":2,
"deltutto<BR>d'accordo":2,
             "moltod'accordo": 1.5,
		     "d'accordo":1,
             "pocod'accordo": 0.5,
		     "abbastanzad'accordo": 0.5,
		     "ne'd'accordone'indisaccordo(NONSTIMOLARE)":0,
		     "nonsaprei":0,
		     "ne'd'accordone'indisaccordo": 0,
		     "ned'accordone'indisaccordo": 0,
		     "nonsaprei\n": 0,
             'nonsaprei&nbsp;':0,
             "indisaccordo":-1,
		     "deltuttoindisaccordo":-2,
             "deltutto<BR>indisaccordo":-2,
             'deltuttodisaccordo':-2,
             "pernullad'accordo":-2,
             'preferiscononrispondere':np.nan,
             'preferisco<BR>non<BR>rispondere':np.nan}

#%%
df_Omni_values = ds_lab_renamed.copy()

#select relevant features from other script
df_Omni_values = df_Omni_values[column_names]
nomi_var_O = df_Omni_values.columns.tolist()

for var in nomi_var_O:
    if '_ac_' in var:
        print(var)
        df_Omni_values[var] = ds_lab_renamed[var].str.replace(r"\s","").replace(diz_aliases_ac)

# %%
df_Omni_values.m_p_int_voto.value_counts()
#%%
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
                'Sinistra Italiana':'Sinistra',
                'Italia Viva di Renzi':'Sinistra',
                'Potere al Popolo':'Sinistra',
                'Rifondazione Comunista':'Sinistra',
                "Fratelli d'Italia-Alleanza Nazionale&nbsp;":"Destra",
                'La Sinistra':'Sinistra'}

# pulitura dati in int voto e autocol
df_Omni_values['m_p_int_voto'] = df_Omni_values['m_p_int_voto'].replace(diz_aliases_int_voto)
df_Omni_values_ready = df_Omni_values[(df_Omni_values.m_p_int_voto!='preferisco non rispondere') & (df_Omni_values.m_p_int_voto!='sono indeciso')&
        (df_Omni_values.m_p_int_voto!='non andrei a votare')&(df_Omni_values.m_p_int_voto!='bianca/nulla')&
        (df_Omni_values.m_p_autocol!='preferisco non rispondere')]
df_Omni_values_ready = df_Omni_values_ready[df_Omni_values_ready['m_p_int_voto'].isin(['Sinistra', 'PD', 'M5S', 'Destra'])]
df_Omni_values_ready.m_p_int_voto.value_counts()

#% identificazione variabili di accordo
variabili_ac = []
for var_ac in df_Omni_values_ready.columns:
    if '_ac_' in var_ac:
        print(var_ac)
        variabili_ac.append(var_ac)

# %%
## preparazione dati pre-train-test-split
omni_pre_tts = df_Omni_values_ready.dropna()

#%% variables mapping manuale
sex_map = {"Uomo":'maschio',"Donna":'femmina'}
settore_map = {"Pubblico":' pubblico',"Privato":' privato'}
ampiezza_map = {"da 30001 a 100000 abitanti":'da 30.001 a 100.000',"oltre 250000 abitanti":'piu` di 250.001',
'da 100001 a 250000 abitanti':'da 100.001 a 250.000','fino a 5000 abitanti':'meno di 5.000',
'da 10001 a 30000 abitanti':'da 10.001 a 30.000','da 5001 a 10000 abitanti':'da 5.001 a 10.000'}

omni_pre_tts['m_sesso'] = omni_pre_tts['m_sesso'].replace(sex_map)
omni_pre_tts['m_p_pubblico_privato'] = omni_pre_tts['m_p_pubblico_privato'].replace(settore_map)
omni_pre_tts['m_p_r_ampiezza6'] = omni_pre_tts['m_p_r_ampiezza6'].replace(ampiezza_map)
omni_pre_tts['m_istat_reg'] = omni_pre_tts['m_istat_reg'].str[3:]


#%%
from rachael_noodles.feature_engineering import XY_encoder
X_omni, y_omni = XY_encoder(df = omni_pre_tts, target = 'm_p_int_voto',
                                    numeric_features = variabili_ac+['m_p_r_eta'],
                                    categorical_features = ['m_p_autocol','m_p_r_ampiezza6','m_istat_reg'],
                                    bool_features = ['m_sesso', 'm_p_pubblico_privato'])


# %% test model performance
print("RF test accuracy: %0.3f" % trained_clf.score(X_omni, y_omni))

# %% classification report
y_omni_pred_best = trained_clf.predict(X_omni)
print('_'+classification_report(y_omni, y_omni_pred_best))

# %% [markdown]
## PLOTTING
# CONFUSION MATRIX
model = trained_clf
fig4, ax4 = plt.subplots(figsize=(10,7))
fig4 = plot_confusion_matrix(model,
    X_omni,y_omni, normalize='true',
    ax = ax4, values_format = '.0%', cmap = 'cividis')
fig4.ax_.set_title('Intenzioni di voto', size = 24)
fig4.im_.set_clim(0, .7)
fig4.ax_.set_xlabel("Predizione", size = 20)
fig4.ax_.set_ylabel("Vero", size = 20)
fig4.im_.colorbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.,0))



#%% RF MDI VS PERMUTATION
model = trained_clf
result = permutation_importance(model, X_omni, y_omni, n_repeats=10,
                                random_state=42)
perm_sorted_idx = result.importances_mean.argsort()

tree_importance_sorted_idx = np.argsort(model.feature_importances_)
tree_indices = np.arange(0, len(model.feature_importances_)) + 0.5

fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.barh(tree_indices,
         model.feature_importances_[tree_importance_sorted_idx], height=0.7)
ax1.set_yticklabels(X_omni.columns[tree_importance_sorted_idx])
ax1.set_yticks(tree_indices)
ax1.set_ylim((0, len(model.feature_importances_)))
plt.tight_layout()

fig, ax2 = plt.subplots(figsize=(10,10))
ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
            labels=X_omni.columns[perm_sorted_idx])
ax2.set_xlabel("Drop in accuracy if permuted", size = 12)
#ax2.axvline(x=0.05, c = 'red')
fig.tight_layout()
plt.show()








#%%
# ################################# da sistemare per salvare il dataset

# %% [markdown]
# ## Salvataggio dati val e lab rinominati in .csv
ds_lab_renamed.to_csv(r"..\..\..\Dati_BB\Walden 97-19\XXX labels rinominato.csv", index=False, encoding='utf-8-sig', sep = ';')
ds_val_renamed.to_csv(r"..\..\..\Dati_BB\Walden 97-19\XXX values rinominato.csv", index=False, encoding='utf-8-sig', sep = ';')

# %% [markdown]
# ### Crea file con value labels da SPSS - mappatura alias

# copia temporanea dizionario diz_qx_var_labels
tmp_diz_qx_vLabels = diz_qx_var_labels.copy()

# crea nuove entries nel dizionario con i keys cambiati tramite conversione
# ed elimina le entries vecchie (del) -> non possibile sostituire keys nei dizionari
for qn in list(diz_qx_var_labels.keys()): # per ogni QX
    tmp_diz_qx_vLabels[df_matrice_convers_QDN['new_label']\
        [df_matrice_convers_QDN.index[df_matrice_convers_QDN['qx'] == qn].tolist()[0]]] = \
            tmp_diz_qx_vLabels[qn]
    del tmp_diz_qx_vLabels[qn] # elimina entry vecchia (qx)


# %%
# salva il nuovo dizionario con i value labels come pandas dataframes
df_value_to_vLabel_map = pd.DataFrame.from_dict(tmp_diz_qx_vLabels).reset_index()

# %% check correttezza
df_value_to_vLabel_map.head()
# %% salva in CSV
df_value_to_vLabel_map.to_csv(r"..\..\..\Dati_BB\Walden 97-19\XXX value to value label map.csv", index=False, encoding='utf-8-sig', sep = ';')

# %%
