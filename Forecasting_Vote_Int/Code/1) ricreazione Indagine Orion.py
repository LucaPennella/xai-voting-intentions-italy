# %%
#%% import libraries
import os
import gc
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#import pyreadstat
import json


# %% check directory
os.chdir(os.path.dirname(__file__)) # change to dir name (excludes file from name) of filepath
os.getcwd()

# %% [markdown]
# # 1) lettura file

df = pd.read_csv('Z:\Dati_BB\Walden 97-19\walden 97-19 integrated v2.csv', sep=';')#,nrwos=100)
print(df.shape)
df.head()


# %% mantieni solo ultimi anni
df_1619 = df[df['m_anno_indagine']>=2017].drop_duplicates(subset='IDU', keep="last")
df_1619.shape

# drop cols with < 70% data
df_1619_Xpc = df_1619.dropna(thresh=0.7*(df_1619.shape[0]), axis='columns')

# %% droppa colonne ph, gli ac e gli op
nomi_var = df_1619_Xpc.columns.tolist()
var_da_rimuovere = ['_op_','_bi_','_freq_','_modalita_','TREND_15',"TREND",'_ac_']
var_da_tenere = []
varsToDrop_0 = [] # crea lista vuota da popolare con le variabili da scartare
for var in nomi_var:
    if df_1619_Xpc[var].isnull().all():
        print(var)
        varsToDrop_0.append(var)
    for snippet in var_da_rimuovere:
        if snippet in var: # se è ph, aggiungi a lista drops
            varsToDrop_0.append(var)

#%% altre variabili non demogradiche da droppare
varsToDrop = varsToDrop_0+['IDU','m_anno_indagine','m_pesos','m_peso',#'m_p_autocol',\
    'm_p_frequenza_cinema','m_p_frequenza_teatro','m_p_frequenza_concerti',
        'm_p_frequenza_mostre','m_p_frequenza_palestra_sport',
            'm_modernizz_vs_regress_Paese', #'m_solo_riduzioneTasse_per_sviluppoEconomico',
                'm_fiducia_proprieIdee_rispetto_avvenimentiMondo','m_p_comune_istat',
                'm_p_data_nascita','m_p_eta_6','m_p_scolarita','m_p_zona_5istat',\
                    'm_p_in_ita_dal','m_p_origini','m_p_nascita_in_italia_genitori','m_p_cittadinanza',
                        'm_p_radio_ore','m_p_lettura_quotidiani','m_p_iscrizione_sindacato',
                            'm_sr_professione','m_sr_statusSocioEconomico_ceto']#,'m_p_int_voto']


#var_da_tenere = ["m_ac_meglio_uguaglianza_vs_merito_singolo",'m_ac_immigrati_risorsa','m_ac_chiesa_nonDovrebbe_condizionare_stato.1','m_ac_modernizzazioneItalia_grazie_UE','m_ac_testamento_biologico']          
var_da_tenere = ["m_ac_italia_paeseInDeclino","m_ac_difesa_scuolaPubblica_insensata",
"m_ac_immigrati_risorsa","m_ac_italia_ipartecipazioneIn_missioniMilitariEstere",
"m_ac_importanza_partiti","m_ac_validita_insegnamChiesa"]

var_da_tenere = ['m_ac_vantaggi_globalizz_economie_mercati', # mandate fuori in Omnibus il 04-03
'm_ac_validita_insegnamChiesa','m_ac_legalizz_drogheLeggere',
'm_ac_italia_ipartecipazioneIn_missioniMilitariEstere','m_ac_repressione_unicaArma_vs_crimin']

var_da_tenere = ['m_ac_nonSicuro_doveVive', 'm_ac_nord_unicoMotore_economiaItaliana', # da nuovo metodo
       'm_ac_italia_ipartecipazioneIn_missioniMilitariEstere',
       'm_ac_vantaggi_globalizz_economie_mercati',
       'm_ac_societa_troppoPermissiva_gay',
       'm_ac_repressione_unicaArma_vs_crimin',
       'm_ac_modelloImprenditorialePrivato_unico_produceRicchezzaPerTutti',
       'm_ac_valori_resistenza_altra_epoca']
# %% rimuovi dal df le variabili nella lista varsToDrop
df_1619_D_TREND_0 = df_1619_Xpc.drop(varsToDrop, axis=1)
df_1619_D_TREND = pd.concat([df_1619_D_TREND_0,df_1619_Xpc[var_da_tenere]], axis=1)
#df_1619_D_TREND = df_1619_Xpc[var_da_tenere+['m_p_autocol','m_p_int_voto']]
df_1619_D_TREND.shape

# %% preparazione alias conversone trend
# diz_aliases_trend = {"in trend ":1,
#                 "slight in trend ":1,
#                 "slight off trend ":0,
#                 "off trend ":0,
#                 'non classificati':np.nan}


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
df_1619_D_TREND_values = df_1619_D_TREND.copy()
nomi_var_2 = df_1619_D_TREND.columns.tolist()

trend_ricalcolato = []

# for var in nomi_var_2:
#     if 'TREND' in var:
#         print(var)
#         df_1619_D_TREND_values[var] = df_1619_D_TREND_values[var].replace(diz_aliases_trend)
        

for var in nomi_var_2:
    if '_ac_' in var:
        print(var)
        df_1619_D_TREND_values[var] = df_1619_D_TREND_values[var].str.replace(r"\s","").replace(diz_aliases_ac)
        
# check if mapping is complete
#pd.unique(df_1619_D_TREND_values.values.ravel('K'))


#%% optional if deciding to remove duplicates
# df_1619_D_TREND_values_unique = df_1619_D_TREND_values.drop_duplicates(subset ="IDU", 
#                      keep = 'last')

# %% cleanup trend names
df_1619_D_TREND_values.columns = df_1619_D_TREND_values.columns.str.replace(r"\s*\(.*\)","")
df_1619_D_TREND_values.columns = df_1619_D_TREND_values.columns.str.replace(r"'","")
df_1619_D_TREND_values.columns = df_1619_D_TREND_values.columns.str.replace(r"\s-","")
#%%
df_1619_D_TREND_values.to_csv(r"..\..\Dati_BB\Walden 97-19\Dati Orion\Walden 16-19 [D+domande+m_p_int_voto].csv", index=False, encoding='utf-8-sig', sep = ';')

# %%
