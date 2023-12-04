# import relevant packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# %% define function
def XY_encoder(df, target, numeric_features = [], categorical_features = [], bool_features = []):
    #define encoders and scalers
    encoder_bin = OneHotEncoder(drop = 'first') #for binary features (drops one)
    encoder_cat  = OneHotEncoder() # for categorical features (no drops)
    MM_scaler = MinMaxScaler()

    X_0 = df.drop(columns = target)
    y = df[target]

    if len(numeric_features)>=1:
        X_numeric = X_0[numeric_features].reset_index(drop=True)
    else:
        X_numeric = pd.DataFrame()

    # dummy booleane
    if len(bool_features) >= 1:
        X_dummies_bool_0 = X_0[bool_features]
        encoder_bin.fit(X_dummies_bool_0)
        X_dummies_bool = encoder_bin.transform(X_dummies_bool_0)
        X_dummies_bool = pd.DataFrame(X_dummies_bool.toarray(), columns=encoder_bin.get_feature_names())
    else:
        X_dummies_bool = pd.DataFrame()

    # dummy categoriche
    if len(categorical_features) >= 1:
        X_dummies_cat_0 = X_0[categorical_features]
        encoder_cat.fit(X_dummies_cat_0)
        X_dummies_cat = encoder_cat.transform(X_dummies_cat_0)
        # trasforma in dataframe
        X_dummies_cat = pd.DataFrame(X_dummies_cat.toarray(), columns=encoder_cat.get_feature_names())
    else:
        X_dummies_cat = pd.DataFrame()


    # scala tutte le variabili con MinMaxScaler
    # aggiungi le non_dummy
    split_features = []
    for group in [X_numeric, X_dummies_bool, X_dummies_cat]:
        if group.shape[1] >= 1:
            split_features.append(group)

    X = pd.concat(split_features, axis=1)

    # scala tutto
    MM_scaler = MinMaxScaler().fit(X)
    X[X.columns] = MM_scaler.transform(X)

    return X, y


# %%
#%% class balancer
def binary_class_balancer(X, y):
    '''input X and y, then balances based on y, returning a new X and y'''

    #reassemble df from X features and y target
    df = pd.concat([X, y.reindex(X.index)], axis=1)
    ## Class count
    classcounts = df[y.name].value_counts().sort_values().reset_index()
    label_class_less, label_class_more = classcounts.iloc[:,0]
    count_class_less, count_class_more = classcounts.iloc[:,1]


    # Divide by class
    df_class_less = df[df[y.name] == label_class_less]
    df_class_more = df[df[y.name] == label_class_more]

    #
    df_class_more_unders = df_class_more.sample(count_class_less)
    df_test_under = pd.concat([df_class_more_unders, df_class_less], axis=0)

    X_unders = df_test_under.drop(columns=y.name)
    y_unders = df_test_under[y.name]

    print('Random under-sampling:')
    print(df_test_under[y.name].value_counts())
    df_test_under[y.name].value_counts().plot(kind='bar', title='Count (target)');
    
    return X_unders, y_unders

# %%
