
from __future__ import print_function
import pandas as pd, pdb, numpy as np, random
import numpy as np
import pandas as pd
#import seaborn as sns
import random
#import psycopg2
#from pandas.io import sql
import scipy
import sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# scikit learn
#from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

def read_data(filename, separator=','):
    #if 'partd' in filename.lower():
    with open(filename,"r") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split(separator))
    #pdb.set_trace()
    skiprows = [idx for idx,ln in enumerate(lines) if len(ln.split(','))!=num_cols]
    data = pd.read_csv(filename, sep = ",", skiprows = skiprows)
    #else:
        #data = pd.read_csv(filename, sep = ",")
    return data

def generate_split_idx(num_data_pts, split_ratio):
    #split_ratio is a list of floats that add upto 1
    assert np.isclose(sum(split_ratio), 1)
    assert all(map(lambda x : x>=0, split_ratio))
    cdf = np.cumsum(split_ratio)
    result = [[] for _ in range(len(split_ratio))]
    for i in range(num_data_pts):
        r = random.uniform(0, 1)
        for binid in range(len(split_ratio)):
            if cdf[binid] > r:
                break
        result[binid].append(i)
    return result

def process_df(data):
    drop_cols = ['nppes_provider_last_org_name', 'nppes_provider_first_name', 'nppes_provider_city', 'nppes_provider_state', 'drug_name']
    #drop colms etc
    #drop invalid rows etc
    data = data.drop(drop_cols, axis=1)
    return data

def summarize(data):
    print ("Number of rows in data set: ", len(data))
    print ("Columns in data set: ", data.columns.values)
    print ("Number of columns: ", len(data.columns.values))

def split_df(data, split_ratio):
    #pdb.set_trace()
    idx_lists = generate_split_idx(len(data), [0.5, 0.25, 0.25])
    return [data.iloc[idxs] for idxs in idx_lists]

def convert_categorical_datacols(data, colname):
    category_map = {}
    ctr = 0
    for item in data[colname]:
        if item not in category_map:
            category_map[item] = ctr
            ctr += 1
    return category_map

def compute_all_cat_maps(data):
    cat_map = {}
    for col in ['generic_name', 'specialty_description', 'label']:  #TODO, find and replace other categorical columns
        category_map = convert_categorical_datacols(data, col)
        cat_map[col] = category_map
    return cat_map

def transform_dataframe_to_numeric(data, colname, category_map):
    data[colname] = data[colname].apply(lambda x : category_map.get(x, -1))
    return data

def transform_all_cat_columns(data, cat_map):
    for colname in cat_map:
        data = transform_dataframe_to_numeric(data, colname, cat_map[colname])
    return data

def labelreturn(col):
    pass

def filterdata(data):
    grouppedata = data.groupby('npi').agg({'total_claim_count':'sum',
    'total_30_day_fill_count':'sum',
    'total_day_supply':'sum',
    'total_drug_cost':'sum',
    'bene_count_ge65':'sum',
    'total_claim_count_ge65':'sum',
    'total_30_day_fill_count_ge65':'sum',
    'total_day_supply_ge65':'sum',
    'total_drug_cost_ge65':'sum',
    'generic_name':'count',
    'specialty_description':'count',
    'label' : 'max'
    })
    #pdb.set_trace()
    return grouppedata


partd_file = "./data/PartD_Prescriber_PUF_NPI_Drug_16.csv"
npiexc_file = "./data/UPDATED.CSV"
npi_exclusion = read_data(npiexc_file)
orig = process_df(read_data(partd_file))
npi_exclusion.columns = map(str.lower, npi_exclusion.columns)
labelled_data = orig.assign(label = orig.npi.isin(npi_exclusion.npi))


summarize(labelled_data)
summarize(labelled_data)

cat_map = compute_all_cat_maps(labelled_data)
data = transform_all_cat_columns(labelled_data, cat_map)
data = filterdata(data)
print("Number of non-frauds ", len(data.loc[data['label'] == 0]))
print("Number of frauds", len(data.loc[data['label'] == 1])) ##dataframe that have rows where label =1
data_label_true = data.loc[data['label'] == 1]
data_label_false = data.loc[data['label'] == 0]
data_label_false = data_label_false.head(50)
df_concat = pd.concat([data_label_true, data_label_false], axis=0)

#train, val, test = split_df(data, [0.5,0.25,0.25])
#train, val, test = split_df(df_concat, [0.5,0.25,0.25])
train = test = data

y_train = train["label"].values
X_train = train.loc[:, train.columns != 'label']

#clf1 = LogisticRegression() # pimp me
#clf2 = RandomForestClassifier(n_estimators =100, max_depth = 10, class_weight = 'auto'
#clf1.fit(x_train,y_train)
#clf2.fit(x_train.toarray(),y_train)
clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
y_scores = clf.predict_proba(X_train)

pdb.set_trace()
print("Accuracy: ", accuracy_score(y_train, y_pred))
print("F1_score", f1_score(y_train, y_pred))
print("roc ", roc_auc_score(y_train, y_scores))
pdb.set_trace()


