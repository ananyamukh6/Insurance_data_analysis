from __future__ import print_function
import pandas as pd, pdb, numpy as np, random

def read_data(filename, separator=','):
    with open(filename,"r") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split(separator))
    skiprows = [idx for idx,ln in enumerate(lines) if len(ln.split(','))!=num_cols]
    data = pd.read_csv(filename, sep = ",", skiprows = skiprows)
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
    drop_cols = ['nppes_provider_last_org_name', 'nppes_provider_first_name', 'nppes_provider_city', 'nppes_provider_state']
    #drop colms etc
    #drop invalid rows etc
    data = data.drop(drop_cols, axis=1)
    return data

def summarize(data):
    print ("Number of rows in data set: ", len(data))
    print ("Columns in data set: ", data.columns.values)
    print ("Number of columns: ", len(data.columns.values))

def split_df(data, split_ratio):
    idx_lists = generate_split_idx(len(data), [0.5, 0.25, 0.25])
    return [data.loc[idxs] for idxs in idx_lists]

def convert_categorical_datacols(data, colname):
    category_map = {}
    ctr = 0
    for item in data[colname]:
        if item not in category_map:
            category_map[item] = ctr
            ctr += 1
    return category_map

def compute_all_cat_maps(data):
    #convert categorical columns
    cat_map = {}
    for col in ['generic_name']:  #TODO, find and replace other categorical columns
        category_map = convert_categorical_datacols(data, col)
        cat_map[col] = category_map
    return cat_map

def transform_dataframe_to_numeric(data, colname, category_map):
    pdb.set_trace()
    data[colname] = data[colname].apply(lambda x : category_map.get(x, -1))
    return data

def transform_all_cat_columns(data, cat_map):
    for colname in cat_map:
        data = transform_dataframe_to_numeric(data, colname, cat_map[colname])
    return data


filename = "/Users/ananymuk/Desktop/Ananya/OracleTrainingLab/capstone/code1/Insurance_data_analysis/data/PartD_Prescriber_PUF_NPI_Drug_16_short2.csv"
data = process_df(read_data(filename))
summarize(data)
train, val, test = split_df(data, [0.5,0.25,0.25])
summarize(data)

cat_map = compute_all_cat_maps(data)
data = transform_all_cat_columns(data, cat_map)

pdb.set_trace()







