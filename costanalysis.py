from utils import *
import pdb

def filterdata(data):
    filtered_data = data.filter(items = ['npi','generic_name', 'specialty_description','total_claim_count','total_drug_cost' ])
    #g = filtered_data.groupby(["npi","generic_name"])
    g = filtered_data.groupby(['npi', 'generic_name']).count()#[["total_claim_count", "total_drug_cost"]].sum()
    #aa has the count of unique ['npi', 'generic_name']
    aa = filtered_data.groupby(['npi','generic_name']).size().reset_index(name='count')
    #bb has the sum of total claim count amnd total grung cost for unique ['npi','generic_name']
    bb = filtered_data.groupby(['npi', 'generic_name','specialty_description'])['total_claim_count','total_drug_cost'].agg(['sum','count'])
    #for i in g.groups:
        #print i, "has", len(g.groups[i])
    pdb.set_trace() 

    return data


filename = "./data/PartD_Prescriber_PUF_NPI_Drug_16_short2.csv"
data = process_df(read_data(filename))
summarize(data)
train, val, test = split_df(data, [0.5,0.25,0.25])
summarize(data)
filterdata(data)