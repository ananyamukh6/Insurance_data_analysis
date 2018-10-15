import pdb
def readData(filename):
    file = open(filename,"r")
    data = file.readlines()
    #pdb.set_trace()
    '''
    for row in range(len(data)):
        #pdb.set_trace()
       # for i in range(len(data[row])):
        print data[row]
        data = data.split("\n")
    '''
    file.close()
    return data

filename = "/Users/ananymuk/Desktop/Ananya/OracleTrainingLab/capstone/code1/data/PartD_Prescriber_PUF_NPI_Drug_16/PartD_Prescriber_PUF_NPI_Drug_16_short2.csv"
data = readData(filename)
for rownum in range(len(data)):
    pdb.set_trace()
    data += data.split("\n")
print data[0]
print data[1]
print data[599]
pdb.set_trace()
print (data[1])
