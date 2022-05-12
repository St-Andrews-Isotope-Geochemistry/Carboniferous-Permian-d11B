import numpy

## Define any preprocessing functions
def importData(name=None):
    import pandas
    data = pandas.read_excel("./Data/Data_Compilation_SrCOB.xlsx",usecols="D:E,I:L,O:Q,S:U,W")
    data = data.rename(columns={"Age_Simulated":"Age"})
    return data
def refineData(data,name):
    if name.lower()=="boron":
        data = data.dropna(subset=["d11B","d11B_Reproducibility","d11B_Uncertainty","d11B_Borate","Age"]).sort_values("Age").reset_index()
    elif name.lower()=="strontium":
        data = data.dropna(subset=["Sr87","Sr87_SD","Age"]).sort_values("Age").reset_index()
    return data
def sortByAge(data):
    return data.sort_values("Age")

def getData(name=None):
    data = importData(name)
    data = refineData(data,name)
    data = sortByAge(data)
    return data

## Define any variables
equally_spaced_ages = numpy.arange(230,330,0.1)
strontium_x = numpy.arange(0.7,0.71,1e-5)
pH_x = numpy.arange(1,14,0.01) 
d11Bsw_x = numpy.arange(-100,100,0.1)
