import numpy
from geochemistry_helpers import Sampling,GaussianProcess

## Define any preprocessing functions
def importData(name=None):
    import pandas
    if name.lower()=="boron":
        data = pandas.read_excel("./Data/Our_Samples.xlsx",usecols="A:B",sheet_name="Matlab")
    elif name.lower()=="strontium":
        data = pandas.read_excel("./Data/Data_Compilation_SrCOB.xlsx",usecols="D:E,K:L",sheet_name="Matlab")
        data = data.rename(columns={"Age_Simulated":"Age"})
    return data
def refineData(data,name):
    if name.lower()=="boron":
        data = data.dropna(subset=["Age","d11B4"]).sort_values("Age").reset_index()
    elif name.lower()=="strontium":
        data = data.dropna(subset=["Age","Sr87","Sr87_SD"]).sort_values("Age").reset_index()
    return data
def sortByAge(data):
    return data.sort_values("Age")

def getData(name=None):
    data = importData(name)
    data = refineData(data,name)
    data = sortByAge(data)
    return data
def getd11BswRange(d11B4,plot=False):
    from cbsyst import boron_isotopes

    d11Bsw_range = boron_isotopes.calculate_d11BT_range(d11B4)
    return d11Bsw_range

## Define any variables
equally_spaced_ages = numpy.arange(250,330,0.1)
strontium_x = numpy.arange(0.7,0.71,1e-5)
pH_x = numpy.arange(1,14,0.01)
carbon_x = numpy.arange(1,100000,1)
d11Bsw_x = numpy.arange(0,100,0.1)
number_of_samples = 500

dic_scenarios = [2000,6000,10000]


# Load the data
data = getData("boron")
d11B4_uncertainty = 0.3
epsilon = 27.2

interpolation_ages = [getData("boron")["Age"].to_numpy(),equally_spaced_ages]

d11Bsw_range = getd11BswRange(data["d11B4"].to_numpy())

## Define priors
# Can reuse the same one if the prior is the same across the time series
d11B4_priors = [Sampling.Sampler(d11Bsw_x,"Gaussian",(d11B4,d11B4_uncertainty),"Monte_Carlo",location=age).normalise() for age,d11B4 in zip(data["Age"].to_numpy(),data["d11B4"].to_numpy(),strict=True)]
pH_prior = [Sampling.Distribution(pH_x,"Gaussian",(8,1)).normalise()]
co2_prior = [Sampling.Distribution(carbon_x,"Flat",(100,20000)).normalise()]
d11Bsw_priors = [Sampling.Sampler(d11Bsw_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(data["Age"].to_numpy(),d11Bsw_range,strict=True)]
d11Bsw_initial_prior = d11Bsw_priors[0]
dic_prior = Sampling.Sampler(carbon_x,"Flat",(1000,10000),"Monte_Carlo").normalise()

d11Bsw_scaling_prior = Sampling.Sampler(numpy.arange(-50,50,0.01),"Flat",(-40,40),"Monte_Carlo").normalise()

