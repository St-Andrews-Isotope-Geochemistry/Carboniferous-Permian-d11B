import numpy
from geochemistry_helpers import Sampling,GaussianProcess

## Define any preprocessing functions
def importData(name=None):
    import pandas
    if name.lower()=="boron":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="A,L,M,O,Q,V,U",names=["horizon","age","age_uncertainty","author","d18O","d11B4_uncertainty","d11B4"],sheet_name="Data")
    elif name.lower()=="strontium":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="C,D,L,M",names=["strontium","strontium_uncertainty","age","age_uncertainty"],sheet_name="Data")
    elif name.lower()=="calcium_magnesium":
        data = pandas.read_excel("./Data/Input/Boundary_Conditions.xlsx",header=1,usecols="A,D,G",names=["age","calcium","magnesium"],sheet_name="Seawater_Relevant")
    return data
def recombineData(data,name):
    if name.lower()=="boron":
        data = data.groupby(["age","horizon"]).mean().reset_index()
        # data["d18O_uncertainty"] = 1 # Assumed
        data["d18O_uncertainty"] = numpy.where(data["d18O"].isnull(),1,0.2)
        data = data.fillna({"d18O":data["d18O"].dropna().mean()})
    elif name.lower()=="strontium":
        pass
    return data
def refineData(data,name):
    if name.lower()=="boron":
        data = data.dropna(subset=["age","d11B4"])
    elif name.lower()=="strontium":
        data = data.dropna(subset=["age","strontium","strontium_uncertainty"])
    return data
def sortByAge(data):
    return data.sort_values("age",ascending=False).reset_index()

def getData(name=None):
    data = importData(name)
    data = recombineData(data,name)
    data = refineData(data,name)
    data = sortByAge(data)
    return data
def getd11BswRange(d11B4,plot=False):
    from cbsyst import boron_isotopes

    d11Bsw_range = boron_isotopes.calculate_d11BT_range(d11B4)
    return d11Bsw_range

## Define any variables
equally_spaced_ages = numpy.arange(340,259.9,-0.1)
strontium_x = numpy.arange(0.7,0.71,1e-5)
temperature_x = numpy.arange(-50,50,0.1)
d18O_x = numpy.arange(-50,50,0.01)
pH_x = numpy.arange(1,14,0.01)
carbon_x = numpy.arange(-1e5,1e6,10)
carbon_logx = numpy.arange(-50,50,0.01)
d11B_x = numpy.arange(-50,100,0.1)
saturation_state_x = numpy.arange(0,1000,0.01)
number_of_samples = 555

initial_dic_edges = [200,6000]
dic_edges = [200,50000]


# Load the data
data = getData("boron")
epsilon = 27.2

interpolation_ages = [getData("boron")["age"].to_numpy(),equally_spaced_ages]

## Species calibration
species_gradient = 0.7735
species_intercept = 5.0936
species_function = lambda measured: (measured-species_intercept)/species_gradient
species_inverse_function = lambda borate: (borate*species_gradient)+species_intercept

## Define priors
# Can reuse the same one if the prior is the same across the time series
d11B4_priors = [Sampling.Sampler(d11B_x,"Gaussian",(d11B4,d11B4_uncertainty),"Monte_Carlo",location=age).normalise() for age,d11B4,d11B4_uncertainty in zip(data["age"].to_numpy(),data["d11B4"].to_numpy(),data["d11B4_uncertainty"].to_numpy(),strict=True)]
pH_prior = [Sampling.Distribution(pH_x,"Gaussian",(8,1)).normalise()]
co2_prior = [Sampling.Distribution(carbon_x,"Flat",(50,8000)).normalise()]
dic_prior = [Sampling.Sampler(carbon_x,"Flat",(dic_edges[0],dic_edges[1]),"Monte_Carlo").normalise()]

strontium_scaling_prior = Sampling.Sampler(numpy.arange(-2,2,0.01),"Flat",(-1,1),"Monte_Carlo").normalise()
d11Bsw_scaling_prior = Sampling.Sampler(numpy.arange(-50,50,0.01),"Flat",(-40,40),"Monte_Carlo").normalise()

d11B4_jitter_samplers = [Sampling.Sampler(numpy.arange(-1,1,0.01),"Gaussian",(0,uncertainty/20),"Monte_Carlo").normalise() for uncertainty in data["d11B4_uncertainty"].to_numpy()]

initial_dic_sampler = [Sampling.Sampler(carbon_x,"Flat",(initial_dic_edges[0],initial_dic_edges[1]),"Monte_Carlo").normalise()]
initial_dic_jitter_sampler = [Sampling.Sampler(numpy.arange(-1e4,1e4,10),"Gaussian",(0,100),"Monte_Carlo").normalise()]

dic_fraction_sampler = [Sampling.Sampler(numpy.arange(-5,5,0.01),"Flat",(-1,2),"Monte_Carlo").normalise()]
dic_fraction_jitter_sampler = [Sampling.Sampler(numpy.arange(-5,5,1e-3),"Gaussian",(0,0.05),"Monte_Carlo").normalise()]
