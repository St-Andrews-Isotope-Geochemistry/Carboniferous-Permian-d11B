import numpy,pandas,cbsyst
from geochemistry_helpers import Sampling,GaussianProcess

def calculate_d11BT_range(d11B4,epsilonB):
    return [(minimum,maximum) for minimum,maximum in zip(d11B4,calculate_d11BT_maximum(d11B4,epsilonB))]
def calculate_d11BT_maximum(d11B4,epsilonB):
    RBO4 = cbsyst.boron_isotopes.d11_to_R11(d11B4)
    alphaB = cbsyst.boron_isotopes.epsilon_to_alpha(epsilonB)

    maximum_RBOT = RBO4*alphaB
    return cbsyst.boron_isotopes.R11_to_d11(maximum_RBOT)

## Define any preprocessing functions
def importData(name=None):
    import pandas
    if name.lower()=="boron":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="A,M,N,P,S,W,X",names=["horizon","age","age_uncertainty","author","d18O","d11B4_uncertainty","d11B4"],sheet_name="Data")
    elif name.lower()=="strontium":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="C,D,M,N",names=["strontium","strontium_uncertainty","age","age_uncertainty"],sheet_name="Data")
    elif name.lower()=="calcium_magnesium":
        data = pandas.read_excel("./Data/Input/Boundary_Conditions.xlsx",header=1,usecols="A,D,E,F,G,H,I",names=["age","calcium","calcium_low","calcium_high","magnesium","magnesium_low","magnesium_high"],sheet_name="Seawater_Relevant")
    elif name.lower()=="oxygen":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="A,M,N,P,S,U",names=["horizon","age","age_uncertainty","author","d18O","notes"],sheet_name="Data")
    elif name.lower()=="carbon":
        data = pandas.read_excel("./Data/Input/Data_Compilation_SrCOB.xlsx",header=1,usecols="L,M,P",names=["age","age_uncertainty","d13C"],sheet_name="Data")
    return data
def recombineData(data,name):
    if name.lower()=="boron":
        data = data.groupby(["age","horizon"]).agg({'author':'first',
                                                    'd11B4':'mean',
                                                    'd11B4_uncertainty':'mean',
                                                    'd18O':'mean'}).reset_index()        
        data["d18O_uncertainty"] = 1
    elif name.lower()=="oxygen":
        data = data.groupby(["age","horizon"]).agg({'author':'first',
                                                    'd18O':'mean'}).reset_index()
        data = data.fillna({"d18O":data["d18O"].dropna().mean()})
    elif name.lower()=="strontium":
        pass
    return data
def refineData(data,name):
    if name.lower()=="boron":
        data["d18O"] = pandas.to_numeric(data["d18O"],errors='coerce')
        data = data.dropna(subset=["age","d11B4","d18O"])
    elif name.lower()=="strontium":
        data = data.dropna(subset=["age","strontium","strontium_uncertainty"])
    elif name.lower()=="oxygen":
        data = data[~(data["notes"]=="Anomalous d18O") | (data["notes"]=="Dubious d18O")]
        data = data[pandas.to_numeric(data["d18O"],errors='coerce').notnull()]
        data = data.dropna(subset=["age","d18O"])
    elif name.lower()=="carbon":
        data = data.dropna(subset=["age","d13C"])
    return data
def sortByAge(data):
    return data.sort_values("age",ascending=False).reset_index()

def getData(name=None):
    data = importData(name)
    data = refineData(data,name)
    data = recombineData(data,name)
    data = sortByAge(data)
    return data
def getd11BswRange(d11B4,epsilon,plot=False):
    from cbsyst import boron_isotopes

    d11Bsw_range = calculate_d11BT_range(d11B4,epsilonB=epsilon)
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
epsilon_x = numpy.arange(10,50,0.01)
number_of_samples = 11111

initial_dic_edges = [200,6000]
dic_edges = [200,50000]


# Load the data
data = getData("boron")

oxygen_data = getData("oxygen")
# epsilon = 27.2
epsilon_distribution_1 = Sampling.Distribution(epsilon_x,"Gaussian",(27.2,0.3)).normalise()
epsilon_distribution_2 = Sampling.Distribution(epsilon_x,"Gaussian",(26.0,0.5)).normalise()
epsilon_sampler = Sampling.Sampler(epsilon_x,"Manual",epsilon_distribution_1.probabilities+epsilon_distribution_2.probabilities,"Monte_Carlo").normalise()
epsilon_jitter_sampler = Sampling.Sampler(numpy.arange(-2,2,0.001),"Gaussian",(0,0.01),"Monte_Carlo").normalise()

interpolation_ages = [getData("boron")["age"].to_numpy(),equally_spaced_ages]

## Species calibration
species_gradient = 0.7735
species_intercept = 5.0936
species_function = lambda measured: (measured-species_intercept)/species_gradient
species_inverse_function = lambda borate: (borate*species_gradient)+species_intercept

## Define priors
# Can reuse the same one if the prior is the same across the time series
# Uncertainty for d11B4 is at 2std, so half it
d11B4_priors = [Sampling.Sampler(d11B_x,"Gaussian",(d11B4,d11B4_uncertainty/2),"Monte_Carlo",location=age).normalise() for age,d11B4,d11B4_uncertainty in zip(data["age"].values,data["d11B4"].values,data["d11B4_uncertainty"].values,strict=True)]
# d11B4_priors[11] = Sampling.Sampler(d11B_x,"Flat",(data["d11B4"][11]-data["d11B4_uncertainty"][11],data["d11B4"][11]+data["d11B4_uncertainty"][11]),"Monte_Carlo",location=data["age"][11]).normalise()
pH_prior = [Sampling.Distribution(pH_x,"Gaussian",(8,1)).normalise()]
co2_prior = [Sampling.Distribution(carbon_x,"Flat",(30,8000)).normalise()]
dic_prior = [Sampling.Sampler(carbon_x,"Flat",(dic_edges[0],dic_edges[1]),"Monte_Carlo").normalise()]

# strontium_scaling_prior = Sampling.Sampler(numpy.arange(-2,2,0.01),"Flat",(-1,1),"Monte_Carlo").normalise()
d11Bsw_scaling_prior = Sampling.Sampler(numpy.arange(-50,50,0.01),"Flat",(-40,40),"Monte_Carlo").normalise()
d11Bsw_scaling_initial = Sampling.Sampler(numpy.arange(-50,50,0.01),"Flat",(12.55-5,12.55+5),"Monte_Carlo").normalise()
d11Bsw_initial_initial = Sampling.Sampler(numpy.arange(-50,50,0.01),"Flat",(37.0-4.0,37.0+10.0),"Monte_Carlo").normalise()

d11Bsw_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,0.1),"Monte_Carlo").normalise()  
d11Bsw_initial_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,0.1),"Monte_Carlo").normalise()

d11B4_jitter_samplers = [Sampling.Sampler(numpy.arange(-1,1,0.01),"Gaussian",(0,uncertainty/20),"Monte_Carlo").normalise() for uncertainty in data["d11B4_uncertainty"].values]
pH_seed_jitter = [numpy.array([0.1]*len(group)) for group in interpolation_ages]

initial_dic_sampler = [Sampling.Sampler(carbon_x,"Flat",(initial_dic_edges[0],initial_dic_edges[1]),"Monte_Carlo").normalise()]
initial_dic_jitter_sampler = [Sampling.Sampler(numpy.arange(-1e4,1e4,10),"Gaussian",(0,50),"Monte_Carlo").normalise()]

dic_fraction_sampler = [Sampling.Sampler(numpy.arange(-5,5,0.01),"Flat",(-1,1),"Monte_Carlo").normalise()]
dic_fraction_jitter_sampler = [Sampling.Sampler(numpy.arange(-5,5,1e-3),"Gaussian",(0,0.05),"Monte_Carlo").normalise()]
