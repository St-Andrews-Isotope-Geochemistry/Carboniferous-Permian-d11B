import numpy
from matplotlib import pyplot

from geochemistry_helpers import Sampling,GaussianProcess
import preprocessing

from cbsyst import boron_isotopes,Csys
from cbsyst.helpers import Bunch
import kgen

number_of_samples = 1000
boron_data = preprocessing.data

temperature_calibration = lambda d18O : 15.7 - 4.36*((d18O-(-1))) + 0.12*((d18O-(-1))**2)


d18O_constraints = [Sampling.Sampler(preprocessing.d18O_x,"Gaussian",(d18O,uncertainty),method="Monte_Carlo",location=location).normalise() for d18O,uncertainty,location in zip(boron_data["d18O"].values,boron_data["d18O_uncertainty"].values,boron_data["age"].values,strict=True)]

pH_uncertainties = []
for d18O_constraint,d11B4_prior in zip(d18O_constraints,preprocessing.d11B4_priors,strict=True):
    d11B4_prior.getSamples(number_of_samples)
    d18O_constraint.getSamples(number_of_samples)
    temperature = temperature_calibration(d18O_constraint.samples)

    # A nominal d11Bsw
    Ks = Bunch(kgen.calc_Ks(TempC=numpy.mean(temperature)))
    d11Bsw = boron_isotopes.calculate_d11BT(8,Ks,d11B4_prior.values[0],27.2)

    Ks = Bunch(kgen.calc_Ks(TempC=temperature,Sal=35,Pres=0))
    pH = boron_isotopes.calculate_pH(Ks,d11Bsw,d11B4_prior.samples,27.2)

    pH_uncertainties += [numpy.nanstd(pH)]

numpy.savetxt("./Data/Output/pH_uncertainties.csv",pH_uncertainties)

a = 5


