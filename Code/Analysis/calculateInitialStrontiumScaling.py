import numpy

from processStrontium import makeStrontiumGP,generateNormalisedStrontium
import preprocessing
from geochemistry_helpers import Sampling,GaussianProcess

from cbsyst import boron_isotopes,Csys
from cbsyst.helpers import Bunch
import kgen

from matplotlib import pyplot

normalised_strontium_gp = generateNormalisedStrontium()
strontium_shapes = normalised_strontium_gp.means

# Draw samples of d11B4 and epsilon
d11B4 = numpy.array([d11B4_prior.getSamples(1).samples[0] for d11B4_prior in preprocessing.d11B4_priors])
epsilon = preprocessing.epsilon_sampler.getSamples(1).samples[0]
# d11B4_likelihood = numpy.sum([d11B4_prior.getLogProbability(d11B4_sample) for d11B4_prior,d11B4_sample in zip(preprocessing.d11B4_priors,d11B4,strict=True)])

# Given d11B4 we can say which d11Bsw's are valid
d11Bsw_range = preprocessing.getd11BswRange(d11B4,epsilon)
d11Bsw_priors = [Sampling.Sampler(preprocessing.d11B_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(preprocessing.data["age"].to_numpy(),d11Bsw_range,strict=True)]

# The first one of those is our tiepoint
d11Bsw_initial_prior = d11Bsw_priors[0]

# Get samples of the initial value and the scaling for Sr->d11Bsw
d11Bsw_initial = d11Bsw_initial_prior.getSamples(1).samples[0]
d11Bsw_scalings = preprocessing.d11Bsw_scaling_prior.getSamples(10000).samples

# Get the probabilities of d11Bsw parameters
# d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability([d11Bsw_initial])
# d11Bsw_scaling_probability = preprocessing.d11Bsw_scaling_prior.getProbability([d11Bsw_scaling])

# Calculate d11Bsw - interpolated
d11Bsws = [[d11Bsw_initial+(d11Bsw_scaling/2)*numpy.squeeze(shape) for shape in strontium_shapes] for d11Bsw_scaling in d11Bsw_scalings]


pyplot.plot(preprocessing.interpolation_ages[0],d11B4)
difference_reference = 1e12
best_parameter = None
for index,d11Bsw in enumerate(d11Bsws):
    d11B4_mean = numpy.mean(d11B4)
    d11Bsw_mean = numpy.mean(d11Bsw[-1])

    offset = d11B4_mean-d11Bsw_mean

    d11Bsw_offset = d11Bsw[0]+offset

    difference = numpy.sum((d11Bsw_offset-d11B4)**2)

    if difference<difference_reference:
        difference_reference = difference
        best_parameter = d11Bsw_scalings[index]
        
        d11Bsw_offset = d11Bsw[-1]+offset

best_d11Bsw = [d11Bsw_initial+(best_parameter/2)*numpy.squeeze(shape) for shape in strontium_shapes]
d11Bsw_mean = numpy.mean(best_d11Bsw[-1])
offset = d11B4_mean-d11Bsw_mean
d11Bsw_offset = best_d11Bsw[-1]+offset

# This gives us the scaling, but what about the offset
# If we want pH approximately of 8 for the background
d11B4_distribution = Sampling.Distribution.fromSamples(d11B4).normalise()
d11B4_quartile = d11B4_distribution.quantile([0.75])[0]

Kb = Bunch({"KB":kgen.calc_K("KB",TempC=21,Ca=0.0188,Mg=0.0576)})
d11Bsw_absolute = boron_isotopes.calculate_d11BT(8.0,Kb,d11B4_quartile,27.2)

pyplot.plot(preprocessing.interpolation_ages[-1],d11Bsw_offset)
pyplot.savefig("test.png")

a = 5