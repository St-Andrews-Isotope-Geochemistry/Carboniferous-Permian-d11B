import numpy
from matplotlib import pyplot

from geochemistry_helpers import Sampling,GaussianProcess

from cbsyst import boron_isotopes,Csys
from cbsyst.helpers import Bunch
import kgen

from processStrontium import makeStrontiumGP,generateNormalisedStrontium
import preprocessing,processCalciumMagnesium

numpy.seterr(all="ignore")

strontium_gp = makeStrontiumGP()
normalised_strontium_gp = generateNormalisedStrontium()

strontium_shapes = normalised_strontium_gp.means

d18O = preprocessing.data["d18O"].to_numpy()
temperature = 15.7 - 4.36*((d18O-(-1))) + 0.12*((d18O-(-1))**2)
temperature_constraints = [Sampling.Distribution(preprocessing.temperature_x,"Gaussian",(temperature,0.1),location=location).normalise() for temperature,location in zip(temperature,preprocessing.data["age"].to_numpy(),strict=True)]
temperature_gp,temperature_mean_gp = GaussianProcess().constrain(temperature_constraints).removeLocalMean(fraction=(10,2))
temperature_gp = temperature_gp.setKernel("rbf",(5,10),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(temperature_mean_gp)

calcium_gp,magnesium_gp = processCalciumMagnesium.calculateCalciumMagnesium()

# Hoist out of preprocessing
# strontium_scaling_prior = preprocessing.strontium_scaling_prior
d11Bsw_priors = None
d11Bsw_initial_prior = None
d11Bsw_scaling_prior = preprocessing.d11Bsw_scaling_prior
pH_prior = preprocessing.pH_prior
co2_prior = preprocessing.co2_prior
dic_prior = preprocessing.dic_prior

d11Bsw_initial_jitter_sampler = preprocessing.d11Bsw_initial_jitter_sampler
d11Bsw_scaling_jitter_sampler = preprocessing.d11Bsw_scaling_jitter_sampler

scaling_edges = numpy.arange(-1,1e3,0.1)
calcium_upscaling_prior = Sampling.Distribution(scaling_edges,"Flat",(0,1.25))
calcium_downscaling_prior = Sampling.Distribution(scaling_edges,"Flat",(0,1.25))

pH_uncertainties = numpy.loadtxt("./Data/Output/pH_uncertainties.csv")

# Hoist out of start loop
def iterated11B4():
    d11B4_accept = False
    while not d11B4_accept:
        d11B4_jitter = numpy.array([d11B4_jitter_sampler.getSamples(1).samples[0] for d11B4_jitter_sampler in preprocessing.d11B4_jitter_samplers])
        d11B4 = markov_chain.final("d11B4_original")+d11B4_jitter
        d11B4_likelihood = numpy.sum([d11B4_prior.getLogProbability(d11B4_sample) for d11B4_prior,d11B4_sample in zip(preprocessing.d11B4_priors,d11B4,strict=True)])

        if d11B4_likelihood<markov_chain.final("probability"):
            probability_difference = 10**(d11B4_likelihood-markov_chain.final("probability"))
            r = numpy.random.random(1)[0]
            if r>probability_difference:
                continue
        
        d11B4_accept = True
    return d11B4

def getInitialSample():
    print("Getting initial sample")
    
    markov_chain = Sampling.MarkovChain()
    initial_sample = Sampling.MarkovChainSample()
    while True:
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
        d11Bsw_initial = preprocessing.d11Bsw_initial_initial.getSamples(1).samples[0]
        d11Bsw_scaling = preprocessing.d11Bsw_scaling_initial.getSamples(1).samples[0]

        # Get the probabilities of d11Bsw parameters
        d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability([d11Bsw_initial])
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability([d11Bsw_scaling])

        # Can already reject the samples if they're 0 - I think this only applies in the iterate step
        # Also the priors are both flat, so don't need to be accounted for in the MCMC
        # if d11Bsw_scaling_probability==0 or d11Bsw_initial_probability==0:
        #     continue
        
        # Calculate d11Bsw - interpolated
        d11Bsws = [d11Bsw_initial+(d11Bsw_scaling/2)*numpy.squeeze(shape) for shape in strontium_shapes]

        # Calculate probability where we have datapoints
        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability([d11Bsw]) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws[0],strict=True)]))
        
        # As we've translated and scaled Sr, it's possible that some d11Bsw's will be impossible
        # If any are impossible reject sample
        if any(d11Bsw_probability==0):
            continue
    
        # Now that we have a valid d11Bsw - calculate pH (which needs Kb)
        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws[0],d11B4,epsilon)

        # Create constraints from the pH values using precalculated pH uncertainty
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,pH_uncertainty),location=location).normalise() for pH_value,location,pH_uncertainty in zip(pH_values,preprocessing.data["age"].values,pH_uncertainties,strict=True)]

        # Put the constraints into a two stage Gaussian process (removing very long term fluctuations and prescribed smoothness for remainder)
        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
        pH_gp = pH_gp.setKernel("rbf",(0.1,3),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
        
        # Generate a seed and use it to draw samples
        pH_seed = pH_gp.generateSeed()
        pH_gp.getSamples(1,seed=pH_seed)

        # Ensure none of the predictions for pH are NaN
        for pH_sample in pH_gp.samples:
            if numpy.any(pH_sample):
                continue
        
        # Given we now have a valid pH evolution, draw samples for starting point and scaling of DIC
        initial_dic = preprocessing.initial_dic_sampler[0].getSamples(1).samples
        dic_fraction = preprocessing.dic_fraction_sampler[0].getSamples(1).samples

        # Calculate what would happen at a constant DIC, and how much CO2 would change relative to the first datapoint
        csys_constant_dic = [Csys(pHtot=numpy.squeeze(pH),DIC=initial_dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=numpy.squeeze(calcium),Mg=numpy.squeeze(magnesium),unit="mol") for pH,temperature,calcium,magnesium in zip(pH_gp.means,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
        relative_co2 = [(csys["CO2"]/csys_constant_dic[0]["CO2"][0])-1 for csys in csys_constant_dic]
        
        # Convert the relative CO2 to absolute DIC
        dic_series = [initial_dic+relative*dic_fraction*initial_dic for relative in relative_co2]
        # Print out for diagnostics
        print("DIC: "+str(numpy.min(dic_series[-1])) +","+ str(initial_dic[0]) +","+ str(numpy.max(dic_series[-1]))+","+str(dic_fraction[0]))

        # Check all the predicted DIC's are resonable (positive and less than 1 million micromol/kg)
        skip = False
        for dic_evolution in dic_series:
            dic_sensible = numpy.all(dic_evolution>0) and numpy.all(dic_evolution<1e6)
            if not dic_sensible:
                skip = True
        if skip:
            continue

        # Now we have valid pH and DIC evolutions, quantify the carbonate system
        csys_variable_dic = [Csys(pHtot=numpy.squeeze(pH),DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=numpy.squeeze(calcium),Mg=numpy.squeeze(magnesium),unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
        
        # Calculate saturation state
        omega = (csys_variable_dic[-1].Ca*csys_variable_dic[-1].CO3)/csys_variable_dic[-1].Ks.KspC
        
        # Calculate how much calcium would have to be wrong to give minimum saturation state of 0.5 and maximum of 15
        # Note this isn't perfect as we already needed calcium above and used a very poorly constrained fit (as this is all that's available)
        calcium_downscaling = numpy.max(omega)/15
        calcium_upscaling = 0.5/numpy.min(omega)

        # Quantify how likely the scalings are
        calcium_downscaling_probability = calcium_downscaling_prior.getProbability([calcium_downscaling])
        calcium_upscaling_probability = calcium_upscaling_prior.getProbability([calcium_upscaling])
        
        # Extract the parameters - in convenient units
        pH = [csys["pHtot"] for csys in csys_variable_dic]
        dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
        co2 = [csys["pCO2"]*1e6 for csys in csys_variable_dic]

        pH_distribution = Sampling.Distribution.fromSamples(pH[0],bin_edges=preprocessing.pH_x).normalise()
        
        # Get the probability of each of the parameters
        epsilon_probability = preprocessing.epsilon_sampler.getProbability([epsilon])
        pH_probability = pH_prior[0].getProbability(pH[0])
        dic_probability = dic_prior[0].getProbability(dic[0]) 
        co2_probability = co2_prior[0].getProbability(co2[0])

        # Reject the sample if any are 0
        if any(dic_probability==0) or any(pH_probability==0) or any(co2_probability==0) or calcium_downscaling_probability==0 or calcium_upscaling_probability==0:
            continue

        # Now that everything is valid, we can calculate some dependent properties 
        # Includes recalculation for updated calcium          
        # Convert the calcium into its downscaled/upscaled value
        if calcium_downscaling>1:
            calcium_adjusted = [calcium/calcium_downscaling for calcium in calcium_gp.means]
        if calcium_upscaling>1:
            calcium_adjusted = [calcium*calcium_upscaling for calcium in calcium_gp.means]

        # Also processing at temperature=25 degrees to eliminate site specific temperature effect
        Kb_25 = [Bunch({"KB":kgen.calc_K("KB",TempC=25,Ca=calcium,Mg=magnesium)}) for calcium,magnesium in zip(calcium_adjusted,magnesium_gp.means,strict=True)]
        # Then convert the interpolated pH to interpolated d11B4
        d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb,d11Bsw,epsilon) for pH,d11Bsw,Kb in zip(pH_gp.means,d11Bsws,Kb_25,strict=True)]
        # And d11B4 into what would hypothetically have been measured
        d11B_measured_interpolated = [preprocessing.species_inverse_function(d11B4) for d11B4 in d11B4_interpolated]
        
        # Recalculate pH constraints with new calcium
        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_adjusted,magnesium_gp.means,strict=True)]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws[0],d11B4,epsilon)
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,pH_uncertainty),location=location).normalise() for pH_value,location,pH_uncertainty in zip(pH_values,preprocessing.data["age"].values,pH_uncertainties,strict=True)]

        # Redo the full GP for interpolated ages
        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
        pH_gp = pH_gp.setKernel("rbf",(0.1,3),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
        
        # Use the same seed to draw samples
        pH_gp.getSamples(1,seed=pH_seed)

        # Redo carbonate chemistry calculations
        csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_adjusted,magnesium_gp.means,strict=True)]
    
        # Pull out variables in typical units
        pH_to_store = pH_gp.samples
        dic_to_store = [csys["DIC"]*1e6 for csys in csys_variable_dic]
        co2_to_store = [csys["pCO2"]*1e6 for csys in csys_variable_dic]
        omega_to_store = [(csys.Ca*csys.CO3)/csys.Ks.KspC for csys in csys_variable_dic]
        calcium_to_store = [calcium*1e3 for calcium in calcium_adjusted]
        magnesium_to_store = [magnesium*1e3 for magnesium in magnesium_gp.means]
        co3_to_store = [csys.CO3*1e6 for csys in csys_variable_dic]
        alkalinity_to_store = [csys.TA*1e6 for csys in csys_variable_dic]

        # Calculate sample probability
        # Don't need DIC or CO2 cumulative because they're flat distributions, either 0 or equal everywhere
        d11Bsw_log_probability = sum(numpy.log(d11Bsw_probability))
        pH_log_probability = sum(numpy.log(pH_probability))
        # pH_quantile_probability = numpy.log(pH_prior[0].getProbability(pH_distribution.quantile([0.75])[0]))
        epsilon_log_probability = numpy.log(epsilon_probability)        
        cumulative_probability = d11Bsw_log_probability+pH_log_probability+epsilon_log_probability

        # Create the sample
        initial_sample = (initial_sample.addField("probability",cumulative_probability,precision=5)
                                        .addField("d11Bsw",d11Bsws,precision=2)
                                        .addField("d11B4_original",d11B4,precision=2)
                                        .addField("d11B4_original_probability",d11B4,precision=5)
                                        .addField("d11B4",d11B4_interpolated,precision=2)
                                        .addField("d11Bm",d11B_measured_interpolated,precision=2)
                                        .addField("epsilon",epsilon,precision=5)
                                        .addField("pH",pH_to_store,precision=3)
                                        .addField("pH_seed",pH_seed,precision=5)
                                        .addField("co2",co2_to_store,precision=0)
                                        .addField("dic",dic_to_store,precision=0)
                                        .addField("d11Bsw_initial",d11Bsw_initial,precision=2)
                                        .addField("d11Bsw_scaling",d11Bsw_scaling,precision=3)
                                        .addField("dic_initial",initial_dic[0],precision=0)
                                        .addField("dic_fraction",dic_fraction[0],precision=3)
                                        .addField("omega",omega_to_store,precision=2)
                                        .addField("calcium",calcium_to_store,precision=2)
                                        .addField("magnesium",magnesium_to_store,precision=2)
                                        .addField("co3",co3_to_store,precision=0)
                                        .addField("alkalinity",alkalinity_to_store,precision=0))
        # Return a chain with this sample in it
        return markov_chain.addSample(initial_sample)

def iterate(markov_chain,number_of_samples):
    while len(markov_chain)<number_of_samples:
        # Create an empty Markov Chain sample
        current_sample = Sampling.MarkovChainSample()

        # Iterate d11B4 on its own Markov Chain to draw reasonable samples
        d11B4 = iterated11B4()

        # Update epsilon
        epsilon_jitter = preprocessing.epsilon_jitter_sampler.getSamples(1).samples[0]
        epsilon = markov_chain.final("epsilon")+epsilon_jitter

        # Given d11B4 we can say which d11Bsw's are valid
        d11Bsw_range = preprocessing.getd11BswRange(d11B4,epsilon)
        d11Bsw_priors = [Sampling.Sampler(preprocessing.d11B_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(preprocessing.data["age"].to_numpy(),d11Bsw_range,strict=True)]
        
        # The first one of those is our tiepoint
        d11Bsw_initial_prior = d11Bsw_priors[0]

        # Update values of the initial value and the scaling for Sr->d11Bsw        
        d11Bsw_initial_jittered = markov_chain.final("d11Bsw_initial") + d11Bsw_initial_jitter_sampler.getSamples(1).samples[0]
        d11Bsw_scaling_jittered = markov_chain.final("d11Bsw_scaling") + d11Bsw_scaling_jitter_sampler.getSamples(1).samples[0]
        
        # Get the probabilities of d11Bsw parameters
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability([d11Bsw_scaling_jittered])
        d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability([d11Bsw_initial_jittered])

        # Can already reject the samples if they're 0 - this is possible if they've been jittered out of their range
        # Also the priors are both flat, so don't need to be accounted for in the MCMC
        if d11Bsw_scaling_probability==0 or d11Bsw_initial_probability==0:
            continue

        # Calculate d11Bsw - interpolated
        d11Bsws = [d11Bsw_initial_jittered+(d11Bsw_scaling_jittered/2)*numpy.squeeze(shape) for shape in strontium_shapes]

        # Calculate probability where we have datapoints
        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability([d11Bsw]) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws[0],strict=True)]))

        # As we've translated and scaled Sr, it's possible that some d11Bsw's will be impossible
        # If any are impossible reject sample
        if any(d11Bsw_probability==0):
            continue

        # Use calcium and magnesium from prior sample as a guesstimate - convert units to mol/kg
        calcium_estimate = [calcium/1e3 for calcium in markov_chain.final("calcium")]
        magnesium_estimate = [magnesium/1e3 for magnesium in markov_chain.final("magnesium")]

        # Now that we have a valid d11Bsw - calculate pH (which needs Kb)
        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_estimate,magnesium_estimate,strict=True)]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws[0],d11B4,epsilon)

        # Create constraints from the pH values using precalculated pH uncertainty
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,pH_uncertainty),location=location).normalise() for pH_value,location,pH_uncertainty in zip(pH_values,preprocessing.data["age"].values,pH_uncertainties,strict=True)]

        # Put the constraints into a two stage Gaussian process (removing very long term fluctuations and prescribed smoothness for remainder)
        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
        pH_gp = pH_gp.setKernel("rbf",(0.1,3),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
        
        # Generate a seed and use it to draw samples
        pH_seed = pH_gp.perturbSeed(markov_chain.final("pH_seed"),preprocessing.pH_seed_jitter)
        pH_gp.getSamples(1,seed=pH_seed)
        
        # Ensure none of the predictions for pH are NaN
        for pH_sample in pH_gp.samples:
            if numpy.any(pH_sample):
                continue

        # Take samples for initial DIC and DIC scaling jitter
        initial_dic_jitter = preprocessing.initial_dic_jitter_sampler[0].getSamples(1).samples
        dic_fraction_jitter = preprocessing.dic_fraction_jitter_sampler[0].getSamples(1).samples

        # Calculate updated DIC parameters
        initial_dic = markov_chain.final("dic_initial") + initial_dic_jitter
        dic_fraction = markov_chain.final("dic_fraction") + dic_fraction_jitter

        # Get the probability of the jittered values - note these are flat so don't need to be in MCMC
        initial_dic_probability = preprocessing.initial_dic_sampler[0].getProbability(initial_dic)
        dic_fraction_probability = preprocessing.dic_fraction_sampler[0].getProbability(dic_fraction)

        # If either parameter is impossible then we must reject this sample
        if not (initial_dic_probability>0 and dic_fraction_probability>0):
            continue

        # Calculate what would happen at a constant DIC, and how much CO2 would change relative to the first datapoint
        csys_constant_dic = [Csys(pHtot=numpy.squeeze(pH),DIC=initial_dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=numpy.squeeze(calcium),Mg=numpy.squeeze(magnesium),unit="mol") for pH,temperature,calcium,magnesium in zip(pH_gp.means,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
        relative_co2 = [(csys["CO2"]/csys_constant_dic[0]["CO2"][0])-1 for csys in csys_constant_dic]
         
        # Convert the relative CO2 to absolute DIC
        dic_series = [initial_dic+relative*dic_fraction*initial_dic for relative in relative_co2]

        skip = False
        for dic_evolution in dic_series:
            dic_sensible = numpy.all(dic_evolution>0) and numpy.all(dic_evolution<1e6)
            if not dic_sensible:
                skip = True
        if skip:
            continue
            
        # Now we have valid pH and DIC evolutions, quantify the carbonate system
        csys_variable_dic = [Csys(pHtot=numpy.squeeze(pH),DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=numpy.squeeze(calcium),Mg=numpy.squeeze(magnesium),unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]

        # Calculate saturation state - using original calcium as this is for down/upscaling
        omega = (calcium_gp.means[-1]*csys_variable_dic[-1].CO3)/csys_variable_dic[-1].Ks.KspC

        # Calculate how much calcium would have to be wrong to give minimum saturation state of 0.5 and maximum of 15
        # Note this isn't perfect as we already needed calcium above and used a very poorly constrained fit (as this is all that's available)
        calcium_downscaling = numpy.max(omega)/15
        calcium_upscaling = 0.5/numpy.min(omega)

        # Quantify the likelihood of those scalings
        calcium_downscaling_probability = calcium_downscaling_prior.getProbability([calcium_downscaling])
        calcium_upscaling_probability = calcium_upscaling_prior.getProbability([calcium_upscaling])

        # Extract the parameters - in convenient units
        pH = [csys["pHtot"] for csys in csys_variable_dic]
        dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
        co2 = [csys["pCO2"]*1e6 for csys in csys_variable_dic]

        pH_distribution = Sampling.Distribution.fromSamples(pH[0],bin_edges=preprocessing.pH_x).normalise()
        
        # Get the probability of each of the parameters
        epsilon_probability = preprocessing.epsilon_sampler.getProbability([epsilon])
        pH_probability = pH_prior[0].getProbability(pH[0])
        dic_probability = dic_prior[0].getProbability(dic[0]) 
        co2_probability = co2_prior[0].getProbability(co2[0])

        # Reject the sample if any are 0
        if any(dic_probability==0) or any(pH_probability==0) or any(co2_probability==0) or calcium_downscaling_probability==0 or calcium_upscaling_probability==0:
            continue
    
        # Now that everything is valid, we can calculate some dependent properties 
        # Includes recalculation for updated calcium          
        # Convert the calcium into its downscaled/upscaled value
        if calcium_downscaling>1:
            calcium_adjusted = [calcium/calcium_downscaling for calcium in calcium_gp.means]
        if calcium_upscaling>1:
            calcium_adjusted = [calcium*calcium_upscaling for calcium in calcium_gp.means]
         
        # Also processing at temperature=25 degrees to eliminate site specific temperature effect
        Kb_25 =  [Bunch({"KB":kgen.calc_K("KB",TempC=25,Ca=calcium,Mg=magnesium)}) for calcium,magnesium in zip(calcium_adjusted,magnesium_gp.means,strict=True)]
        # Then convert the interpolated pH to interpolated d11B4
        d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb,d11Bsw,epsilon) for pH,d11Bsw,Kb in zip(pH_gp.means,d11Bsws,Kb_25,strict=True)]
        # And d11B4 into what would hypothetically have been measured
        d11B_measured_interpolated = [preprocessing.species_inverse_function(d11B4) for d11B4 in d11B4_interpolated]

        # Recalculate pH constraints with new calcium
        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_adjusted,magnesium_gp.means,strict=True)]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws[0],d11B4,epsilon)
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,pH_uncertainty),location=location).normalise() for pH_value,location,pH_uncertainty in zip(pH_values,preprocessing.data["age"].values,pH_uncertainties,strict=True)]

        # Redo the full GP for interpolated ages
        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
        pH_gp = pH_gp.setKernel("rbf",(0.1,3),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
        
        # Use the same seed to draw samples
        pH_gp.getSamples(1,seed=pH_seed)

        # Redo carbonate chemistry calculations
        csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_adjusted,magnesium_gp.means,strict=True)]
    
        # Pull out variables in typical units
        pH_to_store = pH_gp.samples
        dic_to_store = [csys["DIC"]*1e6 for csys in csys_variable_dic]
        co2_to_store = [csys["pCO2"]*1e6 for csys in csys_variable_dic]
        omega_to_store = [(csys.Ca*csys.CO3)/csys.Ks.KspC for csys in csys_variable_dic]
        magnesium_to_store = [magnesium*1e3 for magnesium in magnesium_gp.means]
        calcium_to_store = [calcium*1e3 for calcium in calcium_adjusted]
        co3_to_store = [csys.CO3*1e6 for csys in csys_variable_dic]
        alkalinity_to_store = [csys.TA*1e6 for csys in csys_variable_dic]
                
        # Calculate sample probability
        # Don't need DIC or CO2 cumulative because they're flat distributions, either 0 or equal everywhere
        d11Bsw_log_probability = sum(numpy.log(d11Bsw_probability))
        pH_log_probability = sum(numpy.log(pH_probability))
        # pH_quantile_probability = numpy.log(pH_prior[0].getProbability(pH_distribution.quantile([0.75])[0]))
        epsilon_log_probability = numpy.log(epsilon_probability) 
        cumulative_probability = d11Bsw_log_probability+pH_log_probability+epsilon_log_probability

        if cumulative_probability<markov_chain.final("probability"):
            probability_difference = 10**(cumulative_probability-markov_chain.final("probability"))
            r = numpy.random.random(1)[0]
            if r>probability_difference:
                continue

        current_sample = (current_sample.addField("probability",cumulative_probability,precision=5)
                                        .addField("d11Bsw",d11Bsws,precision=2)
                                        .addField("d11B4_original",d11B4,precision=2)
                                        .addField("d11B4_original_probability",d11B4,precision=5)
                                        .addField("d11B4",d11B4_interpolated,precision=2)
                                        .addField("d11Bm",d11B_measured_interpolated,precision=2)
                                        .addField("epsilon",epsilon,precision=5)
                                        .addField("pH",pH_to_store,precision=3)
                                        .addField("pH_seed",pH_seed,precision=5)
                                        .addField("co2",co2_to_store,precision=0)
                                        .addField("dic",dic_to_store,precision=0)
                                        .addField("d11Bsw_initial",d11Bsw_initial_jittered,precision=2)
                                        .addField("d11Bsw_scaling",d11Bsw_scaling_jittered,precision=3)
                                        .addField("dic_initial",initial_dic[0],precision=0)
                                        .addField("dic_fraction",dic_fraction[0],precision=3)
                                        .addField("omega",omega_to_store,precision=2)
                                        .addField("calcium",calcium_to_store,precision=2)
                                        .addField("magnesium",magnesium_to_store,precision=2)
                                        .addField("co3",co3_to_store,precision=0)
                                        .addField("alkalinity",alkalinity_to_store,precision=0))
        
        markov_chain = markov_chain.addSample(current_sample)
        print(len(markov_chain))
    return markov_chain

markov_chain = getInitialSample()
markov_chain = iterate(markov_chain,preprocessing.number_of_samples)
markov_chain.toJSON("./Data/Output/markov_chain.json")

print("End")

