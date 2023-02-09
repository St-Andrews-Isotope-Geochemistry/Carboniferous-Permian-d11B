import numpy
from matplotlib import pyplot

from geochemistry_helpers import Sampling,GaussianProcess

from cbsyst import boron_isotopes,Csys
from cbsyst.helpers import Bunch
import kgen

from processStrontium import makeStrontiumGP,generateNormalisedStrontium
from processLithium import makeLithiumGP,generateNormalisedLithium
import preprocessing,processCalciumMagnesium

def calculatepH(plot=False):
    def isSamplePossible(d11Bsw_probability,):
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling_jittered)
        d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability(d11Bsw_initial_jittered)
        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_initial_probability==0:
            return False
        return True
    
    strontium_gp = makeStrontiumGP()
    lithium_gp = makeLithiumGP()

    normalised_strontium_gp = generateNormalisedStrontium()
    normalised_lithium_gp = generateNormalisedLithium()

    strontium_shapes = normalised_strontium_gp.means
    lithium_shapes = normalised_lithium_gp.means

    strontium_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-1,1,1e-3),"Gaussian",(0,0.05),"Monte_Carlo").normalise() 
    lithium_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-1,1,1e-3),"Gaussian",(0,0.05),"Monte_Carlo").normalise()  
    d11Bsw_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,0.5),"Monte_Carlo").normalise()  
    d11Bsw_initial_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,1),"Monte_Carlo").normalise()

    d18O = preprocessing.data["d18O"].to_numpy()
    temperature = 15.7 - 4.36*((d18O-(-1))) + 0.12*((d18O-(-1))**2)
    temperature_constraints = [Sampling.Distribution(preprocessing.temperature_x,"Gaussian",(temperature,0.1),location=location).normalise() for temperature,location in zip(temperature,preprocessing.data["age"].to_numpy(),strict=True)]
    temperature_gp,temperature_mean_gp = GaussianProcess().constrain(temperature_constraints).removeLocalMean(fraction=(10,2))
    temperature_gp = temperature_gp.setKernel("rbf",(1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(temperature_mean_gp)

    calcium_gp,magnesium_gp = processCalciumMagnesium.calculateCalciumMagnesium()


    # Hoist out of preprocessing
    strontium_scaling_prior = preprocessing.strontium_scaling_prior
    lithium_scaling_prior = preprocessing.lithium_scaling_prior
    d11Bsw_priors = None
    d11Bsw_initial_prior = None
    d11Bsw_scaling_prior = preprocessing.d11Bsw_scaling_prior
    pH_prior = preprocessing.pH_prior
    co2_prior = preprocessing.co2_prior
    dic_prior = preprocessing.dic_prior

    scaling_edges = numpy.arange(-1,1e3,0.1)
    calcium_upscaling_prior = Sampling.Distribution(scaling_edges,"Flat",(0,3))
    calcium_downscaling_prior = Sampling.Distribution(scaling_edges,"Flat",(0,3))

    # Hoist out of start loop
    total_count = 0
    rejects = 0
    count = 0
    cumulative_probability = 0
    d11Bsws = []
    d11Bsw_to_store = []
    d11Bsw_initial = tuple()
    d11Bsw_initial_reject = tuple()
    d11Bsw_scaling = tuple()
    d11Bsw_scaling_reject = tuple()
    
    markov_chain = Sampling.MarkovChain()
    d11B4_markov_chain = Sampling.MarkovChain()

    # Get initial sample
    initial_d11B4_sample = Sampling.MarkovChainSample()
    initial_sample = Sampling.MarkovChainSample()
    while True:
        print("Getting initial sample")

        d11B4 = numpy.array([d11B4_prior.getSamples(1).samples[0] for d11B4_prior in preprocessing.d11B4_priors])
        d11B4_likelihood = numpy.sum([d11B4_prior.getLogProbability(d11B4_sample) for d11B4_prior,d11B4_sample in zip(preprocessing.d11B4_priors,d11B4,strict=True)])

        d11Bsw_range = preprocessing.getd11BswRange(d11B4)
        d11Bsw_priors = [Sampling.Sampler(preprocessing.d11B_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(preprocessing.data["age"].to_numpy(),d11Bsw_range,strict=True)]
        d11Bsw_initial_prior = d11Bsw_priors[0]
        
        strontium_scaling_prior.getSamples(1)
        lithium_scaling_prior.getSamples(1)

        strontium_scaling = strontium_scaling_prior.samples[0]
        lithium_scaling = lithium_scaling_prior.samples[0]

        d11Bsw_initial_prior.getSamples(1)
        d11Bsw_scaling_prior.getSamples(1)

        d11Bsw_initial = d11Bsw_initial_prior.samples[0]
        d11Bsw_scaling = d11Bsw_scaling_prior.samples[0]

        shapes = [strontium*strontium_scaling + lithium*lithium_scaling for strontium,lithium in zip(strontium_shapes,lithium_shapes,strict=True)]
        normalised_shapes = [(shape-min(shapes[-1][0]))/((max(shapes[-1][0])-min(shapes[-1][0]))/2)-1 for shape in shapes]

        d11Bsws = d11Bsw_initial+(d11Bsw_scaling/2)*numpy.squeeze(normalised_shapes[0])
    
        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
        d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability(d11Bsw_initial)
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling)

        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_initial_probability==0:
            pass
        else:
            Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
            pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
            pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.05),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["age"].to_numpy(),strict=True)]
            
            pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
            pH_gp = pH_gp.setKernel("rbf",(0.1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
            pH = pH_gp.means[0]

            if not numpy.any(numpy.isnan(pH)):
                initial_dic = preprocessing.initial_dic_sampler[0].getSamples(1).samples
                dic_fraction = preprocessing.dic_fraction_sampler[0].getSamples(1).samples

                csys_constant_dic = [Csys(pHtot=pH,DIC=initial_dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,temperature,calcium,magnesium in zip(pH_gp.means,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                relative_co2 = [(csys["CO2"][0]/csys_constant_dic[0]["CO2"][0][0])-1 for csys in csys_constant_dic]
                
                dic_series = [initial_dic+relative*dic_fraction*initial_dic for relative in relative_co2]
                dic_sensible = numpy.all(dic_series[0]>0) and numpy.all(dic_series[1]>0) and numpy.all(dic_series[0]<1e6) and numpy.all(dic_series[1]<1e6)
                print("DIC: "+str(numpy.min(dic_series[-1])) +","+ str(initial_dic[0]) +","+ str(numpy.max(dic_series[-1]))+","+str(dic_fraction[0]))

                if dic_sensible:
                    csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                    
                    dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
                    co2_samples = [csys["pCO2"]*1e6 for csys in csys_variable_dic]
                    omega = (csys_variable_dic[-1].Ca*csys_variable_dic[-1].CO3)/csys_variable_dic[-1].Ks.KspC
                    calcium_downscaling = numpy.max(omega)/15
                    calcium_upscaling = 0.5/numpy.min(omega)

                    calcium_downscaling_probability = calcium_downscaling_prior.getProbability(calcium_downscaling)
                    calcium_upscaling_probability = calcium_upscaling_prior.getProbability(calcium_upscaling)

                    co2 = co2_samples[0]
                    
                    dic_probability = numpy.array([dic_prior[0].getProbability(dic) for dic in dic[0]])
                    pH_probability = numpy.array([pH_prior[0].getProbability(pH) for pH in pH[0]])
                    co2_probability = numpy.array([co2_prior[0].getProbability(co2) for co2 in co2[0]])

                    if any(dic_probability==0) or any(pH_probability==0) or any(co2_probability==0) or calcium_downscaling_probability==0 or calcium_upscaling_probability==0:
                        pass
                    else:
                        d11Bsw_to_store = [d11Bsw_initial+(d11Bsw_scaling/2)*numpy.squeeze(shape) for shape in normalised_shapes]
                        d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))

                        Kb_25 = [Bunch({"KB":kgen.calc_K("KB",TempC=25,Ca=calcium,Mg=magnesium)}) for calcium,magnesium in zip(calcium_gp.means,magnesium_gp.means,strict=True)]
                        d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb,d11Bsw,preprocessing.epsilon) for pH,d11Bsw,Kb in zip(pH_gp.means,d11Bsw_to_store,Kb_25,strict=True)]

                        if calcium_downscaling>1:
                            calcium_gp.means = [calcium/calcium_downscaling for calcium in calcium_gp.means]
                        if calcium_upscaling>1:
                            calcium_gp.means = [calcium*calcium_upscaling for calcium in calcium_gp.means]
                        
                        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
                        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.05),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["age"].to_numpy(),strict=True)]
                        
                        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
                        pH_gp = pH_gp.setKernel("rbf",(0.1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
                        pH_log_probability = numpy.array([pH_prior[0].getLogProbability(pH) for pH in pH[0]])
                        pH_cumulative_probability = sum(pH_log_probability)

                        csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                    
                        dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
                        co2_samples = [csys["pCO2"]*1e6 for csys in csys_variable_dic]
                        omega = [(csys.Ca*csys.CO3)/csys.Ks.KspC for csys in csys_variable_dic]
                        
                        # Don't need DIC or CO2 cumulative because they're flat distributions, either 0 or equal everywhere
                        #co2_cumulative_probability = sum(numpy.log(co2_probability))
                        
                        cumulative_probability = d11Bsw_cumulative_probability+pH_cumulative_probability # +co2_cumulative_probability
                        break
        
    initial_d11B4_sample = (initial_sample.addField("probability",d11B4_likelihood)
                                          .addField("d11B4",[d11B4]))
    d11B4_markov_chain = d11B4_markov_chain.addSample(initial_d11B4_sample)

    initial_sample = (initial_sample.addField("probability",cumulative_probability)
                                    .addField("d11Bsw",d11Bsw_to_store)
                                    .addField("d11B4",d11B4_interpolated)
                                    .addField("pH",pH_gp.means)
                                    .addField("co2",co2_samples)
                                    .addField("dic",dic)
                                    .addField("strontium_scaling",strontium_scaling)
                                    .addField("lithium_scaling",lithium_scaling)
                                    .addField("d11Bsw_initial",d11Bsw_initial)
                                    .addField("d11Bsw_scaling",d11Bsw_scaling)
                                    .addField("dic_initial",initial_dic[0])
                                    .addField("dic_fraction",dic_fraction[0])
                                    .addField("omega",omega)
                    )
    markov_chain = markov_chain.addSample(initial_sample)

    while len(markov_chain)<preprocessing.number_of_samples:        
        total_count+=1
        print(len(markov_chain))
        current_d11B4_sample = Sampling.MarkovChainSample()
        current_sample = Sampling.MarkovChainSample()

        d11B4_accept = False
        while not d11B4_accept:
            d11B4_jitter = numpy.array([d11B4_jitter_sampler.getSamples(1).samples[0] for d11B4_jitter_sampler in preprocessing.d11B4_jitter_samplers])
            d11B4 = d11B4_markov_chain.final("d11B4")[0]+d11B4_jitter
            d11B4_likelihood = numpy.sum([d11B4_prior.getLogProbability(d11B4_sample) for d11B4_prior,d11B4_sample in zip(preprocessing.d11B4_priors,d11B4,strict=True)])

            if d11B4_likelihood>d11B4_markov_chain.final("probability"):
                d11B4_accept = True
            else:
                probability_difference = 10**(d11B4_likelihood-d11B4_markov_chain.final("probability"))
                r = numpy.random.random(1)[0]
                if r<=probability_difference:
                    d11B4_accept = True

        strontium_scaling_jitter_sampler.getSamples(1)
        lithium_scaling_jitter_sampler.getSamples(1)

        calcium_gp,magnesium_gp = processCalciumMagnesium.calculateCalciumMagnesium()

        strontium_scaling = markov_chain.final("strontium_scaling") + strontium_scaling_jitter_sampler.samples[0]
        lithium_scaling = markov_chain.final("lithium_scaling") + lithium_scaling_jitter_sampler.samples[0]

        d11Bsw_range = preprocessing.getd11BswRange(d11B4)
        d11Bsw_priors = [Sampling.Sampler(preprocessing.d11B_x,"Flat",(range[0],range[1]),"Monte_Carlo",location=age).normalise() for age,range in zip(preprocessing.data["age"].to_numpy(),d11Bsw_range,strict=True)]
        d11Bsw_initial_prior = d11Bsw_priors[0]
        
        d11Bsw_initial_jitter_sampler.getSamples(1)
        d11Bsw_initial_jittered = markov_chain.final("d11Bsw_initial") + d11Bsw_initial_jitter_sampler.samples[0]
        d11Bsw_scaling_jitter_sampler.getSamples(1)
        d11Bsw_scaling_jittered = markov_chain.final("d11Bsw_scaling") + d11Bsw_scaling_jitter_sampler.samples[0]


        shapes = [strontium*strontium_scaling + lithium*lithium_scaling for strontium,lithium in zip(strontium_shapes,lithium_shapes,strict=True)]
        normalised_shapes = [(shape-min(shapes[-1][0]))/((max(shapes[-1][0])-min(shapes[-1][0]))/2)-1 for shape in shapes]


        d11Bsws = d11Bsw_initial_jittered+(d11Bsw_scaling_jittered/2)*numpy.squeeze(normalised_shapes[0])

        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.05),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["age"].to_numpy(),strict=True)]
        
        pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
        pH_gp = pH_gp.setKernel("rbf",(0.1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
        pH = pH_gp.means[0]

        if not numpy.any(numpy.isnan(pH)) and strontium_scaling_prior.getProbability(strontium_scaling)>0 and lithium_scaling_prior.getProbability(lithium_scaling)>0:
            initial_dic_jitter = preprocessing.initial_dic_jitter_sampler[0].getSamples(1).samples
            dic_fraction_jitter = preprocessing.dic_fraction_jitter_sampler[0].getSamples(1).samples

            initial_dic = markov_chain.final("dic_initial") + initial_dic_jitter
            dic_fraction = markov_chain.final("dic_fraction") + dic_fraction_jitter

            initial_dic_probability = preprocessing.initial_dic_sampler[0].getProbability(initial_dic)
            dic_fraction_probability = preprocessing.dic_fraction_sampler[0].getProbability(dic_fraction)

            dic_sensible = initial_dic_probability>0 and dic_fraction_probability>0
            if dic_sensible:
                csys_constant_dic = [Csys(pHtot=pH,DIC=initial_dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,temperature,calcium,magnesium in zip(pH_gp.means,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                relative_co2 = [(csys["CO2"][0]/csys_constant_dic[0]["CO2"][0][0])-1 for csys in csys_constant_dic]
                
                dic_series = [initial_dic+relative*dic_fraction*initial_dic for relative in relative_co2]
                dic_sensible = numpy.all(dic_series[0]>0) and numpy.all(dic_series[1]>0)  and numpy.all(dic_series[0]<1e6) and numpy.all(dic_series[1]<1e6)
            
            if dic_sensible:
                csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                
                dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
                co2_samples = [csys["pCO2"]*1e6 for csys in csys_variable_dic]

                omega = (csys_variable_dic[-1].Ca*csys_variable_dic[-1].CO3)/csys_variable_dic[-1].Ks.KspC
                calcium_downscaling = numpy.max(omega)/15
                calcium_upscaling = 0.5/numpy.min(omega)

                calcium_downscaling_probability = calcium_downscaling_prior.getProbability(calcium_downscaling)
                calcium_upscaling_probability = calcium_upscaling_prior.getProbability(calcium_upscaling)

                co2 = co2_samples[0]

                d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
                d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling_jittered)
                d11Bsw_initial_probability = d11Bsw_initial_prior.getProbability(d11Bsw_initial_jittered)
                if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_initial_probability==0  or calcium_downscaling_probability==0 or calcium_upscaling_probability==0:
                    impossible = True
                else:
                    d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))
                    dic_probability = numpy.array([dic_prior[0].getProbability(dic) for dic in dic[0]])
                    pH_probability = numpy.array([pH_prior[0].getProbability(pH) for pH in pH[0]])
                    co2_probability = numpy.array([co2_prior[0].getProbability(co2) for co2 in co2[0]])
                    if any(pH_probability==0) or any(co2_probability==0):
                        impossible = True
                    else:
                        impossible = False
            else:
                impossible = True
            
            keep = False
            if not impossible:
                pH_log_probability = numpy.array([pH_prior[0].getLogProbability(pH) for pH in pH[0]])
                cumulative_probability = d11Bsw_cumulative_probability+sum(pH_log_probability)
                if cumulative_probability>markov_chain.final("probability"):
                    keep = True
                else:
                    probability_difference = 10**(cumulative_probability-markov_chain.final("probability"))
                    r = numpy.random.random(1)[0]
                    if r<=probability_difference:
                        keep = True            

            if keep:
                d11Bsw_to_store = [d11Bsw_initial_jittered+(d11Bsw_scaling_jittered/2)*numpy.squeeze(shape) for shape in normalised_shapes]

                if calcium_downscaling>1:
                    calcium_gp.means = [calcium/calcium_downscaling for calcium in calcium_gp.means]
                if calcium_upscaling>1:
                    calcium_gp.means = [calcium*calcium_upscaling for calcium in calcium_gp.means]
                        
                Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0],Ca=calcium,Mg=magnesium)}) for temperature,calcium,magnesium in zip(temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
                pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
                pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.05),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["age"].to_numpy(),strict=True)]
                
                pH_gp,pH_mean_gp = GaussianProcess().constrain(pH_constraints).removeLocalMean(fraction=(2,2))
                pH_gp = pH_gp.setKernel("rbf",(0.1,5),specified_mean=0).query(preprocessing.interpolation_ages).addLocalMean(pH_mean_gp)
                pH_log_probability = numpy.array([pH_prior[0].getLogProbability(pH) for pH in pH[0]])
                pH_cumulative_probability = sum(pH_log_probability)

                csys_variable_dic = [Csys(pHtot=pH,DIC=dic/1e6,T_in=numpy.squeeze(temperature),T_out=numpy.squeeze(temperature),Ca=calcium,Mg=magnesium,unit="mol") for pH,dic,temperature,calcium,magnesium in zip(pH_gp.means,dic_series,temperature_gp.means,calcium_gp.means,magnesium_gp.means,strict=True)]
            
                dic = [csys["DIC"]*1e6 for csys in csys_variable_dic]
                co2_samples = [csys["pCO2"]*1e6 for csys in csys_variable_dic]
                omega = [(csys.Ca*csys.CO3)/csys.Ks.KspC for csys in csys_variable_dic]
                      
                pH_to_store = pH_gp.means
                co2_to_store = co2_samples

                Kb_25 =  [Bunch({"KB":kgen.calc_K("KB",TempC=25,Ca=calcium,Mg=magnesium)}) for calcium,magnesium in zip(calcium_gp.means,magnesium_gp.means,strict=True)]
                d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb,d11Bsw,preprocessing.epsilon) for pH,d11Bsw,Kb in zip(pH_gp.means,d11Bsw_to_store,Kb_25,strict=True)]
                
                current_d11B4_sample = (current_sample.addField("probability",d11B4_likelihood)
                                                      .addField("d11B4",[d11B4]))
                d11B4_markov_chain = d11B4_markov_chain.addSample(current_d11B4_sample)

                current_sample = (current_sample.addField("probability",cumulative_probability)
                                                .addField("d11Bsw",d11Bsw_to_store)
                                                .addField("d11B4",d11B4_interpolated)
                                                .addField("pH",pH_to_store)
                                                .addField("co2",co2_to_store)
                                                .addField("dic",dic)
                                                .addField("strontium_scaling",strontium_scaling)
                                                .addField("lithium_scaling",lithium_scaling)
                                                .addField("d11Bsw_initial",d11Bsw_initial_jittered)
                                                .addField("d11Bsw_scaling",d11Bsw_scaling_jittered)
                                                .addField("dic_initial",initial_dic[0])
                                                .addField("dic_fraction",dic_fraction[0])
                                                .addField("omega",omega)
                    )
                
                markov_chain = markov_chain.addSample(current_sample)
                count += 1
            else:
                d11Bsw_initial_reject = d11Bsw_initial_reject+(d11Bsw_initial_jittered,)
                d11Bsw_scaling_reject = d11Bsw_scaling_reject+(d11Bsw_scaling_jittered,)
                rejects+=1

    print(str(round(rejects/total_count,4)*100)+"% rejected")
    print(str(round(len(markov_chain)/total_count,4)*100)+"% accepted")
    
    markov_chain.toJSON("./Data/Output/markov_chain.json")

    
    # exec(open("Code/Analysis/plot.py").read())

    print("End")


calculatepH()