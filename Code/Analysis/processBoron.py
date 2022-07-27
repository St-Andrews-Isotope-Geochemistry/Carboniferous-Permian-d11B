from geochemistry_helpers import Sampling,GaussianProcess

def calculatepH(plot=False):
    import numpy
    from matplotlib import pyplot

    from geochemistry_helpers import Sampling,GaussianProcess
    
    from cbsyst import boron_isotopes,Csys
    from cbsyst.helpers import Bunch
    import kgen

    from processStrontium import makeStrontiumGP,generateNormalisedStrontium
    import preprocessing

    def isSamplePossible(d11Bsw_probability,):
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling_jittered)
        d11Bsw_minimum_probability = d11Bsw_initial_prior.getProbability(d11Bsw_minimum_jittered)
        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_minimum_probability==0:
            return False
        return True

    strontium_gp = makeStrontiumGP()
    normalised_strontium_gp = generateNormalisedStrontium()

    #d11B4_gp,d11B4_background = GaussianProcess().constrain(preprocessing.d11B4_priors).setKernel("rbf",(1,3)).removeLocalMean()
    #d11B4_gp.query(preprocessing.interpolation_ages).addLocalMean(d11B4_background)
    d11B4_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
    dic_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)
    d11Bsw_gp = GaussianProcess().constrain(preprocessing.d11Bsw_priors).setKernel("rbf",(1,10)).query(preprocessing.interpolation_ages)
    pH_gp = GaussianProcess().constrain(preprocessing.pH_prior).setQueryLocations(preprocessing.interpolation_ages)    
    co2_gp = GaussianProcess().constrain(preprocessing.co2_prior).setQueryLocations(preprocessing.interpolation_ages)


    d11Bsw_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,0.5),"Monte_Carlo").normalise()  
    d11Bsw_minimum_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,1),"Monte_Carlo").normalise()

    d18O = preprocessing.data["d18O"].to_numpy()
    temperature = 15.7 - 4.36*((d18O-(-1))) + 0.12*((d18O-(-1))**2)
    temperature_constraints = [Sampling.Distribution(preprocessing.temperature_x,"Gaussian",(temperature,0.1),location=location).normalise() for temperature,location in zip(temperature,preprocessing.data["Age"].to_numpy(),strict=True)]
    temperature_gp = GaussianProcess().constrain(temperature_constraints).setKernel("rbf",(1,1)).query(preprocessing.interpolation_ages)

    # Hoist out of preprocessing
    d11B4 = preprocessing.data["d11B4"].to_numpy()
    d11Bsw_priors = preprocessing.d11Bsw_priors
    d11Bsw_initial_prior = preprocessing.d11Bsw_initial_prior
    d11Bsw_scaling_prior = preprocessing.d11Bsw_scaling_prior
    pH_prior = preprocessing.pH_prior
    co2_prior = preprocessing.co2_prior

    d11Bsw_range = preprocessing.d11Bsw_range

    # Hoist out of start loop
    total_count = 0
    accepts = 0
    rejects = 0
    count = 0
    cumulative_probability = 0
    d11Bsws = []
    d11Bsw_to_store = []
    d11Bsw_minimum = tuple()
    d11Bsw_minimum_reject = tuple()
    d11Bsw_scaling = tuple()
    d11Bsw_scaling_reject = tuple()
    
    while True:
        d11Bsw_initial_prior.getSamples(1)
        d11Bsw_scaling_prior.getSamples(1)

        d11Bsw_minimum = (d11Bsw_initial_prior.samples[0],)
        d11Bsw_scaling = (d11Bsw_scaling_prior.samples[0],)

        d11Bsws = d11Bsw_minimum[0]+d11Bsw_scaling[0]*numpy.squeeze(normalised_strontium_gp.means[0])
    
        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
        d11Bsw_minimum_probability = d11Bsw_initial_prior.getProbability(d11Bsw_minimum)
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling)

        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_minimum_probability==0:
            pass
        else:
            Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0])}) for temperature in temperature_gp.means]
            pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
            pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.01),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["Age"].to_numpy(),strict=True)]
            pH_gp = GaussianProcess().constrain(pH_constraints).setKernel("rbf",(0.1,3)).query(preprocessing.interpolation_ages)
            pH = pH_gp.means[0][0]

            if not numpy.any(numpy.isnan(pH)):
                co2_lower = Csys(pHtot=pH,DIC=preprocessing.dic_edges[0]/1e6,T_in=temperature,T_out=temperature,unit="mol")["pCO2"]*1e6
                co2_upper = Csys(pHtot=pH,DIC=preprocessing.dic_edges[1]/1e6,T_in=temperature,T_out=temperature,unit="mol")["pCO2"]*1e6

                co2_constraints = [Sampling.Distribution(preprocessing.carbon_logx,"Flat",numpy.log((lower,upper)),location=location).normalise() for lower,upper,location in zip(co2_lower,co2_upper,preprocessing.data["Age"].to_numpy(),strict=True)]
                log_co2_lines_gp = GaussianProcess().constrain(co2_constraints).setKernel("rbf",(1,3)).query(preprocessing.interpolation_ages)
                
                log_co2_lines_gp.getSamples(1)
                co2 = numpy.exp(log_co2_lines_gp.samples[0][0])
                co2_samples = [numpy.exp(sample) for sample in log_co2_lines_gp.samples]
                
                d11Bsw_to_store = [d11Bsw_minimum[0]+d11Bsw_scaling[0]*numpy.squeeze(norm_mean) for norm_mean in normalised_strontium_gp.means]
                d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))
                
                pH_probability = pH_prior[0].getLogProbability(pH)
                co2_probability = numpy.array([co2_prior[0].getProbability(co2) for co2 in co2])

                if any(pH_probability==0) or any(co2_probability==0):
                    pass
                else:
                    Kb_25 = Bunch({"KB":kgen.calc_K("KB",TempC=25)})
                    d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb_25,d11Bsw,preprocessing.epsilon) for pH,d11Bsw in zip(pH_gp.means,d11Bsw_to_store,strict=True)]
                    dic = [Csys(pHtot=pH[0],pCO2=numpy.exp(co2[0])/1e6,T_in=temperature[0],T_out=temperature[0],unit="mol")["DIC"]*1e6 for pH,co2,temperature in zip(pH_gp.means,log_co2_lines_gp.samples,temperature_gp.means,strict=True)]
                    pH_cumulative_probability = sum(pH_probability)
                    co2_cumulative_probability = sum(numpy.log(co2_probability))
                    cumulative_probability = d11Bsw_cumulative_probability+pH_cumulative_probability+co2_cumulative_probability
                    break
        
    probability_store = (cumulative_probability,)
    d11Bsw_sample_store = (d11Bsw_to_store,)


    #pH_to_store = [boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,preprocessing.epsilon) for d11B4,d11Bsws in zip(d11B4_gp.means,d11Bsw_to_store,strict=True)]
    d11B4_sample_store = (d11B4_interpolated,)
    pH_sample_store = (pH_gp.means,)
    co2_sample_store = (co2_samples,)
    dic_sample_store = (dic,)

    #co2_to_store = [Csys(pHtot=pH,DIC=preprocessing.dic_scenarios[0]/1e6)["pCO2"]*1e6 for pH in pH_to_store]
    #co2_sample_store = (co2_to_store,)

    while count<preprocessing.number_of_samples:
        total_count+=1

        print(count)
        d11Bsw_scaling_jitter_sampler.getSamples(1)
        d11Bsw_scaling_jittered = d11Bsw_scaling[-1]+d11Bsw_scaling_jitter_sampler.samples[0]
        d11Bsw_minimum_jitter_sampler.getSamples(1)
        d11Bsw_minimum_jittered = d11Bsw_minimum[-1] + d11Bsw_minimum_jitter_sampler.samples[0]

        d11Bsws = d11Bsw_minimum_jittered+d11Bsw_scaling_jittered*numpy.squeeze(normalised_strontium_gp.means[0])

        Kb = [Bunch({"KB":kgen.calc_K("KB",TempC=temperature[0])}) for temperature in temperature_gp.means]
        pH_values = boron_isotopes.calculate_pH(Kb[0],d11Bsws,d11B4,preprocessing.epsilon)
        pH_constraints = [Sampling.Distribution(preprocessing.pH_x,"Gaussian",(pH_value,0.01),location=location).normalise() for pH_value,location in zip(pH_values,preprocessing.data["Age"].to_numpy(),strict=True)]
        pH_gp = GaussianProcess().constrain(pH_constraints).setKernel("rbf",(0.1,3)).query(preprocessing.interpolation_ages)
        pH = pH_gp.means[0][0]

        if not numpy.any(numpy.isnan(pH)):
            co2_lower = Csys(pHtot=pH,DIC=preprocessing.dic_edges[0]/1e6,T_in=temperature,T_out=temperature,unit="mol")["pCO2"]*1e6
            co2_upper = Csys(pHtot=pH,DIC=preprocessing.dic_edges[1]/1e6,T_in=temperature,T_out=temperature,unit="mol")["pCO2"]*1e6

            co2_constraints = [Sampling.Distribution(preprocessing.carbon_logx,"Flat",numpy.log((lower,upper)),location=location).normalise() for lower,upper,location in zip(co2_lower,co2_upper,preprocessing.data["Age"].to_numpy(),strict=True)]
            log_co2_lines_gp = GaussianProcess().constrain(co2_constraints).setKernel("rbf",(1,3)).query(preprocessing.interpolation_ages)
            log_co2_lines_gp.getSamples(1)
            co2 = numpy.exp(log_co2_lines_gp.samples[0][0])
            co2_samples = [numpy.exp(sample) for sample in log_co2_lines_gp.samples]

            d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
            d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling_jittered)
            d11Bsw_minimum_probability = d11Bsw_initial_prior.getProbability(d11Bsw_minimum_jittered)
            if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_minimum_probability==0:
                impossible = True
            else:
                d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))
                pH_probability = pH_prior[0].getLogProbability(pH)
                co2_probability = numpy.array([co2_prior[0].getProbability(co2) for co2 in co2])
                if any(pH_probability==0) or any(co2_probability==0):
                    impossible = True
                else:
                    impossible = False
            
            keep = False
            if not impossible:
                cumulative_probability = d11Bsw_cumulative_probability+sum(pH_probability)+sum(numpy.log(co2_probability))
                if cumulative_probability>probability_store[-1]:
                    keep = True
                else:
                    probability_difference = 10**(cumulative_probability-probability_store[-1])
                    r = numpy.random.random(1)[0]
                    if r<=probability_difference:
                        keep = True            

            if keep:
                d11Bsw_to_store = [d11Bsw_minimum_jittered+d11Bsw_scaling_jittered*numpy.squeeze(norm_mean) for norm_mean in normalised_strontium_gp.means]
                pH_to_store = pH_gp.means
                co2_to_store = co2_samples
                Kb_25 = Bunch({"KB":kgen.calc_K("KB",TempC=25)})
                d11B4_interpolated = [boron_isotopes.calculate_d11B4(pH,Kb_25,d11Bsw,preprocessing.epsilon) for pH,d11Bsw in zip(pH_gp.means,d11Bsw_to_store,strict=True)]
                    
                dic = [Csys(pHtot=pH[0],pCO2=co2[0]/1e6,T_in=temperature[0],T_out=temperature[0],unit="mol")["DIC"]*1e6 for pH,co2,temperature in zip(pH_gp.means,co2_samples,temperature_gp.means,strict=True)]
                    
                d11Bsw_minimum = d11Bsw_minimum+(d11Bsw_minimum_jittered,)
                d11Bsw_scaling = d11Bsw_scaling+(d11Bsw_scaling_jittered,)
                probability_store = probability_store+(cumulative_probability,)
                d11Bsw_sample_store += (d11Bsw_to_store,)
                d11B4_sample_store += (d11B4_interpolated,)
                pH_sample_store += (pH_to_store,)
                co2_sample_store += (co2_to_store,)
                dic_sample_store += (dic,)

                accepts+=1
                count += 1
            else:
                d11Bsw_minimum_reject = d11Bsw_minimum_reject+(d11Bsw_minimum_jittered,)
                d11Bsw_scaling_reject = d11Bsw_scaling_reject+(d11Bsw_scaling_jittered,)
                rejects+=1

    print(str(round(rejects/total_count,4)*100)+"% rejected")
    print(str(round(accepts/total_count,4)*100)+"% accepted")
    
    d11B4_gp.fromMCMCSamples(d11B4_sample_store)
    d11Bsw_gp.fromMCMCSamples(d11Bsw_sample_store)
    pH_gp.fromMCMCSamples(pH_sample_store)
    co2_gp.fromMCMCSamples(co2_sample_store)
    dic_gp.fromMCMCSamples(dic_sample_store)
    
    exec(open("Code/Analysis/plot.py").read())

    print("End")


calculatepH()