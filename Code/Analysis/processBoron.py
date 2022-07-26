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

    d11B4_gp,d11B4_background = GaussianProcess().constrain(preprocessing.d11B4_priors).setKernel("rbf",(1,3)).removeLocalMean()
    d11B4_gp.query(preprocessing.interpolation_ages).addLocalMean(d11B4_background)
    d11Bsw_gp = GaussianProcess().constrain(preprocessing.d11Bsw_priors).setKernel("rbf",(1,10)).query(preprocessing.interpolation_ages)
    pH_gp = GaussianProcess().constrain(preprocessing.pH_prior).setQueryLocations(preprocessing.interpolation_ages)    
    co2_gp = GaussianProcess().constrain(preprocessing.co2_prior).setQueryLocations(preprocessing.interpolation_ages)

    d11Bsw_scaling_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,0.5),"Monte_Carlo").normalise()  
    d11Bsw_minimum_jitter_sampler = Sampling.Sampler(numpy.arange(-10,10,1e-3),"Gaussian",(0,1),"Monte_Carlo").normalise()
    dic_jitter_sampler = Sampling.Sampler(numpy.arange(-1000,1000,1),"Gaussian",(0,500),"Monte_Carlo").normalise()

    # Hoist out of preprocessing
    d11B4 = preprocessing.data["d11B4"].to_numpy()
    d11Bsw_priors = preprocessing.d11Bsw_priors
    d11Bsw_initial_prior = preprocessing.d11Bsw_initial_prior
    d11Bsw_scaling_prior = preprocessing.d11Bsw_scaling_prior
    pH_prior = preprocessing.pH_prior
    co2_prior = preprocessing.co2_prior
    dic_prior = preprocessing.dic_prior

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
        dic_prior.getSamples(1)

        d11Bsw_minimum = (d11Bsw_initial_prior.samples[0],)
        d11Bsw_scaling = (d11Bsw_scaling_prior.samples[0],)
        dic = (dic_prior.samples[0],)

        d11Bsws = d11Bsw_minimum[0]+d11Bsw_scaling[0]*numpy.squeeze(normalised_strontium_gp.means[0])
    
        d11Bsw_probability = numpy.squeeze(numpy.array([d11Bsw_prior.getProbability(d11Bsw) for d11Bsw_prior,d11Bsw in zip(d11Bsw_priors,d11Bsws,strict=True)]))
        d11Bsw_minimum_probability = d11Bsw_initial_prior.getProbability(d11Bsw_minimum)
        d11Bsw_scaling_probability = d11Bsw_scaling_prior.getProbability(d11Bsw_scaling)
        dic_probability = dic_prior.getProbability(dic)

        if any(d11Bsw_probability==0) or d11Bsw_scaling_probability==0 or d11Bsw_minimum_probability==0:
            pass
        else:
            Kb = Bunch({"KB":kgen.calc_K("KB")})
            pH = boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,preprocessing.epsilon)
            co2 = Csys(pHtot=pH,DIC=dic[0]/1e6)["pCO2"]*1e6

            d11Bsw_to_store = [d11Bsw_minimum[0]+d11Bsw_scaling[0]*numpy.squeeze(norm_mean) for norm_mean in normalised_strontium_gp.means]
            d11Bsw_cumulative_probability = sum(numpy.log(d11Bsw_probability))
            
            pH_probability = pH_prior[0].getLogProbability(pH)
            co2_probability = numpy.array([co2_prior[0].getProbability(co2) for co2 in co2])

            if any(pH_probability==0) or any(co2_probability==0):
                pass
            else:
                pH_cumulative_probability = sum(pH_probability)
                co2_cumulative_probability = sum(numpy.log(co2_probability))
                cumulative_probability = d11Bsw_cumulative_probability+pH_cumulative_probability+co2_cumulative_probability
                break
       
    probability_store = (cumulative_probability,)
    d11Bsw_sample_store = (d11Bsw_to_store,)

    pH_to_store = [boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,preprocessing.epsilon) for d11B4,d11Bsws in zip(d11B4_gp.means,d11Bsw_to_store,strict=True)]
    pH_sample_store = (pH_to_store,)

    co2_to_store = [Csys(pHtot=pH,DIC=preprocessing.dic_scenarios[0]/1e6)["pCO2"]*1e6 for pH in pH_to_store]
    co2_sample_store = (co2_to_store,)

    while count<preprocessing.number_of_samples:
        total_count+=1

        print(count)
        d11Bsw_scaling_jitter_sampler.getSamples(1)
        d11Bsw_scaling_jittered = d11Bsw_scaling[-1]+d11Bsw_scaling_jitter_sampler.samples[0]
        d11Bsw_minimum_jitter_sampler.getSamples(1)
        d11Bsw_minimum_jittered = d11Bsw_minimum[-1] + d11Bsw_minimum_jitter_sampler.samples[0]
        dic_jitter_sampler.getSamples(1)
        dic_jittered = dic[-1] + dic_jitter_sampler.samples[0]

        d11Bsws = d11Bsw_minimum_jittered+d11Bsw_scaling_jittered*numpy.squeeze(normalised_strontium_gp.means[0])

        Kb = Bunch({"KB":kgen.calc_K("KB")})
        pH = boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,preprocessing.epsilon)
        co2 = Csys(pHtot=pH,DIC=dic_jittered/1e6)["pCO2"]*1e6

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
            pH_to_store = [boron_isotopes.calculate_pH(Kb,d11Bsws,d11B4,preprocessing.epsilon) for d11B4,d11Bsws in zip(d11B4_gp.means,d11Bsw_to_store,strict=True)]
            co2_to_store = [Csys(pHtot=pH,DIC=dic_jittered/1e6)["pCO2"]*1e6 for pH in pH_to_store]
            
            d11Bsw_minimum = d11Bsw_minimum+(d11Bsw_minimum_jittered,)
            d11Bsw_scaling = d11Bsw_scaling+(d11Bsw_scaling_jittered,)
            dic = dic+(dic_jittered,)
            probability_store = probability_store+(cumulative_probability,)
            d11Bsw_sample_store += (d11Bsw_to_store,)
            pH_sample_store += (pH_to_store,)
            co2_sample_store += (co2_to_store,)

            accepts+=1
            count += 1
        else:
            d11Bsw_minimum_reject = d11Bsw_minimum_reject+(d11Bsw_minimum_jittered,)
            d11Bsw_scaling_reject = d11Bsw_scaling_reject+(d11Bsw_scaling_jittered,)
            rejects+=1

    print(str(round(rejects/total_count,4)*100)+"% rejected")
    print(str(round(accepts/total_count,4)*100)+"% accepted")
    
    d11Bsw_gp.fromMCMCSamples(d11Bsw_sample_store)
    pH_gp.fromMCMCSamples(pH_sample_store)
    co2_gp.fromMCMCSamples(co2_sample_store)
    
    exec(open("Code/Analysis/plot.py").read())
    print("End")


calculatepH()