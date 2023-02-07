import pandas,numpy

from geochemistry_helpers import Sampling,GaussianProcess
import preprocessing

def quantilify(dataframe,gp,values):
    for value in values:
        dataframe[str(value)] = numpy.squeeze(gp.quantile(value/100,group=-1))
    return dataframe

markov_chain = Sampling.MarkovChain().fromJSON("./Data/Output/markov_chain.json")

# d11B4_gp = GaussianProcess().setQueryLocations([preprocessing.interpolation_ages[0]],preprocessing.d11B_x)
d11B4_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
dic_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)
d11Bsw_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
pH_gp = GaussianProcess().constrain(preprocessing.pH_prior).setQueryLocations(preprocessing.interpolation_ages)    
co2_gp = GaussianProcess().constrain(preprocessing.co2_prior).setQueryLocations(preprocessing.interpolation_ages)

# d11B4_gp.fromMCMCSamples(d11B4_markov_chain.accumulate("d11B4"))
d11B4_gp.fromMCMCSamples(markov_chain.accumulate("d11B4"))
d11Bsw_gp.fromMCMCSamples(markov_chain.accumulate("d11Bsw"))
pH_gp.fromMCMCSamples(markov_chain.accumulate("pH"))
co2_gp.fromMCMCSamples(markov_chain.accumulate("co2"))
dic_gp.fromMCMCSamples(markov_chain.accumulate("dic"))

quantiles = [5,16,50,84,95]

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
d11B4_dataframe = quantilify(empty_dataframe,d11B4_gp,quantiles)
d11B4_dataframe = d11B4_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
d11Bsw_dataframe = quantilify(empty_dataframe,d11Bsw_gp,quantiles)
d11Bsw_dataframe = d11Bsw_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
pH_dataframe = quantilify(empty_dataframe,pH_gp,quantiles)
pH_dataframe = pH_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
co2_dataframe = quantilify(empty_dataframe,co2_gp,quantiles)
co2_dataframe = co2_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
dic_dataframe = quantilify(empty_dataframe,dic_gp,quantiles)
dic_dataframe = dic_dataframe.sort_values(by="age",ascending=True)

with pandas.ExcelWriter("./Data/Output/metrics.xlsx") as writer:
    d11B4_dataframe.to_excel(writer,sheet_name="d11B4")
    d11Bsw_dataframe.to_excel(writer,sheet_name="d11Bsw")
    pH_dataframe.to_excel(writer,sheet_name="pH")
    co2_dataframe.to_excel(writer,sheet_name="co2")
    dic_dataframe.to_excel(writer,sheet_name="dic")
a = 5

