import pandas,numpy,openpyxl

from geochemistry_helpers import Sampling,GaussianProcess
from processStrontium import makeStrontiumGP
from process_d18O_d13C import generate_d18O,generate_d13C
import preprocessing

def quantilify(dataframe,gp,values,group=-1):
    for value in values:
        dataframe[str(value)] = numpy.squeeze(gp.quantile(value/100,group=group))
    return dataframe


strontium_gp = makeStrontiumGP()
d18O_gp = generate_d18O()
d13C_gp = generate_d13C()

markov_chain = Sampling.MarkovChain().fromJSON("./Data/Output/markov_chain.json").burn(10)

# Redo the calcium to change units to mmol/kg
ion_x = numpy.arange(0,1000.1,0.1)
calcium_magnesium = preprocessing.importData("calcium_magnesium")
calcium_magnesium["calcium"] = calcium_magnesium["calcium"]
calcium_constraints =  [Sampling.Sampler(ion_x,"Gaussian",(calcium,5e-3),"Monte_Carlo",location=age).normalise() for age,calcium in zip(calcium_magnesium["age"].to_numpy(),calcium_magnesium["calcium"].to_numpy(),strict=True) if not numpy.isnan(calcium)]

d11B4_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
d11Bm_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
d11Bsw_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.d11B_x)
pH_gp = GaussianProcess().constrain(preprocessing.pH_prior).setQueryLocations(preprocessing.interpolation_ages)    
co2_gp = GaussianProcess().constrain(preprocessing.co2_prior).setQueryLocations(preprocessing.interpolation_ages)
dic_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)
alkalinity_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)
saturation_state_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,numpy.arange(0,50,0.01))
calcium_gp = GaussianProcess().constrain(calcium_constraints).setQueryLocations(preprocessing.interpolation_ages)
co3_gp = GaussianProcess().setQueryLocations(preprocessing.interpolation_ages,preprocessing.carbon_x)

d11B4_gp.fromMCMCSamples(markov_chain.accumulate("d11B4"))
d11Bm_gp.fromMCMCSamples(markov_chain.accumulate("d11Bm"))
d11Bsw_gp.fromMCMCSamples(markov_chain.accumulate("d11Bsw"))
pH_gp.fromMCMCSamples(markov_chain.accumulate("pH"))
co2_gp.fromMCMCSamples(markov_chain.accumulate("co2"))
dic_gp.fromMCMCSamples(markov_chain.accumulate("dic"))
alkalinity_gp.fromMCMCSamples(markov_chain.accumulate("alkalinity"))
saturation_state_gp.fromMCMCSamples(markov_chain.accumulate("omega"))
calcium_gp.fromMCMCSamples(markov_chain.accumulate("calcium"))
co3_gp.fromMCMCSamples(markov_chain.accumulate("co3"))


quantiles = [5,16,50,84,95]

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = strontium_gp.means[-1][0]
strontium_dataframe = quantilify(empty_dataframe,strontium_gp,quantiles)
strontium_dataframe = strontium_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = d18O_gp.means[-1][0]
d18O_dataframe = quantilify(empty_dataframe,d18O_gp,quantiles)
d18O_dataframe = d18O_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = d13C_gp.means[-1][0]
d13C_dataframe = quantilify(empty_dataframe,d13C_gp,quantiles)
d13C_dataframe = d13C_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = d11B4_gp.means[-1]
d11B4_dataframe = quantilify(empty_dataframe,d11B4_gp,quantiles)
d11B4_dataframe = d11B4_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = d11Bm_gp.means[-1]
d11Bm_dataframe = quantilify(empty_dataframe,d11Bm_gp,quantiles)
d11Bm_dataframe = d11Bm_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = d11Bsw_gp.means[-1]
d11Bsw_dataframe = quantilify(empty_dataframe,d11Bsw_gp,quantiles)
d11Bsw_dataframe = d11Bsw_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = pH_gp.means[-1]
pH_dataframe = quantilify(empty_dataframe,pH_gp,quantiles)
pH_dataframe = pH_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = co2_gp.means[-1]
co2_dataframe = quantilify(empty_dataframe,co2_gp,quantiles)
co2_dataframe = co2_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = dic_gp.means[-1]
dic_dataframe = quantilify(empty_dataframe,dic_gp,quantiles)
dic_dataframe = dic_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = alkalinity_gp.means[-1]
alkalinity_dataframe = quantilify(empty_dataframe,alkalinity_gp,quantiles)
alkalinity_dataframe = alkalinity_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = saturation_state_gp.means[-1]
saturation_state_dataframe = quantilify(empty_dataframe,saturation_state_gp,quantiles)
saturation_state_dataframe = saturation_state_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = calcium_gp.means[-1]
calcium_dataframe = quantilify(empty_dataframe,calcium_gp,quantiles)
calcium_dataframe = calcium_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.equally_spaced_ages
empty_dataframe["mean"] = co3_gp.means[-1]
co3_dataframe = quantilify(empty_dataframe,co3_gp,quantiles)
co3_dataframe = co3_dataframe.sort_values(by="age",ascending=True)

with pandas.ExcelWriter("./Data/Output/metrics_100kyr.xlsx") as writer:
    d11B4_dataframe.to_excel(writer,sheet_name="d11B4")
    d11Bm_dataframe.to_excel(writer,sheet_name="d11Bm")
    d11Bsw_dataframe.to_excel(writer,sheet_name="d11Bsw")
    pH_dataframe.to_excel(writer,sheet_name="pH")
    co2_dataframe.to_excel(writer,sheet_name="co2")
    dic_dataframe.to_excel(writer,sheet_name="dic")
    alkalinity_dataframe.to_excel(writer,sheet_name="alkalinity")
    saturation_state_dataframe.to_excel(writer,sheet_name="saturation_state")
    calcium_dataframe.to_excel(writer,sheet_name="calcium")
    co3_dataframe.to_excel(writer,sheet_name="co3")
    strontium_dataframe.to_excel(writer,sheet_name="strontium")
    d18O_dataframe.to_excel(writer,sheet_name="d18O")
    d13C_dataframe.to_excel(writer,sheet_name="d13C")


empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = strontium_gp.means[0][0]
strontium_dataframe = quantilify(empty_dataframe,strontium_gp,quantiles,group=0)
strontium_dataframe = strontium_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = d18O_gp.means[0][0]
d18O_dataframe = quantilify(empty_dataframe,d18O_gp,quantiles,group=0)
d18O_dataframe = d18O_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = d13C_gp.means[0][0]
d13C_dataframe = quantilify(empty_dataframe,d13C_gp,quantiles,group=0)
d13C_dataframe = d13C_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = d11B4_gp.means[0]
d11B4_dataframe = quantilify(empty_dataframe,d11B4_gp,quantiles,group=0)
d11B4_dataframe = d11B4_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = d11Bm_gp.means[0]
d11Bm_dataframe = quantilify(empty_dataframe,d11Bm_gp,quantiles,group=0)
d11Bm_dataframe = d11Bm_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = d11Bsw_gp.means[0]
d11Bsw_dataframe = quantilify(empty_dataframe,d11Bsw_gp,quantiles,group=0)
d11Bsw_dataframe = d11Bsw_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = pH_gp.means[0]
pH_dataframe = quantilify(empty_dataframe,pH_gp,quantiles,group=0)
pH_dataframe = pH_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = co2_gp.means[0]
co2_dataframe = quantilify(empty_dataframe,co2_gp,quantiles,group=0)
co2_dataframe = co2_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = dic_gp.means[0]
dic_dataframe = quantilify(empty_dataframe,dic_gp,quantiles,group=0)
dic_dataframe = dic_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = alkalinity_gp.means[0]
alkalinity_dataframe = quantilify(empty_dataframe,alkalinity_gp,quantiles,group=0)
alkalinity_dataframe = alkalinity_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = saturation_state_gp.means[0]
saturation_state_dataframe = quantilify(empty_dataframe,saturation_state_gp,quantiles,group=0)
saturation_state_dataframe = saturation_state_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = calcium_gp.means[0]
calcium_dataframe = quantilify(empty_dataframe,calcium_gp,quantiles,group=0)
calcium_dataframe = calcium_dataframe.sort_values(by="age",ascending=True)

empty_dataframe = pandas.DataFrame()
empty_dataframe["age"] = preprocessing.interpolation_ages[0]
empty_dataframe["mean"] = co3_gp.means[0]
co3_dataframe = quantilify(empty_dataframe,co3_gp,quantiles,group=0)
co3_dataframe = co3_dataframe.sort_values(by="age",ascending=True)

with pandas.ExcelWriter("./Data/Output/metrics.xlsx") as writer:
    d11B4_dataframe.to_excel(writer,sheet_name="d11B4")
    d11Bm_dataframe.to_excel(writer,sheet_name="d11Bm")
    d11Bsw_dataframe.to_excel(writer,sheet_name="d11Bsw")
    pH_dataframe.to_excel(writer,sheet_name="pH")
    co2_dataframe.to_excel(writer,sheet_name="co2")
    dic_dataframe.to_excel(writer,sheet_name="dic")
    alkalinity_dataframe.to_excel(writer,sheet_name="alkalinity")
    saturation_state_dataframe.to_excel(writer,sheet_name="saturation_state")
    calcium_dataframe.to_excel(writer,sheet_name="calcium")
    co3_dataframe.to_excel(writer,sheet_name="co3")
    strontium_dataframe.to_excel(writer,sheet_name="strontium")
    d18O_dataframe.to_excel(writer,sheet_name="d18O")
    d13C_dataframe.to_excel(writer,sheet_name="d13C")
