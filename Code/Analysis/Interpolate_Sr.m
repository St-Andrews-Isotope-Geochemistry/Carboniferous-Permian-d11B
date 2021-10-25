clear

run("Import_Data_Compilation.m");


%%
number_of_samples = 10000;

valid_strontium.boolean = ~isnan(DataCompilationSrCOB.Sr87)& ~isnan(DataCompilationSrCOB.Sr87_SD) & ~isnan(DataCompilationSrCOB.Age);

valid_strontium.age = DataCompilationSrCOB.Age(valid_strontium.boolean);
valid_strontium.Sr87 = DataCompilationSrCOB.Sr87(valid_strontium.boolean);
valid_strontium.Sr87_uncertainty = DataCompilationSrCOB.Sr87_SD(valid_strontium.boolean);

strontium_window = linspace(0.7,0.8,10001);

for strontium_index = 1:height(valid_strontium.age)
    valid_strontium.distributions(strontium_index,1) = Geochemistry_Helpers.Distribution(strontium_window,"Gaussian",[valid_strontium.Sr87(strontium_index),valid_strontium.Sr87_uncertainty(strontium_index)]).normalise();
    valid_strontium.distributions(strontium_index,1).location = valid_strontium.age(strontium_index);
end

interpolation_ages = linspace(min(valid_strontium.age),max(valid_strontium.age),1000)';

%%
gp = Geochemistry_Helpers.GaussianProcess("rbf",interpolation_ages);
gp.observations = valid_strontium.distributions';
gp.runKernel([0.00003,5.1]);
gp.getSamples(number_of_samples);


%%
figure(1);
clf
hold on
gp.plotSamples(1:100);

for strontium_index = 1:height(valid_strontium.age)
    plot([valid_strontium.age(strontium_index),valid_strontium.age(strontium_index)],valid_strontium.Sr87(strontium_index)+[-2*valid_strontium.Sr87_uncertainty(strontium_index),+2*valid_strontium.Sr87_uncertainty(strontium_index)],'g-');
end
plot(valid_strontium.age,valid_strontium.Sr87,'rx');

set(gca,'XDir','Reverse');
xlabel("Age");
ylabel("Sr");