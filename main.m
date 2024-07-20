rng("default")
% In most Versions of MATLAB, rng("default") equals rng(0,"twister")

% We use 100,000 sequences to form train set as recommended in *Deep LPPLS: Forecasting of temporal critical points in natural*.
% And we use 20,000 sequences to form validation set.
cvp = cvpartition(size(dataStorage,2),"HoldOut", 20000);

XTrain = dlarray(dataStorage(:,cvp.training),"CB");
[NormLabel,labelC,labelS] = normalize(labelStorage(:,cvp.training),2,"range");
NormLabel(1,:) = NormLabel(1,:);
NormLabel(2,:) = NormLabel(2,:);

YTrain = dlarray(NormLabel,"CB");

XVal = dlarray(dataStorage(:,cvp.test),"CB");
ValidationLabel = (labelStorage(:,cvp.test) - labelC) ./ labelS;
ValidationLabel(1,:) = ValidationLabel(1,:);
ValidationLabel(2,:) = ValidationLabel(2,:);
YVal = dlarray(ValidationLabel,"CB");

% Bayes Opt
vars = [
    optimizableVariable('numUnits_1st', [10, 1000], 'Type', 'integer')
    optimizableVariable('ifnorm', [0, 1], 'Type', 'integer')
    optimizableVariable('numUnits_1', [10, 1000], 'Type', 'integer')
    optimizableVariable('numUnits_2', [10, 1000], 'Type', 'integer')
    optimizableVariable('numLayers_1',[1, 7], 'Type', 'integer')
    optimizableVariable('numLayers_2', [2, 3], 'Type', 'integer')
];
 
% Obj Function
minfn = @(params) bayesObj(XTrain, YTrain, XVal, YVal, params);

% Bayes Optimization
results = bayesopt(minfn, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus');

% Display the best hyperParams
bestParams = bestPoint(results)

numUnits_1 = bestParams.numUnits_1;
numUnits_2 = bestParams.numUnits_2;
numUnits_1st = bestParams.numUnits_1st;
net = dlnetwork;
tempNet = [
    featureInputLayer(252,"Name","input","Normalization","rescale-zero-one")
    fullyConnectedLayer(numUnits_1st,"Name","fc_1")
    geluLayer("Name","gelu_0")];
net = addLayers(net,tempNet);

tempNet = [
    fullyConnectedLayer(numUnits_1,"Name","fc_2")
    preluLayer("Name","prelu_1")
    fullyConnectedLayer(numUnits_1,"Name","fc_3")
    preluLayer("Name","prelu_2")
    fullyConnectedLayer(numUnits_1,"Name","fc_4")
    preluLayer("Name","prelu_+1")
    fullyConnectedLayer(numUnits_1,"Name","fc_+1")
    preluLayer("Name","prelu_+2")
    fullyConnectedLayer(numUnits_1,"Name","fc_+2")
    preluLayer("Name","prelu_+3")
    fullyConnectedLayer(numUnits_1,"Name","fc_5")
    layerNormalizationLayer("Name","layernorm_4")];
net = addLayers(net,tempNet);

tempNet = [
    fullyConnectedLayer(numUnits_2,"Name","std_1")
    geluLayer("Name","gelu_2")
    fullyConnectedLayer(numUnits_2,"Name","std_2")
    geluLayer("Name","gelu_3")
    fullyConnectedLayer(2,"Name","op_2")];
net = addLayers(net,tempNet);

tempNet = [
    fullyConnectedLayer(numUnits_1,"Name","Res")
    layerNormalizationLayer("Name","layernorm_5")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition")
    preluLayer("Name","prelu_4")
    fullyConnectedLayer(2,"Name","op_1")
    ];
net = addLayers(net,tempNet);

tempNet = depthConcatenationLayer(2,"Name","depthcat");
net = addLayers(net,tempNet);

clear tempNet;

net = connectLayers(net,"gelu_0","fc_2");
net = connectLayers(net,"gelu_0","std_1");
net = connectLayers(net,"gelu_0","Res");
net = connectLayers(net,"layernorm_4","addition/in1");
net = connectLayers(net,"layernorm_5","addition/in2");
net = connectLayers(net,"op_1","depthcat/in1");
net = connectLayers(net,"op_2","depthcat/in2");
net = initialize(net);
glnn = net;

plot(glnn)

rng("default")
options = trainingOptions("adam", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=5,...
    LearnRateDropFactor=0.2,...
    VerboseFrequency=5000,...
    MaxEpochs=30, ...
    MiniBatchSize=6, ...
    InitialLearnRate=5e-6,...
    Shuffle="every-epoch",...
    ValidationData={XVal,YVal},...
    ValidationFrequency=5000,...
    ValidationPatience=20,...
    OutputNetwork="best-validation",...
    ExecutionEnvironment="gpu",...
    Plots="training-progress");
netTrained = trainnet(XTrain,YTrain,glnn,"mse",options);
