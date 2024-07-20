rng(0)
cvp = cvpartition(size(dataStorage,2),"HoldOut", 20000);

XTrain = dlarray(dataStorage(:,cvp.training),"TBC");
[NormLabel,labelC,labelS] = normalize(labelStorage(:,cvp.training),2,"range");

YTrain = dlarray(NormLabel,"CB");

XVal = dlarray(dataStorage(:,cvp.test),"TBC");
ValidationLabel = (labelStorage(:,cvp.test) - labelC) ./ labelS;
YVal = dlarray(ValidationLabel,"CB");

vars = [
    optimizableVariable('numLayers', [1, 5], 'Type', 'integer')
    optimizableVariable('ifrescale', [0, 1], 'Type', 'integer')
    optimizableVariable('numChannels', [1, 512], 'Type', 'integer')
    optimizableVariable('numHeads', [3, 12], 'Type', 'integer')
    optimizableVariable('numKeyChannels', [1, 128], 'Type', 'integer')
    optimizableVariable('BatchSize', [2, 64], 'Type', 'integer')...
];
 
minfn = @(params) bayesObj_transformer(XTrain, YTrain, XVal, YVal, params);
results = bayesopt(minfn, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus');

bestParams = bestPoint(results)

net = dlnetwork;
tempNet = sequenceInputLayer(1,"Name","sequence","Normalization","rescale-zero-one");
net = addLayers(net,tempNet);

tempNet = positionEmbeddingLayer(3,252,"Name","positionembed");
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition")
    selfAttentionLayer(4,128,"Name","selfattention_0","AttentionMask","none")
    selfAttentionLayer(4,128,"Name","selfattention_1","AttentionMask","none")
    indexing1dLayer("last", "Name", "indexing")
    fullyConnectedLayer(4,"Name","fc")];
net = addLayers(net,tempNet);
clear tempNet;

net = connectLayers(net,"sequence","positionembed");
net = connectLayers(net,"sequence","addition/in2");
net = connectLayers(net,"positionembed","addition/in1");
net = initialize(net);
plot(net);

rng(0)
options = trainingOptions("adam", ...
    LearnRateSchedule="none", ...
    VerboseFrequency=5000,...
    MaxEpochs=30, ...
    MiniBatchSize=32, ...
    InitialLearnRate=1e-3,...
    Shuffle="every-epoch",...
    ValidationData={XVal,YVal},...
    ValidationFrequency=5000,...
    ValidationPatience=20,...
    OutputNetwork="best-validation",...
    ExecutionEnvironment="gpu",...
    Plots="training-progress");
netTrained = trainnet(XTrain,YTrain,net,"mse",options);
