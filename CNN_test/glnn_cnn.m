rng("default")
cvp = cvpartition(size(dataStorage,2),"HoldOut", 20000);

XTrain = dlarray(dataStorage(:,cvp.training),"TBC");
[NormLabel,labelC,labelS] = normalize(labelStorage(:,cvp.training),2,"range");
NormLabel(1,:) = 10*NormLabel(1,:);
NormLabel(2,:) = 5*NormLabel(2,:);

YTrain = dlarray(NormLabel,"CB");

XVal = dlarray(dataStorage(:,cvp.test),"TBC");
ValidationLabel = (labelStorage(:,cvp.test) - labelC) ./ labelS;
ValidationLabel(1,:) = ValidationLabel(1,:);
ValidationLabel(2,:) = ValidationLabel(2,:);
YVal = dlarray(ValidationLabel,"CB");

vars = [
    optimizableVariable('numLayers', [1, 5], 'Type', 'integer')
    optimizableVariable('ifrescale', [0, 1], 'Type', 'integer')
    optimizableVariable('numChannels', [1, 512], 'Type', 'integer')
    optimizableVariable('numHeads', [3, 12], 'Type', 'integer')
    optimizableVariable('numKeyChannels', [1, 128], 'Type', 'integer')
    optimizableVariable('BatchSize', [2, 64], 'Type', 'integer')...
];
 
minfn = @(params) bayesObj_cnn(XTrain, YTrain, XVal, YVal, params);

results = bayesopt(minfn, vars, ...
    'MaxObjectiveEvaluations', 100, ...
    'IsObjectiveDeterministic', false, ...
    'AcquisitionFunctionName', 'expected-improvement-plus');

bestParams = bestPoint(results);

net = dlnetwork;
tempNet = [
    sequenceInputLayer(1,"Name","sequence")
    convolution1dLayer(3,32,"Name","conv1d","Padding","same")
    batchNormalizationLayer("Name","bn_conv1")
    reluLayer("Name","activation_1_relu")];
net = addLayers(net,tempNet);

tempNet = [
    maxPooling1dLayer(5,"Name","maxpool1d","Padding","same")
    convolution1dLayer(3,32,"Name","conv1d_1","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2a")
    reluLayer("Name","activation_2_relu")
    convolution1dLayer(3,32,"Name","conv1d_2","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2b")
    reluLayer("Name","activation_3_relu")
    convolution1dLayer(3,32,"Name","conv1d_3","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution1dLayer(3,32,"Name","conv1d_4","Padding","same")
    batchNormalizationLayer("Name","bn2a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","add_1")
    reluLayer("Name","activation_4_relu")];
net = addLayers(net,tempNet);

tempNet = [
    convolution1dLayer(3,32,"Name","conv1d_5","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2a")
    reluLayer("Name","activation_41_relu")
    convolution1dLayer(3,32,"Name","conv1d_6","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2b")
    reluLayer("Name","activation_42_relu")
    convolution1dLayer(3,32,"Name","conv1d_7","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch2c")];
net = addLayers(net,tempNet);

tempNet = [
    convolution1dLayer(3,32,"Name","conv1d_8","Padding","same")
    batchNormalizationLayer("Name","bn5a_branch1")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","add_14")
    reluLayer("Name","activation_43_relu")
    convolution1dLayer(3,32,"Name","conv1d_9","Padding","same")
    batchNormalizationLayer("Name","bn5c_branch2a")
    reluLayer("Name","activation_49_relu")
    globalAveragePooling1dLayer
    fullyConnectedLayer(1000,"Name","fc1000")
    fullyConnectedLayer(4,"Name","fc4")];
net = addLayers(net,tempNet);

clear tempNet;

net = connectLayers(net,"activation_1_relu","maxpool1d");
net = connectLayers(net,"activation_1_relu","conv1d_4");
net = connectLayers(net,"bn2a_branch2c","add_1/in1");
net = connectLayers(net,"bn2a_branch1","add_1/in2");
net = connectLayers(net,"activation_4_relu","conv1d_5");
net = connectLayers(net,"activation_4_relu","conv1d_8");
net = connectLayers(net,"bn5a_branch2c","add_14/in1");
net = connectLayers(net,"bn5a_branch1","add_14/in2");
net = initialize(net);
plot(net);

rng("default")
options = trainingOptions("adam", ...
    LearnRateSchedule="none", ...
    VerboseFrequency=5000,...
    MaxEpochs=30, ...
    MiniBatchSize=32, ...
    InitialLearnRate=3e-4,...
    Shuffle="every-epoch",...
    ValidationData={XVal,YVal},...
    ValidationFrequency=5000,...
    ValidationPatience=20,...
    OutputNetwork="best-validation",...
    ExecutionEnvironment="gpu",...
    Plots="training-progress");
netTrained = trainnet(XTrain,YTrain,net,"mse",options);
