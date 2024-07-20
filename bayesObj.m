function valError = bayesObj(xTrain, yTrain, xVal, yVal, params)
% Deep LPPLS (Nielsen et al., 2024) uses grid search, here we employ Bayesian optimization which is generally more efficient.
rng("default")
input_size = 252;
net = dlnetwork;

if(params.ifnorm==0)
    tempNet = [
        featureInputLayer(input_size,"Name","input","Normalization","rescale-zero-one")
        fullyConnectedLayer(params.numUnits_1st,"Name","fc_1")
        geluLayer("Name","gelu_0")];
else
    tempNet = [
        featureInputLayer(input_size,"Name","input","Normalization","rescale-zero-one")
        fullyConnectedLayer(params.numUnits_1st,"Name","fc_1")
        layerNormalizationLayer
        geluLayer("Name","gelu_0")];
end
net = addLayers(net,tempNet);

switch params.numLayers_1
    case 1
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            layerNormalizationLayer("Name","layernorm_4")];
    case 2 
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
    case 3
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_4")
            preluLayer("Name","prelu_3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
    case 4
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_3")
            preluLayer("Name","prelu_2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_4")
            preluLayer("Name","prelu_3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
    case 5
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_3")
            preluLayer("Name","prelu_2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_4")
            preluLayer("Name","prelu_3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+1")
            preluLayer("Name","prelu_+1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
    case 6
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_3")
            preluLayer("Name","prelu_2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_4")
            preluLayer("Name","prelu_3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+1")
            preluLayer("Name","prelu_+1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+2")
            preluLayer("Name","prelu_+2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
    case 7
        tempNet = [
            fullyConnectedLayer(params.numUnits_1,"Name","fc_2")
            preluLayer("Name","prelu_1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_3")
            preluLayer("Name","prelu_2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_4")
            preluLayer("Name","prelu_3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+1")
            preluLayer("Name","prelu_+1")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+2")
            preluLayer("Name","prelu_+2")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_+3")
            preluLayer("Name","prelu_+3")
            fullyConnectedLayer(params.numUnits_1,"Name","fc_5")
            layerNormalizationLayer("Name","layernorm_4")];
end
net = addLayers(net,tempNet);

switch params.numLayers_2
    case 1
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(2,"Name","op_2")];
    case 2
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_2")
            geluLayer("Name","gelu_3")
            fullyConnectedLayer(2,"Name","op_2")];
    case 3
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_2")
            geluLayer("Name","gelu_3")
            fullyConnectedLayer(params.numUnits_2,"Name","std_3")
            geluLayer("Name","gelu_4")
            fullyConnectedLayer(2,"Name","op_2")];
    case 4
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_2")
            geluLayer("Name","gelu_3")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+1")
            geluLayer("Name","gelu_+1")
            fullyConnectedLayer(params.numUnits_2,"Name","std_3")
            geluLayer("Name","gelu_4")
            fullyConnectedLayer(2,"Name","op_2")];
    case 5
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_2")
            geluLayer("Name","gelu_3")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+1")
            geluLayer("Name","gelu_+1")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+2")
            geluLayer("Name","gelu_+2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_3")
            geluLayer("Name","gelu_4")
            fullyConnectedLayer(2,"Name","op_2")];
    case 6
        tempNet = [
            fullyConnectedLayer(params.numUnits_2,"Name","std_1")
            geluLayer("Name","gelu_2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_2")
            geluLayer("Name","gelu_3")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+1")
            geluLayer("Name","gelu_+1")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+2")
            geluLayer("Name","gelu_+2")
            fullyConnectedLayer(params.numUnits_2,"Name","std_+3")
            geluLayer("Name","gelu_+3")
            fullyConnectedLayer(params.numUnits_2,"Name","std_3")
            geluLayer("Name","gelu_4")
            fullyConnectedLayer(2,"Name","op_2")];
end
net = addLayers(net,tempNet);



tempNet = [
    fullyConnectedLayer(params.numUnits_1,"Name","Res")
    layerNormalizationLayer("Name","layernorm_5")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","addition")
    preluLayer("Name","prelu_4")
    fullyConnectedLayer(2,"Name","op_1")];
net = addLayers(net,tempNet);

tempNet = depthConcatenationLayer(2,"Name","depthcat");
net = addLayers(net,tempNet);

net = connectLayers(net,"gelu_0","fc_2");
net = connectLayers(net,"gelu_0","Res");
net = connectLayers(net,"gelu_0","std_1");
net = connectLayers(net,"op_2","depthcat/in2");
net = connectLayers(net,"layernorm_4","addition/in1");
net = connectLayers(net,"layernorm_5","addition/in2");
net = connectLayers(net,"op_1","depthcat/in1");
net = initialize(net);
glnn = net;
%plot(net)
options = trainingOptions("adam", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=5,...
    LearnRateDropFactor=0.2,...
    ValidationData={xVal,yVal},...
    ValidationFrequency=5000,...
    ValidationPatience=3,...
    OutputNetwork="best-validation",...
    Verbose=false,...
    MaxEpochs=30, ...
    MiniBatchSize=6, ...
    InitialLearnRate=5e-6,...
    Shuffle="every-epoch",...
    ExecutionEnvironment="gpu",...
    Plots="none");

netTrained = trainnet(xTrain,yTrain,glnn,"mse",options);
YPred = predict(netTrained, xVal);
valError = extractdata(mean((YPred - yVal).^2,"all"));
end
