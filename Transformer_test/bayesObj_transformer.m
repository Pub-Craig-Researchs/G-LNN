function valError = bayesObj_transformer(xTrain, yTrain, xVal, yVal, params)
rng("default")

net = dlnetwork;

if(params.ifrescale==0)
    tempNet = [
        sequenceInputLayer(1,"Name","input","Normalization","rescale-zero-one")];
else
    tempNet = [
        sequenceInputLayer(1,"Name","input","Normalization","zscore")];
end
net = addLayers(net,tempNet);

tempNet = positionEmbeddingLayer(params.numChannels,252,"Name","positionembed");
net = addLayers(net,tempNet);
numKeyChannels = params.numKeyChannels*params.numHeads;
switch params.numLayers
    case 1
        tempNet = [
            additionLayer(2,"Name","addition")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_0","AttentionMask","none")
            indexing1dLayer("last", "Name", "indexing")
            fullyConnectedLayer(4,"Name","fc")];
    case 2 
        tempNet = [
            additionLayer(2,"Name","addition")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_0","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_end","AttentionMask","none")
            indexing1dLayer("last", "Name", "indexing")
            fullyConnectedLayer(4,"Name","fc")
            ];
    case 3
       tempNet = [
            additionLayer(2,"Name","addition")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_0","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_1","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_end","AttentionMask","none")
            indexing1dLayer("last", "Name", "indexing")
            fullyConnectedLayer(4,"Name","fc")];
    case 4
       tempNet = [
            additionLayer(2,"Name","addition")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_0","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_1","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_2","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_end","AttentionMask","none")
            indexing1dLayer("last", "Name", "indexing")
            fullyConnectedLayer(4,"Name","fc")];
    case 5
        tempNet = [
            additionLayer(2,"Name","addition")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_0","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_1","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_2","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_3","AttentionMask","none")
            selfAttentionLayer(params.numHeads,numKeyChannels,"Name","selfattention_end","AttentionMask","none")
            indexing1dLayer("last", "Name", "indexing")
            fullyConnectedLayer(4,"Name","fc")];

end
net = addLayers(net,tempNet);

net = connectLayers(net,"input","addition/in1");
net = connectLayers(net,"input","positionembed");
net = connectLayers(net,"positionembed","addition/in2");
net = initialize(net);
glnn = net;
%plot(net)
options = trainingOptions("adam", ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=5,...
    LearnRateDropFactor=0.2,...
    Verbose=false,...
    MaxEpochs=30, ...
    MiniBatchSize=params.BatchSize, ...
    InitialLearnRate=1e-3,...
    Shuffle="every-epoch",...
    ExecutionEnvironment="gpu",...
    Plots="none");

netTrained = trainnet(xTrain,yTrain,glnn,"mse",options);
YPred = predict(netTrained, xVal);
valError = extractdata(mean((YPred - yVal).^2,"all"));
end
