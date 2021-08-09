imds = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\train\','IncludeSubFolders',true,'LabelSource','foldernames');
imdsTest = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\test\','IncludeSubFolders',true,'LabelSource','foldernames');
imdsVal = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\val\','IncludeSubFolders',true,'LabelSource','foldernames');

imds = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\TB_Chest_Radiography_Database\train','IncludeSubFolders',true,'LabelSource','foldernames');
imdsTest = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\TB_Chest_Radiography_Database\test','IncludeSubFolders',true,'LabelSource','foldernames');
imdsVal = imageDatastore('C:\Users\Computer\Desktop\M.Eng\EE8204 Neural Networks\Final Project\TB_Chest_Radiography_Database\val','IncludeSubFolders',true,'LabelSource','foldernames');

w = 96;
h = 96;

auimds  = augmentedImageDatastore([w,h],imds,'ColorPreprocessing','rgb2gray');
auimdsTest  = augmentedImageDatastore([w,h],imdsTest,'ColorPreprocessing','rgb2gray');
auimdsVal  = augmentedImageDatastore([w,h],imdsVal,'ColorPreprocessing','rgb2gray');

%imds.Labels = repelem(1,numImages);

%imdsFinal = imageDatastore([imds.Files; imds.Labels.Files]);

% figure
% numImages = 1341;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i)
%     imshow(imds.Files{perm(i)});
%     drawnow;
% end

miniBatchSize = 128;
imds.ReadSize = miniBatchSize;

layers = [
    imageInputLayer([w h 1],'Normalization','none')
    convolution2dLayer(5,96)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(2);
    softmaxLayer
    classificationLayer];
 
maxEpochs = 2;

options = trainingOptions('adam', ...
    'Plots','training-progress', ...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',miniBatchSize,...
    'ValidationData',auimdsVal);

% 
% options = trainingOptions('adam', ...
%     'Plots','training-progress', ...
%     'MaxEpochs',maxEpochs,...
%     'MiniBatchSize',miniBatchSize);
 
net = trainNetwork(auimds,layers,options);

%[XTest,YTest] = imdsTest;
YPred = classify(net,auimdsTest,'MiniBatchSize',miniBatchSize);


acc = sum(YPred == imdsTest.Labels)./numel(imdsTest.Labels);

% 
% 
% [XTrain,~,YTrain] = digitTrain4DArrayData;
% [XValidation,~,YValidation] = digitTest4DArrayData;
% numTrainImages = numel(YTrain);
% figure
% idx = randperm(numTrainImages,20);
% for i = 1:numel(idx)
%     subplot(4,5,i)    
%     imshow(XTrain(:,:,:,idx(i)))
% end
% figure
% histogram(YTrain)
% axis tight
% ylabel('Counts')
% xlabel('Rotation Angle')
% layers = [
%     imageInputLayer([28 28 1])
%     convolution2dLayer(3,8,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     averagePooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     averagePooling2dLayer(2,'Stride',2)
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer
%     dropoutLayer(0.2)
%     fullyConnectedLayer(1)
%     regressionLayer];
% miniBatchSize  = 128;
% validationFrequency = floor(numel(YTrain)/miniBatchSize);
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MaxEpochs',30, ...
%     'InitialLearnRate',1e-3, ...
%     'LearnRateSchedule','piecewise', ...
%     'LearnRateDropFactor',0.1, ...
%     'LearnRateDropPeriod',20, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',{XValidation,YValidation}, ...
%     'ValidationFrequency',validationFrequency, ...
%     'Plots','training-progress', ...
%     'Verbose',false);
% net = trainNetwork(XTrain,YTrain,layers,options);
% net.Layers
% YPredicted = predict(net,XValidation);
% predictionError = YValidation - YPredicted;
% thr = 10;
% numCorrect = sum(abs(predictionError) < thr);
% numValidationImages = numel(YValidation);
% 
% accuracy = numCorrect/numValidationImages
% squares = predictionError.^2;
% rmse = sqrt(mean(squares))
% figure
% scatter(YPredicted,YValidation,'+')
% xlabel("Predicted Value")
% ylabel("True Value")
% 
% hold on
% plot([-60 60], [-60 60],'r--')
% idx = randperm(numValidationImages,49);
% for i = 1:numel(idx)
%     image = XValidation(:,:,:,idx(i));
%     predictedAngle = YPredicted(idx(i));  
%     imagesRotated(:,:,:,i) = imrotate(image,predictedAngle,'bicubic','crop');
% end
% idx = randperm(numValidationImages,49);
% for i = 1:numel(idx)
%     image = XValidation(:,:,:,idx(i));
%     predictedAngle = YPredicted(idx(i));  
%     imagesRotated(:,:,:,i) = imrotate(image,predictedAngle,'bicubic','crop');
% end