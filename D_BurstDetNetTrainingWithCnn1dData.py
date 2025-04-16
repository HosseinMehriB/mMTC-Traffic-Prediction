# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Tue Jun  6 17:05:14 2023

@author: Hossein Mehri

This code trains the burst detection network using processed data bunches from
both actual and predicted traffic patterns.

This code executes the third of four steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.
The CNN-1D-based model only uses the traditional rolling algorithm and the output
is used for performance comparison with RNN-based models that can use FLSP algorithm.


 --------------------------      ------------------      vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
|GeneratingTrafficPatterns| --> |CNN-1DNetTraining| --> |BurstDetNetTrainingWithCNN-1DData| -->
--------------------------      ------------------      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     --------------------------------------
--> |EvaluationAndPlottingResultsForCNN-1D|
    --------------------------------------

The code loads required the trained model from "B_traindCNN1D" folder.
The trained network is stored in "D_trainedBurstNetAndEvalsForCnn1D" folder.

You can modify the following parameters:
    - epochs: Number of iterations over the training streams;
    - dropoutVal: Droupout value;  
    - validationDataSize: Validation data size;
    - trainingDataSize: Training data size.

Citation:
If you find this code useful in your research, please consider citing our paper:

H. Mehri, H. Mehrpouyan, and H. Chen, "RACH Traffic Prediction in Massive Machine Type Communications," IEEE Transactions on Machine Learning in Communications and Networking, vol. 3, pp. 315–331, 2025.

[IEEE Xplore Link](https://ieeexplore.ieee.org/document/10891603) | DOI: 10.1109/TMLCN.2025.3542760

```
@ARTICLE{10891603,
  author={Mehri, Hossein and Mehrpouyan, Hani and Chen, Hao},
  journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
  title={RACH Traffic Prediction in Massive Machine Type Communications}, 
  year={2025},
  volume={3},
  number={},
  pages={315-331},
  keywords={Prediction algorithms;Long short term memory;Machine learning;Accuracy;3GPP;Predictive models;Complexity theory;Base stations;Traffic control;Telecommunication traffic;Massive machine-type communications;internet of things;machine learning;traffic prediction;smart cities},
  doi={10.1109/TMLCN.2025.3542760}}
```

"""

import torch
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from time import sleep
from MLNetworks import linearNetwork2
from MLNetworks import dataManagement, burstTargetDataGen, cnn1dPreprocess
from MLNetworks import textColorFormat as TC

# This function gets the linear network and training data and trains the network.
def trainLinearNet(burstDetectionNet, feed, labels, epochs, iTp, Tp, TpVec, \
                   maxExtraIter=5, printExtraInfo=False):
    """
    This function receives the feed data and labels indicating the bursty area,
    then trains and returns the given burst detection network.

    burstDetectionNet, lossHist, (Precision,Recall,F1Score) = \
        trainLinearNet(burstDetectionNet, feed, labels, epochs, iTp, Tp, TpVec)

    Parameters
    ----------
    burstDetectionNet : linearNetwork
        A burst detection network which is goinf to be trained.
    feed : tensor
        Torch tensor of size [sequences, slices, sliceSize], which is fed to the
        burst detection network.
    labels : tensor
        Labels of each slice in the training data indicating if we have a bursty
        traffic or not.
    epochs : int
        Indicates the iterations over the training streams.
    iTp : int
        Indicates the number of Tp. Just used to show the progress of the simulation.
    Tp : float
        Indicates the prediction duration at each step. Just used to show the 
        progress of the simulation.
    TpVec : list(int)
        list of Tps. Just used to show the progress of the simulation.
    maxExtraIter : int
        Maximum repeatition of the training (each time, 'epochs' iterations) when 
        error value is high. The default is 5.

    Returns
    -------
    burstDetectionNet : linearNetwork
        Trained burst detection network.
    lossHist : list(float)
        History of the training loss.
    (Precision,Recall,F1Score) : tuple(list(float))
        History of three accuracy metrics:
            - Precision = TruePositive/(TruePositive+FalsePositive) => Are all predicted events really bursty event?
            - Recall = TruePositive/(TruePositive+FalseNegative)    => How many percent of events are detected?
            - F1-Score = (2*Precision*Recall)/(Precision+Recall)    => Balance of Recall and Precision.
    """
    lossHist=[]  # Container for training loss values.
    criterion = nn.MSELoss() # Mean square error criterion for loss calculation
    optimizer = optim.Adam(burstDetectionNet.parameters(),lr = 0.000005) # model.parameters() gives the parameters that ...
    burstDetectionNet.train() # Training mode activation.
    Precision=[] # Container for precision metric.
    Recall=[]    # Container for Recall metric.
    F1Score=[]   # Container for F1-Score metric.
    exIter=0     # Indicator of extra iterations when it is needed.
    extraIterNote='' # To inform the extra iterations.
    while(True): # We continue training until reaching to a reliable accuracy.
        desc=f'Tp= {Tp} ({iTp+1} out of {len(TpVec)})'+extraIterNote
        for epoch in tqdm(range(epochs),colour='yellow',desc=desc):
            lossTemp=0 # To accumulate the loss value for each sequence.
            truePositive=0 # Correct predicted '1's (used to calculate new metrics).
            falsePositive=0 # Incorrect predicted '1's (used to calculate new metrics).
            falseNegative=0 # Missed '1's in the predictions (used to calculate new metrics).
            onesOfOut=0     # Indicates the '1's in the output sequence (used to determine accuracy).
            onesOfTar=0.001 # Indicates the '1's in the target sequence (used to determine accuracy).
            for i,(inputTensor,target) in enumerate(zip(feed,labels)):
                output=burstDetectionNet(inputTensor) # Get the output
                truePositive+=sum(target[:output.size(0)]*(output.squeeze()>0.5)).item() # True Positives
                falsePositive+=sum((-1*target[:output.size(0)]+1)*(output.squeeze()>0.5)).item() # False Positives
                falseNegative+=sum((-1*(output.squeeze()>0.5)+1)*target[:output.size(0)]).item() # False Negatives
                if (epoch==epochs-1): # Calculating the acuutacy metrics for the last epoch
                    indexTar=(target[:output.size(0)]).nonzero(as_tuple=True) # Non zero elements of target
                    indexOut=(output>0.5).nonzero(as_tuple=True) # Non zero elements of output
                    onesOfOut+=indexOut[0].size(0)-indexTar[0].size(0)
                    onesOfTar+=indexTar[0].size(0)
                    # Printing the results of last epoch (optional).
                    if i==0: # Printing the total number of detected and target '1's in a sequence:
                        print('\n')
                    if printExtraInfo:
                        print(f"indexOut ones: {indexOut[0].size(0):2.0f}, indexTar:",\
                          f'{indexTar[0].size(0):2.0f} --> '+\
                          f'diff = {indexOut[0].size(0)-indexTar[0].size(0): 2.0f}')
                    if (i+1)==feed.shape[0]: # Printing the average after the last sequence:
                        print(f'\t\t\033[38;5;11mDiff average: \033[48;5;9m{onesOfOut/(i+1): 2.2f}'+\
                              f' ({onesOfOut/onesOfTar: 3.2%})\033[0m\n')
                        sleep(0.2)
                loss=criterion(output.squeeze(),target[:output.size(0)])
                lossTemp+=loss.item()
                loss.backward()
                optimizer.step()
            lossHist.append(lossTemp/(i+1))
            Precision.append(truePositive/(truePositive+falsePositive+0.01))
            Recall.append(truePositive/(truePositive+falseNegative+0.01))
            F1Score.append(2*Precision[-1]*Recall[-1]/(Precision[-1]+Recall[-1]+0.01))
        # Stop training only if:
            # - The missed bursty points are less than 5%;
            # - We want to train the network only one time (epochs==1);
            # - We reached to the maximum number of iterations (max=maxExtraIter).
        if (abs(onesOfOut/onesOfTar)<0.05 or epochs==1 or exIter>=maxExtraIter): 
            break
        else: # Otherwise, conduct extra iterations to reach to the minimum accuracy.
            exIter+=1
            extraIterNote=f'\033[48;5;6m(extra iteration number {exIter} '+\
                f'(max={maxExtraIter}))\033[0m'
            print('+++...\n')
    return burstDetectionNet, lossHist, (Precision,Recall,F1Score)

# This function uses "weighted error method" to calculate the loss value
def myCriterion(estimatedData,targetData,weight=30):
    classZeroIndices=(targetData<0.5).nonzero(as_tuple=True)[0]
    classOneIndices=(targetData>0.5).nonzero(as_tuple=True)[0]
    classZeroErrors=sum((estimatedData[classZeroIndices]-targetData[classZeroIndices])**2)
    classOneErrors=sum((estimatedData[classOneIndices]-targetData[classOneIndices])**2)
    return (classZeroErrors+classOneErrors*weight)/estimatedData.size(0)

# This function uses "weighted error method" to calculate the loss value
def myCriterion2(estimatedData,targetData,weight=1):
    classZeroIndices=(targetData<0.5).nonzero(as_tuple=True)[0]
    classOneIndices=(targetData>0.5).nonzero(as_tuple=True)[0]
    classZeroErrors=sum((estimatedData[classZeroIndices]-targetData[classZeroIndices])**2)/(classZeroIndices.size(0)+classOneIndices.size(0)/100)
    classOneErrors=sum((estimatedData[classOneIndices]-targetData[classOneIndices])**2)/(classOneIndices.size(0)+0.1)
    return classZeroErrors+classOneErrors*weight

#%% Main function:
if __name__=="__main__":
    epochs=40 # Number of iterations over the training streams.
    trainingDataSize=100 # Training dataset size
    validationDataSize=10 # Validation dataset size #### Not implemented in this code
    testDataSize=40
    maxExtraIter=5 # Maximum extra trainings of burst detection network when error rate is high.
    frameSize=0.005 # Each time slot duration    
    dropoutVal = 0.4 # Dropout probability of burst detection network
    linearNetLayers=3 # Number of linear layers of burst detection network
    GPU_flag=False # GPU flag
    (cnn1dTrainedModels,cnn1dAddr)=dataManagement(path='B_traindCNN1D',\
                                                  choice=True,\
                                                  returnAddress=True)
    allLoadedAddress=cnn1dTrainedModels.get('allLoadedAddress') # Loading the container of loaded files' address
    allLoadedAddress.update({'addressLoadCNN1D':cnn1dAddr}) # Adding the address of trained CNN-1D model
    Tstep=cnn1dTrainedModels.get('Tfeed') # Fresh data size which determines the step size
    TpVec=cnn1dTrainedModels.get('TpVec') # 
    nType=cnn1dTrainedModels.get('nType') # Prediction model type
    if GPU_flag:
        device=torch.device('cuda') # Laod on GPU.
    else:
        device=torch.device('cpu') # Laod on CPU.
    # Loading traffic patterns:
    dataLoad=dataManagement(address=allLoadedAddress.get('patternDataAddress'))
    Pattern=dataLoad.get('Pattern') # List containing simulation results: [detected, attempted,congestion,freeP]
    trainingSet=np.random.choice(range(len(Pattern)-100-validationDataSize),\
                                 trainingDataSize, replace=False) # Random choose
    chosenValStreams=[i for i in range(len(Pattern)-100-validationDataSize,\
                                        len(Pattern)-100)]
    chosenTestStreams=np.random.choice(range(len(Pattern)-100,len(Pattern)),\
                                 testDataSize, replace=False) # Random choose    
    #%% Generating predictions for each Tp:
    trainedBurstData=dict()
    for iTp,Tp in enumerate(TpVec):
        # Generating the input data and labels of burst prediction network using the original data:
        labels=[]
        feed=[]
        for idx in tqdm(trainingSet, colour='blue',\
                              desc='Training data processing from original data '+\
                                  f'for Tp: {Tp} ({iTp+1} out of {len(TpVec)})'):
            sequence=Pattern[idx]
            # Extracting the burst labels from congestion sequence (batchSize=25 just to find bursty area!):
            [beacon,_,feedD]=burstTargetDataGen(sequence[0], sequence[2],\
                                      25,Tp,Tstep,frameSize,onlyPredData=True) # sequence[0]: traffic; sequence[2]: congestion
            labels.append(torch.from_numpy(np.array(beacon,copy=True))) # Burst labels
            feed.append(feedD.detach().clone()) # Feed data
        labels=torch.stack(labels)
        feed=torch.stack(feed)
        #%% Creating and training the burst detection network:
        sliceSize=int(feed[0].size(-1)) # input sequection slice size.
        burstDetectionNet=linearNetwork2(sliceSize, outputSize=1, layers=linearNetLayers,\
                                         dropoutVal=dropoutVal, GPU_flag=GPU_flag,\
                                         track_states = False)
        burstDetectionNet.to(device)
        burstDetectionNet.double()
        burstDetectionNet.train() # Training mode activation.
        [burstDetectionNet, trainLossHist, trainAccuracy]=trainLinearNet(\
                                         burstDetectionNet, feed.to(device), labels.to(device), \
                                         epochs, iTp, Tp, TpVec, maxExtraIter=maxExtraIter)
        #%% Validating the trained network using the processed original data:
        burstDetectionNet.eval() # Evaluation mode activation.
        criterion = nn.MSELoss() # Mean square error criterion for loss calculation
        valLoss=[]
        for idx in tqdm(chosenValStreams, colour='cyan',\
                              desc=f'Validation test for Tp: {Tp} ({iTp+1} '+\
                                  f'out of {len(TpVec)})'):
            sequence=Pattern[idx]
            # Extracting the burst labels from congestion sequence (batchSize=25 just to find bursty area!):
            [beacon,_,feedD]=burstTargetDataGen(sequence[0], sequence[2],\
                                      25,Tp,Tstep,frameSize,onlyPredData=True) # sequence[0]: traffic; sequence[2]: congestion
            beacon=torch.from_numpy(np.array(beacon,copy=True))
            output=burstDetectionNet(feedD) # Getting output
            valLoss.append(criterion(output.squeeze(),beacon[:output.size(0)]).item())
            
        #%% Now load the trained CNN-1D network and use predicted data for test:
        # cnn1dTrainedModels
        trainedCNN1D=cnn1dTrainedModels[Tp].get('model') # Loading the trained CNN network
        windowSize=cnn1dTrainedModels[Tp].get('windowSize')
        Tfeed=cnn1dTrainedModels[Tp].get('Tfeed')
        # preprocess the data for CNN-1D:
        steps=int(Tp/frameSize) # Prediction steps in [time slots]
        feedSize=int(Tfeed/frameSize) # Feed data size in [time slots] (determines the resolution of input sequences)
        burstDetTestLossMSE=0
        accuracyMetrics={'Precision':0, 'Recall':0, 'F1_score':0}
        #%
        cnnPredTestLoss=0 # Calculating the traffic prediction loss on test data using CNN-1D network 
        for i,loc in enumerate(tqdm(chosenTestStreams, colour='magenta',\
                                    desc=f'Testing models for Tp: {Tp} ({iTp+1} '+\
                                        f'out of {len(TpVec)})')):
            # Generating the training tensors (input and target) from the selected patterns:
            [precessedInTraffic, precessedTargetTraffic]=cnn1dPreprocess\
                                    (np.expand_dims(Pattern[loc][0],axis=0),\
                                    windowSize=windowSize, feedSize=feedSize, steps=steps)
            [precessedInCongestion, precessedTargetCongestion]=cnn1dPreprocess\
                                    (np.expand_dims(Pattern[loc][2],axis=0),\
                                    windowSize=windowSize, feedSize=feedSize, steps=steps)
            # The ML network gets traffic and congestion values as input features. The \
            # input shape will be [trainingDataSize, batches, featureSize, windowsSize]:
            inputs=torch.cat((precessedInTraffic[0],precessedInCongestion[0]),1) # Appending traffic and congestion tensors.
            targets=torch.cat((precessedTargetTraffic[0],precessedTargetCongestion[0]),1)
            # preprocess the data for burst detection network            
            [beacon,_,_]=burstTargetDataGen(Pattern[loc][0], Pattern[loc][2],\
                                      25,Tp,Tstep,frameSize,onlyPredData=True) # sequence[0]: traffic; sequence[2]: congestion
            startPoint=int(windowSize/int(Tstep/frameSize)) # Ignoring the first labels regarding the feed window data
            beacon=torch.from_numpy(np.array(beacon,copy=True))[startPoint:]
            # Generating the predictions using CNN-1D and feeding the output to burst detection network:
            preds=trainedCNN1D(inputs)
            # Calculating the traffic prediction loss:
            cnnPredTestLoss+=(criterion(preds[:,:,-feedSize:],targets[:,:,-feedSize:]).item())/testDataSize
            predictedBurst=burstDetectionNet(preds,reshape=False).squeeze() # Getting output without reshaping
            predictedBurst[predictedBurst<0.5]=0 # Below 0.5 => '0'
            predictedBurst[predictedBurst>0.5]=1 # Above 0.5 => '1'
            beacon=beacon[:len(predictedBurst)]
            burstDetTestLossMSE+=criterion(predictedBurst,beacon).item()/testDataSize
            
            truePositive=sum(beacon*predictedBurst).item() # True Positives
            falsePositive=sum((-1*beacon+1)*predictedBurst).item() # False Positives
            falseNegative=sum((-1*predictedBurst+1)*beacon).item() # False Negatives
            precision=truePositive/(truePositive+falsePositive+0.001)
            recall=truePositive/(truePositive+falseNegative+0.001)
            f1Score=2*precision*recall/(precision+recall+0.001)
            accuracyMetrics['Precision']+=precision/testDataSize
            accuracyMetrics['Recall']+=recall/testDataSize
            accuracyMetrics['F1_score']+=f1Score/testDataSize
        print(TC(f'\nF1-Score of Tp={Tp}: {accuracyMetrics["F1_score"]:.2f}.\n\n','ynn'))        
        trainedBurstData[Tp]={'burstDetectionNet':burstDetectionNet,\
                              'trainLossHist':trainLossHist,\
                              'trainAccuracy':trainAccuracy, 'valLoss':valLoss,\
                              'cnnPredTestLoss':cnnPredTestLoss,\
                              'burstDetTestLoss':burstDetTestLossMSE,\
                              'accuracyMetrics':accuracyMetrics}
    #%% Printing the simulation results and saving the trained models:
    description_B=TC('\nAccumulated descriptions of B2_Cnn1DNetTraining phase:\n\n\n','ryu')
    description_D=TC('\n\nBurst detection network is trained on ','y00')+TC('original','y0u')+\
        TC(' data and validated by the ','y00')+TC('original','y0u')+\
        TC(' data. Then this network is tested by ','y00')+TC('predicted','y0u')+\
        TC(' data generated by the CNN-1D network (Fixed window size (v3)). '+\
           'The network parameters are as follows:\n\t','y00')+\
            f'- Training epochs: {epochs};\n\t'+\
            f'- Maximum extra iteration in training step: {maxExtraIter} (x epochs);\n\t'+\
            f'- Training data size: {trainingDataSize};\n\t'+\
            f'- Validation data size: {validationDataSize};\n\t'+\
            f'- Test data size: {testDataSize};\n\t'+\
            f'- Linear layers of burst detection network: {linearNetLayers};\n\t'+\
            f'- Dropout value: {dropoutVal};\n\t'+\
            '- Tfeed (Tstep): fresh data size collected at each step: '+\
            f'{Tstep} second(s);\n\t'+\
            f'- TpVec: {TpVec};\n'
    description_D+=TC('\nSimulation results for traffic predcition and burst ','y00')+\
        TC('detection networks when using ','y00')+TC('Test','y0u')+TC(' data:\n\t','y00')+'-'*109+\
        '\n\t'+TC('|\tTp\t|\tTest Traf. Pred. MSE (CNN-1D)\t|\tBurst Det. Net. MSE\t|\tPrecision\t|\t'+\
        'Recall\t\t|\tF1-score\t|','my0')+'\n\t'+'+'*109+'\n\t'
    theme=[[0,5,0],[0,6,0],[0,2,0],[0,1,0]]
    for iTp, Tp in enumerate(TpVec):
        txt=f'|\t{Tp}\t'+\
            f'|\t\t\t\t{trainedBurstData[Tp]["cnnPredTestLoss"]: 5.2f}'+'\t'*5+\
            f'!\t\t\t{trainedBurstData[Tp]["burstDetTestLoss"]: 5.2f}'+'\t'*3+\
            f'!\t{trainedBurstData[Tp]["accuracyMetrics"]["Precision"]: 5.2f}'+'\t'*2+\
            f'!\t{trainedBurstData[Tp]["accuracyMetrics"]["Recall"]: 5.2f}'+'\t'*2+\
            f'!\t{trainedBurstData[Tp]["accuracyMetrics"]["F1_score"]: 5.2f}'+'\t'*2+'!'
            
        description_D+=TC(txt,theme[iTp])+'\n\t'+'-'*109+'\n\t'
        description_B+=cnn1dTrainedModels[Tp]['Description']+'\n'+'+'*110+'\n\n'
    print(description_B)
    print(description_D)
    
    data={'trainedBurstData':trainedBurstData, 'epochs':epochs, \
          'trainingDataSize':trainingDataSize, 'validationDataSize':validationDataSize,\
          'testDataSize':testDataSize, 'maxExtraIter':maxExtraIter, \
          'dropoutVal':dropoutVal, 'linearNetLayers':linearNetLayers, 'Tstep':Tstep,\
          'TpVec':TpVec, 'allLoadedAddress':allLoadedAddress, \
          'description_D2':description_D, 'description_B2':description_B}               
    dataManagement(data=data, save=True, fileFormat='.pt', fileName='BurstNetAndEvals_'+\
                    nType+f'_WS{windowSize}', version='_v3', path='D_trainedBurstNetAndEvalsForCnn1D')
