# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Tue Jun  6 17:23:46 2023

@author: Hossein Mehri

This code evaluates the performance of Online-Prediction algorithm and compares
it with traditional Rolling algorithm. This comparison includes both accuracy 
of predictions and simulation time and results returns as a table. Moreover,
the performance of burst detection network is evaluated using the output of 
traind RNN network over the test data. Finally, a few exemplary test sequences 
are chosen to generate the prediction results for plotting purposes.

This code executes the fifth of six steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.

 --------------------------      ---------------      ------------------------------------
|GeneratingTrafficPatterns| --> |RnnNetTraining| --> |GeneratingBurstDetNetFeedDataForRNN| -->
--------------------------      ---------------      ------------------------------------
     -------------------------------      vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
--> |BurstDetNetTrainingWithRNNData| --> |EvaluatingNetworksAndGeneratingResultsForRNN| -->
    -------------------------------      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     ------------------------------------------
--> |PlottingResultsAndExportingToMatlabForRNN|
    ------------------------------------------

The code loads required data from "A_generatedTraffic", "B_trainedRNN",
"C_burstFeedDataForRnn", and "D_trainedBurstNetForRnn" folders.
The results are stored in "E_EvaluationResultsForRnn" folder.

You can modify the following parameters:
    - testDataSize: Number of sequences chosen to evaluate the accuracy of 
                    predictions by traffic prediction and burst detection networks.
    - comparisionDataSize: Number of sequences chosen randomly to compare the 
                           performance of Classical Rolling prediction and propose
                           Online predcition (FLSP) algorithms.

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
import copy
import numpy as np
from tqdm import tqdm
from scipy import signal
from datetime import datetime
from MLNetworks import rnnNetwork
from MLNetworks import dataManagement
from MLNetworks import  textColorFormat as TC
from MLNetworks import linearNetwork
from B_RnnNetTraining import preprocess

def tab(rep=1):
    '''
    A function to return variable number of Tab character in a string.

    Parameters
    ----------
    rep : int, optional
        Number of '\t' repeatitions. The default is 1.

    Returns
    -------
    string : str
        returns a string.
    '''
    return '\t'*rep

# To evaluate the performance of trained networks on test data, we use the last 
# 100 sequences out of 500 sequences as a test data set.
if __name__=='__main__':
    """ In "C_GenerateBurstDetectionNetFeedData.py", trained RNN network is used
    to generate predictions bunches for all of 500 sequences for Tp=2 seconds.
    These bunches of data can be used to generate predicted sequence for Tp=2
    or cropped for Tp=1.5, Tp=1, and Tp=0.5 seconds scenarios."""
    nType='GRU'
    # The only parameter that can be change for this code is the number of test sequences:
    testDataSize=100 # Number of sequences chosen randomly for test set.
    comparisionDataSize=30 # Number of sequences chosen randomly to compare the performance\
                # of Classical Rolling prediction and propose Online predcition algorithms.
    seqForPlotting=np.random.choice(range(400,500),2,replace=False) # Chosing two random streams for plotting purposes.
    startTime=datetime.now() # To record time.
    ## Loading the trained burst detection network:
    (burstNetData_D, burstNetAddress)=dataManagement(path='D_trainedBurstNetForRnn', \
                                            includesStr=nType,map_location='cpu', \
                                            returnAddress=True)
    burstPredictionNetworks=burstNetData_D.get('burstPredictionNetworks') # List of trained networks.
    allLoadedAddress=burstNetData_D.get('allLoadedAddress') # Loading the container of loaded files' address
    allLoadedAddress.update({'burstNetAddress':burstNetAddress}) # Adding the address of trained burst detection model
    ## Load predicted bunches of traffic by RNN network:
    RNNData_C=dataManagement(address=allLoadedAddress['rnnDataAddress']) # RNN data: generated by a trained RNN network Using FLSP.
    rnnPredictedBunches=torch.stack(RNNData_C.get('linearNetInputs'))
    """
    bunch=rnnPredictedBunches[0,0,:] --> shape:[1000] contains ([100-400-100-400]):
          - 100 points of ground truth traffic; - 100 points of ground truth congestion;
          - 400 points of predicted traffic;    - 400 points of predicted congestion.
    """
    RNNRollData_C=dataManagement(address=allLoadedAddress['rnnRollDataAddress']) # RNN data: generated by a trained RNN network.
    rnnRollPredictedBunches=RNNRollData_C.get('linearNetInputsRoll')
    bufferSizes=RNNRollData_C.get('bufferSizes')
    ## Loading the second set of data containing the labels of bursty areas:
    origData_C=dataManagement(address=allLoadedAddress['origDataAddress']) # Original data: generated using the ground truth sequences.
    burstLabels=origData_C.get('burstLabels') # Loading the burst labels.
    ## Lading the traffic patterns:
    dataLoad_A=dataManagement(address=allLoadedAddress['patternDataAddress']) # Loading the ground truth patterns   
    # List containing simulation results: [500][detected, attempted,congestion,freeP] - converted to np.array:
    patternArray=np.array(dataLoad_A.get('Pattern'))[:,[0,2],:] # Only traffic and congestion sequences 
    patternArray=signal.savgol_filter(patternArray,97,2) # Making data smooth.
    #%% Extracting the simulation parameters:
    batchSize=origData_C.get('batchSize') # Batch size
    TpVec=burstNetData_D.get('TpVec') # List of Tps
    Tseed=RNNData_C.get('Tseed')  # We should consider the seed length in chosing the labels for RNN data.
    Tstep=origData_C.get('Tstep') # Frequency of collecting new data as well as fresh data duration.
    TpMax=RNNData_C.get('TpMax')  # Maximum Tp used by traffic prediction network.
    frameSize=0.005 # A fixed parameter
    totalNumberOfPatterns=patternArray.shape[0] # Total number of available sequences
    # To consider the seed data used for initializing the RNN network, we should\
    # ignore first labels corresponding to the seed data:
    afterSeedLabelPointer=int(int(Tseed/frameSize/batchSize)/int(Tstep/frameSize/batchSize))
    seedSamples=int(Tseed/frameSize/batchSize)*batchSize # Seed length should be considered in error calculation.
    seedLength=int(Tseed/frameSize/batchSize) # Seed length in batches
    batchesAhead=[int(Tp/frameSize/batchSize) for Tp in TpVec] # List of batches ahead for each Tp.
    batchesAheadMax=int(TpMax/frameSize/batchSize)
    sliceSize=int(Tstep/frameSize/batchSize) # Size of each slice of fresh data
    stepLength=sliceSize # They are the same, just using both names for compatibaility with previous codes.
    #%% Starting the simulation by chosing random sequences from the FLSP generated sequences and initilizing the burst detection network's variables: 
    # The test sequences are chosen from the last 100 sequences as they are not \
    # used in the training phase: 
    chosenTestStreams=np.random.choice(range(totalNumberOfPatterns-100,\
                                             totalNumberOfPatterns),testDataSize,\
                                             replace=False)
    criterion = nn.MSELoss() # Using MSE metric to calculate the error.
    traffPredLoss=dict([(Tp,0) for Tp in TpVec]) # Container of traffic prediction loss for each Tp.
    burstDetLoss=dict([(Tp,0) for Tp in TpVec])  # Container of burst prediction loss for each Tp.
    """ Accuracy metrics for imbalanced data:
     - Precision = TruePositive/(TruePositive+FalsePositive) => Are all predicted events really bursty event?
     - Recall = TruePositive/(TruePositive+FalseNegative)    => How many percent of events are detected?
     - F1-Score = (2*Precision*Recall)/(Precision+Recall)    => Balance of Recall and Precision."""
    accuracyMetrics={'Precision':dict([(Tp,0) for Tp in TpVec]),\
                     'Recall':dict([(Tp,0) for Tp in TpVec]),\
                     'F1_score':dict([(Tp,0) for Tp in TpVec])}
    # Generating the predicted streams by croping and concatenating the predicted bunches:
    for j,(stream,pattern) in enumerate(tqdm(zip(rnnPredictedBunches[chosenTestStreams],\
                              torch.from_numpy(patternArray[chosenTestStreams])),\
                              total=testDataSize,colour='green',\
                              desc='Evaluating Traffic/Burst Prediction Networks')):
        for iTp,Tp in enumerate(TpVec):
            predTraffSeq=stream[:,batchesAhead[iTp]*batchSize:\
                             (batchesAhead[iTp]+stepLength)*batchSize].reshape(-1)
            predCongSeq=stream[:,(batchesAheadMax+stepLength+batchesAhead[iTp])*batchSize:\
                             (batchesAheadMax+stepLength+batchesAhead[iTp]+stepLength)*batchSize].reshape(-1)
            actualTraffSeq=pattern[0,(seedLength+batchesAhead[iTp])*batchSize:]
            actualCongSeq=pattern[1,(seedLength+batchesAhead[iTp])*batchSize:]
            lossTraff=criterion(predTraffSeq[:len(actualTraffSeq)],actualTraffSeq[:len(predTraffSeq)])
            lossCong=criterion(predCongSeq[:len(actualCongSeq)],actualCongSeq[:len(predCongSeq)])
            traffPredLoss[Tp]+=(lossTraff+lossCong)/2/testDataSize # Average loss for each TP
            ## Burst detection network performance evaluation:
            predictedBurst=((burstPredictionNetworks[iTp](stream)>0.5)*1)\
                .reshape(-1) # Predictions.
            targetBurst=torch.from_numpy(burstLabels[iTp][chosenTestStreams[j]]\
                                         [afterSeedLabelPointer:])[:len(predictedBurst)]
            burstDetLoss[Tp]+=criterion(predictedBurst,targetBurst)/testDataSize # Averaging loss
            # Calculating the accuracy metrics for burst detection network:
            truePositive=sum(targetBurst*predictedBurst).item() # True Positives
            falsePositive=sum((-1*targetBurst+1)*predictedBurst).item() # False Positives
            falseNegative=sum((-1*predictedBurst+1)*targetBurst).item() # False Negatives
            precision=truePositive/(truePositive+falsePositive)
            recall=truePositive/(truePositive+falseNegative)
            f1Score=2*precision*recall/(precision+recall)
            accuracyMetrics['Precision'][Tp]+=precision/testDataSize
            accuracyMetrics['Recall'][Tp]+=recall/testDataSize
            accuracyMetrics['F1_score'][Tp]+=f1Score/testDataSize
    #%% Calculating the accuracy metrics of Rolling methodusing the already predicted bunches:
    chosenTestStreams=np.linspace(0,99,100,dtype=np.int32) # All the test streams (100 streams)
    accuracyMetricsRoll=dict()
    for bufferSize in bufferSizes:
        accuracyMetricsRoll[bufferSize]={'Precision':dict([(Tp,0) for Tp in TpVec]),\
                         'Recall':dict([(Tp,0) for Tp in TpVec]),\
                         'F1_score':dict([(Tp,0) for Tp in TpVec])}
        testDataSizeRoll=len(rnnRollPredictedBunches[bufferSize])
        for j, stream in enumerate(tqdm(rnnRollPredictedBunches[bufferSize], colour='cyan',\
                                        desc='Burst detection network evaluation using'\
                                        f' Rolling output (BS: {bufferSize//stepLength})')):
            for iTp,Tp in enumerate(TpVec):
                predictedBurstRoll=((burstPredictionNetworks[iTp](stream)>0.5)*1)\
                    .reshape(-1) # Predictions.
                # The test streams are located at 400-500 indices:
                targetBurst=torch.from_numpy(burstLabels[iTp][chosenTestStreams[j]+400]\
                                             [afterSeedLabelPointer:])[:len(predictedBurst)]
                # Calculating the accuracy metrics for burst detection network:
                truePositive=sum(targetBurst*predictedBurstRoll).item() # True Positives
                falsePositive=sum((-1*targetBurst+1)*predictedBurstRoll).item() # False Positives
                falseNegative=sum((-1*predictedBurstRoll+1)*targetBurst).item() # False Negatives
                precision=truePositive/(truePositive+falsePositive)
                recall=truePositive/(truePositive+falseNegative)
                f1Score=2*precision*recall/(precision+recall)
                accuracyMetricsRoll[bufferSize]['Precision'][Tp]+=precision/testDataSizeRoll
                accuracyMetricsRoll[bufferSize]['Recall'][Tp]+=recall/testDataSizeRoll
                accuracyMetricsRoll[bufferSize]['F1_score'][Tp]+=f1Score/testDataSizeRoll        
    #%% Comparing the online prediction with classical rolling methods in terms of time complexity and accuracy:
    # Loading the trained RNN network and use it for both scenarios:
    # trainedRnnNet_B=dataManagement(address=RNNData_C.get('addressLoadRNN'))
    trainedRnnNet_B=dataManagement(address=allLoadedAddress.get('addressLoadRNN'))
    rnnModel=trainedRnnNet_B.get('model') # This is the trained traffic prediction.
    rnnModelRolling=copy.deepcopy(rnnModel) # The same model as above, used for classic rolling prediction model.
    if rnnModel.GPU_flag: # Moving the networks on the desired processing device
        device=torch.device('cuda') # Laod on GPU.
        revDev=torch.device('cpu')
    else:
        device=torch.device('cpu') # Laod on CPU.
        revDev=torch.device('cpu') # Always return on CPU!
    rnnModel.to(device) # Move the model to the desired device.
    rnnModelRolling.to(device) # Move the model to the desired device.
    chosenComparisonStreams=np.random.choice(range(totalNumberOfPatterns-100,\
                                             totalNumberOfPatterns),comparisionDataSize,\
                                             replace=False)
    criterion = nn.MSELoss() # Using MSE metric to calculate the error.
    lossOnline=dict([(Tp,0) for Tp in TpVec]) # Stores the online prediction errors.
    onlinePredStartTime=datetime.now() # It is used to calculate the time complexity of two methdos.
    for j, pattern in enumerate(patternArray[chosenComparisonStreams,:,:]):
        # We make the predictions for the maximum length and then crop it for shorter predictions:
        [feedTraf,tarTraf]=preprocess(pattern[0],batchSize) # Preparing the detected traffic data
        [feedCong,tarCong]=preprocess(pattern[1],batchSize) # Preparing the congestion data
        feedData=torch.cat((feedTraf,feedCong),dim=2) # Total feed data (concatenate traffic and congestion)
        targetData=torch.cat((tarTraf,tarCong),dim=2) # Total target data
        seed=feedData[:seedLength]
        output=dict([(Tp,[]) for Tp in TpVec]) # List to store predicted bunches.
        rnnModel.init_hidden(batchSize) # Initializing the RNN network.
        rnnModel.eval() # Evaluation mode activation.
        totalNumberOfSlices=int(feedData[seedLength:].size(0)/stepLength)
        # Generating the prediction:
        with torch.no_grad():
            outMax=rnnModel(seed.to(device),steps=batchesAheadMax, eval=True)
            for iTp,Tp in enumerate(TpVec):
                output[Tp].append(outMax[:seedLength+batchesAhead[iTp]-1].clone())
            for i in tqdm(range(totalNumberOfSlices),colour='cyan',\
                          desc=f'Online forecasting stream {j+1} out of {comparisionDataSize}         '):
                outMax=rnnModel(feedData[seedLength+i*stepLength:seedLength+(i+1)*stepLength].to(device),\
                              steps=batchesAheadMax, eval=True)
                for (Tp,BA) in zip(TpVec,batchesAhead): # cropping the desired size according to Tp
                    output[Tp].append(outMax[BA-1:BA-1+stepLength]) 
        for Tp in TpVec: # Error calculating
            prediction=torch.cat(output[Tp])[:targetData.size(0)].to(revDev)
            tarTensor=targetData[:prediction.size(0)]
            lossOnline[Tp]+=(criterion(prediction,tarTensor).item())/comparisionDataSize
    onlinePredEndTime=datetime.now() # End time of online prediction method.
    onlineSimulationTime=onlinePredEndTime-onlinePredStartTime
    # Buffer size to store historical data in batches:
    bufferSizes=[BS*stepLength for BS in [0,1,2,3,4]] 
    lossRllingTotal=dict([(bufferSize,0) for bufferSize in bufferSizes]) # Container for rolling loss values.
    rollingSimTime=dict([(bufferSize,0) for bufferSize in bufferSizes]) # Rolling simulation time.
    for bufferSize in bufferSizes:
        rollingPredStartTime=datetime.now() # Start time of rolling method simulation.
        lossRolling=dict([(Tp,0) for Tp in TpVec]) # Stores the rolling prediction errors.
        for j, pattern in enumerate(patternArray[chosenComparisonStreams,:,:]):
            # We make the predictions for the maximum length and then crop it for shorter predictions:
            [feedTraf,tarTraf]=preprocess(pattern[0],batchSize) # Preparing the detected traffic data
            [feedCong,tarCong]=preprocess(pattern[1],batchSize) # Preparing the congestion data
            feedData=torch.cat((feedTraf,feedCong),dim=2) # Total feed data (concatenate traffic and congestion)
            targetData=torch.cat((tarTraf,tarCong),dim=2) # Total target data
            seed=feedData[:seedLength]
            rollingOut=dict([(Tp,[]) for Tp in TpVec]) # List to store predicted bunches.
            rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network.
            rnnModelRolling.eval() # Evaluation mode activation.
            totalNumberOfSlices=int(feedData[seedLength:].size(0)/stepLength)
            # The input sequence to the RNN network in rolling method is sum of buffer\
            # and fresh data: inputSize = bufferSize + StepLength (here: (3*4)+4=16 batches)
            with torch.no_grad():
                # bufferSize=3*stepLength # Buffer size to store historical data in batches
                outMaxRolling=rnnModelRolling(seed.to(device),steps=batchesAheadMax,\
                                              eval=True)
                for iTp,Tp in enumerate(TpVec):
                    rollingOut[Tp].append(outMaxRolling[:seedLength+batchesAhead[iTp]-1].clone())
                for i in tqdm(range(totalNumberOfSlices),colour='yellow',\
                              desc=f'Rolling forecasting stream {j+1} out of'+\
                                  f' {comparisionDataSize} (BS: {bufferSize//stepLength})'):
                    rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network before feeding data.
                    outMaxRolling=rnnModelRolling(feedData[seedLength-bufferSize+i*stepLength:\
                                                    seedLength+(i+1)*stepLength].to(device),\
                                          steps=batchesAheadMax, eval=True)
                    for (Tp,BA) in zip(TpVec,batchesAhead): # cropping the desired size according to Tp 
                        rollingOut[Tp].append(outMaxRolling[BA+bufferSize-1:BA-1+bufferSize+stepLength])
            for Tp in TpVec: # Error calculation:
                predictionRolling=torch.cat(rollingOut[Tp])[:targetData.size(0)].to(revDev)
                tarTensor=targetData[:predictionRolling.size(0)]
                lossRolling[Tp]+=(criterion(predictionRolling,tarTensor).item())/comparisionDataSize
        lossRllingTotal[bufferSize]=lossRolling
        rollingSimTime[bufferSize]=datetime.now()-rollingPredStartTime
    #%% Generating predictions for two exemplary sequences for plotting purposes:
    onlinePrediction=dict()
    for j, pattern in enumerate(patternArray[seqForPlotting,:,:]):
        # We make the predictions for the maximum length and then crop it for shorter predictions:
        [feedTraf,tarTraf]=preprocess(pattern[0],batchSize) # Preparing the detected traffic data
        [feedCong,tarCong]=preprocess(pattern[1],batchSize) # Preparing the congestion data
        feedData=torch.cat((feedTraf,feedCong),dim=2) # Total feed data (concatenate traffic and congestion)
        targetData=torch.cat((tarTraf,tarCong),dim=2) # Total target data
        seed=feedData[:seedLength]
        output=dict([(Tp,[]) for Tp in TpVec]) # List to store predicted bunches.
        rnnModel.init_hidden(batchSize) # Initializing the RNN network.
        rnnModel.eval() # Evaluation mode activation.
        totalNumberOfSlices=int(feedData[seedLength:].size(0)/stepLength)
        # Generating the prediction:
        with torch.no_grad():
            outMax=rnnModel(seed.to(device),steps=batchesAheadMax, eval=True)
            for iTp,Tp in enumerate(TpVec):
                output[Tp].append(outMax[:seedLength+batchesAhead[iTp]-1].clone())
            for i in tqdm(range(totalNumberOfSlices),colour='cyan',\
                          desc=f'Online forecasting stream {j+1} out of {comparisionDataSize}         '):
                outMax=rnnModel(feedData[seedLength+i*stepLength:seedLength+(i+1)*stepLength].to(device),\
                              steps=batchesAheadMax, eval=True)
                for (Tp,BA) in zip(TpVec,batchesAhead): # cropping the desired size according to Tp
                    output[Tp].append(outMax[BA-1:BA-1+stepLength]) 
            onlinePrediction[f'Sequence {seqForPlotting[j]}']=output # Storing the predcited traffic.
    rollingPredinction=dict([(f'bufferSize {bufferSize}',dict()) for bufferSize \
                             in bufferSizes])
    for bufferSize in bufferSizes:
        for j, pattern in enumerate(patternArray[seqForPlotting,:,:]):
            # We make the predictions for the maximum length and then crop it for shorter predictions:
            [feedTraf,tarTraf]=preprocess(pattern[0],batchSize) # Preparing the detected traffic data
            [feedCong,tarCong]=preprocess(pattern[1],batchSize) # Preparing the congestion data
            feedData=torch.cat((feedTraf,feedCong),dim=2) # Total feed data (concatenate traffic and congestion)
            targetData=torch.cat((tarTraf,tarCong),dim=2) # Total target data
            seed=feedData[:seedLength]
            rollingOut=dict([(Tp,[]) for Tp in TpVec]) # List to store predicted bunches.
            rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network.
            rnnModelRolling.eval() # Evaluation mode activation.
            totalNumberOfSlices=int(feedData[seedLength:].size(0)/stepLength)
            # The input sequence to the RNN network in rolling method is sum of buffer\
            # and fresh data: inputSize = bufferSize + StepLength (here: (3*4)+4=16 batches)
            with torch.no_grad():
                outMaxRolling=rnnModelRolling(seed.to(device),steps=batchesAheadMax,\
                                              eval=True)
                for iTp,Tp in enumerate(TpVec):
                    rollingOut[Tp].append(outMaxRolling[:seedLength+batchesAhead[iTp]-1].clone())
                for i in tqdm(range(totalNumberOfSlices),colour='yellow',\
                              desc=f'Rolling forecasting stream {j+1} out of '+\
                                  f'{comparisionDataSize} (BS: {bufferSize//stepLength})'):
                    rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network before feeding data.
                    outMaxRolling=rnnModelRolling(feedData[seedLength-bufferSize+i*stepLength:\
                                                    seedLength+(i+1)*stepLength].to(device),\
                                          steps=batchesAheadMax,eval=True)
                    for (Tp,BA) in zip(TpVec,batchesAhead): # cropping the desired size according to Tp 
                        rollingOut[Tp].append(outMaxRolling[BA+bufferSize-1:BA-1+bufferSize+stepLength])
                rollingPredinction[f'bufferSize {bufferSize}'][f'Sequence {seqForPlotting[j]}']=\
                    rollingOut
    #%% Storing the results of the simulation:
    bgColor=['\033[48;5;5m','\033[48;5;6m','\033[48;5;2m','\033[48;5;1m'] # Color for background.
    colorEnd='\033[0m' # Ending color effect (site: https://en.wikipedia.org/wiki/ANSI_escape_code#Colors).
    lossOnlineStr=', '.join([f'{lossOnline[Tp]:2.2f}' for Tp in TpVec])
    rollingExplanation=''
    for bufferSize in bufferSizes:
        lossRollingStr=', '.join([f'{lossRllingTotal[bufferSize][Tp]:2.2f}' for Tp in TpVec])
        timeDiff=(rollingSimTime[bufferSize].seconds-onlineSimulationTime.seconds)/\
            rollingSimTime[bufferSize].seconds
        lossCompare=', '.join([f'{(lossRllingTotal[bufferSize][Tp]-lossOnline[Tp])/lossRllingTotal[bufferSize][Tp]:2.2%}' for Tp in TpVec])
        rollingExplanation+='-'*84+'\n'+bgColor[bufferSize//stepLength%2]+\
            f'Rolling \t(buffer: {bufferSize*batchSize:3})\t|\t\t'+\
            str(rollingSimTime[bufferSize]).split('.')[0]+\
            '\t'*3+'|'+'\t'*1+lossRollingStr+'  \t'*2+'|\n'+colorEnd+\
            '-'*84+'\n'+bgColor[bufferSize//stepLength%2]+\
            f'\033[38;5;11mComparison\t(buffer: {bufferSize*batchSize:3})\t|\t\t {timeDiff:2.2%}'+\
            '\t'*3+'|'+'\t'*1+lossCompare+' \t|\n'+colorEnd
    explanation='\n\t\033[38;5;11mComparison of traffic preciction using FLSP and'+\
        ' Rolling algorithms in terms of MSE:\033[0m\n'+\
        '\t\t\t\t\t\t\t'+'_'*60+'\n'+'\t'*7+'|\tTime Consumption'+'\t'*1+'|'+'\t'*2+\
        '  Prediction Loss'+'\t'*3+'|\n'+'-'*84+'\n'+bgColor[2]+'Online\t\t\t\t\t\t|\t\t'+\
        str(onlineSimulationTime).split('.')[0]+'\t'*3+'|'+'\t'*1+ lossOnlineStr+\
        '\t'*3+'|\n'+colorEnd+rollingExplanation+'_'*84+'\n'+\
        bgColor[3]+'Note 1:'+colorEnd+' The input to the prediction network is '+\
            'the union of buffered data and freshly collected data of size stepLength '+\
            f'({stepLength} batches or {stepLength*batchSize} time slots).\n'+\
        bgColor[3]+'Note 2:'+colorEnd+' The percentages show the improvements of '+\
            'the proposed online algorithm compared to the traditional rolling method.\n'    
    Description='\n\033[38;5;11mThis is a result of trained model evaluation to predict the network '+\
        'traffic and bursty traffic using '+nType+' model and contains the following items:\033[0m\n\t'+\
        f'- "batchSize": Data is fed to ML network as batches. Value: {batchSize};\n\t'+\
        f'- "TpVec": List of predition durations at each step. Value: {TpVec};\n\t'+\
        f'- "Tseed": Seed data size which is used to initialize the RNN network. Value: {Tseed};\n\t'+\
        f'- "Tstep": The size of fresh data which is frequently collected from the network. Value: {Tstep};\n\t'+\
        '- "allLoadedAddress": A dictionary including all the required addresses to load data;\n\t'+\
        '- "testDataSize": Number of test sequences for evaluating the traffic '+\
            f'prediction and burst detection networks. Value: {testDataSize};\n\t'+\
        '- "comparisionDataSize": Number of streams used for comparison of Online'+\
            f' and Rolling prediction methods. Value: {comparisionDataSize};\n\t'+\
        '- "traffPredLoss": Loss value for RNN network when predicting the traffic'+\
            ' in online mode;\n\t'+\
        '- "burstDetLoss": Loss value of burst detection network;\n\t'+\
        '- "lossOnline": Loss value of traffic predcition network using Online method'+\
            ' over the comparison data sequences;\n\t'+\
        '- "lossRllingTotal": Loss value of traffic predcition network using '+\
            'Classical Rolling method over the comparison data sequences;\n\t'+\
        '- "onlineSimulationTime": Simulation time of Online algorithm;\n\t'+\
        '- "rollingSimTime": Simulation time of Classical Rolling algorithm.\n\t'+\
        '- "onlinePrediction": Predictions for exeplary sequences for different '+\
            'Tps using Online method.\n\t'+\
        '- "rollingPredinction": Predictions for exeplary sequences for different'+\
            ' buffer sizes and Tps using Rolling method.\n\t'+\
        '- "bufferSizes": The list of buffer sizes (in batches) used in Rolling '+\
            'method. Note that the input to the prediction network is union of '+\
            f'buffered and freshly collected data! Value: {bufferSizes}\n'+\
            explanation
    # Adding the accuracy metrics to the description results:
    explanation2=TC('\n\tValidation metrics of burst detection network using outputs of'+\
                    ' FLSP and Rolling algorithms:\n',[11,20,1])+'_'*84+'\n'
    explanation2+=f'|{tab()}Tp{tab(2)}|{tab(2)}Pred. Alg.{tab(3)}|{tab()}Precision{tab()}|'+\
        f'{tab()}Recall{tab()}|{tab()}F1-score{tab()}|\n'
    for Tp in [1,2]:
        explanation2+='-'*84+'\n'
        explanation2+=TC(f'|{tab()}{Tp} [sec]{tab()}|{tab(3)}FLSP{tab(3)}|{tab(2)}'+\
                         f'{accuracyMetrics["Precision"][Tp]:.2f}{tab()}|{tab()}'+\
                         f'{accuracyMetrics["Recall"][Tp]:.2f}{tab()}|{tab(2)}'+\
                         f'{accuracyMetrics["F1_score"][Tp]:.2f}{tab()}|\n',[15,2,1])
        explanation2+='-'*84+'\n'
        buff=bufferSizes[1]
        explanation2+=TC(f'|{tab()}{Tp} [sec]{tab()}|{tab()}Rolling (buff='+\
                         f'{buff*batchSize} [ts]){tab()}|{tab(2)}'+\
                         f'{accuracyMetricsRoll[buff]["Precision"][Tp]:.2f}{tab()}|{tab()}'+\
                         f'{accuracyMetricsRoll[buff]["Recall"][Tp]:.2f}{tab()}|{tab(2)}'+\
                         f'{accuracyMetricsRoll[buff]["F1_score"][Tp]:.2f}{tab()}|\n',\
                         [15,5,1])
        buff=bufferSizes[2]
        explanation2+='-'*84+'\n'
        explanation2+=TC(f'|{tab()}{Tp} [sec]{tab()}|{tab()}Rolling (buff='+\
                         f'{buff*batchSize} [ts]){tab()}|{tab(2)}'+\
                         f'{accuracyMetricsRoll[buff]["Precision"][Tp]:.2f}{tab()}|{tab()}'+\
                         f'{accuracyMetricsRoll[buff]["Recall"][Tp]:.2f}{tab()}|{tab(2)}'+\
                         f'{accuracyMetricsRoll[buff]["F1_score"][Tp]:.2f}{tab()}|\n',\
                         [15,6,1])
    explanation2+='-'*84+'\n'
    explanation2+=TC('Note 1:',[20,1,1])+' Test data is used to calculate these results.'
    Description+=explanation2
    data={'batchSize':batchSize, 'TpVec': TpVec, 'Tseed':Tseed, 'Tstep':Tstep,\
          'allLoadedAddress':allLoadedAddress, 'testDataSize':testDataSize,\
          'comparisionDataSize':comparisionDataSize, 'traffPredLoss':traffPredLoss,\
          'burstDetLoss':burstDetLoss, 'accuracyMetrics':accuracyMetrics,\
          'accuracyMetricsRoll':accuracyMetricsRoll,\
          'lossOnline':lossOnline, 'lossRllingTotal': lossRllingTotal, \
          'onlineSimulationTime':onlineSimulationTime,'seqForPlotting':seqForPlotting,\
          'rollingSimTime':rollingSimTime,'onlinePrediction':onlinePrediction,\
          'rollingPredinction':rollingPredinction,'bufferSizes':bufferSizes,\
              'Description':Description}
    dataManagement(data=data, save=True, fileFormat='.pt', fileName='E_EvaluationResults_'+\
                   nType, version='_v3',path='E_EvaluationResultsForRnn')
    print(Description)
    print(f'\nTotal simulation time: {str(datetime.now()-startTime).split(".")[0]}.')
