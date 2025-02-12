# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Tue Jun  6 17:05:14 2023

@author: Hossein Mehri

This code trains the burst detection network using processed data bunches from
both actual and predicted traffic patterns.

This code executes the forth of six steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.

 --------------------------      ---------------      ------------------------------------
|GeneratingTrafficPatterns| --> |RnnNetTraining| --> |GeneratingBurstDetNetFeedDataForRNN| -->
--------------------------      ---------------      ------------------------------------
     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv      ---------------------------------------------
--> |BurstDetNetTrainingWithRNNData| --> |EvaluatingNetworksAndGeneratingResultsForRNN| -->
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ---------------------------------------------
     ------------------------------------------
--> |PlottingResultsAndExportingToMatlabForRNN|
    ------------------------------------------

The code loads required data from "C_burstFeedDataForRnn" folder.
The trained network is stored in "D_trainedBurstNetForRnn" folder.

You can modify the following parameters:
    - epochs: Number of iterations over the training streams;
    - dropoutVal: Droupout value;  
    - validationDataSize: Validation data size;
    - trainingDataSize: Training data size.

"""

import torch
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
from time import sleep
from datetime import datetime
from MLNetworks import linearNetwork
from MLNetworks import dataManagement, getAddress

# This function gets the linear network and training data and trains the network.
def trainLinearNet(burstDetectionNet, feed, labels, epochs, iTp, Tp, TpVec):
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
            Precision.append(truePositive/(truePositive+falsePositive))
            Recall.append(truePositive/(truePositive+falseNegative))
            F1Score.append(2*Precision[-1]*Recall[-1]/(Precision[-1]+Recall[-1]))
        # Stop training only if:
            # - The missed bursty points are less than 5%;
            # - We want to train the network only one time (epochs==1);
            # - We reached to the maximum number of iterations (max=5).
        if (abs(onesOfOut/onesOfTar)<0.05 or epochs==1 or exIter>=5): 
            break
        else: # Otherwise, conduct extra iterations to reach to the minimum accuracy.
            exIter+=1
            extraIterNote=f'\033[48;5;6m(extra iteration number {exIter} (max=5))\033[0m'
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
    nType='GRU'
    starttime=datetime.now()
    print('Start time: ',starttime)
    ''' Data source addresses:
        # RNN data contains predicted bunches of traffic generated by RNN network.
        # RNN data contains address of RNN network for reference.
        # Tstep, batch size, and TpMax is also included.
        # Data sequences of burstLabels, presentation, and feed data are included in original data.'''
    # RNN data: generated by a trained RNN network:
    (RNNData,addRNNData)=dataManagement(path='C_burstFeedData',includesStr=\
                                          'linearInputFrom'+nType+'Output(', \
                                              returnAddress=True)
    # Original data: generated using the ground truth sequences:
    (origData,addOrigData)=dataManagement(path='C_burstFeedData',includesStr=\
                                          'linearInputFromOriginalTraffic', \
                                              returnAddress=True)
    addRNNDataRoll=getAddress(path='C_burstFeedData', includesStr='linearInputFrom'+\
                              nType+'OutputRoll', newest=True) # Just getting the address of rolling data
    allLoadedAddress=RNNData.get('allLoadedAddress') # Loading the container of loaded files' address
    allLoadedAddress.update({'rnnDataAddress':addRNNData, 'origDataAddress':addOrigData,\
                             'rnnRollDataAddress':addRNNDataRoll}) # Adding the address of processed data sets for burst detection model
    linearInputsRNN=RNNData.get('linearNetInputs') # The predicted bunches of data from RNN network
    linearInputsOrig=origData.get('feedData') # Feed data extracted from the original traffic \
                                              # streams (shape: [iTp(4)][stream(500)][slices, sliceSize]).
    burstLabels=origData.get('burstLabels') # The labels of each step for Tps in the TpVec.
    # Simulation parameters
    batchSize=origData.get('batchSize') # Batch size
    Tstep=origData.get('Tstep') # Frequency of collecting new data as well as fresh data duration.
    Tseed=RNNData.get('Tseed')  # We should consider the seed length in chosing the labels for RNN data.
    totalNumberOfPatterns=len(linearInputsRNN) # Total number of available patterns.
    epochs=40 # Number of iterations over the training streams.
    frameSize=0.005 # Each time slot duration    
    dropoutVal = 0.4 #  
    # Validation parameters:
    validationDataSize=20
    # Load TpVec or use the folling TpVec:
    TpVec=origData.get('TpVec') # Predicting Tp seconds ahead at each step of prediction.
    if TpVec is None:
        # NEVER CHANGE THE TpVec OR RE-GENERATE THE burst labels,... sequences!!
        TpVec=[0.5,1,1.5,2] # Predicting Tp seconds ahead during the validation
    """
    "linearInputsRNN" contains 500 sequences which we divide them into three sets:
        - Training set: Strems are randomely chosen from first '400-validationDataSize' samples.
        - Validation set: Fixed streams located between '400-validationDataSize' and '400' positions.
        - Testing set: Strems are randomely chosen from last '100' samples.
    """
    trainingDataSize=90 # Training data size
    chosenTrainStreams=np.random.choice(range(totalNumberOfPatterns-100-validationDataSize),\
                                 trainingDataSize, replace=False) # Random choose
    chosenValStreams=[i for i in range(totalNumberOfPatterns-100-validationDataSize,\
                                       totalNumberOfPatterns-100)]
    #%% Training the burst detection network <<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    burstPredictionNetworks=[] # An empty list to store the trained models for each Tp in the TpVec.
    lossHist=[] # List of loss history of each Tp.
    validationLoss={'[Tp]':'[Loss Value]'} # A dictionary to store the validation loss values
    starttime2=datetime.now()
    trainingAccuracyMetrics=[] # History of accuracy metrics of training (shape:[Tps,3{P,R,F1},epochs+])
    validationAccuracyMetrics=[] # shape:[Tps,3{P,R,F1}]
    for iTp,Tp in enumerate(TpVec):
        # Converting the training and validation data into the desired types:
        feed=torch.zeros(trainingDataSize,linearInputsOrig[iTp][0].size(0),\
                         linearInputsOrig[iTp][0].size(1),dtype=torch.float64)
        labels=torch.zeros(trainingDataSize,burstLabels[iTp][0].size,dtype=torch.float64)
        for i,loc in enumerate(chosenTrainStreams):
            feed[i]=linearInputsOrig[iTp][loc] # Transferring the sequences from list to tensor
            labels[i]=torch.from_numpy(burstLabels[iTp][loc]) # Transferring the sequences from list to tensor
        valFeed=torch.zeros(validationDataSize,linearInputsOrig[iTp][0].size(0),\
                            linearInputsOrig[iTp][0].size(1),dtype=torch.float64)
        valLabels=torch.zeros(validationDataSize,burstLabels[iTp][0].size,dtype=torch.float64)
        for i,loc in enumerate(chosenValStreams):
            valFeed[i]=linearInputsOrig[iTp][loc] # Transferring the sequences from list to tensor
            valLabels[i]=torch.from_numpy(burstLabels[iTp][loc]) # Transferring the sequences from list to tensor
        # Creating the burst detection neural network:
        sliceSize=int(feed[0].size(-1)) # input sequection slice size.
        burstDetectionNet=linearNetwork(sliceSize, outputSize=1, dropoutVal=dropoutVal,\
                            GPU_flag=False, track_states = False)
        burstDetectionNet.double()
        [burstDetectionNet, lossH, accuracy]=trainLinearNet(burstDetectionNet,\
                                             feed, labels, epochs, iTp, Tp, TpVec)
        lossHist.append(lossH) # Storing the training loss history of current Tp.
        trainingAccuracyMetrics.append(accuracy) # Storing accuracy history (Precision,Recall,F1Score).
        print('Total elapsed time after training from the begining: ',datetime.now()-starttime)
        print('\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        burstPredictionNetworks.append(burstDetectionNet) # Store the trained network for current Tp.
        # Evaluating the trained burst detection network:
        burstDetectionNet.eval() # Evaluation mode activation.
        criterion=nn.MSELoss() # Loss calculation criteria
        valLoss=0 # To accumulate the loss value for each stream.
        truePositive=0  # Used to calculate the accuracy metrics.
        falsePositive=0 # Used to calculate the accuracy metrics.
        falseNegative=0 # Used to calculate the accuracy metrics.
        desc=f'Validating over original data for Tp = {Tp} ({iTp+1} out of {len(TpVec)})'
        for i,(labels,feed) in enumerate(tqdm(zip(valLabels,valFeed),\
                                             total=validationDataSize,\
                                             desc=desc, colour='cyan')):
            output=burstDetectionNet(feed) # Getting the detected bursts
            valLoss+=(criterion(output.squeeze(),labels[:output.size(0)]).item())/validationDataSize
            truePositive+=sum(labels[:output.size(0)]*(output.squeeze()>0.5)).item() # True Positives
            falsePositive+=sum((-1*labels[:output.size(0)]+1)*(output.squeeze()>0.5)).item() # False Positives
            falseNegative+=sum((-1*(output.squeeze()>0.5)+1)*labels[:output.size(0)]).item() # False Negatives
        validationLoss['Tp='+str(Tp)]=valLoss*100/(i+1) # Storing the validation loss in percentage.
        Precision=truePositive/(truePositive+falsePositive+0.00001)
        Recall=truePositive/(truePositive+falseNegative+0.00001)
        F1Score=2*Precision*Recall/(Precision+Recall+0.00001)
        validationAccuracyMetrics.append([Precision,Recall,F1Score])
    endTime1=datetime.now()
    #%% Generating the data stream using the RNN network:
    rnnTrainingDataSize=7 # 6 training streams and 1 final evaluation stream for presentation purposes.
    rnnValidationDataSize=20 # Validation data size for RNN data
    randomStreams=np.random.choice(range(totalNumberOfPatterns-100-rnnValidationDataSize),\
                                 rnnTrainingDataSize, replace=False) # Random choose
    # Validation data from RNN outputs:
    valRandomStreams=[i for i in range(totalNumberOfPatterns-100-rnnValidationDataSize,\
                                       totalNumberOfPatterns-100)]
    #%% Training the linear network with the output of the RNN network:
    rnnLossHist=[] # List of loss history of each Tp using RNN output.
    rnnTrainAccuracyMetrics=[] # History of accuracy metrics of RNN training (shape:[Tps,3{P,R,F1},1])
    rnnValAccuracyMetrics=[]   # shape:[Tps,3{P,R,F1}]
    validationLossRNN={'[Tp]':'[Loss Value]'} # A dictionary to store the validation loss values
    # We should consider the seed data during the initializing the RNN network.\
    # We should ignore first labels corresponding to the seed data:
    afterSeedLabelPointer=int(int(Tseed/frameSize/batchSize)/int(Tstep/frameSize/batchSize))
    for iTp,Tp in enumerate(TpVec):
        # Extracting the feed data from RNN predictions:
        rnnFeed=torch.zeros(rnnTrainingDataSize,linearInputsRNN[0].size(0),\
                                 linearInputsRNN[0].size(1),dtype=torch.float64)
        rnnLabels=torch.zeros(rnnTrainingDataSize,burstLabels[iTp][0][afterSeedLabelPointer:].size,\
                              dtype=torch.float64)
        for i,loc in enumerate(randomStreams):
            rnnFeed[i]=linearInputsRNN[loc]
            rnnLabels[i]=torch.from_numpy(burstLabels[iTp][loc][afterSeedLabelPointer:])
        # Retraining the burst detection network with output of rnnNetwork:
        [burstPredictionNetworks[iTp], rnnLossH,accuracy]=trainLinearNet(\
                                                                         burstPredictionNetworks[iTp],\
                        rnnFeed, rnnLabels, 1,iTp,Tp,TpVec) # Only '1' epoch
        rnnLossHist.append(rnnLossH) # Storing the loss history for RNN data.
        rnnTrainAccuracyMetrics.append(accuracy)# Storing accuracy metrics (Precision,Recall,F1Score).
        # Validation on RNN data:
        rnnValFeed=torch.zeros(rnnValidationDataSize, linearInputsRNN[0].size(0),\
                                 linearInputsRNN[0].size(1), dtype=torch.float64)
        rnnValLabels=torch.zeros(rnnValidationDataSize,burstLabels[iTp][0].size,\
                                 dtype=torch.float64)
        for i,loc in enumerate(valRandomStreams):
            rnnValFeed[i]=linearInputsRNN[loc]
            rnnValLabels[i]=torch.from_numpy(burstLabels[iTp][loc])
        
        valLoss=0
        truePositive=0
        falsePositive=0
        falseNegative=0
        desc=f'Validating over RNN data for Tp = {Tp} ({iTp+1} out of {len(TpVec)})'
        for i,(labels,feed) in enumerate(tqdm(zip(rnnValLabels,rnnValFeed),\
                                              total=rnnValidationDataSize,\
                                              desc=desc, colour='cyan')):
            output=burstPredictionNetworks[iTp](feed) # Getting the detected bursts
            valLoss+=(criterion(output.squeeze(),labels[:output.size(0)]).item())/rnnValidationDataSize
            truePositive+=sum(labels[:output.size(0)]*(output.squeeze()>0.5)).item() # True Positives
            falsePositive+=sum((-1*labels[:output.size(0)]+1)*(output.squeeze()>0.5)).item() # False Positives
            falseNegative+=sum((-1*(output.squeeze()>0.5)+1)*labels[:output.size(0)]).item() # False Negatives
        validationLossRNN['Tp='+str(Tp)]=valLoss*100/(i+1)
        Precision=truePositive/(truePositive+falsePositive)
        Recall=truePositive/(truePositive+falseNegative)
        F1Score=2*Precision*Recall/(Precision+Recall)
        rnnValAccuracyMetrics.append([Precision,Recall,F1Score])
    print('Training is done! Total elapsed time: ',datetime.now()-starttime,'\n')
    
    #%% Storing the trained burst detection network and training history:
    data={'burstPredictionNetworks':burstPredictionNetworks, 'TpVec':TpVec,\
                'Tstep':Tstep, 'lossHist':lossHist,'rnnLossHist':rnnLossHist,
                'trainingAccuracyMetrics':trainingAccuracyMetrics,
                'validationAccuracyMetrics':validationAccuracyMetrics,\
                'rnnTrainAccuracyMetrics':rnnTrainAccuracyMetrics,\
                'rnnValAccuracyMetrics':rnnValAccuracyMetrics,\
                'validationLossRNN':validationLossRNN, 'allLoadedAddress':allLoadedAddress}
    dataManagement(data=data, save=True, fileFormat='.pt', fileName='trainedBurstNet_'+\
                   nType, version='', path='D_trainedBurstNetForRnn')