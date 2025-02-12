# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Wed Jul 28 19:06:58 2021

@author: Hossein Mehri

This code runs the predictions for a specific time and stores it in a file. This 
data can be used for training of Burst prediction network and or to calculate the 
accuracy of RNN models in predicting the traffic pattern. The outputs are generated
as bunches of data from both actual and predicted traffic patterns for all sequences
and store them as two separate files in "C_burstFeedData" folder. 

This code executes the third of six steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.

 --------------------------      ---------------      vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
|GeneratingTrafficPatterns| --> |RnnNetTraining| --> |GeneratingBurstDetNetFeedDataForRNN| -->
--------------------------      ---------------      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     -------------------------------      ---------------------------------------------
--> |BurstDetNetTrainingWithRNNData| --> |EvaluatingNetworksAndGeneratingResultsForRNN| -->
    -------------------------------      ---------------------------------------------
     ------------------------------------------
--> |PlottingResultsAndExportingToMatlabForRNN|
    ------------------------------------------
    
The code loads required data from "A_generatedTraffic" and "B_trainedRNN" folders.
The generated files are stored in "C_burstFeedDataForRnn" folder.
    
You can modify the following parameters:
    - TpVec: A list showing the prediction time at each step.
    
"""


import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from MLNetworks import rnnNetwork
from MLNetworks import dataManagement, preprocess, burstTargetDataGen

    



#%% The main function:
if __name__=="__main__":
    TpVec=[0.5,1,1.5,2] # Predicting Tp seconds ahead at each step
    starttime=datetime.now()
    print('Start time: ',starttime)
    # Loading the trained RNN model and corresponding file address:
    (resLoad,addressR)=dataManagement(path='B_trainedRNN',returnAddress=True, choice=True) # Loading the last file in the folder
    allLoadedAddress=resLoad.get('allLoadedAddress') # Loading the container of loaded files' address
    allLoadedAddress.update({'addressLoadRNN':addressR}) # Adding the address of trained RNN model
    # Description=resLoad.get('Description') # Description of the trained model
    RNNmodel=resLoad.get('model') # Loading the trained RNN network
    if RNNmodel.GPU_flag:
        device=torch.device('cuda') # Laod on GPU.
    else:
        device=torch.device('cpu') # Laod on CPU.
    RNNmodel.to(device) # Move the model to the desired device.
    frameSize=0.005 # Each time slot duration
    batchSize=resLoad.get('batchSize') # Overriding the batchSize to match the trained model conditions. 
    Tstep=resLoad.get('Tstep') # Fresh data collection frequency (and length of the fresh data).
    Tseed=resLoad.get('Tseed') # The initializing data length in seconds
    seedLength=int(Tseed/frameSize/batchSize) # The initializing data length in batches
    # Data load:
    dataLoad=dataManagement(address=allLoadedAddress.get('patternDataAddress'))
    # dataLoad=torch.load('A_generatedTraffic/157_MTC_Traffic(500)(900Sec)__Tue_Jun06_1654.pt') # Loading the data   
    Pattern=dataLoad.get('Pattern') # List containing simulation results: [detected, attempted,congestion,freeP]
    totalNumberOfPatterns=np.shape(Pattern)[0] # Total number of available patterns.
    TpMax=2 # Predicting Tp seconds ahead during at each step (we make prediction for the maximum time,\
            # then crop it for the shorter times).
    #%% Generating the data streams for burst detection network using the FLSP algorithm:
    startTime=datetime.now()
    linearNetInputs=[] # To store the predicted data from RNN network
    for streamNum in range(totalNumberOfPatterns):
        [inputsv,_]=preprocess(Pattern[streamNum][0],batchSize) # Preparing the data
        [congestv,_]=preprocess(Pattern[streamNum][2],batchSize) # Preparing the congestion data
        # Concatenating the traffic and congestion streams to create seed and feed data.
        seed=torch.cat((inputsv[:seedLength].cuda(),congestv[:seedLength].cuda()),2) 
        feedData=torch.cat((inputsv[seedLength:].cuda(),congestv[seedLength:].cuda()),2) # Concatenating the traffic and congestion streams
        # Consider the maximum Tp to reduce the total time of the prediction:
        Tp=TpMax # We run the predictions for the maximum Tp and then crop it for shorter predictions
        sliceSize=int(Tstep/frameSize/batchSize) # Slice size (new data at each step) in bathces unit
        batchesAhead=int(Tp/frameSize/batchSize) # Size of predicted data in batches
        totalSlices=int(feedData.size(0)/sliceSize) # Total number of possible slices in the evaluation data
        # Preparing the RNN model and starting the predictions:
        RNNmodel.eval() # Change to evaluation mode
        RNNmodel.init_hidden(batchSize) # Initializing the model
        with torch.no_grad():
            _=RNNmodel(seed,steps=3, eval = True).cpu() # Initializing the model using seed data.
        # Creating the containers of the results:
        linearInput=torch.zeros((totalSlices,(batchesAhead+sliceSize)*batchSize*2),\
                                device=device,dtype=torch.float64)
        # Generating the predictions of RNN network:
        endTime1=datetime.now()
        with torch.no_grad():
            for i in tqdm(range(0,totalSlices),colour='green',\
                          desc=f'Stream {streamNum+1} out of {totalNumberOfPatterns} streams'):
                RNNOutput = RNNmodel(feedData[i*sliceSize:(i+1)*sliceSize],\
                                      steps=batchesAhead, eval = True) # Get the output of RNN net. for each slice.
                dataBunch=torch.cat((feedData[i*sliceSize:(i+1)*sliceSize],\
                                      RNNOutput[-batchesAhead:]),0) # Concatenating the output and input of RNN to get the feed data to linear net.
                linearInput[i]=torch.cat((dataBunch[:,:,0].reshape(-1),\
                                          dataBunch[:,:,1].reshape(-1)),0) # Changing the shape in a way that firsr half is traffic and second half is congestion stream.
        linearNetInputs.append(linearInput.cpu())
        print('Total elapsed time: ', datetime.now()-startTime,'\n\n')
        
    #%% Generating the data streams for burst detection network using the Rolling algorithm:
    rnnModelRolling=RNNmodel # The same model as above, used for classic rolling prediction model.
    stepLength=int(Tstep/frameSize/batchSize) # Same as slice size!
    testDataSize=50 # Generating the data bunches only for test sequences for Rolling algorithm.
    batchesAheadMax=int(TpMax/frameSize/batchSize) # Size of predicted data in batches
    bufferSizes=[BS*stepLength for BS in [0,1,2,3,4]]
    batchesAhead=int(TpMax/frameSize/batchSize) # Size of predicted data in batches
    linearNetInputsRoll=dict([(bufferSize,[]) for bufferSize in bufferSizes]) # Container for rolling loss values.
    for bufferSize in bufferSizes:
        for j, pattern_ in enumerate(Pattern[400:]):
            # We make the predictions for the maximum length and then crop it for shorter predictions:
            [feedTraf,tarTraf]=preprocess(pattern_[0],batchSize) # Preparing the detected traffic data
            [feedCong,tarCong]=preprocess(pattern_[1],batchSize) # Preparing the congestion data
            feedData=torch.cat((feedTraf,feedCong),dim=2) # Total feed data (concatenate traffic and congestion)
            targetData=torch.cat((tarTraf,tarCong),dim=2) # Total target data
            seed=feedData[:seedLength] # For Rolling algorithm, seed data is useless as we initialize the states at each step!
            rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network.
            rnnModelRolling.eval() # Evaluation mode activation.
            totalNumberOfSlices=int(feedData[seedLength:].size(0)/stepLength)
            linearInput=torch.zeros((totalNumberOfSlices,(batchesAhead+stepLength)*batchSize*2),\
                                    device=device,dtype=torch.float64)
            input('\nstop....')
            # The input sequence to the RNN network in rolling method is sum of buffer\
            # and fresh data: inputSize = bufferSize + StepLength (here: (3*4)+4=16 batches)
            with torch.no_grad():            
                for i in tqdm(range(totalNumberOfSlices),colour='yellow',\
                              desc=f'Rolling forecasting stream {j+1} out of'+\
                                  f' {testDataSize} (BS: {bufferSize//stepLength})'):
                    rnnModelRolling.init_hidden(batchSize) # Initializing the RNN network before feeding data.
                    RNNOutput=rnnModelRolling(feedData[seedLength-bufferSize+i*stepLength:\
                                                    seedLength+(i+1)*stepLength].to(device),\
                                          steps=batchesAheadMax, eval=True)
                    dataBunch=torch.cat((feedData[seedLength+i*stepLength:seedLength+(i+1)*stepLength],\
                                          RNNOutput[-batchesAhead:].cpu()),0) # Concatenating the output and input of RNN to get the feed data to linear net.
                    linearInput[i]=torch.cat((dataBunch[:,:,0].reshape(-1),\
                                              dataBunch[:,:,1].reshape(-1)),0)
            linearNetInputsRoll[bufferSize].append(linearInput.cpu())
    print('Total elapsed time: ', datetime.now()-startTime,'\n\n')
    #%% Generating the input data and labels of burst prediction network using the original data:
    burstLabels=[]
    exactCongAreaData=[]
    feedData=[]
    for iTp,Tp in enumerate(TpVec):
        burstLabel=[]
        exactCongAreaD=[]
        feedD=[]
        for sequence in tqdm(Pattern, colour='yellow',\
                              desc=f'Original data processing for Tp: {Tp} ({iTp+1} out of {len(TpVec)})'):
            # Extracting the burst labels from congestion sequence:
            [beacon,exactCongArea,feed]=burstTargetDataGen(sequence[0], sequence[2],\
                                      batchSize,Tp,Tstep,frameSize,onlyPredData=False) # sequence[0]: traffic; sequence[2]: congestion
            burstLabel.append(np.array(beacon,copy=True)) # Burst labels
            exactCongAreaD.append(np.array(exactCongArea,copy=True)) # Presentation data
            feedD.append(feed.detach().clone()) # Feed data
        burstLabels.append(burstLabel)
        exactCongAreaData.append(exactCongAreaD)
        feedData.append(feedD)
    
    #%% Saving the results of RNN model:
    description_C='In the third step of the current project, data bunches are'+\
        ' extracted from the actual and predicted patterns. The outputs are stored'+\
        ' into three different files as below:\n'+\
        '\t1) Processed original data and burst labels for all 500 streams;\n'+\
        '\t2) Predicted bunches of data using the FLSP algorithm for all 500 streams;\n'+\
        '\t3) Predicted bunches of data using the Rolling algorithm only for test'+\
            ' streams (last 100 streams).'
    # Saving the processed original sequences and burst labels:
    dataOrig = {'Tstep':Tstep,'batchSize':batchSize,'burstLabels':burstLabels,\
                'exactCongAreaData':exactCongAreaData,'feedData':feedData, \
                'TpVec':TpVec, 'description_C':description_C, 'allLoadedAddress':\
                        allLoadedAddress}
    dataManagement(data=dataOrig, save=True, fileFormat='.pt', \
                   fileName='linearInputFromOriginalTraffic(500)',\
                   version='',path='C_burstFeedData')
    # Saving the predicted bunches using the FLSP algorithm:
    # dataRNN={'TpMax':TpMax,'Tstep':Tstep, 'Tseed':Tseed,'batchSize':batchSize,\
    #       'linearNetInputs':linearNetInputs,'addressLoadRNN':addressR,\
    #       'description_C':description_C, 'allLoadedAddress':allLoadedAddress}
    dataRNN={'TpMax':TpMax,'Tstep':Tstep, 'Tseed':Tseed,'batchSize':batchSize,\
          'linearNetInputs':linearNetInputs, 'allLoadedAddress':allLoadedAddress,\
              'description_C':description_C}
    dataManagement(data=dataRNN, save=True, fileFormat='.pt', fileName='linearInputFrom'+\
                   RNNmodel.nType+'Output(500)',version='',path='C_burstFeedData')
    # Saving the predicted bunches using the Rolling algorithm:
    # dataRNNRoll={'TpMax':TpMax,'Tstep':Tstep, 'Tseed':Tseed,'batchSize':batchSize,\
    #       'linearNetInputsRoll':linearNetInputsRoll,'addressLoadRNN':addressR,\
    #           'bufferSizes':bufferSizes, 'description_C':description_C}
    dataRNNRoll={'TpMax':TpMax,'Tstep':Tstep, 'Tseed':Tseed,'batchSize':batchSize,\
          'linearNetInputsRoll':linearNetInputsRoll, 'allLoadedAddress':allLoadedAddress,\
              'bufferSizes':bufferSizes, 'description_C':description_C}
    dataManagement(data=dataRNNRoll, save=True, fileFormat='.pt', \
                   fileName='linearInputFrom'+RNNmodel.nType+'OutputRoll(500)',\
                   version='',path='C_burstFeedDataForRnn')
    
    