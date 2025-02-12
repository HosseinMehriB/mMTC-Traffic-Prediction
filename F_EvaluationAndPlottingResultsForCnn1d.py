# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Mon Jul 29 17:43:00 2024

@author: HOSSEINMEHRI

This code examines the simulation time required by the CNN-1D based traffic prediction
models to generate the results for the given set of sequences. The results then
is compared to the RNN-based prediction models. Moreover, the accuracy metrics
as well as an exemplary generated results are provided in this code.

Simulations are performed in two forms:
    - Non-sequential: In this case, whole the input data is given to the CNN-1D
        model at once. In this case, CNN-1D models can leverage the matrix multiplication
        properties of the CNN models and can generate the predictions in a short time.
    - Sequential: In this case the data is fed to the CNN-1D model in short lengthed
        batches of size 'feedSize', resempling a practical live network where 
        fresh data is collected frequently from the network in a fixed sized batches.

Note that as the goal of this work is comparing the performance of the models when
they are employed in a live network, so the results of the 'Sequential' data feeding
model provides the fair comaprison results.

This code executes the fourth of four steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.
The CNN-1D-based model only uses the traditional rolling algorithm and the output
is used for performance comparison with RNN-based models that can use FLSP algorithm.

 --------------------------      ------------------      ---------------------------------- 
|GeneratingTrafficPatterns| --> |CNN-1DNetTraining| --> |BurstDetNetTrainingWithCNN-1DData| -->
--------------------------      ------------------      ----------------------------------
     vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
--> |EvaluationAndPlottingResultsForCNN-1D|
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code loads required the trained model from "D_trainedBurstNetAndEvalsForCnn1D"
and "B_traindCNN1D" folders.

You can modify the following parameters:
    - testDataSize: Number of test sequences. It should match the number of sequences
        used for testing RNN-based models for a fair comparison.
    

"""

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt 

from MLNetworks import dataManagement, cnn1dPreprocess, burstTargetDataGen
from MLNetworks import textColorFormat as TC

#%% Loading the trained models:
# Put the file name of trained models with corresponding window size as a key.\
# You can find them in "D_trainedBurstNetAndEvalsForCnn1D" folder.
addr={'W100':'212_BurstNetAndEvals_CNN-1D_WS100__Tue_Jul30_1515_v3.pt',\
      'W200':'211_BurstNetAndEvals_CNN-1D_WS200__Mon_Jul29_2115_v3.pt',\
      'W300':'212_BurstNetAndEvals_CNN-1D_WS300__Tue_Jul30_1427_v3.pt',\
      'W400':'212_BurstNetAndEvals_CNN-1D_WS400__Tue_Jul30_1437_v3.pt'}
dataLoad=dataManagement(address='A_generatedTraffic/157_MTC_Traffic(500)(900Sec)__Tue_Jun06_1654.pt')
Pattern=dataLoad.get('Pattern') # List containing traffic patterns: [detected, attempted,congestion,freeP]
#%% Loading the trained models for each given window size and Tp in TpVec:
for key in addr.keys():
    resLoad_D=dataManagement(address='D_trainedBurstNetAndEvalsForCnn1D/'+addr[key]) # Loading the trained model
    allLoadedAddress=resLoad_D.get('allLoadedAddress') # Loading the address dictionary including address of employed CNN-1D models
    resLoad_B2Add=allLoadedAddress.get('addressLoadCNN1D')
    resLoad_B2Add=resLoad_B2Add.replace('B2_traindCNN1D','B_traindCNN1D')
    resLoad_B2=dataManagement(address=resLoad_B2Add) # Loading the corresponding CNN-1D model
    testDataSize=100 
    frameSize=0.005
    GPU_flag=True
    
    TpVec=resLoad_B2['TpVec']
    chosenTestStreams=np.random.choice(range(len(Pattern)-100,len(Pattern)),\
                                 testDataSize, replace=False) # Random choose
    if GPU_flag:
        device=torch.device('cuda') # Laod on GPU.
    else:
        device=torch.device('cpu') # Laod on CPU.    
    # Loading the CNN-1D model:
    models={}
    for Tp in TpVec:
        models[Tp]=resLoad_B2[Tp]['model'].to(device)
    #%% Non-sequential data feeding simulation:
    starttime=datetime.now() # Record the simulation time.
    note=('\n'+TC('='*100,'ryu'))*2
    note+=('\n\n\t\t\t\t')
    note+=TC('Traffic prediction simulation for window size of: ','rcn')+TC(f'{key}\n\n','rcu')
    note+=TC('v'*100,'ryn')
    print('\n\n',note)
    for iTp, Tp in enumerate(TpVec):
        windowSize=resLoad_B2[Tp]['windowSize']
        Tfeed=resLoad_B2[Tp]['Tfeed']
        feedSize=int(Tfeed/frameSize)
        steps=int(Tp/frameSize) # Prediction steps in [time slots]
        for i,loc in enumerate(tqdm(chosenTestStreams, colour='magenta',\
                                    desc=f'Testing models for Tp: {Tp} ({iTp+1} '+\
                                        f'out of {len(TpVec)}) - Non-sequential - W-{windowSize}')):
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
            # To simulate a real-life simulation where data is collected sequentially and fed to the model,
            # we do the same here.
            preds=models[Tp](inputs.to(device))
        # Printing CNN-1D-based traffic prediction and burst detection prediction results:
            # resLoad_D2['description_B2']: Traffic prediction results;
            # resLoad_D2['description_D2']: Burst detection network results.
    endTime=datetime.now()
    print(f'\n\nTotal simulation time for {TC("Non-Sequential","nnu")} mode for W = {TC(windowSize,"ynu")}: {TC(endTime-starttime,"rbu")}')
    print('\n'+'='*90+'\n\n\n')
    
    #%% Sequential data feeding simulation:
    starttime=datetime.now() # Record the simulation time.
    for iTp, Tp in enumerate(TpVec):
        windowSize=resLoad_B2[Tp]['windowSize']
        Tfeed=resLoad_B2[Tp]['Tfeed']
        feedSize=int(Tfeed/frameSize)
        steps=int(Tp/frameSize) # Prediction steps in [time slots]
        for i,loc in enumerate(tqdm(chosenTestStreams, colour='magenta',\
                                    desc=f'Testing models for Tp: {Tp} ({iTp+1} '+\
                                        f'out of {len(TpVec)}) - Sequential - W-{windowSize}')):
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
            # To simulate a real-life simulation where data is collected sequentially and fed to the model,
            # we do the same here.
            
            for i in range(inputs.shape[0]):
                preds=models[Tp](inputs[i:i+1].to(device))
    endTime=datetime.now()

    print(f'\n\nTotal simulation time for {TC("Sequential","nnu")} mode for W = {TC(windowSize,"ynu")}: {TC(endTime-starttime,"rbu")}\n')
    print('----------------------------------------------------------------------------------------\n\n')
    print(resLoad_D['description_B'],'\n',resLoad_D['description_D']) # Printing the evaluation results which is done is previous step
    
#%% Plotting an example of the predictions:
Tp=1 # Plotting for Tp=1
loc=435 # A random sequence
windowSize=resLoad_B2[Tp]['windowSize']
Tfeed=resLoad_B2[Tp]['Tfeed']
feedSize=int(Tfeed/frameSize)
steps=int(Tp/frameSize) # Prediction steps in [time slots]
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
# To simulate a real-life simulation where data is collected sequentially and fed to the model,
# we do the same here.
preds=models[Tp](inputs.to(device))
# Prepare data to plot:
predTraff=preds[:,0,-feedSize:].reshape(-1).detach().cpu().numpy()
predCong=preds[:,1,-feedSize::].reshape(-1).detach().cpu().numpy()
targetTraff=targets[:,0,-feedSize::].reshape(-1).detach().cpu().numpy()
targetCong=targets[:,1,-feedSize::].reshape(-1).detach().cpu().numpy()
#%% Loading the burst detection network and feeding the predicted traffic pattern to it:
burstDetectionNet=resLoad_D['trainedBurstData'][Tp]['burstDetectionNet']
burstDetectionNet.cuda()
predictedBurst=burstDetectionNet(preds,reshape=False).squeeze() # Getting output without reshaping
predictedBurst[predictedBurst<0.5]=0 # Below 0.5 => '0'
predictedBurst[predictedBurst>0.5]=1 # Above 0.5 => '1'
predictedBurst=predictedBurst.repeat_interleave(feedSize).detach().cpu().numpy() # Expanding labels to fit time slots
# Generating the bursty area labels:
[_,exactCongArea,_]=burstTargetDataGen(Pattern[loc][0], Pattern[loc][2],\
                          25,Tp,Tfeed,frameSize,onlyPredData=True) # sequence[0]: traffic; sequence[2]: congestion
exactCongArea=exactCongArea[windowSize+feedSize:] # Ignoring first points that are for feed data and window size
#%% Plotting the results:
startTime=(feedSize+windowSize)*frameSize
xAxis=np.linspace(startTime,startTime+targetCong.shape[0]*frameSize,targetCong.shape[0])
fig0=plt.figure(figsize=[9,9])
plt.subplot(3,1,1)
plt.plot(xAxis,targetTraff,'C0',linewidth=2,label='Ground Truth')
plt.plot(xAxis,predTraff,'C1',linewidth=2,label='CNN-1D - Rolling')
title=f'Traffic prediction using CNN-1D for window size = {windowSize} [time slots], '+\
    f'T$fresh$ = {Tfeed} [sec], T$p$ = {Tp} [sec]'
plt.title(title)
plt.ylabel('Detected preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.grid()

plt.subplot(3,1,2)
plt.plot(xAxis,targetCong,'C0',linewidth=2,label='Ground Truth')
plt.plot(xAxis,predCong,'C1',linewidth=2,label='CNN-1D - Rolling')
plt.ylabel('Congested preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.grid()

plt.subplot(3,1,3)
plt.plot(xAxis,predCong,'C1',linewidth=2,label='CNN-1D - Rolling')
plt.plot(xAxis,exactCongArea*75,'--C4',linewidth=2,label='Expedted Cong Area')
plt.plot(xAxis,predictedBurst*65,'#38f416',linestyle='--',linewidth=2,label='Predicted Cong Area')
# plt.xlim([130,145])
plt.ylabel('Congested preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.grid()
fig0.set_tight_layout('tight') # Solving overlapping labels.
plt.show()

print('\n\nPlotting only a short range where there is a bursty traffic:\n')

criticalPoint=np.where(predictedBurst)[0][0] # Finding the first point that we have bursty traffic
low=xAxis[criticalPoint-1000] # xlim low limit
high=xAxis[criticalPoint+2000] # xlim high limit
fig1=plt.figure(figsize=[9,9])
plt.subplot(3,1,1)
plt.plot(xAxis,targetTraff,'C0',linewidth=2,label='Ground Truth')
plt.plot(xAxis,predTraff,'C1',linewidth=2,label='CNN-1D - Rolling')
title=f'Traffic prediction using CNN-1D for window size = {windowSize} [time slots], '+\
    f'T$fresh$ = {Tfeed} [sec], T$p$ = {Tp} [sec]'
plt.title(title)
plt.ylabel('Detected preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.xlim([low,high])
plt.grid()

plt.subplot(3,1,2)
plt.plot(xAxis,targetCong,'C0',linewidth=2,label='Ground Truth')
plt.plot(xAxis,predCong,'C1',linewidth=2,label='CNN-1D - Rolling')
plt.ylabel('Congested preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.xlim([low,high])
plt.grid()

plt.subplot(3,1,3)
plt.plot(xAxis,predCong,'C1',linewidth=2,label='CNN-1D - Rolling')
plt.plot(xAxis,exactCongArea*75,'--C4',linewidth=2,label='Expedted Cong Area')
plt.plot(xAxis,predictedBurst*65,'#38f416',linestyle='--',linewidth=2,label='Predicted Cong Area')
plt.xlim([low,high])
plt.ylabel('Congested preambles')
plt.xlabel('Time [sec]')
plt.legend(loc='upper right')
plt.grid()
fig1.set_tight_layout('tight') # Solving overlapping labels.
plt.show()