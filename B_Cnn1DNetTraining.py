# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Mon Jul  1 15:52:57 2024

@author: Hossein Mehri

This code trains the CNN-1D network using the traffic patterns generated by 
"A_GeneratingTrafficPatterns.py". The patterns are first processed as batches,
then are fed to the CNN-1D network.

This code executes the second of four steps in predicting the mMTC network's traffic
using the Fast LiveStream Predictor (FLSP) algorithm, a new live forecasting algorithm.
The CNN-1D-based model only uses the traditional rolling algorithm and the output
is used for performance comparison with RNN-based models that can use FLSP algorithm.

 --------------------------      vvvvvvvvvvvvvvvvvv      ----------------------------------
|GeneratingTrafficPatterns| --> |CNN-1DNetTraining| --> |BurstDetNetTrainingWithCNN-1DData| -->
--------------------------      ^^^^^^^^^^^^^^^^^^      ----------------------------------
     --------------------------------------
--> |EvaluationAndPlottingResultsForCNN-1D|
    --------------------------------------

The code loads required data from "A_generatedTraffic" folder.
The trained network stored in "B_traindCNN1D" folder.

You can modify the following parameters:
    - cnnLayers: Number of CNN-1D layers;
    - windowSize: Windows size of CNN-1D model in time slots unit;
    - cnnOutFeatureSize: The featrue size of output sequence of CNN-1D layer;
    - cnnKernelSize: The one dimentional kernel size of CNN-1D layers;
    - maxpoolSize: The maxPool layer kernel size after each CNN-1D layer;
    - linearLayers: Number of linear network layers after CNN-1D network;
    - linearHidSize: Size of each linear layer;
    - dropoutVal: Dropout probability between DenseNet layers;
    - batchSize: Batch size of input data to ML network;
    - repeating: Number of iterations over the training dataset;
    - TpVec: A list shwoing prediction duration in the validation phase;
    - Tstep: The frequency of collecting new data from the network in live network;
    - Tseed: The initializing data length;
    - trainingDataSize: Number of patterns used for training;
    - validationDataSize: Number of patterns used for validation;
    - validationFreq: Running validation every validationFreq epochs.

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

from datetime import datetime
from MLNetworks import CNNTimeSeriesPredictor
from MLNetworks import dataManagement, cnn1dPreprocess, countParameters
    
def checkModelHyperPars(cnnLayers, cnnOutFeatureSize, cnnKernelSize, linearLayers,\
                linearHidSize):
    """
    This function modifies the machine learning model parameters to make sure 
    they meet the needed number of layers. Note that the size of last layer of
    linear network is determined by the output shape of the network. So it is
    excluded in this function.
    
        [cnnOutFeatureSize, cnnKernelSize, linearHidSize]=checkModelHyperPars(cnnLayers,\
            cnnOutFeatureSize, cnnKernelSize, linearLayers, linearHidSize)

    Parameters
    ----------
    cnnLayers : int
        Determines the number of CNN-1D layers.
    cnnOutFeatureSize : tuple(int)
        A list containing the output feature size of each CNN-1D layer.
    cnnKernelSize : tuple(int)
        A list containing the kernel size of each CNN-1D layer.
    linearLayers : int
        Determines the number of linear layers including the last layer.
    linearHidSize : tuple(int)
        A list containing the layer size of linear network (excluding the last
        layer.)

    Returns
    -------
    modified hyperparameters : tuple
        The modified lists of the hyper parameters are returned as a result.

    """
    # The difference of expected and actual size:
    featureSizeDiff=len(cnnOutFeatureSize)-cnnLayers
    if featureSizeDiff>0: # if we have more numbers than expected
        print('\nThe CNN feature sizes is reduced to match the CNN size:')
        print(f'\t-> Changed from {cnnOutFeatureSize} to ', end='')
        cnnOutFeatureSize=cnnOutFeatureSize[:cnnLayers] # Get only first portion
        print(f'{cnnOutFeatureSize}.')
    elif featureSizeDiff<0: # if we have less numbers than expected
        print('The CNN feature sizes is extended to match the CNN size:')
        print(f'\t-> Changed from {cnnOutFeatureSize} to ', end='')
        for i in range(-1*featureSizeDiff):
            cnnOutFeatureSize+=[cnnOutFeatureSize[-1]] # Duplicate the last number
        print(f'{cnnOutFeatureSize}.')
    kernelSizeDiff=len(cnnKernelSize)-cnnLayers
    if kernelSizeDiff>0: # if we have more numbers than expected
        print('The CNN kernel sizes is reduced to match the CNN size:')
        print(f'\t-> Changed from {cnnKernelSize} to ', end='')
        cnnKernelSize=cnnKernelSize[:cnnLayers] # Get only first portion
        print(f'{cnnKernelSize}.')
    elif kernelSizeDiff<0: # if we have less numbers than expected
        print('The CNN kernel sizes is extended to match the CNN size:')
        print(f'\t-> Changed from {cnnKernelSize} to ', end='')
        for i in range(-1*kernelSizeDiff):
            cnnKernelSize+=[cnnKernelSize[-1]] # Duplicate the last number
        print(f'{cnnKernelSize}.')
    # The difference of expected and actual size:
    linearLayers=linearLayers-1 # Excluding the last layer which is determined by output data size
    linearSizeDiff=len(linearHidSize)-linearLayers
    if linearSizeDiff>0: # if we have more numbers than expected
        print('The linear layers sizes is reduced to match the linear network size. '+\
              '(The last layer is determined by output shape!):')
        print(f'\t-> Changed from {linearHidSize} to ', end='')
        linearHidSize=linearHidSize[:linearLayers] # Get only first portion
        print(f'{linearHidSize} (excluding the last layer!).')
    elif linearSizeDiff<0: # if we have less numbers than expected
        print('The linear layer sizes is extended to match the linear network size. '+\
              '(The last layer is determined by output shape!):')
        print(f'\t-> Changed from {linearHidSize} to ', end='')
        for i in range(-1*linearSizeDiff):
            linearHidSize+=[linearHidSize[-1]] # Duplicate the last number
        print(f'{linearHidSize} (excluding the last layer!).')
    return [cnnOutFeatureSize, cnnKernelSize,linearHidSize]

#%% Main Function:
if __name__=="__main__":
    # Simulation parameters:
    TpVec=[0.5,1,1.5,2] # Predicting Tp seconds ahead during the validation
    Tfeed=0.5 # The size of the feed data collected at each step of online prediction in seconds (same as Tstep)
    repeating=1  # Number of iterations over the training dataset.
    trainingDataSize=300 # Number of patterns used for training.
    validationDataSize=15 # Number of patterns used for validation.
    validationFreq=30    # Running validation every validationFreq epochs
    windowSize=400 # Windows size in time slots unit.
    # ML network settings that can be changed:
    nType='CNN-1D' # Network type: [LSTM, GRU]
    cnnLayers=3 # Needed to be considered in model input!!!
    cnnOutFeatureSize=[10,20,30] # The featrue size of output sequence of CNN-1D layer.
    cnnKernelSize=[3,5,7] # The one dimentional kernel size of CNN-1D layer.
    cnnPadding='same' # To retain the sequence size.
    maxpoolSize=2 # The maxPool layer kernel size after CNN-1D
    linearLayers=3 # Needed to be considered in model input!!!
    linearHidSize=[1000,500] # The hidden layer size of linear layer after CNN-1D (currently we consider only one hidden layer in the linear network!)
    dropoutVal = 0.3 # Dropout probability between linear network layers.
    GPU_flag=True # GPU flag
    # Fixed parameters:
    frameSize=0.005 # Duration of each time slot.
    # Checking the model parameters:
    [cnnOutFeatureSize, cnnKernelSize, linearHidSize]=checkModelHyperPars(cnnLayers,\
        cnnOutFeatureSize, cnnKernelSize, linearLayers, linearHidSize)
    # Load traffic patterns:
    (dataLoad, patternDataAddress)=dataManagement(path='A_generatedTraffic',\
                                                  includesStr='MTC_Traffic',\
                                                  returnAddress=True)
    # Create a container which is passed to other softwares and includes the address ...
    # of all loaded files. They may update it and pass to the next software:
    allLoadedAddress={'patternDataAddress':patternDataAddress}
    """
    Loading the "Pattern" list containing generated traffic patterns as follows:
    Shape of "Pattern": (500, 4, 180000) --> (# of patterns, 4 types of patterns, length of each pattern)
    Types of patterns are [detected, attempted,congested,freeP] and explained below:
    Pattern[:,0,:]: Detected packets by the base station (successfully delivered);
    Pattern[:,1,:]: Total attempts (transmissions) done by all devices to access the network;
    Pattern[:,2,:]: Count of congested preambles out of avilable preambles (usually: out of 53);
    Pattern[:,3,:]: Count of not used preambles at each time slot (usually: out of 53).
    NOTE: We use only "detected" and "congested" patterns in this work and the ...\
          rest of them can be used for presentation purposes.
    """
    Pattern=dataLoad.get('Pattern') # Loading the patterns
    """
    Extracting the detected packets and congested preambles pattern:
    "Pattern" contains 500 sequences which we divide them into three sets:
        - Training set: Strems are randomely chosen from first '400-validationDataSize' samples.
        - Validation set: Fixed streams located between '400-validationDataSize' and '400' positions.
        - Testing set: Strems are randomely chosen from last '100' samples.
    """
    starttime=datetime.now() # Record the simulation time.
    print('Simulation start time: ',str(starttime).split(".")[0])
    #%% Running the simulation for each Tp value in TpVec and saving the results:
    allData=dict() # To store the data of each Tp
    for Tp in TpVec:
        # Preparing the input data and simulation parameters:
        # windowSize=int(windowSizeToTp*Tp/frameSize) # The window size of CNN-1D in [time slots]
        steps=int(Tp/frameSize) # Prediction steps in [time slots]
        feedSize=int(Tfeed/frameSize) # Feed data size in [time slots] (determines the resolution of input sequences)
        epochs=repeating*trainingDataSize  # Old definition of epochs
        trainingSet=np.random.choice(range(len(Pattern)-100-validationDataSize),\
                                     trainingDataSize, replace=False) # Random choose
        detectedTraffic=np.zeros([trainingDataSize,np.shape(Pattern)[2]]) 
        congestion=np.zeros_like(detectedTraffic)
        for i,loc in enumerate(trainingSet):
            detectedTraffic[i,:]=Pattern[loc][0] # Detected traffic is in the first position
            congestion[i,:]=Pattern[loc][2] # Congestion is in the third position
        validationSet=[i for i in range(len(Pattern)-100-validationDataSize,len(Pattern)-100)]
        valDetectedTraffic=np.zeros([validationDataSize,np.shape(Pattern)[2]])
        valCongestion=np.zeros_like(valDetectedTraffic)
        for i,loc in enumerate(validationSet):
            valDetectedTraffic[i,:]=Pattern[loc][0] # Detected traffic is in the first position
            valCongestion[i,:]=Pattern[loc][2] # Congestion is in the third position
        # Generating the training tensors (input and target) from the selected patterns:
        [precessedInTraffic, precessedTargetTraffic]=cnn1dPreprocess(detectedTraffic,\
                                 windowSize=windowSize, feedSize=feedSize, steps=steps)
        [precessedInCongestion, precessedTargetCongestion]=cnn1dPreprocess(congestion,\
                                 windowSize=windowSize, feedSize=feedSize, steps=steps)
        
        # The ML network gets traffic and congestion values as input features. The \
        # input and output shape will be [trainingDataSize, batches, featureSize, windowsSize]:
        inputs=torch.cat((precessedInTraffic,precessedInCongestion),2) # Appending traffic and congestion tensors.
        targets=torch.cat((precessedTargetTraffic,precessedTargetCongestion),2) # Appending target tensors.
        # Preparing a small portion of validation data for validation purposes:
        [preprocessedValTraff, preprocessedValTarget]=cnn1dPreprocess(\
                                          valDetectedTraffic[:,0:40000],\
                                          windowSize=windowSize, feedSize=feedSize,\
                                          steps=steps)
        [preprocessedValCong, preprocessedValCongTar]=cnn1dPreprocess(\
                                          valCongestion[:,0:40000],\
                                          windowSize=windowSize,feedSize=feedSize,\
                                          steps=steps)
        # Concatenating the traffic and congestion patterns to create input/output data:
        valInputs=torch.cat((preprocessedValTraff, preprocessedValCong),2)
        valTargets=torch.cat((preprocessedValTarget, preprocessedValCongTar),2)
        # Creating the ML network:
        model = CNNTimeSeriesPredictor(featureSize=inputs.shape[-2], cnnOutFeatureSize=\
                                       cnnOutFeatureSize, windowSize=windowSize,\
                                       kernelSize=cnnKernelSize,padding=cnnPadding,\
                                       maxPoolSize=maxpoolSize,linearSize=linearHidSize,\
                                       dropoutVal=dropoutVal,steps=steps,GPU_flag=GPU_flag)
        model.double()
        # Moving the data on the desired processign device:
        if model.GPU_flag:
            model.cuda()
            device=torch.device('cuda') # Used to move tensors to the desired device (GPU).
        else:
            device=torch.device('cpu') # Used to move tensors to the desired device (CPU).
        model.train() # Training mode activation.
        criterion = nn.MSELoss() # Using MSE metric to calculate the error.
        optimizer = optim.Adam(model.parameters(),lr = 0.0001) # model.parameters() returns weights to be optimized.
        trainingLoss=np.zeros(epochs)  # Training error history
        starttimeTp=datetime.now() # Record the simulation time.
        starttimeModel=datetime.now() # Record the simulation time.
        print(f'\nTarining CNN-1D model for Tp={Tp} at {str(starttimeModel).split(".")[0]}.')
        # Training the ML network:
        valLoss=[] # A list to store the validation MSE values
        for e in range(repeating):
            for i,(inputSeq,targetSeq) in enumerate(tqdm(zip(inputs,targets),total=trainingDataSize,\
                                                 desc='Training CNN-1D network over the training data',\
                                                    colour='cyan')):
                model.train()   # Training mode activation.
                optimizer.zero_grad() # Initializing the gradients 
                out = model(inputSeq.to(device)) # Get the output
                loss = criterion(out,targetSeq.to(device)) # calculate the loss
                trainingLoss[e*trainingDataSize+i]=loss.item() # Add up the loss of each sub sequence.
                loss.backward()  # Backward propagating the loss
                optimizer.step() # Optimizing the net. parameters
                # Evaluating the performance of prediction over the validation data\
                # every validationFreq iterations. Evaluations done for online prediction\
                # mode for each specific 'Tp' in 'TpVec':
                if(i%validationFreq==0 and i!=0):
                    with torch.no_grad(): # No gradient calculation --> faster + less memory.
                        valSubLoss=0 # Initializing the validation loss for each sequence.
                        for (valInput, valTarget) in zip(valInputs,valTargets):
                            model.eval()
                            output=model(valInput.to(device))
                            valSubLoss += criterion(output[:,:,-feedSize:].to(device), \
                                                    valTarget[:,:,-feedSize:].to(device)).item()
                        valLoss.append(valSubLoss/validationDataSize)  # Validation error of first set 
        valLoss=np.array(valLoss) # Converting to numpy array.
        # Storing the results:
        print(f'\nTotal elapsed time for Tp={Tp}: {str(datetime.now()-starttimeModel).split(".")[0]}.\n')
        Description='\033[38;5;11mThis is a result of training a CNN-1D model '+\
            'to predict the network traffic with following settings and fixed window '+\
            'size (version 3):\033[0m\n\t'+\
            f'- {nType} network with {cnnLayers} layers, output feature sizes of'+\
            f' {cnnOutFeatureSize}, and kernel sizes of {cnnKernelSize};\n\t'+\
            f'- Linear network with {linearLayers} layers and hidden sizes of'+\
            f' {linearHidSize} (excluding the input and output size);\n\t'+\
            f'- Training sequences: {trainingDataSize};\n\t'+\
            f'- Total training iterations: {epochs};\n\t'+\
            f'- Validation sequences: {validationDataSize};\n\t'+\
            f'- Window size: {windowSize};\n\t'+\
            f'- Tstep (Tfeed) for valdiation: {Tfeed} second(s);\n\t'+\
            f'- Tp value: {Tp} second(s);\n\t'+\
            f'- Simulation time: {str(datetime.now()-starttimeTp).split(".")[0]}.\n\n'+\
            '\033[38;5;11mResults and discusstions:\033[0m\n\t'+\
            f'- Parameter count: {countParameters(list(model.parameters())):,};\n\t'+\
            f'- Last trainig MSE loss value: {trainingLoss[-1]:.2f};\n\t'+\
            f'- Validation MSE loss list: {[f"{loss:.2f}" for loss in valLoss]}.'
        print(Description,'\n--------------------------\n')
        data={'model':model, 'epochs':epochs,'losshist':trainingLoss,\
              'Description':Description, 'trainingDataSize':trainingDataSize,\
              'repeating':repeating, 'valLoss':valLoss, 'Tp':Tp,\
              'Tfeed':Tfeed,'windowSize':windowSize, 'dropoutVal':dropoutVal} # data to be stored.
        allData[Tp]=data    
    #%%############################################################################
    # Including the address of loaded traffic patterns and other important data:
    allData.update({'allLoadedAddress':allLoadedAddress, 'Tfeed':Tfeed, \
                    'TpVec':TpVec, 'nType':nType}) 
    # Storing the data on disk:
    dataManagement(data=allData, save=True, fileFormat='.pt', \
                   fileName=f'B2_TrainedCNN1D_WS{windowSize}',version='_v3',\
                   path='B_traindCNN1D')
    print('End time: ',str(datetime.now()).split(".")[0],'.')
    print('Total elapsed time: ',str(datetime.now()-starttime).split(".")[0],'.')
    
