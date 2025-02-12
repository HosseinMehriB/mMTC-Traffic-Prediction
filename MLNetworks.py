# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Thu May 25 10:41:31 2023

@author: HOSSEINMEHRI

This file contains the machine learning methods used in traffic prediction task.
"""

import torch
from torch import nn
from datetime import datetime
import os
import pickle
from scipy.io import savemat
import numpy as np
from scipy import signal


#%% Define required functions - Section one: Essential Functions
def getAddress(path,newest=True,includesStr=''):
    """
    address = getAddress(path,newest=True,includesStr='')
    
    This function is used to choose a file in the desired path. It can print a
    list of files ore can return the lates saved file.

    Parameters
    ----------
    path : string
        The desired path to look for a file.
    newest : Boolean, optional
        If True, it returns the latest stored file, otherwise, it prints a list
        of files in the path and you can choose one. The default is True.
    includesStr : string, optional
        Only lists and returns addresses that includes the given string in its
        name. The default is '', which means return all addresses.

    Returns
    -------
    address : string
        The address of the desired file.

    """
    files = os.listdir(path)
    # Filtering out the undesired files:
    files = [basename for basename in files if basename.find(includesStr)>-1]
    paths = [os.path.join(path, basename) for basename in files]
    if newest:
        address=max(paths, key=os.path.getctime)
    else:
        print('\n')
        for i in range(len(files)):
            print('{0:2d})--> {1}'.format(i,files[i]))
        choice=input('Choose a file to load: ')
        if choice=='': # If nothing entered, we choose the last item.
            choice=-1
        address=paths[int(choice)]
    return address

def dataManagement(address=None,data=None,save=False,fileFormat='.pickle',choice=False,\
                   fileName='',version='vx', path='Results', map_location='cpu',\
                       includesStr='', returnAddress=False):
    """
    This function is used to load or save files in pickle format.
    save:
        dataManagement(address=None,data=data,save=True,choice=False,fileFormat='.pickle')
    load:
        data = dataManagement(address=None,data=None,save=False,fileFormat='.pickle',choice=False)
    
    Parameters
    ----------
    address : string, optional
        Only used when loading data. If None, we load the lates saved file or 
        print the list of files based on 'choice' setting. The default is None.
    data : Object, optional
        The data to be saved. The default is None.
    save : Boolean, optional
        If True, saves the given data, else, loads data. The default is 'False'.
    fileFormat : string, optional
        You can choose '.pt', '.mat', or '.pickle', to save using pytorch or pickle. The 
        default is '.pickle'
    choice : Boolean, optional
        If True and 'address' is None, it prints the list of files in the path
        to be loaded. Otherwise, it loads the latest stored file.  The default 
        is 'False'.
    fileName : string, optional
        This is a decribing section of file name which helps to find the desired
        file easier. The default is "''".
    path : string, optional
        The main directory of loading/saving the files. The default is 'Results'.
    map_location : string, optional
        Load files on CPU or GPU. Options: 'cpu', 'gpu'. The default is 'gpu'.
    includesStr : string, optional
        Only lists and returns addresses that includes the given string in its
        name. This option only works if you are loading a file. The default is '',
        which means return all addresses. 
    returnAddress : Boolean, optional
        If True, it returns the address used for loading a data.
        
    Returns
    -------
    data : Object
        If 'save' is false, this function loads and returns the file.

    """
    if map_location=='cpu':
        map_location=torch.device('cpu')
    else:
        map_location=torch.device('cuda')
    if save:
        TT=datetime.now() # Data and time of saving the file
        address1=path+'/'+TT.strftime("%j")+'_'+fileName+'__'+TT.strftime("%a")+\
            '_'+TT.strftime("%b")+TT.strftime("%d")+'_'+TT.strftime("%H")+TT.strftime("%M")
        fileType=version+fileFormat
        address=address1+fileType
        # If there is file with the same name, we need to change the address
        count=1
        while (os.path.isfile(address)):
            address=address1+'('+str(count)+')'+fileType
            count+=1
        if (fileFormat=='.pickle'):
            pickle.dump(data, open(address, "wb"),protocol=pickle.HIGHEST_PROTOCOL)
        elif (fileFormat=='.mat'):
            savemat(address,data)
        else:
            torch.save(data, address)
        print(f'File saved to "{address}"!\n')
        return None
    else:
        if (address is None) or (choice):
            address=getAddress(path,newest=(not choice), includesStr=includesStr)
        address=address.replace('\\','/') # Making it compatible with Linux systems
        if ('.pickle' in address):
            data=pickle.load(open(address,"rb"))
            print(f'Data loaded from {address}!\n')
        elif('.pt' in address):
            data=torch.load(address,map_location=map_location)
            print(f'Data loaded from {address}!\n')
        else:
            print('\n\033[48;5;9m\033[38;5;11mERROR:\033[0m \033[38;5;9mUnknown',\
                  f'file format: \033[0m\033[03m{address}\033[0m')
            data=None
        if returnAddress:
            return data, address
        else:
            return data
#%%
def textColorFormat(text, setting=['','',0]):
    """
    Changes the color of foreground and background as well as text style (Pro version).
        
        textColorFormat('Example Text!','rgu')
    
    Usually called 'TC' using aliasing:
        import textColorFormat from * as TC
    or:
        TC=textColorFormat
    
    Parameters
    ----------
    text : str
        Input text.
    settings : str/list(int)
        Settings of foreground, background, and font style. Can be:
            
            I) A three-char string where the first char is text color, second is 
            background color, and third is the font style. See below for details:
                Char meaning = {'r':red, 'g':green, 'y':yellow, 'b':blue, 
                                'm':magenta, 'c':cyan, 'w':white, 'n':normal font,
                                'i':italic font,'u':underline, '0': neutral}
            
            II) A list of integers of size three:
                    --> settings : [text Color, background color, font style]
                For detailed information, see (section 8-bit): 
                        https://en.wikipedia.org/wiki/ANSI_escape_code
                
                A summary of codes are as below:
                color codes (bright) = {0: null, 8: Gray, 9: red, 10: green, 11: yellow,
                                        12: blue, 13: magenta, 14: cyan, 15: white}
                color codes (normal) = {0: null, 1: red, 2: green, 3: olive, 
                                        4: blue, 5: purple, 6: teal, 7: light gray}
                neutral color = {'': neutral}
                style codes = {0: Reset, 1: Normal, 3: Italic font, 4: Underline}
    Returns
    -------
    str
        Formated text as str.

    """
    shortCode={'r':9, 'g':10, 'y':11, 'b':12, 'm':13, 'c':14, 'w':15, 'n':1, 'i':3,'u':4, '0':0, 'n':0}
    if isinstance(setting,list):
        if setting[0]!=0:
            textColor= f'\033[38;5;{setting[0]}m'
        else:
            textColor=''
        if setting[1]!=0:
            backgroundColor= f'\033[48;5;{setting[1]}m'
        else:
            backgroundColor=''
        if setting[2]!=0:
            style=f'\033[{setting[2]};1;1m'
        else:
            style=''
    elif isinstance(setting,str):
        code=shortCode[setting[0]]
        if code!=0:
            textColor= f'\033[38;5;{code}m'
        else:
            textColor=''
        code=shortCode[setting[1]]
        if code!=0:
            backgroundColor= f'\033[48;5;{code}m'
        else:
            backgroundColor=''
        code=shortCode[setting[2]]
        if code!=0:
            style=f'\033[{code};1;1m'
        else:
            style=''
    return f"{textColor}{backgroundColor}{style}{text}\033[0m"
# Short name for textColorFormat:
TC=textColorFormat # Aliasing for short naming
#%%
def countParameters(pars):
    """
    count = countParameters(list(model.parameters()))
    Calculates the total parameters of the given model.
    
    Parameters
    ----------
    pars : list
        list of parameters. Models usually return a generator, which need to be
        change to list by list() function before feeding to countParameters().

    Returns
    -------
    count : int
        Total number of parameters.

    """
    if type(pars) == list:
        return sum(countParameters(subitem) for subitem in pars)
    else:
        return pars.numel()
#%% Section two: Data Procesing Functions
# Defining data preprocessing function to prepare it for LSTM network    
def preprocess(stream, batchSize=1):
    """
    This function get a stream as numpy array and returns batched torch tensors 
    of feed and target.
    
    (feed, target) = preprocess(stream, batchSize=1)
    
    Parameters
    ----------
    stream : numpy.array
        Raw input sequential stream.
    batchSize : int, optional
        Batch size. The default is 1.

    Returns
    -------
    feed : tensor
        Batched feed stream of shape [batches-1, batchSize, 1].
    target : tensor
        Batched target stream of shape [batches-1, batchSize, 1].

    """
    stream=signal.savgol_filter(stream,97,2) # Smoothing the input stream.
    length=stream.size # Input stream length.
    residual=length % batchSize # Residual value is needed during reshaping process
    feed=np.reshape(stream[:length-residual],(-1,batchSize,1)) # Resulted shape: [batches, batchSize, 1]
    target=torch.tensor(feed[1:,:,:]) # Target stream is one batch ahead of feed stream.
    feed=torch.tensor(feed[:-1,:,:])  # Removing the last batch.
    return feed, target

# This function gets the raw list of inputs and transfers them to the desired tensors -- Jan 17
def list2Tensor(rawData,batchSize):
    """
    This function gets a set of sequential streams as a numpy ndarray and generates
    input and expected output to and from the ML network based on given batch size
    and returns the results as torch tensors.
    
    (batchedInput, batchedTarget) = list2Tensor(rawData, batchSize)

    Parameters
    ----------
    rawData : numpy.ndarray
        Contains all the sequential streams.
    batchSize : int
        Batch size which data is shaped based on that.

    Returns
    -------
    batchedInput : tensor
        The feed data to the ML network shaped based on the given batch size.
    batchedTarget : tensor
        The target data that the ML network is expected to return.

    """
    dataLength=rawData.shape[0] # Number of streams
    firstTensor=preprocess(rawData[0,:],batchSize) # Creating the first data sample to get the shapes.
    batchedInput=torch.zeros((dataLength,firstTensor[0].size(0),batchSize,1),dtype=torch.double)
    batchedTarget=torch.zeros((dataLength,firstTensor[0].size(0),batchSize,1),dtype=torch.double)
    batchedInput[0]=firstTensor[0] # 
    batchedTarget[0]=firstTensor[1]
    # Preprocessing all the data samples and expected outputs (targets)
    for i in range(dataLength-1):
        [batchedInput[i+1],batchedTarget[i+1]]=preprocess(rawData[i+1,:],batchSize)
    return batchedInput,batchedTarget

# This function prepares the raw data for CNN-1D network:
def cnn1dPreprocess(rawData, windowSize=2000, feedSize=200, steps=1000):
    """
    This function gets a set of sequential streams as a numpy ndarray and generates
    input and expected output to and from the one dimensional CNN based on given
    parameters and returns the results as torch tensors.

    Parameters
    ----------
    rawData : numpy.ndarray
        Contains all the sequential streams.
    windowSize : int, optional
        The size of input sequence to CNN-1D network. The default is 2000.
    feedSize : int, optional
        Determines the resolution of input data to CNN-1D. It is equal to the size
        of feed data at each time. The default is 200.
    steps : int, optional
        Size of the predicted output. The default is 1000.

    Returns
    -------
    inputTensor : tensor
        The input data to CNN-1D.
    targetTensor : tensor
        The target output data from CNN-1D network.

    """
    (totalStreams,streamSize)=rawData.shape
    slices=int((streamSize-windowSize-steps)/feedSize)+1
    inputTensor=torch.zeros([totalStreams,slices,1,windowSize],dtype=torch.double)
    targetTensor=torch.zeros([totalStreams,slices,1,steps],dtype=torch.double)
    for i, stream in enumerate(rawData):
        stream=signal.savgol_filter(stream,97,2)
        for idx in range(slices):
            inputTensor[i,idx,0,:]=torch.from_numpy(stream[idx*feedSize:idx*feedSize+windowSize])
            targetTensor[i,idx,0,:]=torch.from_numpy(stream[idx*feedSize+windowSize:idx*feedSize+windowSize+steps])
    return inputTensor, targetTensor

# This function is used to extract the bursty traffic labels:
def burstTargetDataGen(detectedTraffic, congestion, batchSize, Tp, Tstep=0.5,\
                       frameSize=0.005, onlyPredData=False):
    """
    This function extracts the bursty region from the congestion sequence. Moreover, 
    it prepares the real data to be feed to the linear network.
    
    (burstLabel, exactCongArea, feedData) = burstTargetDataGen(detectedTraffic,\
                        congestion, batchSize, Tp, Tstep=0.5, frameSize=0.005)

    Parameters
    ----------
    detectedTraffic : numpy.array
        Traffic sequence. Used to create the feed data.
    congestion : numpy.array
        Congestion sequence. It is used to extract the bursty area labels.
    batchSize : int
        Batch size.
    Tp : float
        Prediction duration at each step.
    Tstep : float, optional
        Frequency of collecting new data as well as duration of the fresh data.
        The default is 0.5.
    frameSize : float, optional
        Duration of each time slot. The default is 0.005.
    onlyPredData : Boolean, optional
        When we want only the predicted data to feed the burst detection network,
        we make this flag as True. Otherwise, it includes the fresh data in the 
        feed data. The default is False.

    Returns
    -------
    burstLabel : numpy.array
        Labels used to train the burst prediction network.
    exactCongArea : numpy.array
        A sequence locating the exact location of bursty area for plotting purposes.
    feedData : tensor
        Data which is fed to the burst prediction network.

    """
    #### This part extracts the bursty region from the congestion stream.
    #### Moreover, It prepares the real data to be feed to the linear network.
    congestTensor=torch.from_numpy(np.copy(congestion)) # Converting numpy array to torch tensor.
    detectedTensor=torch.from_numpy(np.copy(detectedTraffic)) # Converting numpy array to torch tensor.
    # Extracting the congested area by investigating the intensity of congestion in\
        # a period of time. Based on experiment, we use a window size of 3 seconds\
        # to calculate the intensity of congetion in a specific time:
    congestion=signal.savgol_filter(congestion,97,2) # Smoothing the congestion data
    intensity=np.zeros(np.size(congestion)) # Averaged congestion over the moving windows.
    windowSize=int(3/frameSize/batchSize)*batchSize # Rounding the windows size according to the batch size
    # Extending the congestion stream for averaging:
    congestion2=np.concatenate((np.zeros(int(windowSize/2)),\
                               congestion,np.zeros(int(windowSize/2)))) 
    for i in range(np.size(congestion)):
        intensity[i]=np.sum(congestion2[i:i+windowSize])
    intensity=signal.savgol_filter(intensity,97,2) # Smoothing the averaged data to remove the outlier points 
    burstyArea=intensity>7000   # During the congestion we will have 'ones' (shape: (180000,))
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # Extract the exact congested time slots from the congested neighborhood:
    burstySlots=(congestion*burstyArea)>23
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # =========================================================================
    """ 
    This part generates the 'burstLabel' and 'exactCongArea' streams.
    # Raw data is a label to each period if there is a fraction of congestion in that period.
    # BurstLabel: label of each priod is 'yes' if there is a bursty period in next (Tpred/Tstep) periods.
    # exactCongArea: The exact location of congested area for presentation purposes.
    # Raw data:     | N | N | N | N | B | B | N | N | N |...
    # BurstLabel :       | no | yes | yes | yes | yes | no | no | no | no |...
    # ExactCongArea: | regu | regu | regu | regu | cong | cong | regu | regu | regu |...
    # BurstLabel is used to train the model. ExactCongArea stream is used only for visual presentation.
    """
    # =========================================================================
    # Generating the burst labels:
    slicesAhead=int(Tp/Tstep) # The prediction windows in terms of Tstep periods.
    sliceSize=int(Tstep/frameSize) # number of time slots in a step
    totalSteps=int(np.size(intensity)/sliceSize) # Total number of periods in the burst stream
    rawData=np.zeros((totalSteps)) # Indicates the bursty slices.
    burstLabel=np.zeros((totalSteps)) # Main labels used for the training phase.
    possibleCongArea=np.zeros(np.size(burstySlots)) # Labels just for possibleCongArea and ploting.
    for i in range(totalSteps):
        rawData[i]=np.sum(burstySlots[i*sliceSize:(i+1)*sliceSize])>0 # Labels of each period
    for i in range(totalSteps):
        isBursty=np.sum(rawData[i+1:i+1+slicesAhead])>0 # Do we have a burst in next stpes?
        burstLabel[i]=isBursty*1 # If we have burst in next steps, assign '1' label to this period.
        # If we have burst in next steps, change next 'stepsAhead' steps to '1'
        possibleCongArea[(i+1)*sliceSize:(i+1+slicesAhead)*sliceSize]=isBursty*1  
    exactCongArea=(congestion*possibleCongArea)>19 # The exact congested area.
    # In this part we prepare the input data to feed into the linear burst prediction network:
    slices=int(congestTensor.size()[0]/sliceSize) # Total number of slices in a stream
    predictedSlices=int(Tp/Tstep)
    congestTensor=congestTensor[0:slices*sliceSize].view(-1,sliceSize) # Giving a new shape to the real data
    detectedTensor=detectedTensor[0:slices*sliceSize].view(-1,sliceSize) # Giving a new shape to the real data
    slicedData=[]
    if onlyPredData: # When we want only predicted data, we exclude the feed data
        noFeed=1 # Ignoring the first slice which is the fresh data
    else:
        noFeed=0 # Includeing the first slice which is the fresh data
    for i in range(slices-predictedSlices):
        temp=torch.cat((detectedTensor[i+noFeed:i+(1+predictedSlices)]\
                        .view(1,-1),\
                        congestTensor[i+noFeed:i+(1+predictedSlices)]\
                        .view(1,-1)),1)
        slicedData.append(temp) # Store the slices in a list
    # Change the list of tensors to a single tensor and squeeze it. This tensor\
        # will be used as input to the bursty prediction network: 
    feedData=torch.stack(slicedData).squeeze() 
    return burstLabel, exactCongArea, feedData

#%%
# RNN network with DenseNet to predict the future:
class rnnNetwork(nn.Module):
    def __init__(self, features, batchSize, rnnLayers, hiddenLayerSize, dropoutVal,\
                 nType='LSTM', track_states = False,GPU_flag=True):
        super(rnnNetwork,self).__init__()
        self.modelVer=f'rnnNetwork: One {rnnLayers}-layer {nType} and one DenseNet with two-layer FFNN'
        self.bs = batchSize
        self.features = features # Dimension of each input sample.
        self.nh = hiddenLayerSize  # The dimension of h_(t-1) coming from previous time slots.
        self.nl = rnnLayers  # Number of LSTM/GRU layers.
        self.track = track_states
        self.nType=nType # Type of RNN network.
        if not nType in ['LSTM', 'GRU']:
            raise TypeError(f"Only {TC('LSTM','00u')} and {TC('GRU','00u')} models are accepted!")
        self.rnn1 = eval(f'nn.{nType}(self.features,self.nh,self.nl)') #(features, hidden_size,num_layers)
        self.dropout = nn.Dropout(p = dropoutVal)    # Dropout value between the LSTM/GRU and FFNN layers.
        self.linear1 = nn.Linear(self.nh,self.nh)    # First linear layer.
        self.linear2 = nn.Linear(self.nh*2,features) # Second linear Layer. 
        self.GPU_flag=GPU_flag # Determines the device on which the ML newtrok is running.
        if GPU_flag: 
            self.device=torch.device('cuda') # Device on which the ML newtrok is running.
        else:
            self.device=torch.device('cpu')  # Device on which the ML newtrok is running.
        self.init_hidden(self.bs)  # Initializing the hidden layers of RNN network.
    
    def init_hidden(self,batchSize): 
        if self.nType == 'LSTM': # RNN: LSTM
            # In LSTM, we get two values from previous time slot, those are: h_(t-1) and C_(t-1):
            self.hiddenStates=[torch.rand(self.nl,batchSize,self.nh,dtype = \
                                          torch.double,device = self.device)*2-1,\
                               torch.rand(self.nl,batchSize,self.nh,dtype = \
                                          torch.double,device = self.device)*2-1]
        else: # RNN: GRU
            # In GRU, we get one value from previous time slot, this is: h_(t-1)
            self.hiddenStates=torch.rand(self.nl,batchSize,self.nh,dtype = \
                                         torch.double,device = self.device)*2-1
            # num_directions = 2 for bidirectional, else 1
            # (rnnLayers*num_directions,batchSize,hidden_size)
    def forward(self, inputs, steps = 1000, eval = False, onlineStream=None):
        predictions = [] # A comtainer to store the predicted batches in prediction phase.
        # Using the previous states without retaining the graph:
        
        if self.nType=='LSTM':
            self.hiddenStates=[self.hiddenStates[0].detach().clone(),
                               self.hiddenStates[1].detach().clone()] 
        else:
            self.hiddenStates=self.hiddenStates.detach().clone()
        outputs,self.hiddenStates = self.rnn1(inputs,self.hiddenStates)
        outputL1=self.linear1(self.dropout(outputs))
        inputL2=torch.cat((self.dropout(outputL1),self.dropout(outputs)),2)
        outputL2 = self.linear2(inputL2) # During the training phase, it will be the same size as target tensor
        # During the evlaution phase, we won't use this tenosr!
        if(eval): # In prediction phase, we predict the sequence for 'steps' batches:
            evalInput = outputL2[-1:].detach().clone() # We feed detached data.
            # Creating a copy of the states and keep them original one untouched:
            if self.nType=='LSTM':
                hidStates=[self.hiddenStates[0].detach().clone(), 
                           self.hiddenStates[1].detach().clone()]
            else:
                hidStates=self.hiddenStates.detach().clone() 
            for i in range(steps-1):
                rnnOut,hidStates = self.rnn1(evalInput,hidStates) # Calculate the output of RNN network
                linear1Out = self.linear1(rnnOut) # Calculate the output of first linear layer
                inputL2=torch.cat((linear1Out,rnnOut),2) # Concatinating the output of first linear layer and RNN layer
                linear2Out=self.linear2(inputL2) # Calculate the output of second linear layer
                predictions += [linear2Out[0]] # Adding the final output of deep neural netowk to the output list (and squeezing!)
                evalInput = linear2Out # Using the final outut of deep neural netowk as input to the network
            predictedTensor=torch.stack(predictions)
            outputL2=torch.cat((outputL2,predictedTensor),0) # Concatinating the actual data to seed section
        return outputL2

# Linear network to predict bursty traffic:
class linearNetwork(nn.Module):
    def __init__(self, inputSize, outputSize=1, dropoutVal=0.4,GPU_flag=True, \
                 track_states = False):
        super(linearNetwork,self).__init__()
        self.GPU_flag=GPU_flag # Determines the device on which the ML newtrok is running.
        self.inLen=inputSize # input data size.
        self.linear1=nn.Linear(inputSize,inputSize//2) # First linear layer size. 
        self.linear2=nn.Linear(inputSize//2,outputSize) # Second linear layer size. 
        self.dropout=nn.Dropout(p = dropoutVal) # Dropout value between the layers.
        self.activation=nn.Tanh() # Activation function.
    def forward(self,inputData):
        inFeatureSize=inputData.size(-1) # Input feature size.
        # This section is for triming the input data:
        inputData=inputData.view(-1,2,int(inFeatureSize/2)) # A) Separating traffic and congestion data.
        # B) Trimming the extra samples of traffic and congestion data and then, \
        # C) reshaping the results to the desired format:
        inputData=inputData[:,:,0:int(self.inLen/2)].reshape(-1,self.inLen) 
        outputData=self.linear1(inputData)
        outputData=self.dropout(outputData)
        outputData=self.activation(outputData)
        outputData=self.linear2(outputData)
        outputData=(self.activation(outputData)+1)/2
        return outputData

# Linear network to predict bursty traffic:
class linearNetwork2(nn.Module):
    def __init__(self, inputSize, outputSize=1, layers=2, dropoutVal=0.4,GPU_flag=True, \
                 track_states = False):
        super(linearNetwork2,self).__init__()
        self.GPU_flag=GPU_flag # Determines the device on which the ML newtrok is running.
        self.inLen=inputSize # input data size.
        sizes=[inputSize]+[(layers-i)*(inputSize//layers) for i in range(1,layers)]\
            +[outputSize]
        dropoutVals=[0]+[dropoutVal]*(layers-1)
        activations=[nn.ReLU()]*(layers-1)+[nn.Tanh()]
        activationFlags=[True for _ in range(layers)]
        self.linears=nn.Sequential(*[linearLayer(sizes[i],sizes[i+1],dropoutVal=\
                                 dropoutVals[i], activation=activationFlags[i],\
                                 activationType=activations[i])\
                                     for i in range(len(sizes)-1)])
    def forward(self,inputData,reshape=True):
        if reshape: # Reshape when input shape is [-1,inFeatureSize] to [-1,2,inFeatureSize/2] for possible trimming and returning back to [-1,self.inLen]
            inFeatureSize=inputData.size(-1) # Input feature size.
            # This section is for triming the input data:
            inputData=inputData.view(-1,2,int(inFeatureSize/2)) # A) Separating traffic and congestion data.
            # B) Trimming the extra samples of traffic and congestion data and then, \
            # C) reshaping the results to the desired format:
        inputData=inputData[:,:,0:int(self.inLen/2)].reshape(-1,self.inLen) 
        outputData=(self.linears(inputData)+1)/2
        return outputData

# A CNN-1D cell:
def cnn1dLayer(inFeatureSize,outFeatureSize,kernelSize,padding,maxPoolSize):
    return nn.Sequential(nn.Conv1d(inFeatureSize, outFeatureSize, kernel_size=kernelSize,\
                           padding=padding),nn.ReLU(),nn.MaxPool1d(maxPoolSize))

def linearLayer(inSize, outSize, dropoutVal=0, activation=True, \
                activationType=nn.ReLU()):
    activLayer=[activationType]*activation
    dropout=[nn.Dropout(p=dropoutVal)]*(dropoutVal>0)
    cell=dropout+[nn.Linear(inSize, outSize)]+activLayer
    return nn.Sequential(*cell)
    
# A CNN-1D model with linear layers to generate the output:
class CNNTimeSeriesPredictor(nn.Module):
    def __init__(self,featureSize=2, cnnOutFeatureSize=[10,20], windowSize=2000, \
                 kernelSize=[3,3], padding='same',maxPoolSize=2, linearSize=[1000,1000], \
                 dropoutVal=0.3, steps=1000,GPU_flag=True):
        super(CNNTimeSeriesPredictor, self).__init__()
        self.GPU_flag=GPU_flag
        self.featureSize=featureSize
        self.steps=steps
        featureSizes=[featureSize]+cnnOutFeatureSize
        self.conv1DLayers=nn.Sequential(*[cnn1dLayer(featureSizes[i],featureSizes[i+1],\
                                         kernelSize[i],padding,maxPoolSize) for i in range(len(kernelSize))])
        self.flatten = nn.Flatten() # output shape: [bs,64*windowSize/2]
        linearSizes=[int(cnnOutFeatureSize[-1]*int(windowSize/(maxPoolSize**len(kernelSize))))]+\
            linearSize + [steps*featureSize]
        activations=[True for _ in range(len(linearSize))]+[False]
        self.linearLayers=nn.Sequential(*[linearLayer(linearSizes[i],linearSizes[i+1],\
                                                      dropoutVal,activations[i]) \
                                          for i in range(len(activations))])
    def forward(self, x):
        batchSize=x.shape[0]
        x = self.conv1DLayers(x)
        x = self.flatten(x)
        x = self.linearLayers(x)
        return x.reshape([batchSize,self.featureSize,self.steps])
