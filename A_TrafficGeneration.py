# -*- coding: utf-8 -*-
__author__ = 'Hossein Mehri'
__license__ = 'MIT'
"""
Created on Sun Feb 28 03:26:49 2021
This code generates bursty traffic (including bith uniform and bursty traffic)
of a massive machine type communication (mMTC) network based on given MTC
groups and the probability of happening a burst in each group.

The generated traffic patterns will be stored in "Traffic" folder as PyTorch
tensors.

@author: Hossein Mehri

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

from datetime import datetime
import os
from scipy import signal, integrate, special
import numpy as np
import torch
from tqdm import tqdm

"""This function produces the pdf of Beta distribution with alpha=3 and beta=4"""
def betaDistribution(t,T=10):
    """
    Parameters
    ----------
    t : float
        Time in which the Beta distribution will be calculated.
    T : float, optional
        Beta distibution duratin. The default is 10.

    Returns
    -------
    pdf : float
        probability density function at time point "t".

    """
    alpha = 3
    beta = 4
    betaFunction = special.beta(alpha,beta)
    pdf = (t**(alpha-1)*(T-t)**(beta-1))/(T**(alpha+beta-1)*betaFunction)
    return pdf

"""This function creates the uniform traffic intensity"""
def uniformTrafficIntensity(T, frameSize, numDevices, packetProbability=1/60):
    """
    Parameters
    ----------
    T : float
        Uniform distribution time.
    frameSize : float
        Duration of each time slot in seconds.
    numDevices : int
        Number of devices in the MTC group.
    packetProbability : float, optional
        DProbability of having a packet in a second. The default is 1/60.

    Returns
    -------
    uniformTraffic : int array
        Returns uniform traffic intensity at each time slot.
    """
    slots=int(T/frameSize)
    packetsPerSlotPerDev=packetProbability*frameSize
    # Generating a unifrom random variable for each device/slot:
    randomSeries=np.random.uniform(0,1,[slots,numDevices]) 
    # Packets per slot/active devices per slot:
    uniformTraffic=np.sum(randomSeries<=packetsPerSlotPerDev,1) 
    return uniformTraffic

"""This function creates the burst traffic intensity"""
def burstTrafficIntensity(numDevices, frameSize, Tb=10):
    """
    Parameters
    ----------
    numDevices : int
        Number of devices in the MTC group.
    frameSize : float
        Duration of each time slot in seconds.
    Tb : float, optional
        Burst traffic duration. The default is 10.

    Returns
    -------
    burstTraffic : int array
        Returns bursty traffic intensity at each time slot.
    """
    slots=int(Tb/frameSize) # number of slots during the burst traffic
    packetsPerSlotPerDev=np.zeros([slots,numDevices]) # Initializing
    tVector=np.linspace(0, Tb-frameSize, slots)  # Time vector of each slot
    for i,t in enumerate(tVector):
        # Packet generation probability at each time slot per device
        packetsPerSlotPerDev[i,:] = integrate.quad(lambda t: \
                                   betaDistribution(t,Tb), t, t+frameSize)[ 0 ]
    randomSeries=np.random.uniform(0,1,[slots,numDevices]) # Generating a unifrom random variable for each device/slot
    burstTraffic = np.sum(randomSeries<=packetsPerSlotPerDev,1) # Packets per slot/active devices per slot
    
    return burstTraffic    # Returning the number of packets/active devices in each time slot


"""This function generates the events riggering time and their duration for each 
type of devices (each network)"""
def eventGenerator(T, frameSize, eventProbability):
    """
    Parameters
    ----------
    T : float
        Observation window in seconds.
    frameSize : float
        Duration of each time slot in seconds.
    eventProbability : float
        Probability of observing an bursty event per second.

    Returns
    -------
    eventsTime : float array
        Duration of each event.
    Tb : int array
        Index of event occurance.
    """
    slots=int(T/frameSize) # Total number of slots
    eventProbabilityPerSlot=eventProbability*frameSize
    events=np.random.uniform(0,1,slots)<=eventProbabilityPerSlot
    eventsTime=np.where(events)[0] # Finding the location of the events
    numEvents=np.sum(events) # Total number of events
    Tb=np.random.uniform(8,15,numEvents)# Burst traffic duration: random value between 8 and 11 seconds
    Tb=Tb-Tb%frameSize # Rounding the burst time with respect to the frame size.
    # This part removes the overlapping events
    lastBurstEnd=0
    ovelappingEvents=[]
    for i,triggerTime in enumerate(eventsTime):
        if(lastBurstEnd>triggerTime):  # We have overlapping
            ovelappingEvents.append(i)    # Recording the overlapping events to be removed later
        else:
            lastBurstEnd=triggerTime+Tb[i]/frameSize # Updating LBD with the end of last valid burst traffic
    # removing the overlapping events
    eventsTime=np.delete(eventsTime,ovelappingEvents)
    Tb=np.delete(Tb,ovelappingEvents)
    return eventsTime, Tb

"""This function only gets the set of devices and their population as well as 
event probability and returns the total number of new packets at each time slot"""
def newArivals(numDevicesVec, eventProbabilities, T, frameSize):
    """
    Parameters
    ----------
    numDevicesVec : int array
        Number of devices in the MTC group.
    eventProbabilities : float array
        Probability of observing an bursty event per second for each group. 
    T : float
        Observation window in seconds.
    frameSize : float
        Duration of each time slot in seconds.

    Returns
    -------
    arrivals : array
        New packets of whole MTC network at each time slot.
    eventsAll : list
        List of arrays including index of event occurance for each group.
    TbsAll : list
        List of arrays including duration of each event for each group.
    """
    slots=int(T/frameSize) # Total number of slots in monitoring window
    arrivals=np.zeros(slots) # Initializing the final arrivals of packets at each time slot
    eventsAll=[]
    TbsAll=[]
    for i,numDevices in enumerate(numDevicesVec):   # Iteratinng over the diferent type of devices
        groupTraffic=uniformTrafficIntensity(T, frameSize, numDevices)   # Generating the base uniform traffic
        # Generating the events and their durations:
        [eventsTime, Tb]=eventGenerator(T, frameSize, eventProbabilities[i])
        eventsAll.append(eventsTime)
        TbsAll.append(Tb)
        # Generating burst traffic for each event
        for k,event in enumerate(eventsTime):
            burstTraffic=burstTrafficIntensity(numDevices, frameSize, Tb[k])
            if (event+int(Tb[k]/frameSize)>=slots): # When event happens at the end of monitoring window
                lastPoint=slots-event
                groupTraffic[event:]=burstTraffic[:lastPoint]
            else:
                # Overwriting the uniform traffic with he burst traffic:
                groupTraffic[event:event+int(Tb[k]/frameSize)]=burstTraffic 
        arrivals+=groupTraffic  # Adding the traffic of the group to the total trafic
    return arrivals,eventsAll,TbsAll

"""Definign the device class"""
class UE():
    def __init__(self):
        self.transmissions = 0 # Number of attemps that a device makes
        self.preamble = 0 # Assigned preamble to the device
        self.backoffCounter = 0 # Backoff timer

"""This fnction gets uniform traffic, burst traffic, and event arrays of all 
type of deveices and add them up and assign a preamble to each of them. It also
generates the congestion patterns as well."""
def actualTrafficPattern(arrivals,frameSize=0.005,backoffBool=True):
    """
    Parameters
    ----------
    arrivals : array
        New packets of whole MTC network at each time slot.
    frameSize : float, optional
        Duration of each time slot in seconds. The default is 0.005.
    backoffBool : bool, optional
        If true, after collision a device waits for some random time and then
        retransmits the collided packet. The default is True.

    Returns
    -------
    successfulUEsPerSlot : array
        Number of successfully transmitted packets at each time slot.
    UEsPerSlot : array
        Total number of devices with a packet to transmit at each time slot.
    congestion : array
        Total number of collided packets at each time slot.
    freePreambles : array
        Total number of unused preamples at each time slot.
    """
    totalPreambles=54          # Total number of available preambles
    backoffWindowMax = 20  # Maximum time that a device waits before retransmission
    preambleTransMax = 10  # Maximum number of retransmissions, then UE discards the packet
    slots = arrivals.shape[0] # Total number of slots
    UEsPerSlot = np.zeros(slots) # Initializing the numbe of active devices in each slot 
    UEs = []   # List of active UEs in each time slots
    successfulUEsPerSlot = np.zeros(slots) # Initializing the total number of successful devices per time slot
    congestion = np.zeros(slots)    # Total number of congested devices per time slot
    freePreambles = np.zeros(slots) # Total number of not chosen preambles per time slot
    for slot , packetsPerSlot in enumerate(arrivals):
        successfulPreambles = np.zeros(totalPreambles) # Preambles chosen successfully without collision
        congestedPreambles = np.zeros(totalPreambles)  # Preambles experiencing collision
        unusedPreambles = np.zeros(totalPreambles)     # Unused preambles
        # Create a dictionary containing the preamble number and UEs using the preamble (initially, 0 UEs).
        preambleCounter = {k:0 for k in range(totalPreambles)} # track how often a preamble is used
        # Create a new device for each packet per slot
        for i in range(int(packetsPerSlot)):
            UEs.append(UE())
        # Assigne a random preamble to each UE with a packet to transmit
        for device in UEs:
            # Only assign preamble when the backoff timer of UE is zero, otherwise,
            # decrease the backoff timer:
            if(device.backoffCounter > 0):
                device.preamble = 99
            else:
                preamble = np.random.randint(0,totalPreambles)
                device.preamble = preamble # Assign a preamble to the device
                preambleCounter[preamble] += 1 # How many users are trying to use this preamble at the same time
                device.transmissions += 1 # Number of attemps that a device makes     
        # Checks for collisions:
        for j, v in enumerate(successfulPreambles):
            j = int(j)
            if(preambleCounter[j] == 1):    # Only one device chosses the preamble: Successful transmission
                successfulPreambles[j] = 1
            if (preambleCounter[j] > 1):    # More than one device chooses the preamble: Collision
                congestedPreambles[j] = 1
            if (preambleCounter[j] == 0):   # No device chooses the preamble
                unusedPreambles[j] = 1
        
        successfulUEsPerSlot[slot] = (sum(successfulPreambles)) # Successful transmissions in each time-slot
        congestion[slot]=(sum(congestedPreambles))  # Collided preambles in each time-slot
        freePreambles[slot]=(sum(unusedPreambles))    # Free preambles in each time-slot
        # Removing the successful UEs from UE list:
        finishedUEs = []
        UEsPerSlotCounter = 0
        for device in UEs:
            if(device.backoffCounter == 0):
                UEsPerSlotCounter += 1 # Counting successful UEs
                if(preambleCounter[device.preamble] == 1): # Removing the successful devices
                    finishedUEs.append(device)
                    # Removing devices that reach to their maximum re-transmisson number:
                elif (device.transmissions == preambleTransMax):
                    finishedUEs.append(device)
                else:       # Assigning backoff timer to unsuccessful devices
                    if(backoffBool):
                        randomBackoff = np.random.randint(0,backoffWindowMax + 1)
                    else:
                        randomBackoff = 0
                    device.backoffCounter = np.ceil(randomBackoff/(frameSize*1000))
            else:  # Reducting the backoff timer of unsuccessful devices
                device.backoffCounter -= 1
        for device in finishedUEs: # Removing the successful devices
            UEs.remove(device)
            del(device)
        UEsPerSlot[slot] = UEsPerSlotCounter # Just all transmitting UEs at each time-slot
    
    return successfulUEsPerSlot, UEsPerSlot, congestion, freePreambles

"""Main function to generate the mMTC traffic"""
if __name__=="__main__":
    numDevicesVec=[15000,8000,3000,2000,2000,15000,8000,3000,2000,2000] # Size of MTC groups
    eventProbabilities=[0.006,0.009,0.09,0.1,0.2,0.004,0.004,0.05,0.1,0.2] # Events per second
    totalStreams=500 # Total number of streams
    T=900  # Total observing window size in seconds
    frameSize=0.005 # Size of each time slot based on 3GPP TR 37.868.
    Intensity=[] # A list to store the expected traffic (arrivals), events indices,
                 # and duration of bursty event.
    Pattern=[]   # A list to store the actual traffic of mMTC network, total 
                 # devices with a packet to transmit, and unused preambles.
    starttime=datetime.now()
    for i in tqdm(range(totalStreams),desc='Generating streams',\
                      position=0,colour='red'):
        starttime=datetime.now()
        arrivals,eventsAll,TbsAll=newArivals(numDevicesVec, eventProbabilities,\
                                             T, frameSize)
        # Smoothing the traffic pattern to increase the prediction accuracy:
        traffic=signal.savgol_filter(arrivals,97,2)
        detected, attempted,congestion,freeP=actualTrafficPattern(arrivals)
        Intensity.append([arrivals,eventsAll,TbsAll])
        Pattern.append([detected, attempted,congestion,freeP])
    # Saving the results in a file while preventing the overwriting issue.
    Description='Generating data for 900 second long monitoring window to create 500 sequences.'
    TT=datetime.now()
    directory='../Traffic'
    # Check if the target director exists:
    if (not os.path.isdir(directory)):
        os.mkdir(directory)
    address1=directory+f'/{TT.strftime("%j")}_{TT.strftime("%a")}_'+\
        f'{TT.strftime("%b")}{TT.strftime("%d")}_{TT.strftime("%H")}{TT.strftime("%M")}'+\
        f'_Samples({totalStreams})({T}Sec)'
    fileType='.pt'
    address=address1+fileType
    # If there is file with the same name, we need to change the address
    count=1
    while (os.path.isfile(address)):
        address=address1+'('+str(count)+')'+fileType
        count+=1
    torch.save({'numDevicesVec':numDevicesVec,'eventProbabilities':eventProbabilities,\
                'T':T,'frameSize':frameSize, 'arrivals':arrivals, 'eventsAll':eventsAll,\
                'TbsAll':TbsAll,'detected':detected, 'attempted':attempted,\
                'congestion':congestion,'freeP':freeP, 'Intensity':Intensity,\
                'Pattern':Pattern},address)
    print('\n\nEnd time: ',datetime.now())
    print('Total elapsed time: ',datetime.now()-starttime)
