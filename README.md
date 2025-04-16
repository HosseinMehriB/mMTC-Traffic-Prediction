# mMTC-Traffic-Prediction
This project aims to predict the traffic pattern and the probability of bursty traffic in mMTC networks using machine learning (ML) models. A new ML algorithm, FLSP (Fast Live Stream Prediction), is proposed for traffic prediction in live scenarios where fresh data is continuously collected from a live network and efficiently leveraged by ML models to make continuous and accurate predictions.

Two types of ML models are used in this project for traffic prediction:

1. **RNN-based models (LSTM and GRU):** Both the traditional Rolling algorithm and the proposed FLSP algorithm are applied to these models. The flowchart below shows the simulation steps for this category:
```
--------------------------      ---------------      ------------------------------------
|GeneratingTrafficPatterns| --> |RnnNetTraining| --> |GeneratingBurstDetNetFeedDataForRNN| -->
--------------------------      ----------------      ------------------------------------
     -------------------------------      ---------------------------------------------
--> |BurstDetNetTrainingWithRNNData| --> |EvaluatingNetworksAndGeneratingResultsForRNN| -->
    -------------------------------      ---------------------------------------------
     ------------------------------------------
--> |PlottingResultsAndExportingToMatlabForRNN|
    ------------------------------------------
```
   
2. **CNN-1D-based model:** Only the traditional Rolling algorithm is applied to this model, and it is used for comparison. The flowchart below shows the simulation steps for this category:
```
--------------------------      ------------------      ----------------------------------
|GeneratingTrafficPatterns| --> |CNN-1DNetTraining| --> |BurstDetNetTrainingWithCNN-1DData| -->
--------------------------      ^^^^^^^^^^^^^^^^^^      ----------------------------------
     --------------------------------------
--> |EvaluationAndPlottingResultsForCNN-1D|
    --------------------------------------
```

## Citation:
**If you find this code useful in your research, please consider citing our paper:** 

H. Mehri, H. Mehrpouyan, and H. Chen, "RACH Traffic Prediction in Massive Machine Type Communications," IEEE Transactions on Machine Learning in Communications and Networking, vol. 3, pp. 315–331, 2025.

[IEEE Xplore Link](https://ieeexplore.ieee.org/document/10891603) | DOI: <ins> 10.1109/TMLCN.2025.3542760 </ins>

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
