# In-Silico Target Prediction by Ensemble Chemogenomic Model based on Multi-Scale Information of Chemical Structures and Protein Sequences

![7777](https://user-images.githubusercontent.com/106001963/169690953-8c947c7b-fe2a-42d0-8560-6585b9c439ad.png)

## Overview
Identification and validation of bioactive small-molecule targets is a significant challenge in drug discovery. In recent years, various in-silico approaches have been proposed to expedite time- and resource-consuming experiments for target detection. Herein, we developed several chemogenomic models for target prediction based on multi-scale information of chemical structures and protein sequences. By combining the information of a compound with multiple protein targets together and putting these compound-target pairs into a well-established model, the scores to indicate whether there are interactions between compounds and targets can be derived, and thus a target prediction task can be completed by sorting the outputted scores. To improve the prediction performance, we constructed several chemogenomic models using multi-scale information of chemical structures and protein sequences, and the ensemble model with the best performance was used as our final model. The model was validated by various strategies and external datasets and the promising target prediction capability of the model, i.e., the fraction of known targets identified in the top-k (1 to 10) list of the potential target candidates suggested by the model, was confirmed. Compared with multiple state-of-art target prediction methods, our model showed equivalent or better predictive ability in terms of the top-k predictions. It is expected that our method can be utilized as a powerful computational tool to narrow down the potential targets for experimental testing.

## Data Resource
**Link:** https://pan.baidu.com/s/1mbVLe4qgNG4k4mW1qLMQJw    

**Code:** 0522

## Implementation
### if you want to predict targets for several compounds, there are three steps for you:   

**step 1:** Train a ensemble chemogenomic model using the available data I uploaded, using the Python script "1_Model_Training.py";   
**step 2:** Calulate descriptors for these compounds for which you wish to predict targets, using the Python scripts "2_cal_descriptors_###";   
**step 3:** using the Python script "3_Target_Prediction"; 
 
 
**Or using the KNIME workfolw "target prediction.knwf"**
  

## Contact
If you have questions or suggestions, please contact: sorchayang@163.com, and oriental-cds@163.com.
