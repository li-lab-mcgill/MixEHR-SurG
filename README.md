# MixEHR-SurG: a joint proportional hazard and guided topic model for inferring mortality-associated topics from electronic health records

MixEHR-SurG is a tool designed to analyze Electronic Health Records (EHR) using topic modeling techniques combined with survival analysis. Specifically, MixEHR-SurG integrates the Cox proportional hazards model to predict patient mortality while enhancing topic interpretability. It achieves this by guiding topic inference based on patient-specific PheCodes, resulting in phenotype topics closely associated with mortality risks. The tool has shown effectiveness in identifying high-risk phenotype topics related to severe cardiac conditions and critical neurological injuries, offering valuable insights for epidemiological studies and healthcare research. MixEHR-SurG consists of four main steps. The training process is highlighted in green, and the prediction process is depicted in purple. In Step 1, we prepossess and aggregate raw EHR data for each patient $j$.  Step 2 involves determining a $K$-dimensional phenotype topic prior, $\boldsymbol{\uppi}_j = (\pi_{j1}, \ldots, \pi_{jK})$, for each patient. Step 3 infers phenotype topic distribution $\boldsymbol{\upphi}_k^{(m)} \in \mathbb{R}^{V^{(m)}}$ for EHR type $m$ in topic $k$ (i.e., the model parameters of \model). This requires inferring the latent topic assignment $z_{ji}\in\{1,\ldots,K\}$ for each EHR token $i$ in patient $j$. In Step 4, the trained model is applied to predict personalized survival function for new patient. 

The proababilistic graphical model of MixEHR-S is shown:


<img src="">



# Relevant Publications

This code is referenced from following paper:

>

# Dataset

MixEHR-SurG was evaluated using a simulated dataset and two real-world EHR datasets: the Quebec Congenital Heart Disease (CHD) dataset, featuring 154,775 subjects with 46,812,368 outpatient claim ICD codes, and the Medical Information Mart for Intensive Care (MIMIC-III), which includes 38,597 subjects with 53,423 multi-modal EHR records such as clinical notes, lab data, prescriptions, drug data, and ICD codes. Due to confidentiality and access requirements, the Quebec CHD dataset is not publicly available, and the MIMIC-III dataset requires specific certification for use. However, we have made our simulated data available in the repository under the path "MixEHR-SurG/data/"

# Code Description

## STEP 1: Process Dataset

The input data file need to be processed into built-in data structure "Corpus". You can use "MixEHR-SurG/code/corpus_Sur.py" for survival supervised model (MixEHR-SurG, MixEHR-Surv) and "MixEHR-SurG/code/corpus.py" for unsupervised model (MixEHR, MixEHR-G) to process dataset and generate a suitable data structure for MixEHR models.

Place dataset to specific path "MixEHR-SurG/data/". You can run following code:

    run(parser.parse_args(['process', '-n', './data/', './store/']))
    
you also need to split the dataset into train/validation/test subset. The data path and detailed split ratio could be edited:
    
    run(parser.parse_args(['split', 'store/', 'store/']))

## STEP 2: Topic Modelling

After process dataset and obtained trainable data, you can run "MixEHR-SurG/code/main.py" to perform topic modelling for each dataset. 
Hyperparameters of training stage include number of latent topics,  training epoches, and parameters related to stochastic learning. We have implemented all the 4 MixEHR-models here. You need first to choose the model you need by importing different file here:

    # MixEHR model:
    from MixEHR import MixEHR
    # MixEHR-G model:
    from MixEHR_Guided import MixEHR
    # MixEHR-Surv model:
    from MixEHR_Surv import MixEHR
    # MixEHR-SurG model:
    from MixEHR_SurG import MixEHR
For unguided models (MixEHR, MixEHR-Surv), you also need to specify the number of topics K=? at the top of the "MixEHR-SurG/code/main.py"

The first argument should be train. Th topics number, data path and epoches could choose adequate value. The execution code is:

    run(parser.parse_args(['train','./store/', './result/']))
    

## STEP 3: Label Prediction

With the saved models stored in training stage, you can used these models to obtain new patient's topic proportion and further more to predict hazard risk (only for MixEHR-SurG and MixEHR-Surv) in "MixEHR-SurG/code/main.py" by the following code
The number of latent topics should be same with the number of saved model (training stage). 
The test set should be used in label prediction task. The execution code is:

    run(parser.parse_args(['predict','./store/', './result/']))
    

## STEP 5: Prepare Your Own Dataset

The required dataset may not be EHR ICD data. Any dataset includes words and diverse data types could be considered. 
For example, ACT code and drug code in EHR data can only be organized into trainable dataset. You 

Your prepared data should have three files: metadata, text data and survival data.
- metadata.txt: the document specify the different type of text data.

                            Headers:
                            index,path,word_colum 

- data.txt: the document (patient) data with ID, word (ICD code), data type (specialist), frequency.

                            Headers:
                            id, word, data_type, frequency 


- patinet_label.csv: for each document ID, whether this document has a corresponding response. In this paper, the response label is disease label and drug useage. It could be death flag and others. 

                            Headers:
                            nam,Surv_time,Censor


We also need to provide the prior matrix for the guided model in prior.txt

                            






