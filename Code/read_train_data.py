from distutils.util import subst_vars
import os
import re
import time

import h5py
import pandas as pd
import numpy as np
from scipy.special import digamma, gammaln
from scipy.stats import norm
from sklearn.metrics import roc_curve, average_precision_score
import pickle
from corpus_Surv import Corpus
from metrics import auc

train_dir = "/Users/yixuanli/Desktop/Now Dataset/train"
c_test = Corpus.read_corpus_from_directory(train_dir)
Survival_times_test_true = np.array([p[0].Survival_time for p in c_test])
test_patients = np.array([p[0].patient_id for p in c_test])
test_patients_df = pd.DataFrame(test_patients)
test_patients_df.to_csv("/Users/yixuanli/Desktop/train.csv",index=False)