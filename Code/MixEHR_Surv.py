import time
import math
from tkinter.tix import ExFileSelectBox
import numpy as np
from corpus import Corpus
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.special import gammaln
from torch import lgamma, digamma
from utils import logsumexp
import pickle
import os

# Package for Cox PH Model
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mini_val = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MixEHR(nn.Module):
    def __init__(self, K, corpus, modality_list, stochastic_VI=True,  batch_size=1000, out='./store/'):
        """
        Arguments:
            corpus: document class.
            words_topic_matrix: V x K matrix, each element represents the existence of  word w for topic k.
            batch_size: batch size for a minibatch
            out: output path
        """
        super(MixEHR, self).__init__()
        self.modalities = modality_list # name of modalites
        self.modaltiy_num = len(modality_list) # number of modaltiy M
        self.out = out  # folder to save experiments

        # Model parameters
        self.stochastic_VI = stochastic_VI
        self.full_batch_generator = Corpus.generator_full_batch(corpus)
        self.batch_size = batch_size # document number in a mini batch
        self.mini_batch_generator = Corpus.generator_mini_batch(corpus, self.batch_size) # default batch size 1000

        # Corpus parameters
        self.C = corpus.C  # C is number of words in the corpus, use for updating gamma for SCVB0
        self.D = corpus.D # document number in full batch
        self.K = K
        self.V = corpus.V  # vocabulary size of words

        # hyper-parameters
        self.alpha = torch.distributions.Gamma(torch.tensor([10.0]), torch.tensor([1.0])).sample([self.K]).T[0] # hyperparameter for prior on topic weight vectors theta
        self.beta = [torch.distributions.Gamma(torch.tensor([2.0]), torch.tensor([500])).sample([self.K,self.V[m]]).T[0]
                        for m in range(self.modaltiy_num)]  # hyperparameter for prior on code vectors beta
        self.alpha_sum = torch.sum(self.alpha)  # scalar value
        self.beta_sum = [torch.sum(beta, axis=0) for beta in self.beta]  # sum over w, K dimensional

        # variational variables
        self.exp_z_avg = torch.zeros(self.D, self.K, dtype=torch.double, requires_grad=False, device=device)
        self.exp_q_z = 0

        # expected tokens
        self.exp_m = torch.rand((self.D, self.K), dtype=torch.double, requires_grad=False, device=device) # suppose a general m_dk across different modalities
        self.exp_n = [torch.rand((self.V[m], self.K), dtype=torch.double, requires_grad=False, device=device)
                      for m in range(self.modaltiy_num)] # exp_n for differnt modality
        # normalization 
        for d in range(self.D):
            self.exp_m[d] /= torch.sum(self.exp_m[d])
        for m in range(self.modaltiy_num):
            for v in range(self.V[m]):
                self.exp_n[m][v] /= torch.sum(self.exp_n[m][v])
        self.exp_m_sum = torch.sum(self.exp_m, axis=1) 
        self.exp_n_sum = [torch.sum(exp_n, axis=0) for exp_n in self.exp_n] # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector for each modality

        # survival variables
        self.Survival_times = np.random.normal(size=self.D)
        self.Censor_indicators = np.ones(self.D)
        self.w_ = torch.tensor(np.random.rand(self.K)) # survival coefficient for cox regression

        # Model parameters 
        self.hyperparameters = ['alpha','beta','w_']

        self.elbo_print = []

    def get_surv_inf(self, docs, indices):
        '''
        get the survinf (Survival_times and Censor_indicator)from doc
        '''
        self.Survival_times = np.random.normal(size=self.D)
        self.Censor_indicators = np.ones(self.D)
        for d_i, (doc_id, doc) in enumerate(zip(indices, docs)):
            self.Survival_times[doc_id] = doc.Survival_time
            self.Censor_indicators[doc_id] = doc.Censor_indicator


    def get_loss(self, batch_indices, batch_C, minibatch, epoch, start_time, m, test = False):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        ELBO = 0
        # E_q[ log p(z | alpha)
        ELBO += lgamma(torch.sum(self.alpha)) - torch.sum(lgamma(self.alpha))
        ELBO += torch.sum(torch.sum(lgamma(self.alpha + self.exp_m[batch_indices]), axis=1) - \
                lgamma(torch.sum(self.alpha_sum + self.exp_m_sum[batch_indices])))
        # E_q[log p(x | z, beta)]
        ELBO += (self.K) * (lgamma(torch.sum(self.beta[m])) - torch.sum(lgamma(self.beta[m])))
        log_sum_terms = 0
        for k in range(self.K):
            log_sum_terms += torch.sum(lgamma(self.beta[m][:,k] + self.exp_n[m][:,k])) - \
                             lgamma(torch.sum(self.beta_sum[m][k] + self.exp_n_sum[m][k]))   
        ELBO += log_sum_terms
        # E_q[log p(time,delta|z,w)]
        ELBO_surv = 0
        if (test == False) and (self.modalities[m] == 'icd'):
            for d in range(self.D):
                ELBO_surv += torch.tensor(self.Censor_indicators[d]) * torch.dot(self.w_,self.exp_m[d])
                ELBO_surv -= torch.tensor(self.H0(self.Survival_times[d]))*torch.exp(torch.dot(self.w_,self.exp_m[d]))
        ELBO += ELBO_surv
        # - E_q[log q(z | gamma)]
        ELBO -= self.exp_q_z
        self.exp_q_z = 0
        print("took %s seconds for minibatch %s" % (time.time() - start_time, minibatch))
        return (ELBO.detach().cpu().numpy().item())   



    def SCVB0(self, batch_BOW, batch_indices, batch_C, batch_cnt_m, iter_n, m):
        temp_exp_n = torch.zeros(self.V[m], self.K, dtype=torch.double, device=device)
        temp_exp_m_batch = torch.zeros(batch_BOW.shape[0], self.K, dtype=torch.double, device=device)
        # M step

        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma = torch.zeros(self.V[m], self.K, dtype=torch.double, device=device) #  V x K
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            temp_gamma_avg = torch.zeros(self.K, dtype=torch.double, device=device)
            temp_gamma_avg = torch.sum(self.gamma[m][doc_id][BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            temp_gamma_avg_sum = torch.sum(temp_gamma_avg)
            temp_gamma_avg /= temp_gamma_avg_sum + mini_val
            temp_gamma[BOW_nonzero] = (self.alpha + self.exp_m[doc_id]) * ((self.beta[m][BOW_nonzero] + self.exp_n[m][BOW_nonzero])) \
                                      / (self.beta_sum[m] + self.exp_n_sum[m])                          
            temp_gamma[BOW_nonzero] *= torch.exp(self.w_ / batch_cnt_m[d_i])
            temp_gamma[BOW_nonzero] /= torch.exp(torch.tensor(self.H0(self.Survival_times[doc_id])) * torch.exp(torch.dot(self.w_,temp_gamma_avg)+ (self.w_ / batch_cnt_m[d_i])))
            # normalization
            temp_gamma_sum = temp_gamma.sum(dim=1).unsqueeze(1)
            temp_gamma /= temp_gamma_sum + mini_val
            # calculate sufficient statistics
            temp_exp_n += temp_gamma * batch_BOW[d_i].unsqueeze(1)
            if (self.modalities[m] == 'icd'):
                temp_exp_m_batch[d_i] += torch.sum(temp_gamma[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            self.exp_q_z += torch.sum(temp_gamma[BOW_nonzero] * torch.log(temp_gamma[BOW_nonzero] + mini_val))
            self.gamma[m][doc_id] = temp_gamma

        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 5), 0.9)
        if self.stochastic_VI:
            if (self.modalities[m] == 'icd'):
                self.exp_m[batch_indices] = (1 - rho) * self.exp_m[batch_indices] + rho * temp_exp_m_batch
                self.exp_m_sum = torch.sum(self.exp_m, dim=1) 
            self.exp_n[m] = (1-rho)*self.exp_n[m] + rho*temp_exp_n*self.C[m]/batch_C
            self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0) # sum over w, exp_n is [V K] dimensionality
        else:
            if (self.modalities[m] == 'icd'):
                self.exp_m[batch_indices] = temp_exp_m_batch
                self.exp_m_sum = torch.sum(self.exp_m, dim=1) 
            self.exp_n[m] = temp_exp_n
            self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0) # sum over w, exp_n is [V K] dimensionality

        # Update Survival parameters
        if (self.modalities[m] == 'icd'):
            tempsurv_exp_m = self.exp_m.clone().detach().cpu().numpy() 
            for d in range(self.D):
                tempsurv_exp_m[d] /= (np.sum(tempsurv_exp_m[d]) + mini_val)
            self.cph_M= CoxPHSurvivalAnalysis()
            self.cph_M.set_params(alpha=0.1)
            self.cph_M.fit(tempsurv_exp_m, self.df_sur)
            self.H0 = self.cph_M.cum_baseline_hazard_
            self.w_ = torch.tensor(self.cph_M.coef_)

        # Update hyper parameters
        self.update_hyperparams(m)

    def SCVB0_un(self, batch_BOW, batch_indices, batch_C, iter_n, m):
        temp_exp_n = torch.zeros(self.V[m], self.K, dtype=torch.double, device=device)
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma = torch.zeros(self.V[m], self.K, dtype=torch.double, device=device) #  V x K
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            # regular word must be regular topic
            temp_gamma[BOW_nonzero] = (self.alpha + self.exp_m[doc_id]) * ((self.beta[m][BOW_nonzero] + self.exp_n[m][BOW_nonzero])) \
                                      / (self.beta_sum[m] + self.exp_n_sum[m])
            # normalization
            temp_gamma_sum = temp_gamma.sum(dim=1).unsqueeze(1)
            temp_gamma /= temp_gamma_sum + mini_val
            # calculate sufficient statistics
            temp_exp_n += temp_gamma * batch_BOW[d_i].unsqueeze(1)
        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 5), 0.9)
        if self.stochastic_VI:
            self.exp_n[m] = (1-rho)*self.exp_n[m] + rho*temp_exp_n*self.C[m]/batch_C
            self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0) # sum over w, exp_n is [V K] dimensionality
        else:
            self.exp_n[m] = temp_exp_n
            self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0) # sum over w, exp_n is [V K] dimensionality
        self.update_hyperparams(m)


    def update_hyperparams(self, m):
        '''
        update hyperparameters pi using Bernoulli trial
        '''
        if (self.modalities[m] == 'icd'):
            alpha_term = torch.zeros(self.K, dtype=torch.double, requires_grad=False, device=device)
            alpha_sum_term = torch.sum(alpha_term)
            for j in range(self.D):
                alpha_term += digamma(self.alpha + self.exp_m[j]) - digamma(self.alpha)
                alpha_sum_term += digamma(self.alpha_sum + self.exp_m_sum[j]) - digamma(self.alpha_sum)
            self.alpha = (1-1 + self.alpha * alpha_term) / (10 + alpha_sum_term)
            self.alpha_sum = torch.sum(self.alpha) 
        beta_term = torch.zeros(self.V[m], self.K, dtype=torch.double, requires_grad=False, device=device)
        beta_sum_term = torch.sum(beta_term, axis=0)
        for k in  range(self.K):
            beta_term[:,k] += digamma(self.beta[m][:,k] + self.exp_n[m][:,k]) - digamma(self.beta[m][:,k])
            beta_sum_term[k] += digamma(self.beta_sum[m][k] + self.exp_n_sum[m][k]) - digamma(self.beta_sum[m][k])
        self.beta[m] = (2-1 + self.beta[m] * beta_term) / (500 + beta_sum_term)
        self.beta_sum[m] = torch.sum(self.beta[m], axis=0)



    def inference(self, max_epoch=500, save_every=50):
        '''
        inference algorithm for topic model, apply stochastic collaposed variational inference for latent variable z,
        and apply stochastic gradient descent for dynamic variables \eta (\alpha)
        '''
        epoch_print = [max_epoch,0]
        for batch, d in enumerate(self.full_batch_generator):
            batch_docs, batch_indices, batch_C = d
            self.gamma = [{doc_id: torch.rand((self.V[m], self.K), dtype=torch.double, requires_grad=False, device=device) for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs))}
                      for m in range(self.modaltiy_num)]
            self.get_surv_inf(batch_docs,batch_indices)
            self.df_sur = np.zeros(self.D, dtype=[('censor','bool'),('time','f8')])
            self.df_sur['censor'] = self.Censor_indicators
            self.df_sur['time'] = self.Survival_times
            x_init = np.random.normal(size=(self.D,self.K))
            cph_init= CoxPHSurvivalAnalysis()
            cph_init.set_params(alpha=0.1)
            cph_init.fit(x_init, self.df_sur)
            self.H0 = cph_init.cum_baseline_hazard_

        for epoch in range(0, max_epoch):
            start_time = time.time()

            print("Training for epoch", epoch)
            if self.stochastic_VI:
                batch_n = (self.D // self.batch_size) + 1
                for minibatch, d in enumerate(self.mini_batch_generator):  # For each epoach, we sample a series of mini_batch data once
                    print("Running for minibatch", minibatch)
                    elbo_batch = 0
                    start_time = time.time()
                    batch_docs, batch_indices, batch_C = d  # batch_C is total number of ICD codes (only) in a minibatch for SCVB0
                    for m in range(self.modaltiy_num):
                        # modaltiy specific BOW matrix, shape is D X V[m]
                        batch_BOW_m = torch.zeros(len(batch_docs), self.V[m], dtype=torch.int, requires_grad=False,
                                                  device=device)  # document number (not M) x V
                        batch_C_m = sum([doc_C[m] for doc_C in batch_C])
                        for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                            for word_id, freq in doc.words_dict[m].items():
                                batch_BOW_m[d_i, word_id] = freq
                        self.SCVB0(batch_BOW_m, batch_indices, batch_C_m, epoch * batch_n + minibatch,m)
                        elbo_batch  += self.get_loss(batch_indices, batch_C_m, minibatch, epoch, start_time, m)
                    elbo = elbo_batch
                    epoch_print.append(elbo)

            else:
                for batch, d in enumerate(self.full_batch_generator):  # For each epoch, use full batch of data
                    print("Running for fullbatch")
                    elbo_batch = 0
                    batch_docs, batch_indices, batch_C = d  # batch_C is total number of ICD codes (only) in a minibatch for SCVB0
                    for m in range(self.modaltiy_num):
                        # modaltiy specific BOW matrix, shape is D X V[m]
                        batch_BOW_m = torch.zeros(len(batch_docs), self.V[m], dtype=torch.int, requires_grad=False,
                                                  device=device)  # document number (not M) x V
                        batch_C_m = sum([doc_C[m] for doc_C in batch_C])
                        batch_cnt_m = torch.zeros(len(batch_docs),dtype=torch.int, requires_grad=False, device=device)
                        for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                            for word_id, freq in doc.words_dict[m].items():
                                batch_BOW_m[d_i, word_id] = freq
                            batch_cnt_m[d_i] = doc.Cd[m]
                        if (self.modalities[m] == 'icd'):
                            self.SCVB0(batch_BOW_m, batch_indices, batch_C_m, batch_cnt_m, batch, m)
                        else:
                            self.SCVB0_un(batch_BOW_m, batch_indices, batch_C_m, batch, m)
                        elbo_batch += self.get_loss(batch_indices, batch_C_m, batch, epoch, start_time, m)
                    elbo = elbo_batch
                    epoch_print.append(elbo) 
            if (epoch+1) % save_every == 0:
                self.save_parameters(epoch)
                pickle.dump(epoch_print, open(os.path.join("./parameters/", 'elbo_training_%s.pkl' % (epoch+1)), 'wb'))
            self.exp_q_z = 0  # update to zero for next minibatch
            print("%s elbo %s diff %s "%(epoch , epoch_print[-1], np.abs(epoch_print[-1] - epoch_print[-2])))
        test_dir = "./store/test"
        c_test = Corpus.read_corpus_from_directory(test_dir)
        self.load_parameters()
        self.predict(c_test)
        return elbo

    def save_parameters(self, epoch):
        torch.save(self.exp_m, "./parameters/exp_m_%s.pt" % (epoch+1))
        for i, modality in enumerate(self.modalities):
            torch.save(self.exp_n[i], "./parameters/exp_n_%s_%s.pt" % (modality, epoch+1))
        torch.save(self.alpha, "./parameters/alpha_%s.pt" % (epoch+1))
        torch.save(self.w_ , "./parameters/w_%s.pt" % (epoch+1))
        for i, modality in enumerate(self.modalities):
            torch.save(self.beta[i], "./parameters/beta_%s_%s.pt" % (modality, epoch+1))



    def load_parameters(self, epoch=200):
        self.exp_m = torch.load("./parameters/exp_m_%s.pt" % (epoch))
        for i, modality in enumerate(self.modalities):
            self.exp_n[i] = torch.load("./parameters/exp_n_%s_%s.pt" % (modality, epoch))
        # normalization 
        for d in range(self.D):
            self.exp_m[d] /= (torch.sum(self.exp_m[d]) + mini_val)
        for m in range(self.modaltiy_num):
            for v in range(self.V[m]):
                self.exp_n[m][v] /= (torch.sum(self.exp_n[m][v]) + mini_val)
        self.exp_m_sum = torch.sum(self.exp_m, axis=1) 
        self.exp_n_sum = [torch.sum(exp_n, axis=0) for exp_n in self.exp_n]
        self.alpha = torch.load("./parameters/alpha_%s.pt" % (epoch))
        for i, modality in enumerate(self.modalities):
            self.beta[i] = torch.load("./parameters/beta_%s_%s.pt" % (modality, epoch))
        self.alpha_sum = torch.sum(self.alpha)
        self.beta_sum = [torch.sum(beta, axis=0) for beta in self.beta]


    def predict(self, corpus):
        self.D = corpus.D
        self.C = corpus.C 
        self.full_batch_generator = Corpus.generator_full_batch(corpus)
        self.exp_z_avg = torch.zeros(self.D, self.K, dtype=torch.double, requires_grad=False, device=device)
        self.exp_q_z = 0
        self.exp_m = torch.rand((self.D, self.K), dtype=torch.double, requires_grad=False, device=device) 
        for d in range(self.D):
            self.exp_m[d] /= torch.sum(self.exp_m[d])
        self.exp_m_sum = torch.sum(self.exp_m, axis=1) 

        # Start calculating the topic assignments for test patients
        for batch, d in enumerate(self.full_batch_generator):  # For each epoch, use full batch of data
            print("Running for fullbatch")
            batch_docs, batch_indices, batch_C = d  # batch_C is total number of ICD codes (only) in a minibatch for SCVB0
            self.get_surv_inf(batch_docs,batch_indices)
            self.df_sur = np.zeros(self.D, dtype=[('censor','bool'),('time','f8')])
            self.df_sur['censor'] = self.Censor_indicators
            self.df_sur['time'] = self.Survival_times
            for m in range(self.modaltiy_num):
            # modaltiy specific BOW matrix, shape is D X V[m]
                batch_BOW_m = torch.zeros(len(batch_docs), self.V[m], dtype=torch.int, requires_grad=False,
                                                  device=device)  # document number (not M) x V
                batch_C_m = sum([doc_C[m] for doc_C in batch_C])
                batch_cnt_m = torch.zeros(len(batch_docs),dtype=torch.int, requires_grad=False, device=device)
                for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                    for word_id, freq in doc.words_dict[m].items():
                        batch_BOW_m[d_i, word_id] = freq
                    batch_cnt_m[d_i] = doc.Cd[m]
                self.SCVB0_test(batch_BOW_m, batch_indices, batch_C_m, batch_cnt_m, batch, m)
            tempsurv_exp_m = self.exp_m.clone().detach().cpu().numpy() 
            for d in range(self.D):
                tempsurv_exp_m[d] /= (np.sum(tempsurv_exp_m[d]) + mini_val)
            Survival_functions = self.cph_M.predict_survival_function(tempsurv_exp_m)
            Hazard_functions = self.cph_M.predict_cumulative_hazard_function(tempsurv_exp_m)
            c_index = self.cph_M.score(tempsurv_exp_m, self.df_sur)
            print("Cindex for test patients: ",c_index)
            with open('./parameters/test/survival_functions.pkl', 'wb') as f:
                pickle.dump(Survival_functions, f)
            with open('./parameters/test/hazard_functions.pkl', 'wb') as f:
                pickle.dump(Hazard_functions, f)
            torch.save(self.exp_m, "./parameters/test/exp_m_testing.pt")
            torch.save(c_index, "./parameters/test/c_index.pt")
            print("Test patients topic saved")

    
    def SCVB0_test(self, batch_BOW, batch_indices, batch_C, batch_cnt_m, iter_n, m):
        temp_exp_m_batch = torch.zeros(batch_BOW.shape[0], self.K, dtype=torch.double, device=device)
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma = torch.zeros(self.V[m], self.K, dtype=torch.double, device=device) #  V x K
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            temp_gamma[BOW_nonzero] = (self.alpha + self.exp_m[doc_id]) * ((self.beta[m][BOW_nonzero] + self.exp_n[m][BOW_nonzero])) \
                                      / (self.beta_sum[m] + self.exp_n_sum[m])
            # normalization
            temp_gamma_sum = temp_gamma.sum(dim=1).unsqueeze(1)
            temp_gamma /= temp_gamma_sum + mini_val
            # calculate sufficient statistics
            temp_exp_m_batch[d_i] += torch.sum(temp_gamma[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            self.exp_q_z += torch.sum(temp_gamma[BOW_nonzero] * torch.log(temp_gamma[BOW_nonzero] + mini_val))
            self.exp_m[batch_indices] = temp_exp_m_batch
            self.exp_m_sum = torch.sum(self.exp_m, dim=1) 

