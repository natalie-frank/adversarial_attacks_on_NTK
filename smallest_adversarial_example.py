#methods for the smallest_adversarial_example experiments

import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle
import numpy.linalg as la
import ntk_methods
import math
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import torch
import sklearn.preprocessing as skp



#given a neural net and an attack method with function signature attack(x,label,epsilon,model), finds the smallest epsilon for which attack(x,label,epsilon,model) is misclassified by the neural net using a binary search
#returns the smallest epsilon for which one can find an adversarial example along with this adversarial example

#x: data point at which we want to find the smallest adversarial perturbation
#label: true label of the data point
#attack: an attack with function signature attack(x,label,epsilon,model) which returns an adversarial example
#upper_bound_guess: an upper bound on the size of the smallest adversarial perturbation 
#strict: if True, upper_bound_guess is treated as a strict upper bound on the size of the smallest adversarial perturbation, otherwise it's assumed to be just a guess (setting this variable as true produces a significant speedup)
#error: stopping condition for the binary search-- the binary search terminates when upper_bound - lower bound <error
#fail_value: this value is returned if an adversarial example is not found
def smallest_adversarial_perturbation(x, label,model, attack, upper_bound_guess=1,strict=False,error=10**-4,fail_value=None):
    pred=model(x)
    _, predicted = pred.max(1)
    if not predicted.eq(label).item():
        adversarial_radius=0
        example=x
    else:
        mx_float=sys.float_info.max
        correct=True#if the upper bound is correct
        under_mx_float=True
        while not strict and under_mx_float and correct:
            if upper_bound_guess>=mx_float:
                under_mx_float=False
            upper_bound_guess=min(upper_bound_guess,mx_float)
            try:
                adversarial_example=attack(x, label, upper_bound_guess,model)
                pred=model(adversarial_example)
                _, predicted = pred.max(1)
                correct=predicted.eq(label).item()
                upper_bound_guess*=2
            except:
                under_mx_float=False
                ub=fail_value
                adversarial_example=[]
                break
        if under_mx_float or strict:
            ub=upper_bound_guess #Upper bound
            lb=0 #lower bound
            while ub-lb>error:
                eps_mid=(ub+lb)/2
                adversarial_example=attack(x, label, eps_mid,model)
                pred=model(adversarial_example)
                _, predicted = pred.max(1)
                correct=predicted.eq(label).item()
                if correct:
                    lb=eps_mid
                else:
                    ub=eps_mid

                
            adversarial_radius=ub
            example=adversarial_example

        else:
            adversarial_radius=fail_value
            example=None

    return adversarial_radius, example



#given a nested list values_list for which values_list[i][j] are scalars, plots a histogram of the aggregate of all the scalars in the nested list, with error bars given by computing the standard deviation of the histograms of values_list[i]


#values_list: A list of numpy arrays
#bins: for specifying the binds of the histogram. If None, determined automatically
#save_file: path at which to save the final plot
#error_bars: whether to plot error bars on the histogram
#x_label: label on x-axis
#fail_value: exclude these instances from the histogram
#rng: range of histogram
#title: title of histogram
def attribute_histogram(values_list,bins=None,save_file=None, error_bars=True,x_label='x',fail_value=None,rng=None,title=None):
    all_values=[v  for values in values_list for v in values if v is not fail_value]
    all_values=np.array(all_values)
    weights=np.ones_like(all_values)*1/len(all_values)
    if bins is None:
        y, binEdges=np.histogram(all_values,weights=weights,range=rng)    
    else:
        y, binEdges=np.histogram(all_values,bins=bins,weights=weights,range=rng)
    hist_vals=[]
    for values in values_list:
        rm_values=[v for v in values if v is not fail_value]
        #hst, edge_vals=np.histogram(rm_values,bins=binEdges,range=rng)
        if bins is None:
            hst, edge_vals=np.histogram(rm_values,range=rng)
        else:
            hst, edge_vals=np.histogram(rm_values,bins=bins,range=rng)
        hist_vals.append(hst)
    hist_vals=np.array(hist_vals)
    stds=np.std(hist_vals,axis=0)/len(all_values)


    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    width      = 0.05
    #hist_plot=plt.hist(all_values,bins=bins,weights=weights,range=rng)
    if error_bars:
        plt.bar(bincenters, y, width=width, color='b', yerr=stds,capsize=5.0)
    else:
        plt.bar(bincenters, y, width=width, color='b')
    plt.ylabel('Average Frequency')
    plt.xlabel(x_label)
    if title is not None:
        plt.title(title)
    if save_file is None:
        plt.show()
    else:
        plt.savefig(save_file+".png",bbox_inches='tight')
    plt.close()








#returns the cosine similarities between the columns of matrices A and B
#A, B: numpy arrays with the same number of rows
def calculate_cossims(A,B):
    A_normalized=skp.normalize(A,axis=0,norm='l2')
    B_normalized=skp.normalize(B,axis=0,norm='l2')
    signed_cossims=np.matmul(np.transpose(A_normalized),B_normalized)
    cossims=np.abs(signed_cossims)
    return cossims


#computes the cosine similarities between the feature mapping of the training data and feature mapping of the smallest adversarial example starting at the test data

#test_loader: a data loader object at which we want to find the smallest adversarial example
#model: a neural net from which we extract the feature mappings
#feature_mapping: feature mapping of the test set, given by derivatives of the net in the parameters
#attack: an attack with function signature attack(x,label,epsilon,model) which returns an adversarial example
#upper_bound_guess: an upper bound on the size of the smallest adversarial perturbation 
#strict: if True, upper_bound_guess is treated as a strict upper bound on the size of the smallest adversarial perturbation, otherwise it's assumed to be just a guess (setting this variable as true produces a significant speedup)
#error: stopping condition for the binary search-- the binary search terminates when upper_bound - lower bound <error
#fail_value: this value is returned if an adversarial example is not found
def cosine_similarities_at_smallest_perturbation(test_loader,model, feature_mapping, attack,upper_bound_guess=1,strict=False,error=10**-4,fail_value=None,save=None):
    xs_correct=[]
    xs_incorrect=[]
    for (x,label) in test_loader:
        radius,adversarial_example=smallest_adversarial_perturbation(x, label,model, attack, upper_bound_guess=upper_bound_guess,strict=strict,error=error,fail_value=fail_value)
        adversarial_ys=model(adversarial_example)
        _, predicted = adversarial_ys.max(1)
        correct_prediction=predicted.eq(label).item()
        if correct_prediction:
            xs_correct.append((adversarial_example,label))
        else:
            xs_incorrect.append((adversarial_example,label))
    if save is not None:
        torch.save(xs_correct,save+"x_correct.pt")
        torch.save(xs_incorrect,save+"x_incorrect.pt")
    
    if xs_correct:
        ftrs_correct=ntk_methods.feature_mapping(model ,xs_correct,data_set=True,return_torch_tensor=False)
        cs_correct=calculate_cossims(ftrs_correct,feature_mapping)
    else:
        cs_correct=np.empty((0,0))
    if xs_incorrect:
        ftrs_incorrect=ntk_methods.feature_mapping(model ,xs_incorrect,data_set=True,return_torch_tensor=False)
        cs_incorrect=calculate_cossims(ftrs_incorrect,feature_mapping)
    else:
        cs_incorrect=np.empty((0,0))

    return (np.ndarray.flatten(cs_correct) ,np.ndarray.flatten(cs_incorrect))
  




#test_loader: a data loader containing the test set
#model: a neural net
#feature_mapping: a feature mapping of the training set
#attack: an adversarial attack with function signature attack(model,x,label) which returns and adversarial example
def cosine_similarities_of_attack(test_loader,model,feature_mapping,attack,save=None):
    cs_correct=[]
    cs_incorrect=[]
    adversarial_examples_correct=[]
    adversarial_examples_incorrect=[]
    for (x,label) in test_loader:
        adversarial_example=attack(model,x,label)
        adversarial_ys=model(adversarial_example)
        _, predicted = adversarial_ys.max(1)
        correct_prediction=predicted.eq(label).item()
        if correct_prediction:
            adversarial_examples_correct.append((adversarial_example,label))
        else:
            adversarial_examples_incorrect.append((adversarial_example,label))
    if save is not None:
        torch.save(adversarial_examples_correct,save+"x_correct.pt")
        torch.save(adversarial_examples_incorrect,save+"x_incorrect.pt")
    if adversarial_examples_correct:
        ftrs_correct=ntk_methods.feature_mapping(model ,adversarial_examples_correct,data_set=True,return_torch_tensor=False)
        cs_correct=calculate_cossims(ftrs_correct,feature_mapping)
    else:
        cs_correct=np.empty((0,0))
    if adversarial_examples_incorrect:
        ftrs_incorrect=ntk_methods.feature_mapping(model ,adversarial_examples_incorrect,data_set=True,return_torch_tensor=False)
        cs_incorrect=calculate_cossims(ftrs_incorrect,feature_mapping)
    else:
        cs_incorrect=np.empty((0,0))
    
    return np.ndarray.flatten(cs_correct), np.ndarray.flatten(cs_incorrect)
    




