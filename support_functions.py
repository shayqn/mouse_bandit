# -*- coding: utf-8 -*-
"""
This file contains functions written by Shay & Gil intended to simplify
creating and scoring models for our ac209a_project 
"""

import numpy as np
import pandas as pd

def score_F1_and_confuse(y_predict,y_test,disp=True,confusion=True):
    '''
    this function creates a confusion table for a given set of predictions
    and testing data
    
    Inputs:
        y_predict: vector of predicted values
        y_test: vector of true values
        disp = True or False : boolean whether to print out confusion table or not
        confusion = True or False: whether to return the confusion table
    
    Outputs:
        acc: accuracy of model
        F1 : F1 score of model
        confusion_table: the confusion table (only if confusion = True)
        
    '''
    acc = np.mean(y_predict == y_test)
    
    trueP = np.sum(y_predict[np.where(y_test==1)]==1).astype(float)
    falseP = np.sum(y_predict[np.where(y_test==0)]==1).astype(float)
    trueN = np.sum(y_predict[np.where(y_test==0)]==0).astype(float)
    falseN = np.sum(y_predict[np.where(y_test==1)]==0).astype(float)
    
    d = {'Predicted NO':[trueN,falseN],'Predicted YES':[falseP,trueP]}
    confusion_table = pd.DataFrame(data=d,index=['True NO','True YES'])
    
    if (((trueP + falseP) == 0) or ((trueP + falseN) == 0)):
        F1 = 0
    else:
        p = trueP / (trueP + falseP)
        r = trueP / (trueP + falseN)
        if (p+r == 0):
            F1 = 0
        else:
            F1 = 2*(p*r) / (p+r)
    
    if disp is True:
        print(confusion_table)
        print('\nF1: %.03f' % F1)
        print('\nScore: %.02f' % acc)
    
    
    if confusion is True:
        return acc,F1,confusion_table
    else:
        return acc,F1
        
###############################################################################

def score_both_and_confuse(y_predict,y_test,disp=True,confusion=True):
    '''
    this function creates a confusion table for a given set of predictions
    and testing data
    
    Inputs:
        y_predict: vector of predicted values
        y_test: vector of true values
        disp = True or False : boolean whether to print out confusion table or not
        confusion = True or False: whether to return the confusion table
    
    Outputs:
        acc_pos: accuracy of model on flu cases
        acc_neg: accuracy of model on healthy cases
        F1 : F1 score of model
        confusion_table: the confusion table (only if confusion = True)
        
    '''
    acc_pos = np.mean(y_predict[y_test==1] == y_test[y_test==1])
    acc_neg = np.mean(y_predict[y_test==0] == y_test[y_test==0])
    
    trueP = np.sum(y_predict[np.where(y_test==1)]==1).astype(float)
    falseP = np.sum(y_predict[np.where(y_test==0)]==1).astype(float)
    trueN = np.sum(y_predict[np.where(y_test==0)]==0).astype(float)
    falseN = np.sum(y_predict[np.where(y_test==1)]==0).astype(float)
    
    d = {'Predicted NO':[trueN,falseN],'Predicted YES':[falseP,trueP]}
    confusion_table = pd.DataFrame(data=d,index=['True NO','True YES'])
    
    if (((trueP + falseP) == 0) or ((trueP + falseN) == 0)):
        F1 = 0
    else:
        p = trueP / (trueP + falseP)
        r = trueP / (trueP + falseN)
        if (p+r == 0):
            F1 = 0
        else:
            F1 = 2*(p*r) / (p+r)
    
    if disp is True:
        print(confusion_table)
        print('\nF1: %.03f' % F1)
        print('\nAccuracy on class 0: %.02f' % acc_neg)
        print('Accuracy on class 1: %.02f\n' % acc_pos)
    
    
    if confusion is True:
        return acc_pos,acc_neg,F1,confusion_table
    else:
        return acc_pos,acc_neg,F1