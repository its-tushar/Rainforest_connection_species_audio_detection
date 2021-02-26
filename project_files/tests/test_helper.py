# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:36:46 2021

@author: tushar
"""

from files.helper import spectrogram,pad,normalise

def test_spectrogram(random_input):
    ''' This function tests the spectrogram function
    Parameters:
        random_input(array): A numpy array of length at least 2048
    Returns:
        Boolean: Return True if spectrogram function works as expected
    '''
    spec=spectrogram(random_input)
    assert(spec.shape[0]==64)
    return True
    
def test_pad(random_input):
    ''' This function test the pad function
    Parameters:
        random_input(array): A numpy array
    Returns:
        Boolean: Return True if pad function works as expected else False
    '''
    pad_data=pad(random_input)
    assert(len(pad_data)==570776)
    return True
    
def test_normalise(random_input):
    ''' This function test the normalise function
    Parameters:
        random_input(array): A 2D numpy array
    Returns:
        Boolean: Return True if normalise function works as expected else False
    '''
    normalised_data=normalise(random_input)
    assert(normalised_data.shape==random_input.shape)
    return True