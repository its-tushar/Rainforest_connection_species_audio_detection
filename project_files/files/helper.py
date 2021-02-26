# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:23:51 2021

@author: tushar
"""
import librosa
import sys
import numpy as np
import logging
logging.basicConfig(filename="logfile.log",format='%(asctime)s %(message)s',filemode='a')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
def spectrogram(pad_signal):
        ''' This function takes padded raw audio as input and converts raw audio data to spectrogram
        Parameters:
            pad_signal(array): This must be an array of padded raw audio of length 570776.
        Returns:
            array:Result after converting raw audio signal to spectrogram
        '''
        try:
            mel = librosa.feature.melspectrogram(y=pad_signal, sr=48000, n_mels=64)
            log_spec = librosa.power_to_db(S=mel, ref=np.max)
            logger.info('Padded data converted to spectrogram with shape {}'.format(log_spec.shape))
            return log_spec
        except:
            logger.error('Raw data was not converted to spectogram successfully')
            sys.exit('Exiting')   

def pad(data):
        ''' This function takes raw audio as input and does padding on raw audio data
        Parameters:
            data(array): This is raw audio data with length<=570776
        Returns:
            array: padded raw audio data with length=570776
        '''
        max_length=570776
        if(len(data)>max_length):
            logger.critical('Something is wrong, length of input data can not be greater than {}'.format(max_length))
        try:    
            k=list(data)
        except:
            logger.error('There is some problem with converting your input data to list')  
            sys.exit('Exiting')  
        logger.info('Length of data before padding {}'.format(len(k)))
        k.extend(0 for i in range(max_length-len(k)))    
        logger.info('Length of data after padding {}'.format(len(k)))
        return (np.array(k))
    
def normalise(spec):
        ''' This function takes spectrogram as input and normalize the spectrogram data
        Parameters:
            spec(array):Spectrogram
        Returns:
            array: Normalised spectrogram
        '''
        mean=-69.59436665357703
        std=19.596147689935908
        try:
            spec=(spec-mean)/std
        except:
            logger.critical('Something wrong with the input spectrogram, unable to normalize it')   
            sys.exit('Exiting') 
        logger.info('Spectogram was successfully normalized')   
        return(spec)