# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:20:30 2021

@author: tushar
"""
import pandas as pd
import librosa
import numpy as np
import datetime
from prettytable import PrettyTable
from keras.models import model_from_json
import sys
import logging
from files.helper import spectrogram,pad,normalise
class final:
    def __init__(self):
        ''' This function loads all required files'''
        #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        try:
            logging.basicConfig(filename="logfile.log",format='%(asctime)s %(message)s',filemode='a')
            self.logger=logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            json_file = open("/content/project_files/model.json", 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(loaded_model_json)
            # load weights into model
            self.model.load_weights("/content/project_files/model.h5")
            self.data=pd.read_csv('/content/project_files/train_tp.csv')
            self.logger.info('All required files and model were loaded successfully')
        except:
            self.logger.error('Required files were not loaded successfully, can not move ahead.')  
            sys.exit('Exiting')
            
    def load_audio(self,id):
        ''' This function take index as input and loads the audio file as time series sequence
        Parameters:
            id(int): index of audio file for which you want to make prediction
        Returns:
            array: Raw audio data
        '''
        try:
            path = '{}.flac'.format(self.data['recording_id'][id])
            audio_signal,sample_rate=librosa.load(path,sr=None) #Audio files are already encoded with sample rate of 48000, so we don't need to resample them.
        except:
            self.logger.error('The file you are trying to use does not exist') 
            sys.exit('Exiting')
        try:                                           
            signal=audio_signal[int(self.data['t_min'][id]*48000):int(self.data['t_max'][id]*48000)+1] 
            duration=self.data['t_max'][id]-self.data['t_min'][id]
            self.logger.info('Audio file was successfully loaded')
        except:
            self.logger.error('Either start time is negative or end time exceeds 60 seconds limit')    
            sys.exit('Exiting')
        return(signal,duration)   

    def predict(self,id):
        ''' This function takes index of recording id as input and return the predicted species id and total time taken
        Parameters:
            id(int):index of audio file for which you want to make prediction
        Returns:
            Table: Information like recording id, duration of audio, predicted species, time taken in tabular format
        '''
        start=datetime.datetime.now()
        raw_signal,duration=self.load_audio(id) # Load audio file
        pad_data=pad(raw_signal) # Pads the data
        spec=spectrogram(pad_data) # 
        spec=spec.reshape(1,64,1115) # Reshapes the data in order to fed to model
        spec=normalise(spec) # Converts raw data to spectrogram
        pred_array=self.model.predict(spec)[0] # Predicts probabilities for each species
        pred_label=list(pred_array).index(max(pred_array)) # Calculate the species ID

        x = PrettyTable(["Recording ID","Duration", "Predicted Species ID",'Time Taken'])
        row = [self.data['recording_id'][id],np.round(duration,3),pred_label,datetime.datetime.now()-start]
        x.add_row(row)
        print(x)    