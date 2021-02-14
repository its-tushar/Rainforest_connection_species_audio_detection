import numpy as np
import pickle
from keras.models import model_from_json
import pandas as pd
import datetime
import warnings
warnings.filterwarnings('ignore')


class final:
    def load_files(self):
        # This function loads all required files
        #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        json_file = open("model_1.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into model
        self.model.load_weights("model_1.h5")
        self.data=pd.read_csv('train_tp_1.csv')
        with open('deploy_data.pickle','rb') as file:
            self.spec_data=pickle.load(file)


    def normalise(self,spec):
        # This function normalize the data
        mean=-69.59436665357703
        std=19.596147689935908
        spec=(spec-mean)/std
        return(spec)
   

    def predict(self,id):
        # This function takes index of recording id as input and return the predicted species id
        self.load_files() # Loads files
        duration=self.data['t_max'][id]-self.data['t_min'][id]
        rec_id=self.data['recording_id'][id]
        start=datetime.datetime.now()
        spec=self.spec_data[id]
        spec=spec.reshape(1,64,1115) # Reshapes the data in order to fed to model
        spec=self.normalise(spec) # Converts raw data to spectrogram
        pred_array=self.model.predict(spec)[0] # Predicts probabilities for each species
        pred_label=list(pred_array).index(max(pred_array)) # Calculate the species ID
        return(rec_id,np.round(duration,3),pred_label,datetime.datetime.now()-start)
