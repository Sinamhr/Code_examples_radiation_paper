# Import 
import keras                        #tensorflow version 2.7.0 is recomended 
from keras.models import load_model

import numpy as np
from MCAE import MCAE


# ----------------------------------------      loading the models       ----------------------------------------  
# loading the MCAE model
MCAE = MCAE.load("MCAE_model")
# showing the MCAE architecture
MCAE.summary()


# loading the AAE model
AAE = load_model("AAE_model.h5")
AAE_encoder = load_model("AAE_encoder.h5")
AAE_decoder = load_model("AAE_decoder.h5")
# showing the AAE architecture 
AAE.summary()


# loading the AAEE model
AAEE = load_model("AAEE_model.h5")
AAEE_encoder = load_model("AAEE_encoder.h5")
AAEE_decoder = load_model("AAEE_decoder.h5")
# showing the AAE architecture 
AAEE.summary()


# loading the FAE model
FAE = load_model("FAE_model.h5")
FAE_encoder = load_model("FAE_encoder.h5")
FAE_decoder = load_model("FAE_decoder.h5")
# showing the FAE architecture
FAE.summary()




#  ----------------------------------------      loading the input data     ----------------------------------------                            

# loading example circulation data point for MCAE
loaded = np.load('Example_input_data_for_MCAE.npz')

# 10 circulation data points consists of daily mean field of Mean Sea Level Pressure (MSLP) and 300-700 layer thickness (τ300−700) 
# data points have been transferred to the Lambert azimuthal equal-area projection
# shape of data (datapoint_id, time steps, x, y, channel), x, y are the projection axes, and "channel" is 0 for the MSLP and 1 for τ300−700 
# the data have been normalized  
MCAE_example_inptut_data = loaded['Sample_data_points_Control_run']
# day of the year for each datapoint
MCAE_example_days = loaded['Sample_data_points_Control_run_days'] 
# month of the year for each datapoint
MCAE_example_month = loaded['Sample_data_points_Control_run_month'] 
# the year for each datapoint
MCAE_example_year = loaded['Sample_data_points_Control_run_year'] 

# Projection latitude and longitude for each pixel
MCAE_grid_lat = loaded['grid_lat']  
MCAE_grid_lon = loaded['grid_lon']  

# mask for data southward of latitude 30 deg North
MCAE_mask = loaded['Mask_MCAE'] 

# max and min values that can be used to denormalize the data points
Min_MSLP = loaded['min_data_P']  
Max_MSLP = loaded['max_data_P']  
Min_T300_700 = loaded['min_data_T']  
Max_T300_700 = loaded['max_data_T']



# loading example circulation data point for AAE, AAEE, and FAE

# Similar structure as introduced for MCAE
loaded = np.load('Example_input_data_for_AAE_and_FAE.npz')

AAE_FAE_example_inptut_data = loaded['Sample_data_points_Control_run'] 
AAE_FAE_example_days = loaded['Sample_data_points_Control_run_days'] 
AAE_FAE_example_month = loaded['Sample_data_points_Control_run_month'] 
AAE_FAE_example_year = loaded['Sample_data_points_Control_run_year'] 
AAE_FAE_grid_lat = loaded['grid_lat']  
AAE_FAEE_grid_lon = loaded['grid_lon']  
AAE_FAE_mask = loaded['Mask_AAE_FAE'] 

# these are the same as the one from MCAE (if you didn't load them yet you can load them now)
#Min_MSLP = loaded['min_data_P']  
#Max_MSLP = loaded['max_data_P']  
#Min_T300_700 = loaded['min_data_T']  
#Max_T300_700 = loaded['max_data_T']



# ----------------------------------------     some processing with the input data        ----------------------------------------   

# making the latent representation of the input data

MCAE_latent_representation = MCAE.encoder.predict(MCAE_example_inptut_data)    # the first 40 dimenstions are the PMSET indices the athers or free indices 
AAE_latent_representation = AAE_encoder.predict(AAE_FAE_example_inptut_data)
AAEE_latent_representation = AAEE_encoder.predict(AAE_FAE_example_inptut_data)
FAE_latent_representation = FAE_encoder.predict(AAE_FAE_example_inptut_data)


# regenerating the input data

MCAE_example_regenerated_data = MCAE.decoder.predict(MCAE_latent_representation)   
AAE_example_regenerated_data = AAE.predict(AAE_FAE_example_inptut_data)
AAEE_example_regenerated_data = AAEE.predict(AAE_FAE_example_inptut_data)
FAE_example_regenerated_data = FAE.predict(AAE_FAE_example_inptut_data)


# you can perform any sort of analysis with the models

