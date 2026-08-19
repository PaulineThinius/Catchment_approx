# -*- coding: utf-8 -*-
"""
LSTM model architecture and training
Adapted from Liesch and Ohmer (2026), DOI: 10.5194/hess-30-1877-2026.

Modifications to the existing code include:

Adaptation of the sequence-generation procedure to skip missing (NaN) target values in the training data while allowing
missing values in the testing data.
Adjustments to support time series with different record lengths and individual training, validation, and testing periods.
Extension of the static-input configuration to allow different types of static features, including environmental attributes,
time-series-derived characteristics, and randomly generated features.
"""

#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
import itertools
import copy
import tensorflow.keras.backend as K
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # '0'=all, '1'=filter INFO, '2'=filter WARNING, '3'=only ERROR
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'


#tf.debugging.set_log_device_placement(True)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memory growth set for GPUs.")
    except RuntimeError as e:
        print("⚠️ Could not set memory growth:", e)


mode_aggregate = sys.argv[1]# "buffer_1" # "catch
mode_features = sys.argv[2] #"static" #"timeseries" # "random"
cluster = sys.argv[3] # "None", "1","2","3","4"


pth_dt_dyn = "./data/dynamic"
base_out = f"./results/{mode_features}/{mode_aggregate}"

pth_out = base_out
    
os.makedirs(pth_out, exist_ok=True)  


#%% functions

# ----- sequentialize  --------------------

def make_sequences(data, n_steps_in, n_input, skip_nan =True):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    #skip_nan=True: for training and validation, no NaN in discharge date are allowed
    #skip_nan=False: for testing, NaN in discharge data are allowed

    X, Y = list(), list()
    # step over the entire history one time step at a time
    k=0

    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + n_steps_in
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        if not np.isnan(seq_y).any() or not skip_nan:
            X.append(seq_x)
            Y.append(seq_y)
        else:
            k += 1
    #print(f'Number of NaN in QobsS: {k}/{len(data)}')
    return np.array(X), np.array(Y)
    
# ----- learning rate scheduling  --------------------

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler implementing linear warmup and cosine decay."""

    def __init__(self, warmup_steps, total_steps, target_lr=0.001, start_lr=0.0, hold=0):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_lr = target_lr
        self.start_lr = start_lr
        self.hold = hold
        self.global_step = 0  # Initialize step counter

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "learning_rate"):
            raise ValueError('Optimizer must have a "learning_rate" attribute.')
        
        # Compute new learning rate
        new_lr = self.lr_warmup_cosine_decay(self.global_step)
        self.model.optimizer.learning_rate.assign(new_lr)  # Assign new LR
        
        print(f"\nEpoch {epoch}: Learning rate is {float(new_lr)}")
        self.global_step += 1  # Increment step counter

    def lr_warmup_cosine_decay(self, global_step):
        """Computes learning rate with linear warmup and cosine decay."""
        if global_step >= self.total_steps:
            return self.target_lr  # Ensure no overshooting beyond total steps

        # Cosine decay
        learning_rate = 0.5 * self.target_lr * (1 + np.cos(np.pi * (global_step - self.warmup_steps - self.hold) / float(self.total_steps - self.warmup_steps - self.hold)))

        # Linear warmup
        warmup_lr = self.target_lr * (global_step / self.warmup_steps)

        # Apply warmup, hold, and decay logic
        if self.hold > 0:
            learning_rate = np.where(global_step > self.warmup_steps + self.hold, learning_rate, self.target_lr)

        learning_rate = np.where(global_step < self.warmup_steps, warmup_lr, learning_rate)
        return learning_rate
    


# ----- clear memory  --------------------

class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        K.clear_session()
        gc.collect()
        
        
# ----- data generator  --------------------
        
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, X_dyn, X_stat, y, batch_size, shuffle=True):
        self.X_dyn = X_dyn
        self.X_stat = X_stat
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(y))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Generate data
        X_dyn_batch = self.X_dyn[batch_indices]
        X_stat_batch = self.X_stat[batch_indices]
        y_batch = self.y[batch_indices]

        return (X_dyn_batch, X_stat_batch), y_batch

    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        
# ----- BatchProgressCallback  --------------------

from keras.callbacks import Callback

class BatchProgressCallback(Callback):
    def on_train_batch_end(self, batch, logs=None):
        print(f"Batch {batch + 1} processed. Loss: {logs.get('loss'):.4f}")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nStarting Epoch {epoch + 1}...\n")
        
# ----- Non-parametric KGE  --------------------        


def kge_np(sim, obs):
    """
    Non-parametric KGE efficiency metric, computed column-wise if sim/obs are DataFrames.

    Parameters
    ----------
    sim : pd.DataFrame or np.ndarray
        Simulated values, can be multi-column.
    obs : pd.DataFrame or np.ndarray
        Observed values, same shape as sim.

    Returns
    -------
    np.ndarray
        KGE_np values for each column.
    """

    # Convert to numpy arrays if pandas
    if isinstance(sim, pd.DataFrame) or isinstance(sim, pd.Series):
        sim = sim.values
    if isinstance(obs, pd.DataFrame) or isinstance(obs, pd.Series):
        obs = obs.values

    # Check shapes
    if sim.shape != obs.shape:
        raise ValueError("'sim' and 'obs' must have the same shape!")

    # If 1D, make it 2D with one column for uniformity
    if sim.ndim == 1:
        sim = sim[:, np.newaxis]
        obs = obs[:, np.newaxis]

    n_cols = sim.shape[1]
    kge_vals = np.full(n_cols, np.nan)

    for i in range(n_cols):
        sim_col = sim[:, i]
        obs_col = obs[:, i]

        # Remove NaN values
        valid_idx = ~np.isnan(sim_col) & ~np.isnan(obs_col)
        sim_col = sim_col[valid_idx]
        obs_col = obs_col[valid_idx]

        if len(sim_col) == 0:
            continue  # skip empty columns

        # Mean values
        mean_sim = np.mean(sim_col)
        mean_obs = np.mean(obs_col)

        # Normalized flow duration curves
        fdc_sim = np.sort(sim_col / (mean_sim * len(sim_col)))
        fdc_obs = np.sort(obs_col / (mean_obs * len(obs_col)))

        # Alpha, beta, r
        RNP_alpha = 1 - 0.5 * np.sum(np.abs(fdc_sim - fdc_obs))
        RNP_beta = mean_sim / mean_obs
        RNP_r, _ = spearmanr(sim_col, obs_col)

        # KGE_np for this column
        kge_vals[i] = 1 - np.sqrt((RNP_alpha - 1) ** 2 + (RNP_beta - 1) ** 2 + (RNP_r - 1) ** 2)

    if kge_vals.size == 1:
        return kge_vals.item()
    return kge_vals


# ----- Scores computation  -------------------- 

def compute_scores(sim_col, obs_col, obs_train, kge_np_func=None):
    """
    Compute all scores for a single pair of sim and obs series.
    
    Parameters:
    - sim_col: pd.Series of simulated values
    - obs_col: pd.Series of observed values
    - nash_ref: reference for NSE computation (optional), same length as obs_col
    - kge_np_func: function(sim, obs) -> float (optional)
    
    Returns:
    - dict of scores
    """
    # Remove rows where obs is NaN
    mask = ~obs_col.isna()
    sim_col = sim_col[mask]
    obs_col = obs_col[mask]

    if len(obs_col) == 0:
        # No valid data, return NaNs
        return {key: np.nan for key in ['NSE','KGE','R2','Bias','MSE','RMSE',
                                        'KGE_I','KGE_np','BE','VE']}

    # Errors
    err = sim_col - obs_col
    err_nash = obs_col - np.nanmean(obs_train)


    # Basic metrics
    MSE = np.mean(err**2)
    RMSE = np.sqrt(MSE)
    Bias = np.mean(err)
    rr = stats.pearsonr(sim_col, obs_col)[0] if len(sim_col) > 1 else np.nan
    NSE = 1 - (np.sum(err**2) / np.sum(err_nash**2))

    # KGE metrics
    alpha = np.std(sim_col) / np.std(obs_col) if np.std(obs_col) != 0 else np.nan
    beta = np.mean(sim_col) / np.mean(obs_col) if np.mean(obs_col) != 0 else np.nan
    gamma = (np.std(sim_col)/np.mean(sim_col)) / (np.std(obs_col)/np.mean(obs_col)) if np.mean(sim_col) != 0 and np.mean(obs_col) != 0 else np.nan
    KGE = 1 - np.sqrt((rr-1)**2 + (alpha-1)**2 + (beta-1)**2)
    KGE_I = 1 - np.sqrt((rr-1)**2 + (gamma-1)**2 + (beta-1)**2)
    KGE_np = kge_np_func(sim_col, obs_col) if kge_np_func is not None else np.nan

    BE = (np.sum(sim_col) - np.sum(obs_col)) / np.sum(obs_col) if np.sum(obs_col) != 0 else np.nan
    VE = 1 - (np.abs(np.sum(err)) / np.sum(obs_col)) if np.sum(obs_col) != 0 else np.nan

    return {'NSE': NSE, 'KGE': KGE, 'R2': rr**2, 'Bias': Bias, 'MSE': MSE,
            'RMSE': RMSE, 'KGE_I': KGE_I, 'KGE_np': KGE_np, 'BE': BE, 'VE': VE}
        

# ----- Helper function inverse scaling  --------------------         
def inverse_transform_predictions(sim_n, IDlen, scalers_y, dt_list_names):
    sim = []

    for i in range(len(dt_list_names)):
        temp = sim_n[IDlen == i, 0].reshape(-1, 1)
        temp = scalers_y[i].inverse_transform(temp).ravel()

        sim.append(pd.DataFrame({
            "ID": i,
            "sim": temp
        }))

    return pd.concat(sim, ignore_index=True)
    

# ----- Helper function split evaluation  --------------------     
def evaluate_split(sim, Y, IDlen, split_name):
    results = []

    for i in range(len(dt_list_names)):
        temp = sim[sim.ID == i]
        temp.columns = (dt_list_names[i] + "_" + temp.columns).tolist()

        temp[dt_list_names[i] + "_obs"] = scalers_y[i].inverse_transform(
            Y[IDlen == i].reshape(-1, 1)
        )

        temp = temp.reset_index(drop=True)
        results.append(temp)

    results = pd.concat(results, axis=1)

    results.to_csv(f"{pth_out_i}/results_{split_name}.csv",
                   float_format="%.4f", sep=";")

    return results

# ----- Helper function median  --------------------     
def median_results(all_results):
    sim_list = [res.filter(like="sim") for res in all_results]
    obs_df = all_results[0].filter(like="obs")  # identical for all seeds

    sim_array = np.stack([sim.values for sim in sim_list], axis=-1)
    sim_median = np.median(sim_array, axis=-1)

    sim_median_df = pd.DataFrame(
        sim_median,
        index=sim_list[0].index,
        columns=sim_list[0].columns
    )

    results_median = pd.concat([sim_median_df, obs_df], axis=1)

    return results_median, sim_median_df, obs_df



#%% load time series data

#-----------------------------------

# get file with start and end dates
datasplit_df = pd.read_csv("./data/datasplit.csv")

# Filtered AlpAKaS-IDs:
alpakas_ids = pd.read_csv('./data/subset_alpakas_ids.csv', header=None)[0].tolist()

# list dynamic time series data files
dt_list_files = [
    f for f in os.listdir(pth_dt_dyn)
    if (
        mode_aggregate in f
        and f.endswith(".csv")
        and any(alp_id in f for alp_id in alpakas_ids)
    )
]

# load dynamic time series
dt_list_dyn = []
for fname in dt_list_files:
    temp = pd.read_csv(
        os.path.join(pth_dt_dyn, fname),
        parse_dates=[0],
        index_col=0,
        decimal=".",
        sep=","
    )
    dt_list_dyn.append(temp)

# get ID names
dt_list_names = [
    fname.rsplit("_dynamic.csv", 1)[0].rsplit("_", 1)[1]
    for fname in dt_list_files
]
dt_list_names = [item.replace("#", "/") for item in dt_list_names]



#-----------------------------------


# save IDs of remaining stations
pd.DataFrame({"ID": dt_list_names}).to_csv("./data/IDremainig.csv", sep = ";")


#-----------------------------------
#%% load static features
    
# static environmental features
if mode_features == "static":
    dt_static_features = pd.read_csv(f'./data/static/{mode_aggregate}_stat_attrs.csv')
# Random features are the same for each catchment representation
elif mode_features == "random":
    dt_static_features = pd.read_csv(f'./data/static/random_attrs.csv')
    if mode_aggregate == "point":
        dt_static_features = dt_static_features.drop(columns=["elev_min", "elev_max"])
else:
    dt_static_features = pd.read_csv(f'./data/static/time_series_attrs.csv')

dt_static_features.set_index('alpakas_id', inplace = True)

# Scaling
scaler_static = MinMaxScaler(feature_range = (-1, 1))
scaler_static.fit(dt_static_features)
dt_static_features_n = scaler_static.transform(dt_static_features)
    

#-----------------------------------

#%% Data preprocessing

n_steps_in = 120
feature_column_names = ["precipitation_nat","temperature_mean_nat","temperature_max_nat","temperature_min_nat","Tsin"]
n_input = len(feature_column_names)

    
#Initialize containers
IDlen_train = []
IDlen_stop = []
IDlen_test = []
IDlen_all = []

datafull = list()

scalers = list()
scalers_y = list()

X_train,Y_train = list(), list()
X_stop,Y_stop = list(), list()
X_test,Y_test = list(), list()
X_all,Y_all = list(), list()
X_train_stat, X_stop_stat, X_test_stat, X_all_stat = list(),list(),list(), list()
dates_all = []
split_dates_used = {}

for i in range(len(dt_list_files)):

    # Define split dates individually for each station
    alpakas_id = dt_list_names[i]
    print(alpakas_id)

    # read from data -split file
    date_start_test = pd.to_datetime(
        datasplit_df.loc[datasplit_df["alpakas_id"]==alpakas_id,"testing_start"].iloc[0]
    )
    date_start_stop = date_start_test - pd.DateOffset(years=2)  # 2 years for stop subset
    date_start_train = pd.to_datetime(
        datasplit_df.loc[datasplit_df["alpakas_id"]==alpakas_id,"training_start"].iloc[0]
    )
    # merge all input data
    tempdata = copy.deepcopy(dt_list_dyn[i]) #deepcopy does not affect the original
    
    tempdata["ID"] = i

    
    ordered_cols = (
    [c for c in feature_column_names if c in tempdata.columns] # order feature columns
        + ["QobsS"] # put target at the end
    )
    
    id_cols = [c for c in ["alpakas_id", "ID"] if c in tempdata.columns]
    
    tempdata = tempdata[id_cols + ordered_cols]

    # save original data for score calculations later on 
    datafull.append(tempdata)

    
    # fit scalers (on train+stop data) and transform (on full) data
    tempdata = tempdata.drop(columns=["alpakas_id","ID"])

    scalers.append(StandardScaler().fit(tempdata[(tempdata.index < date_start_test)]))
    scalers_y.append(StandardScaler().fit(pd.DataFrame(tempdata[(tempdata.index < date_start_test)]["QobsS"])))
    tempdata_n = scalers[i].transform(tempdata)
    
    # Split data
    tempdata_n_stop = tempdata_n[(tempdata.index >= date_start_stop) & (tempdata.index < date_start_test)]

    # Fallback if no validation data avilable in the defined window
    if np.isnan(tempdata_n_stop[:, -1]).all():
        # last non-NA QobsS BEFORE stop_start
        last_valid_idx = tempdata.loc[
            tempdata.index < date_start_stop, tempdata.columns[-1]
        ].last_valid_index()
    
        if last_valid_idx is None:
            raise ValueError("No non-NA Q found before stop_start")
    
        stop_end = last_valid_idx
        date_start_stop = stop_end - pd.DateOffset(years=2) + pd.DateOffset(days=1)
    
        # rebuild StopData with corrected window
        tempdata_n_stop = tempdata_n[(tempdata.index >= date_start_stop) & (tempdata.index <= stop_end)]

    split_dates_used[alpakas_id] = {
        "train_start":date_start_train,
        "train_end": date_start_stop - pd.Timedelta(days=1),
        "stop_start": date_start_stop,
        "test_start": date_start_test,
    }
    tempdata_n_train = tempdata_n[(tempdata.index < date_start_stop)]    
    tempdata_n_test = tempdata_n[(tempdata.index >= date_start_test)]

    
    # extend stop + Testdata to be able to fill sequence later
    tempdata_n_stop_ext = np.concatenate([tempdata_n_train[-n_steps_in:], tempdata_n_stop], axis=0)
    tempdata_n_test_ext = np.concatenate([tempdata_n_stop[-n_steps_in:], tempdata_n_test], axis=0)
    
    # sequentialize data and add static features
    # train data
    temp_x, temp_y = make_sequences(np.asarray(tempdata_n_train), n_steps_in, n_input)

    X_train.append(temp_x); Y_train.append(temp_y[:,0])
    X_train_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                  len(temp_x), axis = 0))
    # ID tracker: Save length of test set to identify individual Mst after modelfit
    IDlen_train.append(np.repeat(i, len(temp_y)))

    # stop data
    temp_x, temp_y = make_sequences(np.asarray(tempdata_n_stop_ext), n_steps_in, n_input)

    X_stop.append(temp_x); Y_stop.append(temp_y[:,0])
    X_stop_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                 len(temp_x), axis = 0))
    IDlen_stop.append(np.repeat(i, len(temp_y)))

    # test data
    temp_x, temp_y = make_sequences(np.asarray(tempdata_n_test_ext), n_steps_in, n_input, skip_nan = False)

    X_test.append(temp_x); Y_test.append(temp_y[:,0])
    X_test_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                 len(temp_x), axis = 0))
    IDlen_test.append(np.repeat(i,len(temp_y)))

    # all data to create simulations with trained model
    temp_x, temp_y = make_sequences(np.asarray(tempdata_n), n_steps_in, n_input, skip_nan = False)

    X_all.append(temp_x); Y_all.append(temp_y[:,0])
    X_all_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                  len(temp_x), axis = 0))
    IDlen_all.append(np.repeat(i, len(temp_y)))
    dates_all.append(tempdata.index[-len(temp_y):])
    

#del temp_x,temp_y,tempdata,tempdata_n,tempdata_n_stop,tempdata_n_stop_ext
#del tempdata_n_test,tempdata_n_test_ext,tempdata_n_train

# Final merge
X_train = np.concatenate(X_train)
X_train_stat = np.concatenate(X_train_stat)
Y_train = np.concatenate(Y_train)
X_stop = np.concatenate(X_stop)
X_stop_stat = np.concatenate(X_stop_stat)
Y_stop = np.concatenate(Y_stop)
X_test = np.concatenate(X_test)
X_test_stat = np.concatenate(X_test_stat)
Y_test = np.concatenate(Y_test)
X_all = np.concatenate(X_all)
X_all_stat = np.concatenate(X_all_stat)
Y_all = np.concatenate(Y_all)
IDlen_test = np.concatenate(IDlen_test)
IDlen_stop = np.concatenate(IDlen_stop)
IDlen_train = np.concatenate(IDlen_train)
IDlen_all = np.concatenate(IDlen_all)

datafull = pd.concat(datafull)
order = datafull["alpakas_id"].drop_duplicates()

datafullwide = (
    datafull[["alpakas_id", "QobsS"]]
    .pivot(columns="alpakas_id", values="QobsS")
    .reindex(columns=order)
)


datafullwide.to_csv(pth_out+'/datafullwide.csv', float_format='%.4f', sep = ";")


#%% Model - Prepare Inputs and Hyperparameters

HP_seeds = [171, 206, 380, 471, 570, 624, 643, 778, 808, 973]

# Hyperparameters
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.3
HPi_targetlr =  0.001
HPi_epochs = 20
HPi_batchsize = 256

all_results_train = []
all_results_val = []
all_results_test = []
all_results_all = []


for ii in range(len(HP_seeds)):
    
    HPi_seed = HP_seeds[ii]
    
    # create folder for outputs
    pth_out_i = os.path.join(pth_out, 'run' + str(HPi_seed))
    os.makedirs(pth_out_i, exist_ok=True)
    
    
    
    #% Modelling
    
    #-----------------------------    
    # Model
    #-----------------------------
    model_path = os.path.join(pth_out_i, 'model.keras') 

    if not os.path.isfile(model_path):
      #take time
      now1 = datetime.now()
      
      # set seed
      np.random.seed(HPi_seed)
      tf.random.set_seed(HPi_seed)
      
      # Callbacks
      model_es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
      model_mc = tf.keras.callbacks.ModelCheckpoint(filepath=pth_out_i+'/model.keras', 
                                                    save_best_only=True)
      
      warmup_steps = 3  # Number of warmup steps
      total_steps = 20  # Total training steps
      target_lr = 0.001  # Peak learning rate  
      lr_callback = CustomLearningRateScheduler(warmup_steps, total_steps, target_lr=target_lr)
            
  
      # Input layers for dynamic and static model strands
      model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
      model_stat_in = tf.keras.Input(shape=(X_train_stat.shape[1],))
      
      # Dynamic model strand
      model_dyn = tf.keras.layers.LSTM(HPi_lstm_size)(model_dyn_in) #, activation='relu'
      model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)
  
      # Static model strand
      model_stat = tf.keras.layers.Dense(HPi_static_size, activation='relu')(model_stat_in)
      model_stat = tf.keras.layers.Dropout(HPi_dropout)(model_stat)
      
      # Combine dynamic and static strands
      model_comb = tf.keras.layers.concatenate([model_dyn, model_stat])
      model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_comb)
      model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
      
      # Define output layer for predictions
      model_output = tf.keras.layers.Dense(units=1, activation='linear', dtype=tf.float32)(model_comb)
      
      # Define model with both dynamic and static inputs
      model = tf.keras.Model(inputs=[model_dyn_in, model_stat_in], outputs=model_output)
      
      # Compile model with appropriate loss function and optimizer
      optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
      model.compile(loss='mse', optimizer=optimizer)
  
  
      # Instantiate data generators

      train_generator = DataGenerator(X_train, X_train_stat, Y_train, HPi_batchsize, shuffle=True)
      val_generator = DataGenerator(X_stop, X_stop_stat, Y_stop, HPi_batchsize, shuffle=False)
      
      # Use fit with generators
      
      model_history = model.fit(train_generator,
                                validation_data=val_generator,
                                epochs=HPi_epochs, verbose=0, 
                                callbacks=[model_es, model_mc, lr_callback,
                                BatchProgressCallback(),
                                ClearMemory()])
          
  
          
      # take time
      now2 = datetime.now()
      timetaken = round((now2-now1).total_seconds())/60
      print('\n timetaken = '+str(timetaken)+'\n')
      
      #% Export results
      # train history
      pd.DataFrame(model_history.history).to_csv(pth_out_i+'/losshistory.csv', float_format='%.4f', sep = ";")
      
      # Minimum val MSE
      MSEvalmin = np.min(model_history.history['val_loss'])
      
      print(f' MSE validation min: {MSEvalmin}')
      
    

    # predict - with saved model checkpoint
    gc.collect()
    tf.keras.backend.clear_session()
    loaded_model = tf.keras.models.load_model(pth_out_i+'/model.keras')
    sim_test_n  = loaded_model.predict([X_test,  X_test_stat],  batch_size=64,verbose=0)
    sim_all_n  = loaded_model.predict([X_all,  X_all_stat],  batch_size=64,verbose=0)
     
    # inverse scaling
    sim_test = inverse_transform_predictions(
        sim_test_n, IDlen_test, scalers_y, dt_list_names)
    
    sim_all = inverse_transform_predictions(
        sim_all_n, IDlen_all, scalers_y, dt_list_names)
    
    
    
    #% Evaluate Model

    results_test  = evaluate_split(sim_test, Y_test, IDlen_test, "test")
    results_all  = evaluate_split(sim_all, Y_all, IDlen_all, "all")

    all_results_test.append(results_test)
    all_results_all.append(results_all)

results_test_median, sim_test_median_df, obs_test_df = median_results(all_results_test)


results_all_median, _, _ = median_results(all_results_all)


for i, alpakas_id in enumerate(dt_list_names):

    # Number of predictions for this ID
    n_i = np.sum(IDlen_all == i)

    # Find this ID's sim/obs columns
    sim_col = f"{alpakas_id}_sim"
    obs_col = f"{alpakas_id}_obs"

    # Extract only the rows that actually belong to this ID
    df_id = results_all_median.loc[
        :n_i - 1,
        [sim_col, obs_col]
    ].copy()

    # Rename columns
    df_id.columns = ["sim", "obs"]

    # Assign the true datetime index for this ID
    df_id.index = dates_all[i]
    df_id.index.name = "date"

    # Add split information
    stop_start = split_dates_used[alpakas_id]["stop_start"]
    test_start = split_dates_used[alpakas_id]["test_start"]
    train_start = split_dates_used[alpakas_id]["train_start"]

    # Remove everything before training starts
    df_id = df_id.loc[df_id.index >= train_start].copy()

    df_id["split"] = "train"
    df_id.loc[df_id.index >= stop_start, "split"] = "stop"
    df_id.loc[df_id.index >= test_start, "split"] = "test"

    # Save
    df_id.to_csv(
        os.path.join(pth_out,f"{alpakas_id}_median_all.csv"),
        sep=";",
        float_format="%.4f"
    )