### script version of Generate DroneRF Features Notebook 

import os
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
import os
from sklearn.model_selection import train_test_split
from spafe.features.lfcc import lfcc
import spafe.utils.vis as vis
from scipy.signal import get_window
import scipy.fftpack as fft
from scipy import signal
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

from loading_functions import *
from file_paths import *
from feat_gen_functions import *

import importlib

# Dataset Info
main_folder = dronerf_raw_path
t_seg = 20
fs = 40e6 #40 MHz
samples_per_segment = int(t_seg/1e3*fs)

# Streaming configuration for chunk-wise feature extraction
chunk_size_samples = samples_per_segment * 4  # four segments per read (~memory/perf tradeoff)
segments_per_batch = 64  # how many segments to accumulate before flushing to disk
stream_dtype = np.float32

n_per_seg = 1024 # length of each segment (powers of 2)
n_overlap_spec = 120
win_type = 'hamming' # make ends of each segment match
high_low = 'L' #'L', 'H' # high or low range of frequency
feature_to_save = ['PSD', 'SPEC', ] # what features to generate and save: SPEC or PSD
format_to_save = ['IMG', 'ARR'] # IMG or ARR or RAW
to_add = True
spec_han_window = np.hanning(n_per_seg)

# Image properties
dim_px = (224, 224) # dimension of image pixels
dpi = 100

# Raw input len
v_samp_len = 10000

# data saving folders
features_folder = dronerf_feat_path
date_string = date.today()
# folder naming: ARR_FEAT_NFFT_SAMPLELENGTH
arr_spec_folder = "ARR_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
arr_psd_folder = "ARR_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_spec_folder = "IMG_SPEC_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
img_psd_folder = "IMG_PSD_"+high_low+'_'+str(n_per_seg)+"_"+str(t_seg)+"/"
raw_folder = 'RAW_VOLT_'+str(v_samp_len)+"_"+str(t_seg)+"/" # high and low frequency stacked together

os.makedirs(features_folder, exist_ok=True)
existing_folders = os.listdir(features_folder)

if high_low == 'H':
    i_hl = 0
elif high_low == 'L':
    i_hl = 1
    
# check if this set of parameters already exists
# check if each of the 4 folders exist
sa_save = False   #spec array
si_save = False   #spec imag
pa_save = False   #psd array
pi_save = False   #psd imag
raw_save = False # raw high low signals

if 'SPEC' in feature_to_save:
    if 'ARR' in format_to_save:
        if arr_spec_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+arr_spec_folder)
            except:
                print('folder already exist - adding')
            sa_save = True
            print('Generating SPEC in ARRAY format')
        else:
            print('Spec Arr folder already exists')
    if 'IMG' in format_to_save:
        if img_spec_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+img_spec_folder)
            except:
                print('folder already exist - adding')
            si_save = True
            print('Generating SPEC in IMAGE format')
        else:
            print('Spec Arr folder already exists')
if 'PSD' in feature_to_save:
    if 'ARR' in format_to_save:
        if arr_psd_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+arr_psd_folder)
            except:
                print('folder already exist - adding')
            pa_save = True
            print('Generating PSD in ARRAY format')
        else:
            print('PSD Arr folder already exists')
    if 'IMG' in format_to_save:
        if img_psd_folder not in existing_folders or to_add:
            try:
                os.mkdir(features_folder+img_psd_folder)
            except:
                print('folder already exist - adding')
            pi_save = True
            print('Generating PSD in IMAGE format')
        else:
            print('PSD Arr folder already exists')

if 'RAW' in feature_to_save:
    if raw_folder in existing_folders or to_add:
        try:
            os.mkdir(features_folder+raw_folder)
        except:
            print('RAW V folder already exists')
        raw_save = True


if all([not sa_save, not si_save, not pa_save, not pi_save, not raw_save]):
    print('Features Already Exist - Do Not Generate')
else:
    print(
        f'Streaming DroneRF segments with chunk_size_samples={chunk_size_samples} '
        f'and segments_per_batch={segments_per_batch}'
    )

    def flush_batch(batch_idx, BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V):
        if not BILABEL:
            return BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V

        if sa_save and F_SPEC:
            save_array_rf(
                features_folder + arr_spec_folder,
                F_SPEC,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'SPEC',
                n_per_seg,
                batch_idx,
            )

        if pa_save and F_PSD:
            save_array_rf(
                features_folder + arr_psd_folder,
                F_PSD,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'PSD',
                n_per_seg,
                batch_idx,
            )

        if raw_save and F_V:
            save_array_rf(
                features_folder + raw_folder,
                F_V,
                BILABEL,
                DRONELABEL,
                MODELALBEL,
                'RAW',
                '',
                batch_idx,
            )

        return [], [], [], [], [], []

    BILABEL = []
    DRONELABEL = []
    MODELALBEL = []
    F_PSD = []
    F_SPEC = []
    F_V = []

    batch_index = 0
    count_in_batch = 0

    segment_iterator = iter_dronerf_segments(
        main_folder,
        t_seg,
        chunk_size_samples=chunk_size_samples,
        dtype=stream_dtype,
    )

    for record in tqdm(segment_iterator, desc='Generating DroneRF features', unit='seg'):
        segment = record['segment']
        d_real = segment[i_hl]

        if raw_save:
            t = np.arange(0, segment.shape[1])
            f_high = interpolate.interp1d(t, segment[0])
            f_low = interpolate.interp1d(t, segment[1])
            tt = np.linspace(0, segment.shape[1] - 1, num=v_samp_len)
            d_v = np.stack((f_high(tt), f_low(tt)), axis=0).astype(np.float32, copy=False)
            F_V.append(d_v)

        if pa_save or pi_save:
            fpsd, Pxx_den = signal.welch(d_real, fs, window=win_type, nperseg=n_per_seg)
            if pa_save:
                F_PSD.append(Pxx_den.astype(np.float32, copy=False))
            if pi_save:
                save_psd_image_rf(
                    features_folder,
                    img_psd_folder,
                    record['ten_label'],
                    batch_index,
                    count_in_batch,
                    Pxx_den,
                    dim_px,
                    dpi,
                )

        if sa_save or si_save:
            if si_save:
                plt.clf()
                fig, ax = plt.subplots(
                    1, figsize=(dim_px[0] / dpi, dim_px[1] / dpi), dpi=dpi
                )
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                ax.axis('tight')
                ax.axis('off')

            Sxx, _, _, _ = plt.specgram(
                d_real,
                NFFT=n_per_seg,
                Fs=fs,
                window=spec_han_window,
                noverlap=n_overlap_spec,
                sides='onesided',
            )
            if si_save:
                save_spec_image_fig_rf(
                    features_folder,
                    img_spec_folder,
                    record['ten_label'],
                    batch_index,
                    count_in_batch,
                    fig,
                    dpi,
                )
            if sa_save:
                F_SPEC.append(interpolate_2d(Sxx, (224, 224)).astype(np.float32, copy=False))

        BILABEL.append(record['bi_label'])
        DRONELABEL.append(record['four_label'])
        MODELALBEL.append(record['ten_label'])

        count_in_batch += 1

        if count_in_batch >= segments_per_batch:
            BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V = flush_batch(
                batch_index, BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V
            )
            batch_index += 1
            count_in_batch = 0

    BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V = flush_batch(
        batch_index, BILABEL, DRONELABEL, MODELALBEL, F_SPEC, F_PSD, F_V
    )

