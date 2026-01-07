"""
Grid score calculations. Adapted from https://github.com/deepmind/grid-cells/blob/d1a2304d9a54e5ead676af577a38d4d87aa73041/scores.py 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Union, Tuple, List, Dict, Any, Callable
from numpy.typing import NDArray

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.ndimage import gaussian_filter, label
import warnings 
import scipy.signal
import scipy.misc
import skimage as sk
import skimage.transform 
# from astropy.convolution import convolve
# from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import pandas as pd
from collections import defaultdict
import numpy.matlib
import pyfftw
import multiprocessing
import numba
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
import copy
from itertools import accumulate


import glob
import seaborn as sns
from scipy.stats import ttest_ind, zscore, linregress, norm, pearsonr, binom_test, median_abs_deviation
from scipy.signal import butter, resample, filtfilt, savgol_filter, convolve2d, sosfiltfilt, iirnotch

from scipy.fftpack import next_fast_len
import statsmodels.api as sm
import statsmodels.formula.api as smf

import joblib
import warnings
import os
import h5py
from matplotlib.backends.backend_pdf import PdfPages


from skimage.measure import regionprops, label
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.filters import gaussian
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mec-cell-classification-master"))
from shuffle_rate_map import shuffle_rate_map
from fourier_transform import (fourier_transform, analyze_fourier, analyze_polar_spectrogram, 
    analyze_fourier_rings, fourier_rings_significance)
from plot_data import (plot_canonical_scoring_method, plot_analyze_correlation_periodicity, plot_fourier_scoring,  
    plot_polar_components, plot_rho_mean_power, plot_ring_power, plot_ring_random_distribution)
from calculate_2d_tuning_curve import calculate_2d_tuning_curve
from calculate_spatial_periodicity import calculate_spatial_periodicity
from calculate_correlation_matrix import calculate_correlation_matrix
from analyze_periodicity import analyze_periodicity, Partition_Data

import spike_utils 

import random 
from collections import Counter

from scipy import stats
import skimage

import re

def plot_unit_density(sw, savepath=None):
    """
    Plot a 2D density histogram of spike waveforms.
    
    Creates a heatmap showing the distribution of voltage values across time
    samples for all spikes in a unit. This visualization helps assess unit quality
    and waveform stability.
    
    Parameters
    ----------
    sw : np.ndarray
        Spike waveforms array of shape (num_spikes, num_samples).
        Each row is a single spike waveform.
    savepath : str, optional
        Path to save the plot as a PDF file. If None, plot is not saved.
    
    Returns
    -------
    None
        Saves plot to file if savepath is provided.
    """
    f, ax = plt.subplots(1,1, figsize=[4,3], dpi=300)
    num_spikes, num_samples = sw.shape
    # --- Step 3: Prepare Bins ---
    min_voltage = np.min(sw)
    max_voltage = np.max(sw)
    num_voltage_bins = 150 # Adjust as needed
    voltage_bins = np.linspace(min_voltage, max_voltage, num_voltage_bins + 1)
    # --- Step 4: Create 2D Histogram ---
    density_matrix = np.zeros((num_samples, num_voltage_bins))

    for i in range(num_spikes):
        waveform = sw[i, :]
        # Find which voltage bin each sample falls into
        # np.digitize is efficient for this
        voltage_bin_indices = np.digitize(waveform, voltage_bins[1:-1]) # Bins excluding the edges for np.digitize

        for t in range(num_samples):
            bin_index = voltage_bin_indices[t]
            # Ensure bin_index is within bounds (0 to num_voltage_bins-1)
            if 0 <= bin_index < num_voltage_bins:
                 density_matrix[t, bin_index] += 1
    # --- Step 5: Visualize ---
    # plt.figure(figsize=(8, 6))

    extent = [0, 64, voltage_bins[0], voltage_bins[-1]]

    # Optional: Apply log scaling for better visibility of low counts (add small value to avoid log(0))
    # plot_data = np.log1p(density_matrix.T) # Transpose because imshow expects (y, x)
    plot_data = density_matrix.T # Or plot raw counts

    ax.imshow(plot_data, aspect='auto', origin='lower', cmap='viridis', extent=extent) # 'hot', 'jet', 'grey_r' are other options

    # Optional: Overlay mean waveform
    # mean_waveform = np.mean(sw, axis=0)
    # ax.plot(mean_waveform, color='red', linewidth=1.5)

    ax.set_ylabel("voltage (µV)") # Adjust unit if needed
    ax.set_xlabel("time (sample)") # Adjust unit if needed

    plt.savefig(f'{savepath}.pdf')
    plt.close()


def expand_range(value):
    """
    Expand a range string (e.g., 'chan1:5') into a list of individual values.
    
    Parameters
    ----------
    value : str
        String in format 'prefixstart:end' (e.g., 'chan1:5').
        If format doesn't match, returns single-item list with original value.
    
    Returns
    -------
    list of str
        List of expanded values (e.g., ['chan1', 'chan2', 'chan3', 'chan4', 'chan5']).
        If format doesn't match, returns [value].
    
    Examples
    --------
    >>> expand_range('chan1:5')
    ['chan1', 'chan2', 'chan3', 'chan4', 'chan5']
    >>> expand_range('bundle10')
    ['bundle10']
    """
    match = re.match(r'([A-Za-z]+)(\d+):(\d+)', value)
    if match:
        prefix, start, end = match.groups()
        return [f"{prefix}{i}" for i in range(int(start), int(end)+1)]
    return [value]

def get_all_spikes_all_subj(all_encoding_df, sync_data, data_dir, f_range=(2,10)):
    """
    Extract spike times, waveforms, and phase data for all subjects and cells.
    
    Processes neural data from multiple subjects, extracting spike times,
    waveforms, and LFP phase information. Computes quality metrics for each unit
    and generates unit density plots.
    
    Parameters
    ----------
    all_encoding_df : pd.DataFrame
        DataFrame containing encoding trial information with timing data.
    sync_data : pd.DataFrame
        DataFrame with synchronization parameters (slope, offset) per subject.
    data_dir : str
        Base directory containing neural data files.
    f_range : tuple, default=(2, 10)
        Frequency range (Hz) for LFP phase extraction.
    
    Returns
    -------
    tuple
        Returns (all_spikes, all_spike_phases, all_subj, all_channel_loc,
        all_channel_num, all_cell_num, all_cell_mni, quality_metrics) where:
        - all_spikes: list of spike time arrays
        - all_spike_phases: list of phase arrays
        - all_subj: list of subject IDs
        - all_channel_loc: list of brain region labels
        - all_channel_num: list of channel numbers
        - all_cell_num: list of cell indices
        - all_cell_mni: list of MNI coordinates
        - quality_metrics: dict with quality metrics per cell
    """
    
    # First, corral all the data: 
    sub_dirs = glob.glob('/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Implantations_20220301/sub*')
    # restrict to new data 
    # sub_dirs = [x for x in sub_dirs if '2023' in x] + [x for x in sub_dirs if '2024' in x]

    sub_names = [x.split('/')[-1] for x in sub_dirs]

    freiburg_subjects_old = [x.split('/')[-1] for x in sub_dirs if (('2023' not in x) & ('2024' not in x))] 
    freiburg_subjects_new = ['sub0127_20220922_102544', 'sub128_20230212_144323', 'sub129_20230511_113542',
    'sub30_20240127_112616', 'sub031_20240223_123631', 'sub32_20240305_164311', 'sub32_20240308_153137']
    bonn_subjects = ['sub019_20230513_182036', 'sub020_20230515_164251', 'sub021_20230610_105606', 'sub022_20230612_114116',
                'sub023_20230818_195634', 'sub024_20230821_151917', 'sub025_20230823_155819', 'sub026_20230827_190344',
                'sub027_20230830_101758', 'sub028_20231113_194728', 'sub029_20231219_140815', 'sub030_20240518_211031',
                'sub031_20240521_181631']

    all_encoding_dfs = [] 
    cell_i = 0
    all_spikes = [] 
    all_spike_waveforms = []
    all_spike_phases = []
    all_subj = [] 
    all_channel_loc = []
    all_channel_num = [] 
    all_cell_num = []
    all_cell_mni = []

    quality_metrics = {'num_units_per_wire': [], 
                       'ici_percentage': [] , 
                       'mean_fr': [], 
                       'A_peak': [], 
                       'median_deviation': [],
                       'peak_snr': []}


    fs = 250 # For now

    for subj in sub_names:
        # subj = sub_names[1]
        encoding_time_in_neural_time = [(x*sync_data.slope[sync_data.subj==subj] + sync_data.offset[sync_data.subj==subj]).values for x in all_encoding_df.time[all_encoding_df.subj==subj]]

        # get all channel folders for subject
        if subj in freiburg_subjects_old:
            chan_dirs = glob.glob(f'{data_dir}/Neural/Micros/{subj}/times_chan*.h5')
        else:
            chan_dirs = next(os.walk(f'{data_dir}/Neural/Micros_New/{subj}'))[1]
            chan_dirs = [x for x in chan_dirs if 'ainp' not in x]

        # get all localizations for subject 
        loc_file = f'{data_dir}/Implantations_20220301/{subj}/Microwires/Micro2Region.txt'
        locs_micro = pd.read_csv(loc_file, sep=' ', header=None).rename(columns={0:'bundle', 1:'region'})
        if subj in freiburg_subjects_old + freiburg_subjects_new:
            locs_micro.bundle = locs_micro.bundle.apply(lambda x: np.arange(int(x.split(':')[0]), 1+int(x.split(':')[-1])))
        elif subj in bonn_subjects:
            locs_micro.bundle = locs_micro['bundle'].apply(expand_range)

        # Get the MNI coordinates for each bundle: 
        # If you want to find out the MNI coordinate of a particular microwire, 
        # proceed as follows: (1) get the microwire label (e.g., “chan1” for a subject from Freiburg); 
        # (2) use the “Micro2Macro.txt” file to get the associated macroelectrode (e.g., “AL”); 
        # and (3) use the FIRST (= innermost) contact of the corresponding macroelectrode (e.g., “AL1”) in the Pylocator file to get the MNI coordinates of the microwire (e.g., “-24.6648348182,-1.14602088062,-20.3921593406”). 
        # Note: Because the microwires protrude from the tip of the macroelectrode, 
        # the MNI coordinate of the first contact is not perfectly accurate, but a reasonable approximation.

        pylocator_dir = f'{data_dir}/Implantations_20220301/{subj}/Pylocator'
        pylocator_file = glob.glob(f'{pylocator_dir}/*')[0]
        pylocator_data = pd.read_csv(pylocator_file, sep=',',header=None).reset_index(drop=True)

        pylocator_data.rename(columns={0: 'depth', 
                             1:'mni.x', 
                             2:'mni.y', 
                             3:'mni.z'}, inplace=True)

        pylocator_data = pylocator_data[pylocator_data.depth.str.endswith('1')]

        pylocator_data.depth = pylocator_data.depth.apply(lambda x: x[:-1])

        # # Get rid of 11's
        pylocator_data = pylocator_data[~pylocator_data.depth.str.endswith('1')]
        pylocator_data = pylocator_data[["depth", "mni.x", "mni.y", "mni.z"]]

        loc_file_macro =f'{data_dir}/Implantations_20220301/{subj}/Microwires//Micro2Macro.txt'
        locs_macro = pd.read_csv(loc_file_macro, sep=' ', header=None).rename(columns={0:'bundle', 1:'depth'})
        #     locs_macro.bundle = locs_macro.bundle.apply(lambda x: np.arange(int(x.split(':')[0]), 1+int(x.split(':')[-1])))
        locs = pd.concat([locs_micro, locs_macro['depth']], axis=1)

        # merge in with the locs dataframe
        locs = locs.merge(pylocator_data, on='depth')

        locs = locs.explode('bundle').reset_index(drop=True).rename(columns={'bundle':'channel_name'})

        if subj in freiburg_subjects_old + freiburg_subjects_new:
            locs['channel_name'] = locs['channel_name'].apply(lambda x: f'chan{x}')


        # Iterate through channels 
        for chan_dir in chan_dirs:

            if subj in freiburg_subjects_old:
                # get the channel # 
                time_file = chan_dir
                channel_num = time_file.split('/')[-1].split('_')[1][:-3]
                # get the channel localization 
                channel_loc = locs[locs.channel_name==channel_num].region.values[0]
                channel_mni = [locs.loc[locs.channel_name==channel_num, 'mni.x'].values[0], 
                                       locs.loc[locs.channel_name==channel_num, 'mni.y'].values[0], 
                                       locs.loc[locs.channel_name==channel_num, 'mni.z'].values[0]]

            else:
                time_file = f'{data_dir}/Neural/Micros_New/{subj}/{chan_dir}/times_datacut.mat'

                # get the channel localization 
                channel_loc = locs[locs.channel_name==chan_dir].region.values[0]
                channel_mni = [locs.loc[locs.channel_name==chan_dir, 'mni.x'].values[0], 
                                       locs.loc[locs.channel_name==chan_dir, 'mni.y'].values[0], 
                                       locs.loc[locs.channel_name==chan_dir, 'mni.z'].values[0]]

            if channel_loc not in ['AMY', 'EC', 'HC', 'PHC', 'TP']:
                continue

            if subj in freiburg_subjects_old:
                f = h5py.File(time_file,'r')
                n1 = f.get('spike_data_sorted')
                spikeTimes = np.array(n1.get('spikeTimes')) # NOTE: THESE ARE IN MILLISECONDS 
                spikeClasses = np.array(n1.get('spikeClasses'))
                spikeClusters = np.array(n1.get('spikeClusters')) 
                spikeWaveforms = np.array(n1.get('spikeWaveforms')) 

            else:
                f = scipy.io.loadmat(time_file)
                spikeTimes = np.array(f['cluster_class'][:,1]) # NOTE: THESE ARE IN MILLISECONDS 
                spikeClusters = np.array(f['cluster_class'][:,0]) 
                spikeWaveforms = np.array(f['spikes'])

                spikeTimes = spikeTimes[spikeClusters>0]
                spikeWaveforms = spikeWaveforms[spikeClusters>0]
                spikeClusters = spikeClusters[spikeClusters>0]

            if len(spikeTimes) == 0: 
                # print('no cells on this channel') 
                continue

            if subj in freiburg_subjects_old:
                lfp_file = f'{data_dir}/Neural/Micros/{subj}/micro_lfp/{channel_num}_lfp.h5'
                # Open the file (read-only by default)
                with h5py.File(lfp_file, 'r') as f:

                    # Get the 500 Hz sampled data 
                    lfp_data = f['lfp'][:]

                f.close()
                # downsample to 250 Hz
                pad = np.ones([next_fast_len(len(lfp_data)) - len(lfp_data)])
                if len(pad) == 0:
                    lfp_data = resample(lfp_data, int(len(lfp_data)/(500/fs)))
                else:
                    lfp_data = resample(np.append(lfp_data, pad), int(len(np.append(lfp_data, pad))/(500/fs)))
                    lfp_data = lfp_data[:-int(len(pad)/(500/fs))]

            else:
                lfp_file = f'{data_dir}/Neural/MicroLFP_NEW/{subj}/{chan_dir}_lfp.p'
                lfp_data = joblib.load(lfp_file)


            # interpolate spikeTimes
            for ts in spikeTimes:
                lfp_data[((ts - 2)//4).astype(int):((ts + 8)//4).astype(int) ]  = np.nan

            s = pd.Series(lfp_data)
            s = s.interpolate(limit_direction='both')

            lfp_data = s.values

            # f_range = (2, 10)

            phase_data = get_phase(lfp_data, freqs=f_range, fs=fs, 
                                              N_seconds=0.1, N_seconds_theta=0.75, threshold=33)

            corrected_phase_data = phase_data['hilbert_phase'].copy()
            corrected_phase_data[phase_data['low_amplitude']] = np.nan

            if len(spikeTimes) == 0: 
                # print('no cells on this channel') 
                continue

            # iterate through cells on a channel 

            quality_metrics['num_units_per_wire'].append(np.unique(spikeClusters).shape[0])

            
            if subj in freiburg_subjects_old:
                chan_dir = chan_dir.split('/')[-1][6:-3]
                    
            for ic in np.unique(spikeClusters):
                st = spikeTimes[spikeClusters==ic]
                sw = spikeWaveforms[spikeClusters==ic]  # account for amplitude rescaling by sorting
                sp = corrected_phase_data[(st//4).astype(int)] # unwrapped phase
                # sp = sp[~np.isnan(sp)] 

                if len(st) <= 100:
                    continue
                cell_i+=1   
                all_spikes.append(st) 
                all_spike_phases.append(sp)
                # all_spike_waveforms.append(sw)
                all_subj.append(subj)
                all_channel_loc.append(channel_loc)
                all_channel_num.append(chan_dir)
                all_cell_num.append(cell_i)
                all_cell_mni.append(channel_mni)

                # get quality metrics: 

                A_peak = np.nanmax(np.abs(sw))
                # mean_waveform = np.nanmean(sw, axis=0)
                # A_peak = np.nanmax(np.abs(np.nanmin(mean_waveform)), np.abs(np.nanmax(mean_waveform)))
                median_deviation = median_abs_deviation(lfp_data)

                if subj in freiburg_subjects_old:
                    A_peak=A_peak*4 # account for amplitude rescaling by Combinato
                # elif A_peak < 25
                
                ici_percentage = np.sum(np.diff(st)<3) / len(st)
                num_units_per_wire = np.unique(spikeClusters).shape[0]
                total_avg_fr = len(st) / all_encoding_df.session_length[all_encoding_df.subj==subj].unique()[0]
                snr = A_peak/median_deviation

                quality_metrics['A_peak'].append(A_peak)
                quality_metrics['median_deviation'].append(median_deviation)
                quality_metrics['ici_percentage'].append(np.sum(np.diff(st)<3) / len(st))
                quality_metrics['mean_fr'].append(total_avg_fr)
                quality_metrics['peak_snr'].append(snr)
                
                # save out a density plot of this spikewaveform
                savepath = f'/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Results/plots/unit_density/{subj}_{channel_loc}_{cell_i}'
                plot_unit_density(sw, savepath=savepath)
     
    
    # fix the empty channels in all_channel_num
    filled_data = list(accumulate(all_channel_num, lambda prev, curr: curr if curr else prev))
    all_channel_num = filled_data

    return all_spikes, all_spike_phases, all_subj, all_channel_loc, all_channel_num, all_cell_num, all_cell_mni, quality_metrics


def corr_maps(map1, map2, maptype='normal'):
    """
    Correlate two ratemaps together, ignoring areas with zero sampling.
    
    Computes Pearson correlation between two spatial maps (ratemaps or grid maps),
    handling NaN values and mismatched shapes by resizing if necessary.
    
    Parameters
    ----------
    map1 : np.ndarray
        First ratemap to correlate.
    map2 : np.ndarray
        Second ratemap to correlate.
    maptype : str, default='normal'
        Type of map. Options:
        - 'normal': Valid pixels are those > 0 or not NaN
        - 'grid': Valid pixels are those not NaN
    
    Returns
    -------
    tuple
        Pearson correlation coefficient and p-value from scipy.stats.pearsonr.
    """
    if map1.shape > map2.shape:
        map2 = skimage.transform.resize(map2, map1.shape, mode='reflect')
    elif map1.shape < map2.shape:
        map1 = skimage.transform.resize(map1, map2.shape, mode='reflect')
    map1 = map1.flatten()
    map2 = map2.flatten()
    if 'normal' in maptype:
        valid_map1 = np.logical_or((map1 > 0), ~np.isnan(map1))
        valid_map2 = np.logical_or((map2 > 0), ~np.isnan(map2))
    elif 'grid' in maptype:
        valid_map1 = ~np.isnan(map1)
        valid_map2 = ~np.isnan(map2)
    valid = np.logical_and(valid_map1, valid_map2)
#     r = np.corrcoef(map1[valid], map2[valid])
    return stats.pearsonr(map1[valid], map2[valid])

def smooth_ratemap(ratemap, smooth_width):
    """
    Smooth a ratemap using Gaussian filtering while preserving NaN values.
    
    Applies Gaussian smoothing to a ratemap, properly handling NaN values
    (unvisited bins) by using a two-pass filtering approach that normalizes
    by the number of valid samples per bin.
    
    Parameters
    ----------
    ratemap : np.ndarray
        Raw ratemap with potential NaN values for unvisited bins.
    smooth_width : float
        Standard deviation of the Gaussian smoothing kernel.
    
    Returns
    -------
    np.ndarray
        Smoothed ratemap with same shape as input, preserving NaN locations.
    """
    nbins = ratemap.shape[0]
    trunc = (((nbins - 1) / 2) - 0.5) / smooth_width
    
    V=ratemap.copy()
    V[np.isnan(ratemap)]=0
    VV=gaussian_filter(V, sigma=smooth_width, mode='constant', truncate=trunc)

    W=0*ratemap.copy()+1
    W[np.isnan(ratemap)]=0
    WW=gaussian_filter(W, sigma=smooth_width, mode='constant', truncate=trunc)

    smoothmap=VV/WW
    
    return smoothmap

def scramble_matrix(matrix):
    """
    Randomly shuffle the values in a matrix while preserving its shape.
    
    Flattens the matrix, shuffles the values randomly, then reshapes to
    original dimensions. Useful for creating spatial surrogates.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to scramble.
    
    Returns
    -------
    np.ndarray
        Scrambled matrix with same shape as input.
    """
    flat = matrix.flatten()
    np.random.shuffle(flat)
    return flat.reshape(matrix.shape)


def draw_heatmap(*args, **kwargs):
    """
    Draw a heatmap from a pandas DataFrame using seaborn-style interface.
    
    Creates a pivot table from the DataFrame and visualizes it as a heatmap.
    Only shows bins with more than 3 data points.
    
    Parameters
    ----------
    *args : tuple
        First two arguments should be column names for index and columns of pivot table.
        Third argument should be the value column name.
    **kwargs : dict
        Additional keyword arguments. Must include 'data' key with pandas DataFrame.
    
    Returns
    -------
    None
        Displays the heatmap plot.
    """
    data = kwargs.pop('data')
    d = data.pivot_table(index=args[0], columns=args[1], values=args[2], dropna=False, aggfunc=np.nanmean)
    mask = data.pivot_table(index=args[0], columns=args[1], values=args[2], dropna=False, aggfunc='count')
    d = d[mask>3]
#     sns.heatmap(d, **kwargs)
    im = plt.imshow(np.array(d.values.tolist()).astype('float'), 
                    aspect='auto', interpolation='hanning', cmap='jet')
    plt.colorbar()
    plt.gca().invert_yaxis()

def jitter(value, j):
    """
    Add random jitter to a value for visualization purposes.
    
    Adds normally distributed noise to a value to prevent overplotting
    in scatter plots or similar visualizations.
    
    Parameters
    ----------
    value : float
        Original value to jitter.
    j : float
        Mean of the jitter distribution.
    
    Returns
    -------
    float
        Jittered value (value + noise where noise ~ N(j, 0.025)).
    """
    return value + np.random.normal(j, 0.025, 1)


def circle_mask(size, radius, in_val=1.0, out_val=0.0):
    """
    Create a circular mask for spatial autocorrelation analysis.
    
    Generates a 2D binary mask with a circular region. Used for analyzing
    grid cell spatial autocorrelograms at different radial distances.
    
    Parameters
    ----------
    size : list or tuple of int
        Shape of the output mask [height, width].
    radius : float
        Radius of the circle in pixels.
    in_val : float, default=1.0
        Value for pixels inside the circle.
    out_val : float, default=0.0
        Value for pixels outside the circle.
    
    Returns
    -------
    np.ndarray
        2D array with circular mask of specified size.
    """
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.linspace(-sz[0], sz[1], size[1])
    x = np.expand_dims(x, 0)
    x = x.repeat(size[0], 0)
    y = np.linspace(-sz[0], sz[1], size[1])
    y = np.expand_dims(y, 1)
    y = y.repeat(size[1], 1)
    z = np.sqrt(x**2 + y**2)
    z = np.less_equal(z, radius)
    vfunc = np.vectorize(lambda b: b and in_val or out_val)
    return vfunc(z)


class GridScorer(object):
    """Class for scoring ratemaps for grid-scores.
    
    To deal with the np.nan's for "univisited" bins: 

    A Gaussian filter which ignores NaNs in a given array U can be easily obtained by applying a standard Gaussian filter to two auxiliary arrays V and W and by taking the ratio of the two to get the result Z.

    Here, V is copy of the original U with NaNs replaced by zeros and W is an array of ones with zeros indicating the positions of NaNs in the original U.

    The idea is that replacing the NaNs by zeros introduces an error in the filtered array which can, however, be compensated by applying the same Gaussian filter to another auxiliary array and combining the two.

    This is because due to the "missing" data the coefficients of VV (below) do not sum up to one anymore and therefore underestimate the "true" value.

    However, this can be compensated by using the second auxiliary array (cf. WW below) which, after filtering with the same Gaussian, just contains the sum of coefficients.

    Therefore, dividing the two filtered auxiliary arrays rescales the coefficients such that they sum up to one while the NaN positions are ignored.
    
    """

    def __init__(self, df=None, mask_parameters=None, nbins=18, smooth_width=1, min_max=False, folds=np.arange(4,11), xdim='valence', ydim='arousal'):
        """
        Initialize GridScorer for computing grid cell scores.
        
        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame containing spatial and firing rate data.
        mask_parameters : iterable, optional
            Iterable of (mask_min, mask_max) tuples for ring masks.
        nbins : int, default=18
            Number of bins per dimension in the ratemap.
        smooth_width : float, default=1
            Standard deviation of Gaussian smoothing kernel.
        min_max : bool, default=False
            Whether to use min-max correction for grid score calculation.
        folds : np.ndarray, default=np.arange(4,11)
            Array of fold symmetries to test (e.g., 4-fold, 6-fold, etc.).
        xdim : str, default='valence'
            Column name for x-dimension in DataFrame.
        ydim : str, default='arousal'
            Column name for y-dimension in DataFrame.
        """
        
        self.xdim = xdim
        self.ydim = ydim
        self.df = df
        self._nbins = nbins
        self._min_max = min_max
        self._folds = folds
        peak_angles = [[*range(360//f, 360//f*(180//(360//f)), 360//f)] for f in folds] 
        peak_angles = [item for sublist in peak_angles for item in sublist]
        trough_angles = [[*range(360//f//2, 360//f*(180//(360//f)), 360//f)] for f in folds] 
        trough_angles = [item for sublist in trough_angles for item in sublist]
        corr_angles = np.sort(peak_angles + trough_angles)
        # these angles allow me to compute all kinds of symmetries later 
        self._corr_angles = corr_angles
        # Create all masks
        self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                                  mask_max))
                       for mask_min, mask_max in mask_parameters]
        # Can change the smoothing width of the gaussian kernel here and see if it changes the results 
        self._smooth_width = smooth_width
        # Mask for hiding the parts of the SAC that are never used
        self._plotting_sac_mask = circle_mask(
            [nbins * 2 - 1,nbins * 2 - 1],
            nbins,
            in_val=1.0,
            out_val=np.nan)
        
        self._trunc = 2/self._smooth_width # Ensures a Gaussian kernel of size 5


    def calculate_ratemap(self):
        """
        Calculate raw and smoothed ratemaps from DataFrame.
        
        Computes 2D binned firing rate maps and applies Gaussian smoothing
        while properly handling NaN values (unvisited bins).
        
        Returns
        -------
        tuple
            (rate_map, smoothmap) where:
            - rate_map: Raw firing rate map (spikes/time)
            - smoothmap: Gaussian-smoothed ratemap
        """
        # This exception is for the visual ratemaps
        if (self.xdim != 'x') and (self.ydim != 'y'):       
            self.df[f'{self.xdim}_bin'] = pd.cut(self.df[self.xdim], bins=self._nbins, labels=np.arange(self._nbins))
            self.df[f'{self.ydim}_bin'] = pd.cut(self.df[self.ydim], bins=self._nbins, labels=np.arange(self._nbins))
            # compute the trial-averaged firing rate in 2d binned space:
            raw_map = self.df.pivot_table(index=f'{self.xdim}_bin', 
                                    columns=f'{self.ydim}_bin', 
                                    values='trial_fr_encoding', dropna=False,
                                          aggfunc=np.nanmean).values #.fillna(0).values
            mask = self.df.pivot_table(index=f'{self.xdim}_bin', 
                                   columns=f'{self.ydim}_bin', 
                                   values='subj', dropna=False,
                                       aggfunc='count')
            
            time_mask = mask * 2 # Turn counts into seconds 
            time_mask = np.nan_to_num(time_mask)
        else:
            self.df['y_bins'] = pd.cut(self.df['y'], bins=self._nbins, labels=np.arange(self._nbins))
            self.df['x_bins'] = pd.cut(self.df['x'], bins=self._nbins, labels=np.arange(self._nbins))
            # compute the trial-averaged firing rate in 2d binned space:
            raw_map = self.df.pivot_table(index='x_bins', 
                                    columns='y_bins', 
                                    values='trial_fr_encoding', dropna=False,
                                          aggfunc=np.nanmean).values #.fillna(0).values
            mask = self.df.pivot_table(index='x_bins', 
                                   columns='y_bins', dropna=False,
                                   values='subj', aggfunc='count')  
            
            time_mask = mask * 0.5 # Turn counts into seconds 
            time_mask = np.nan_to_num(time_mask)

        
        rate_map = raw_map/time_mask
        
        # "A Gaussian filter which ignores NaNs in a given array U can be easily obtained by applying a standard Gaussian filter to two auxiliary arrays V and W and by taking the ratio of the two to get the result Z."

        V=rate_map.copy()
        V[np.isnan(rate_map)]=0
        VV=gaussian_filter(V, sigma=self._smooth_width, mode='constant', truncate=self._trunc)
            
        W=0*rate_map.copy()+1
        W[np.isnan(rate_map)]=0
        WW=gaussian_filter(W, sigma=self._smooth_width, mode='constant', truncate=self._trunc)

        Z=VV/WW
        
        return rate_map, Z
    
    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return (circle_mask(n_points, mask_max * self._nbins) *
                (1 - circle_mask(n_points, mask_min * self._nbins)))
        
        
    def grid_score(self, corr, folds=6):
        """
        Compute grid score for a specific fold symmetry.
        
        Calculates grid score as the difference between peak and trough
        correlations at expected angles for the given fold symmetry.
        
        Parameters
        ----------
        corr : dict
            Dictionary mapping angles (in degrees) to correlation values.
        folds : int, default=6
            Fold symmetry to compute (e.g., 6 for hexagonal grid).
        
        Returns
        -------
        float
            Grid score for the specified fold symmetry.
        """
        # Expected peak correlations of sinusoidal modulation
        peak_correlations = []
        # Expected trough correlations of sinusoidal modulation
        trough_correlations = []

        peak_angles = [*range(360//folds, 360//folds*(180//(360//folds)), 360//folds)]
        trough_angles = [*range(360//folds//2, 360//folds*(180//(360//folds)), 360//folds)]

        for peak in peak_angles: 
            peak_correlations.append(corr[peak])
        for trough in trough_angles: 
            trough_correlations.append(corr[trough])
        
        if self._min_max:
            gridScore = min(peak_correlations) - max(trough_correlations)
        else:
            gridScore = (np.nansum(peak_correlations) / len(peak_correlations)) - (np.nansum(trough_correlations) / len(trough_correlations))

        return gridScore

    def calculate_sac(self, seq1):
        """
        Calculate spatial autocorrelogram (SAC) of a ratemap.
        
        Computes Pearson correlation between the ratemap and itself at all
        spatial lags, properly handling NaN values (unvisited bins).
        
        Parameters
        ----------
        seq1 : np.ndarray
            Input ratemap (can contain NaNs).
        
        Returns
        -------
        np.ndarray
            Spatial autocorrelogram with shape (2*nbins-1, 2*nbins-1).
        """
        seq2 = seq1

        def filter2(b, x):
            stencil = np.rot90(b, 2)
            return convolve2d(x, stencil, mode='full')

        seq1 = np.nan_to_num(seq1)
        seq2 = np.nan_to_num(seq2)

        ones_seq1 = np.ones(seq1.shape)
        ones_seq1[np.isnan(seq1)] = 0
        ones_seq2 = np.ones(seq2.shape)
        ones_seq2[np.isnan(seq2)] = 0

        seq1[np.isnan(seq1)] = 0
        seq2[np.isnan(seq2)] = 0

        seq1_sq = np.square(seq1)
        seq2_sq = np.square(seq2)

        seq1_x_seq2 = filter2(seq1, seq2)
        sum_seq1 = filter2(seq1, ones_seq2)
        sum_seq2 = filter2(ones_seq1, seq2)
        sum_seq1_sq = filter2(seq1_sq, ones_seq2)
        sum_seq2_sq = filter2(ones_seq1, seq2_sq)
        n_bins = filter2(ones_seq1, ones_seq2)
        n_bins_sq = np.square(n_bins)

        std_seq1 = np.power(
            np.subtract(
                np.divide(sum_seq1_sq, n_bins),
                (np.divide(np.square(sum_seq1), n_bins_sq))), 0.5)
        std_seq2 = np.power(
            np.subtract(
                np.divide(sum_seq2_sq, n_bins),
                (np.divide(np.square(sum_seq2), n_bins_sq))), 0.5)
        covar = np.subtract(
            np.divide(seq1_x_seq2, n_bins),
            np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
        x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        x_coef = np.real(x_coef)
        x_coef = np.nan_to_num(x_coef)
        
        return x_coef

    def rotated_sacs(self, sac, angles):
        return [
            scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.nansum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.nansum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.nansum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.nansum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
            
        Gs = {f'{f}':np.nan for f in self._folds}
        for f in self._folds: 
            Gs[f'{f}'] = self.grid_score(corrs, folds=f)
        return Gs 

    def get_scores(self, rate_map):
        """Get summary of scrores for grid cells."""
        sac = self.calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        dd_GS = defaultdict(list)
        # iterate through each possible mask for the autocorrelation to calculate the grid score
        for mask, mask_params in self._masks:
            Gs = self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for key, value in Gs.items():
                dd_GS[key].append(value)   
        
        # Pick the highest grid score for each of the masks 
        max_Gs = {f'{f}':np.nan for f in self._folds}
        max_masks = {f'{f}':np.nan for f in self._folds}
        # utilize the same mask for all the symmetries: 
        max_ind = np.argmax(dd_GS['6'])
        for f in self._folds: 
            # max_ind = np.argmax(dd_GS[f'{f}'])
            max_Gs[f'{f}'] = dd_GS[f'{f}'][max_ind]
            max_masks[f'{f}'] = self._masks[max_ind][1]

        return (max_Gs, max_masks, sac)

    def plot_ratemap(self, ratemap, ax=None, title=None, *args, **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """
        Plot a ratemap.
        
        Parameters
        ----------
        ratemap : np.ndarray
            Firing rate map to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, uses current axes.
        title : str, optional
            Plot title.
        *args, **kwargs
            Additional arguments passed to imshow.
        """
        
        
        if ax is None:
            ax = plt.gca()
        # Plot the ratemap
        ax.imshow(ratemap, *args, **kwargs)
        # ax.pcolormesh(ratemap, *args, **kwargs)
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

    def plot_sac(self,
               sac,
               mask_params=None,
               ax=None,
               title=None,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot spatial autocorrelogram."""
        if ax is None:
            ax = plt.gca()
        # Plot the sac
        useful_sac = sac * self._plotting_sac_mask
        ax.imshow(useful_sac, interpolation='hamming', *args, **kwargs)
        # ax.pcolormesh(useful_sac, *args, **kwargs)
        # Plot a ring for the adequate mask
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
            plt.Circle(
                  (center, center),
                  mask_params[0] * self._nbins,
                  # lw=bump_size,
                  fill=False,
                  edgecolor='k'))
            ax.add_artist(
              plt.Circle(
                  (center, center),
                  mask_params[1] * self._nbins,
                  # lw=bump_size,
                  fill=False,
                  edgecolor='k'))
        ax.axis('off')
        if title is not None:
            ax.set_title(title)


def distance_between_points(point, points_list):
    """
    Calculate Euclidean distances from a point to a list of points.
    
    Parameters
    ----------
    point : tuple or list
        (x, y) coordinates of the reference point.
    points_list : list of tuples or list of lists
        List of (x, y) coordinates to compute distances to.
    
    Returns
    -------
    np.ndarray
        Array of Euclidean distances from point to each point in points_list.
    """
    x1, y1 = point
    distances = []

    for x2, y2 in points_list:
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(distance)

    return np.array(distances)

def find_grid_fields(smoothmap, adj_dist_thresh=2, field_thresh=0.66, peak_thresh_z=1, field_max_thresh=4, gain=2, size_thresh=3):
    """
    Detect firing fields in a smoothed ratemap using clustering and peak detection.
    
    Identifies grid cell firing fields by finding local maxima and clustering
    adjacent high-firing regions. Uses a depth-first search approach to find
    connected components above threshold.
    
    Parameters
    ----------
    smoothmap : np.ndarray
        Smoothed firing rate map.
    adj_dist_thresh : float, default=2
        Minimum distance threshold between field peaks (in bins).
    field_thresh : float, default=0.66
        Threshold for field detection as fraction of maximum rate.
    peak_thresh_z : float, default=1
        Z-score threshold for peak detection.
    field_max_thresh : float, default=4
        Maximum distance from peak for field pixels (in bins).
    gain : float, default=2
        Power to raise the map before field detection (enhances contrast).
    size_thresh : int, default=3
        Minimum number of pixels required for a valid field.
    
    Returns
    -------
    peaks : list of lists
        List of [x, y] coordinates for each detected field peak.
    adjacents : np.ndarray
        Array of arrays, where each array contains [y, x] coordinates
        of pixels belonging to each field.
    """

    def find_clustered_coordinates(arr, threshold=0.33):

        max_value = np.max(arr)

        coordinates = []
        visited = set()

        # Find the maximum value's indices
        max_indices = np.argwhere(arr == max_value)

        # Define possible neighbor directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

        def dfs(x, y):
            if (x, y) in visited:
                return

            visited.add((x, y))
            coordinates.append([y, x])

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < arr.shape[0]
                    and 0 <= ny < arr.shape[1]
                    and arr[nx, ny] > threshold  * max_value
                ):
                    dfs(nx, ny)

        # Iterate over all maximum value indices
        for idx in max_indices:
            x, y = idx
            dfs(x, y)

        return [max_indices[0][1], max_indices[0][0]], coordinates

    squared_map = smoothmap.copy()**gain
    copied_map = squared_map.copy()
    peaks = [] 
    adjacents = [] 

    max_indices, coordinates = find_clustered_coordinates(copied_map, threshold=field_thresh)

    while zscore(squared_map)[max_indices[1], max_indices[0]] > peak_thresh_z:
        max_indices, coordinates = find_clustered_coordinates(copied_map, threshold=field_thresh)
        too_far_coords = np.where(distance_between_points(max_indices, coordinates)>field_max_thresh)[0]
        for index in sorted(too_far_coords, reverse=True):
            del coordinates[index]
        # Check if the returned max_indices are next to an existing coordinate:
        visited = peaks + [item for row in adjacents for item in row]
        all_peak_dists = distance_between_points(max_indices, visited)
        copied_map[max_indices[1], max_indices[0]] = 0
        for coords in coordinates:
            copied_map[coords[1], 
                       coords[0]] = 0
        if all(all_peak_dists > adj_dist_thresh): 
            if len(visited) > 0:
                all_adj_dists = cdist(coordinates, visited)
                all_close_adj = np.unique(np.where(all_adj_dists <= adj_dist_thresh)[0])
                for index in sorted(all_close_adj, reverse=True):
                    del coordinates[index]

            all_peak_adj_dists = distance_between_points(max_indices, coordinates)

            if len(coordinates) >= size_thresh:
                peaks.append(max_indices)
                adjacents.append(coordinates)

    return peaks, np.array(adjacents)

def euclidean_distance(a, b):
    """
    Calculate Euclidean distance between two points.
    
    Parameters
    ----------
    a : array-like
        First point coordinates.
    b : array-like
        Second point coordinates.
    
    Returns
    -------
    float
        Euclidean distance between points a and b.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def move_clusters(smoothmap, centroids, cluster_membership, nbins=15, random_state=None):
    """
    Randomly relocate clusters (firing fields) in a ratemap while preserving structure.
    
    Moves cluster centroids to random positions and relocates cluster members
    using the same displacement vectors. Used for creating spatial surrogates
    that preserve local field structure but randomize global positions.
    
    Parameters
    ----------
    smoothmap : np.ndarray
        Smoothed ratemap containing firing fields.
    centroids : list of tuples
        List of (x, y) coordinates for cluster centroids.
    cluster_membership : list of lists
        List where each element is a list of (x, y) coordinates belonging to that cluster.
    nbins : int, default=15
        Size of the ratemap (assumed square).
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.
    
    Returns
    -------
    np.ndarray
        New ratemap with clusters relocated to random positions.
    """
    size = nbins
    new_array = -999 * np.ones([size, size])
    smmap = smoothmap.copy()

    displacement_vectors = []
    n_members = []

    rng = np.random.default_rng(random_state)

    # Step 1: Move centroids and store displacement vectors
    for cluster_idx, centroid in enumerate(centroids):
        new_x = rng.integers(0, size)
        new_y = rng.integers(0, size)

        displacement = (new_x - centroid[0], new_y - centroid[1])
        displacement_vectors.append(displacement)

        new_array[new_x, new_y] = smoothmap[centroid[0], centroid[1]]
        smmap[centroid[0], centroid[1]] = np.nan

        # Sort cluster members by distance to centroid
        cluster = cluster_membership[cluster_idx]
        try:
            sorted_members = sorted(cluster, key=lambda x: euclidean_distance(x, centroid))
        except TypeError:
            sorted_members = sorted(cluster.tolist(), key=lambda x: euclidean_distance(x, centroid))
        cluster_membership[cluster_idx] = sorted_members
        n_members.append(len(sorted_members))

    # Step 2: Move cluster members using displacement vectors
    max_members = np.max(n_members)
    for pixel in range(max_members):
        for cluster_idx, centroid in enumerate(centroids):
            if pixel >= len(cluster_membership[cluster_idx]):
                continue

            member = cluster_membership[cluster_idx][pixel]

            if np.all(member == centroid):
                continue

            displacement = displacement_vectors[cluster_idx]
            new_coords = (member[0] + displacement[0], member[1] + displacement[1])
            new_coords = (new_coords[0] % size, new_coords[1] % size)

            # Find available coordinates with a buffer
            avail_coords = np.array(list(zip(*np.where(new_array == -999))))
            used_coords = np.array(list(zip(*np.where(new_array != -999))))

            if used_coords.size > 0:
                avail_used_dists = cdist(avail_coords, used_coords)
                all_good_avail = np.unique(np.where(avail_used_dists > 2)[0])
                avail_coords = avail_coords[all_good_avail]

            # Fallback if none meet the distance requirement
            if len(avail_coords) == 0:
                avail_coords = np.array(list(zip(*np.where(new_array == -999))))

            if len(avail_coords) == 0:
                continue  # should be very rare / only in pathological cases

            dists = distance_between_points(new_coords, avail_coords)
            best_coords = tuple(avail_coords[np.argmin(dists)])

            new_array[best_coords[0], best_coords[1]] = smoothmap[member[0], member[1]]
            smmap[member[0], member[1]] = np.nan

    # Step 3: Fill in the remaining values
    fill_coords = list(zip(*np.where(new_array == -999)))
    remain_coords = list(zip(*np.where(~np.isnan(smmap))))

    if len(fill_coords) != len(remain_coords):
        min_len = min(len(fill_coords), len(remain_coords))
        fill_coords = fill_coords[:min_len]
        remain_coords = remain_coords[:min_len]

    for target_coord, source_coord in zip(fill_coords, remain_coords):
        new_array[target_coord[0], target_coord[1]] = smmap[source_coord[0], source_coord[1]]

    return new_array

def compute_spatial_metrics_new(spatial_input, 
                              xdim = 'valence', 
                              ydim = 'arousal',
                              nbins=15,
                              folds=np.arange(4,11), 
                              smooth_width=1,
                              post_stimulus=[0, 2], 
                              mask_parameters=None,
                              leave_one_out=False,
                              filename=None, 
                              foldername=None): 
    """
    Same function as the original with a couple of major changes. 
    
    1. Adds field computation for each cell 
    2. leave one out method leaves one out list and re-computes the method for cell categorization
    3. Remove surrogate methods and stick with label surrogate which can be done outside the function 
    
    In general, I try to clean up and speed up everything as well.
    
    spatial_input can be either encoding_df or a smoothmap. check before doing anything
    """

    if isinstance(spatial_input, pd.DataFrame):
        encoding_df = spatial_input
        
        # compute the trial-averaged firing rate in 2d binned space:
        encoding_df[f'{xdim}_bin'] = pd.cut(encoding_df[xdim], bins=nbins, labels=np.arange(nbins))
        encoding_df[f'{ydim}_bin'] = pd.cut(encoding_df[ydim], bins=nbins, labels=np.arange(nbins))

        # use the pivot table to get the 2d binned firing rate:
        spatial_spiking = encoding_df.pivot_table(index=[f'{xdim}_bin'], 
                                columns=[f'{ydim}_bin'], dropna=False,
                                values='trial_fr_encoding', aggfunc=np.nanmean).values #.fillna(0).values
        
        mask = encoding_df.pivot_table(index=[f'{xdim}_bin'], 
                            columns=[f'{ydim}_bin'], dropna=False,
                            values='subj', aggfunc='count')

        time_mask = mask * (post_stimulus[1]-post_stimulus[0]) # Turn occupancy count into seconds 
        time_mask = np.nan_to_num(time_mask)

        # Initialize grid computer: 
        gc = GridScorer(encoding_df, 
                        xdim=xdim, 
                        ydim=ydim,
                        mask_parameters=mask_parameters, 
                        nbins=nbins, 
                        smooth_width=smooth_width, 
                        min_max=False, 
                        folds=folds)

        ratemap, smoothmap = gc.calculate_ratemap()
        
        # manually compute spatial information 
        rate = np.nansum(spatial_spiking * time_mask) / np.nansum(time_mask)
        # Compute the occupancy probability, per bin
        occ_prob = time_mask / np.nansum(time_mask)

        # use a mask for nonzero values
        nz = np.nonzero(spatial_spiking)
        SI = np.nansum(occ_prob[nz] * spatial_spiking[nz] * np.log2(spatial_spiking[nz] / rate)) / rate
        
    elif isinstance(spatial_input, np.ndarray):
        smoothmap = spatial_input
        
        gc = GridScorer(df=None, 
                        mask_parameters=mask_parameters, 
                        nbins=nbins, 
                        smooth_width=smooth_width, 
                        min_max=False, 
                        folds=folds)
        
        SI = np.nan

    
    # COMPUTE FIELDS AND METRICS
    # ----------------------------------------
    try:
        peaks, fields = find_grid_fields(smoothmap, adj_dist_thresh=2, field_thresh=0.66, peak_thresh_z=1,
                                         field_max_thresh=4, gain=2)
    except:
        peaks = np.nan 
        fields = np.nan

    # Compute grid scores for all kind of modulations [6-fold, 10-fold, 8-fold, 4-fold] 
    gc_info = gc.get_scores(smoothmap)

    Gs = {f'{f}':[] for f in folds}
    max_masks = {f'{f}':[] for f in folds}
    
    for f in folds: 
        Gs[f'{f}'].append(gc_info[0][f'{f}'])
        max_masks[f'{f}'].append(gc_info[1][f'{f}'])
            
    return peaks, fields, SI, Gs, max_masks


def compute_spatial_metrics(encoding_df, 
                   spikeTimes, 
                   ev_times, 
                   nbins=18, 
                   folds = np.arange(4,11), 
                   smooth_width=1, 
                   post_stimulus=[0, 1.5], 
                   surr_num=500, 
                   surr_types=['label', 'circ'], 
                   resid=False,
                   filename=None,
                   foldername=None,
                   random_seed=1):
    
    """
    Compute spatial metrics (spatial information, gridness score, Fourier power) using surrogate techniques.
    
    encoding_df: pandas dataframe from above with data from ONE cell
    
    spikeTimes: array with spike times (s) not exceeding ms resolution (Fs=1000). Used for circ-shuffling
    
    ev_times: array with event times (s). Used to compute circularly shuffled firing rate per event 
    
    nbins: integer denoting the number of bins to slice emotional space up into 
    
    folds: the different symmetries to test for gridness 
    
    smooth_width: float denoting the width of the gaussian smoothing kernel 
    
    post_stimulus: the amount of time (s) after the stimulus to look at firing rate.
    
    surr_num: the number of surrogates to use to establish the spatial metrics
    
    surr_types: list of surrogate methods that we can employ. label = swap labels, circ = rotate the spike train 
    
    resid: if True, use residuals after accounting for effect of emotion on overall firing rate. Default is False
    
    """
    
    starts = [0.2] * 15
    ends = np.linspace(0.4, 1.2, num=15)
    mask_parameters = zip(starts, ends.tolist())
        
    if resid: 
        # Make binned firing rate using the residuals after regressing out category modulations of firing rate 
#         modelcat = smf.ols(formula='zscored_encoding_fr ~ C(category)', data=encoding_df).fit()
#         d = encoding_df.pivot_table(index='spatial_valence_bins', 
#                                     columns='spatial_arousal_bins', 
#                                     values='fr_cat_resid', aggfunc=np.nanmean)
        
#         d = np.abs(d)
#         mask = encoding_df.pivot_table(index='spatial_valence_bins', 
#                                        columns='spatial_arousal_bins', 
#                                        values='trial_fr_encoding', aggfunc='count')

        encoding_df['trial_fr_encoding'] = encoding_df['fr_cat_resid']

    spatial_spiking = encoding_df.pivot_table(index='spatial_valence_bins', 
                            columns='spatial_arousal_bins', dropna=False,
                            values='trial_fr_encoding', aggfunc=np.nanmean).values #.fillna(0).values
    mask = encoding_df.pivot_table(index='spatial_valence_bins', 
                           columns='spatial_arousal_bins', dropna=False,
                           values='subj', aggfunc='count')

    time_mask = mask * (post_stimulus[1]-post_stimulus[0]) # Turn counts into seconds 
    time_mask = np.nan_to_num(time_mask)

    # SI:
    # Calculate average firing rate of the neuron, dividing out by total occupancy time
    # Note: this recomputes total spike (basically, de-normalizing)
    rate = np.nansum(spatial_spiking * time_mask) / np.nansum(time_mask)

    # Compute the occupancy probability, per bin
    occ_prob = time_mask / np.nansum(time_mask)

    # Calculate the spatial information, using a mask for nonzero values
    nz = np.nonzero(spatial_spiking)
    SI = np.nansum(occ_prob[nz] * spatial_spiking[nz] * np.log2(spatial_spiking[nz] / rate)) / rate

    # Gridness:
    # Technique 1: 

    starts = [0.2] * 15
    ends = np.linspace(0.4, 1.2, num=15)
    mask_parameters = zip(starts, ends.tolist())

    # Uses my technique for computing gridness 
    gc = GridScorer(encoding_df, mask_parameters, nbins=nbins, smooth_width=smooth_width, min_max=False, folds=folds)
    ratemap, smoothmap = gc.calculate_ratemap()

    # Compute grid scores for all kind of modulations [6-fold, 10-fold, 8-fold, 4-fold] 
    gc_info = gc.get_scores(smoothmap)

    Gs1 = {f'{f}':[] for f in folds}
    for f in folds: 
        Gs1[f'{f}'].append(gc_info[0][f'{f}'])

    # Technique 2: note that this computes a correlation matrix after circularly rotating the map 6 degrees at at time. 
    correlationMatrix = calculate_correlation_matrix(smoothmap)
    rotations, correlations, Gs2, circularMatrix, threshold = calculate_spatial_periodicity(correlationMatrix, folds=folds)

    # Get rid of the -inf: 
    for f in folds: 
        if np.isneginf(Gs2[f'{f}']):
            Gs2[f'{f}'] = np.nan

    length = np.ceil(encoding_df.session_length.unique()[0])
    fs = 100 # Hz 
    dt = 1/fs
    spike_train = np.zeros(int(length * fs) + 1).astype(int)
    inds = [int(ind * fs) for ind in spikeTimes if ind * fs <= spike_train.shape[-1]]
    spike_train[inds] = 1

    # Get the behavioral indices of every sample point 
    pos_arousal = np.nan * np.zeros_like(spike_train) 
    pos_valence = np.nan * np.zeros_like(spike_train)

    ix = np.arange(len(spike_train))
    for trial, event in enumerate(ev_times): 
        ix_to_keep_event = np.where(np.logical_and((ix/fs)>=(event+post_stimulus[0]), 
                                                   (ix/fs)<=(event+post_stimulus[1])))[0].astype(int)

        pos_arousal[ix_to_keep_event] = encoding_df.loc[trial, 'arousal']
        pos_valence[ix_to_keep_event] = encoding_df.loc[trial, 'valence']

    adjustedRateMap = smoothmap - np.nanmean(smoothmap)
    meanFr = np.nanmean(ratemap)

    fourierSpectrogram = np.fft.fft2(adjustedRateMap, s=[256, 256])
    # normalize by environment size and mean firing rate
    fourierSpectrogram = fourierSpectrogram / (meanFr * math.sqrt(smoothmap.shape[0] * smoothmap.shape[1]))
    fourierSpectrogram = np.fft.fftshift(fourierSpectrogram)
    fourierSpectrogram = np.absolute(fourierSpectrogram)
    # normalize to 0-1 range: 
#         fourierSpectrogram = (fourierSpectrogram - np.min(fourierSpectrogram)) / (np.max(fourierSpectrogram) - np.min(fourierSpectrogram))
    # Should I only take the max in a certain range (i.e. around 3+-1?)
#         print(fourierSpectrogram.shape)

    maxPower = np.nanmax(fourierSpectrogram)

    # initialize space for the surrogate data 
    surr_Gs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    zGs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    pGs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}

    surr_Gs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    zGs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    pGs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}

    surr_SIs = {f'{x}':[] for x in surr_types}
    zSIs = {f'{x}':[] for x in surr_types}
    pSIs = {f'{x}':[] for x in surr_types}

    surr_FPs = {f'{x}':[] for x in surr_types}
    zFPs = {f'{x}':[] for x in surr_types}
    pFPs = {f'{x}':[] for x in surr_types}

    
    for surr_type in surr_types: # 'circ'
        trunc = (((nbins - 1) / 2) - 0.5) / smooth_width
        # iterate through types of surrogate

        if surr_type=='spatial':
            surr_ratemap = np.ones_like(ratemap) * np.nan
            x_y = [(x,y) for x,y in zip(np.where(~np.isnan(ratemap))[0], np.where(~np.isnan(ratemap))[1])]
            vals = [ratemap[x,y] for x,y in zip(np.where(~np.isnan(ratemap))[0], np.where(~np.isnan(ratemap))[1])]
            # make the surrogate array and shuffle it 
            repeated_vals = numpy.matlib.repmat(vals, surr_num, 1)
            shuffled_vals = repeated_vals.copy()
            [np.random.shuffle(x) for x in shuffled_vals]
        elif surr_type=='circ': 
            # use a minimum of 3 second pushes to circularly shift the spike train 
            circ_shuffled_spikes = spike_utils.shuffle_circular(spikeTimes, shuffle_min=1250, n_shuffles=surr_num)

        for surr_i in range(surr_num): 
                
            random_state = np.random.seed(seed=surr_i)

            # iterate through number of surrogates 
            surr_df = encoding_df.copy()
            if surr_type=='label':
                # Surrogate 1: Shuffle trial labels mismatching "space" and firing rate 
                surr_df['trial_fr_encoding'] = surr_df['trial_fr_encoding'].sample(frac=1, random_state=random_state).values
            if surr_type =='sample':
                # 8/15/2024: new type of surrogate samplign random arousal and valence values for each trial from space
                surr_df['valence'] = np.random.uniform(low=encoding_df['valence'].min(), 
                                       high=encoding_df['valence'].max(), 
                                       size=len(surr_df['valence']))
                surr_df['arousal'] = np.random.uniform(low=encoding_df['arousal'].min(), 
                                       high=encoding_df['arousal'].max(), 
                                       size=len(surr_df['arousal']))  
            elif surr_type == 'label_cat': 
                for category in surr_df.category.unique():
                    surr_df.loc[surr_df.category==category, 'trial_fr_encoding'] = surr_df.loc[surr_df.category==category, 'trial_fr_encoding'].sample(frac=1).values
            elif surr_type=='circ': 
                 # Surrogate 2: Compute circ_trial_fr using the event times 
                encoding_fr = []
                for event in ev_times: 
                    st_to_keep_event = np.where(np.logical_and(circ_shuffled_spikes[surr_i, :]>=event+post_stimulus[0], 
                                                               circ_shuffled_spikes[surr_i, :]<=event+post_stimulus[1]))[0].astype(int)
                    encoding_fr.append(len(st_to_keep_event)/(post_stimulus[1]-post_stimulus[0]))

                surr_df['trial_fr_encoding'] = encoding_fr   

            surr_raw_map = surr_df.pivot_table(index='spatial_valence_bins', 
                                    columns='spatial_arousal_bins', dropna=False,
                                    values='trial_fr_encoding', aggfunc=np.nanmean).values #.fillna(0).values
            surr_mask = surr_df.pivot_table(index='spatial_valence_bins', 
                                   columns='spatial_arousal_bins', dropna=False,
                                   values='trial_fr_encoding', aggfunc='count')

            surr_time_mask = surr_mask * (post_stimulus[1]-post_stimulus[0])  # Turn counts into seconds 
            surr_time_mask = np.nan_to_num(surr_time_mask)

            # surr SI
            surr_rate = np.nansum(surr_raw_map * surr_time_mask) / np.nansum(surr_time_mask)

            # Compute the occupancy probability, per bin
            surr_occ_prob = surr_time_mask / np.nansum(surr_time_mask)

            # Calculate the spatial information, using a mask for nonzero values
            surr_nz = np.nonzero(surr_raw_map)
            surr_SI = np.nansum(surr_occ_prob[surr_nz] * surr_raw_map[surr_nz] * np.log2(surr_raw_map[surr_nz] / surr_rate)) / surr_rate
            surr_SIs[f'{surr_type}'].append(surr_SI)


            starts = [0.2] * 15
            ends = np.linspace(0.4, 1.2, num=15)
            mask_parameters = zip(starts, ends.tolist())

            # Uses my technique for computing gridness 
            surr_gc = GridScorer(surr_df, 
                             mask_parameters, 
                             nbins=nbins, 
                             smooth_width=smooth_width, 
                             min_max=False, folds=folds)

            surr_ratemap, surr_smoothmap = surr_gc.calculate_ratemap()
            if surr_type=='spatial': 
#                     countmap = spatial_spiking * time_mask
#                     surr_map = np.zeros_like(countmap)
#                     x_y = [(x,y) for x,y in zip(np.where(mask.values>0)[0], np.where(mask.values>0)[1])]
#                     total_spikes = int(np.nansum(countmap))
#                     random_fill_spots = random.choices(range(len(x_y)), k=total_spikes)
#                     random_fill_count = Counter(random_fill_spots)
#                     for ix, pixel in enumerate(x_y): 
#                         surr_map[pixel[0], pixel[1]] = random_fill_count[ix]
#                     surr_ratemap = surr_map/time_mask
                for ix, pixel in enumerate(x_y):
                    surr_ratemap[pixel[0], pixel[1]] = shuffled_vals[surr_i, :][ix]
                V=surr_ratemap.copy()
                V[np.isnan(surr_ratemap)]=0
                VV=gaussian_filter(V, sigma=smooth_width, mode='constant', truncate=trunc)

                W=0*surr_ratemap.copy()+1
                W[np.isnan(surr_ratemap)]=0
                WW=gaussian_filter(W, sigma=smooth_width, mode='constant', truncate=trunc)

                surr_smoothmap=VV/WW           

                # surr SI
                surr_map = surr_ratemap * time_mask
                surr_rate = np.nansum(surr_map) / np.nansum(time_mask)

                # Compute the occupancy probability, per bin
                surr_occ_prob = time_mask / np.nansum(time_mask)

                # Calculate the spatial information, using a mask for nonzero values
                surr_nz = np.nonzero(surr_ratemap)
                surr_SI = np.nansum(surr_occ_prob[surr_nz] * surr_ratemap[surr_nz] * np.log2(surr_ratemap[surr_nz] / surr_rate)) / surr_rate
                # replace surr_SI 
                surr_SIs[f'{surr_type}'][surr_i] = surr_SI

            # Surrogate Fourier analysis 
#                 surr_spike_train = np.zeros(int(length * fs) + 1).astype(int)
#                 surr_inds = [int(ind * fs) for ind in circ_shuffled_spikes[surr_i, :] if ind * fs <= surr_spike_train.shape[-1]]
#                 surr_spike_train[surr_inds] = 1

            surr_adjustedRateMap = surr_smoothmap - np.nanmean(surr_smoothmap)
            surr_meanFr = np.nanmean(surr_ratemap)

            surr_fourierSpectrogram = np.fft.fft2(surr_adjustedRateMap, s=[256, 256])
            surr_fourierSpectrogram = surr_fourierSpectrogram / (surr_meanFr * math.sqrt(surr_smoothmap.shape[0] * surr_smoothmap.shape[1]))
            surr_fourierSpectrogram = np.fft.fftshift(surr_fourierSpectrogram)
            surr_fourierSpectrogram = np.absolute(surr_fourierSpectrogram)
#                 surr_fourierSpectrogram = (surr_fourierSpectrogram - np.min(surr_fourierSpectrogram)) / (np.max(surr_fourierSpectrogram) - np.min(surr_fourierSpectrogram))


            surr_maxPower = np.nanmax(surr_fourierSpectrogram)

            surr_FPs[f'{surr_type}'].append(surr_maxPower)


            # Compute grid scores for all kind of modulations [6-fold, 10-fold, 8-fold, 4-fold] 
            surr_gc_info = surr_gc.get_scores(surr_smoothmap)

            for f in folds: 
                surr_Gs1[f'{surr_type}'][f'{f}'].append(surr_gc_info[0][f'{f}'])

            # Technique 2: note that this computes a correlation matrix after circularly rotating the map 6 degrees at at time. 
            surr_correlationMatrix = calculate_correlation_matrix(surr_smoothmap)
            try:
                surr_rotations, surr_correlations, sGs2, surr_circularMatrix, surr_threshold = calculate_spatial_periodicity(surr_correlationMatrix, folds=folds)
            except:
                print('failed surrogate')
                continue

            # Get rid of the -inf: 
            for f in folds: 
                if np.isneginf(sGs2[f'{f}']):
                    sGs2[f'{f}'] = np.nan

            for f in folds:
                surr_Gs2[f'{surr_type}'][f'{f}'].append(sGs2[f'{f}'])


        # Compute the zscored true value compared to the surrogates of this type
        for k in zGs1[f'{surr_type}'].keys():
            zGs1[f'{surr_type}'][k].append(((Gs1[k][0] - np.nanmean(surr_Gs1[f'{surr_type}'][k])) / np.nanstd(surr_Gs1[f'{surr_type}'][k])))
            pGs1[f'{surr_type}'][k].append(sum(surr_Gs1[f'{surr_type}'][k] > Gs1[k][0]) / surr_num)

            zGs2[f'{surr_type}'][k].append(((Gs2[k] - np.nanmean(surr_Gs2[f'{surr_type}'][k])) / np.nanstd(surr_Gs2[f'{surr_type}'][k])))
            pGs2[f'{surr_type}'][k].append(sum(np.array(surr_Gs2[f'{surr_type}'][k]) > Gs2[k]) / surr_num)


        zSIs[f'{surr_type}'].append((SI - np.nanmean(surr_SIs[f'{surr_type}'])) / np.nanstd(surr_SIs[f'{surr_type}']))
        pSIs[f'{surr_type}'].append(sum(surr_SIs[f'{surr_type}'] > SI) / surr_num)

        zFPs[f'{surr_type}'].append((maxPower - np.nanmean(surr_FPs[f'{surr_type}'])) / np.nanstd(surr_FPs[f'{surr_type}']))
        pFPs[f'{surr_type}'].append(sum(surr_FPs[f'{surr_type}'] > maxPower) / surr_num)

                
    Gs_df1_z = pd.DataFrame.from_dict(zGs1).reset_index().rename(columns={'index':'symmetry'})
    Gs_df1_p = pd.DataFrame.from_dict(pGs1).reset_index().rename(columns={'index':'symmetry'})
    
    Gs_df2_z = pd.DataFrame.from_dict(zGs2).reset_index().rename(columns={'index':'symmetry'})
    Gs_df2_p = pd.DataFrame.from_dict(pGs2).reset_index().rename(columns={'index':'symmetry'})
        
    SI_df1 = pd.DataFrame.from_dict(zSIs)
    SI_df2 = pd.DataFrame.from_dict(pSIs)
    
    FP_df1 = pd.DataFrame.from_dict(zFPs)
    FP_df2 = pd.DataFrame.from_dict(pFPs)
    
    for surr_type in surr_types:

        Gs_df1_z.rename(columns={f'{surr_type}': f'zG_{surr_type}_shift'}, inplace=True)
        Gs_df1_z[f'zG_{surr_type}_shift'] = Gs_df1_z[f'zG_{surr_type}_shift'].apply(lambda x: x[0])
        Gs_df1_p.rename(columns={f'{surr_type}': f'pG_{surr_type}_shift'}, inplace=True)
        Gs_df1_p[f'pG_{surr_type}_shift'] = Gs_df1_p[f'pG_{surr_type}_shift'].apply(lambda x: x[0])
        
        Gs_df2_z.rename(columns={f'{surr_type}': f'zG_{surr_type}_shift'}, inplace=True)
        Gs_df2_z[f'zG_{surr_type}_shift'] = Gs_df2_z[f'zG_{surr_type}_shift'].apply(lambda x: x[0])
        Gs_df2_p.rename(columns={f'{surr_type}': f'pG_{surr_type}_shift'}, inplace=True)
        Gs_df2_p[f'pG_{surr_type}_shift'] = Gs_df2_p[f'pG_{surr_type}_shift'].apply(lambda x: x[0])
        

        SI_df1.rename(columns={f'{surr_type}': f'zSI_{surr_type}_shift'}, inplace=True)
        SI_df2.rename(columns={f'{surr_type}': f'pSI_{surr_type}_shift'}, inplace=True)
        
        FP_df1.rename(columns={f'{surr_type}': f'zFP_{surr_type}_shift'}, inplace=True)
        FP_df2.rename(columns={f'{surr_type}': f'pFP_{surr_type}_shift'}, inplace=True)

    Gs_df1 = Gs_df1_z.merge(Gs_df1_p)
    Gs_df1['raw_Gs'] = np.nan
    for key in Gs1.keys():
        Gs_df1.raw_Gs[Gs_df1.symmetry==key] = Gs1[key]
    
    Gs_df2 = Gs_df2_z.merge(Gs_df2_p)
    Gs_df2['raw_Gs'] = np.nan
    for key in Gs2.keys():
        Gs_df2.raw_Gs[Gs_df2.symmetry==key] = Gs2[key]
    
    SI_df = SI_df1.merge(SI_df2, right_index=True, left_index=True)
    
    FP_df = FP_df1.merge(FP_df2, right_index=True, left_index=True)
    
    
#     plot_canonical_scoring_method(smoothmap, np.nanmean(ratemap), correlationMatrix, rotations, correlations, Gs1['6'], circularMatrix, threshold,
#                                   filename, foldername)   
        
#     isPeriodic = FP_df['pFP_label_shift'][0]<=0.05
#     plot_fourier_scoring(fourierSpectrogram, fourierSpectrogram, maxPower, isPeriodic, 
#                          filename, foldername)
    
    return Gs_df1, Gs_df2, SI_df, FP_df




def compute_spatial_metrics_visual(distractor_df, 
                   spikeTimes, 
                   ev_times, 
                   nbins=15, 
                   folds = np.arange(4,11), 
                   smooth_width=1, 
                   post_stimulus=[0, 1.5], 
                   surr_num=500, 
                   surr_types=['label', 'circ'], 
                   resid=False,
                   filename=None,
                   foldername=None):
    
    """
    Compute spatial metrics (spatial information, gridness score, Fourier power) using surrogate techniques.
    
    distractor_df: pandas dataframe from above with data from ONE cell
    
    spikeTimes: array with spike times (s) not exceeding ms resolution (Fs=1000). Used for circ-shuffling
    
    ev_times: array with event times (s). Used to compute circularly shuffled firing rate per event 
    
    nbins: integer denoting the number of bins to slice emotional space up into 
    
    folds: the different symmetries to test for gridness 
    
    smooth_width: float denoting the width of the gaussian smoothing kernel 
    
    post_stimulus: the amount of time (s) after the stimulus to look at firing rate.
    
    surr_num: the number of surrogates to use to establish the spatial metrics
    
    surr_types: list of surrogate methods that we can employ. label = swap labels, circ = rotate the spike train 
    
    resid: if True, use residuals after accounting for effect of emotion on overall firing rate. Default is False
    
    """
    
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    mask_parameters = zip(starts, ends.tolist())

    spatial_spiking = distractor_df.pivot_table(index='x_bins', 
                            columns='y_bins', dropna=False,
                            values='trial_fr_encoding', aggfunc=np.nanmean).values #.fillna(0).values
    mask = distractor_df.pivot_table(index='x_bins', 
                           columns='y_bins', dropna=False,
                           values='trial_fr_encoding', aggfunc='count')

    time_mask = mask * (post_stimulus[1]-post_stimulus[0]) # Turn counts into seconds 
    time_mask = np.nan_to_num(time_mask)

    # SI:
    # Calculate average firing rate of the neuron, dividing out by total occupancy time
    # Note: this recomputes total spike (basically, de-normalizing)
    rate = np.nansum(spatial_spiking * time_mask) / np.nansum(time_mask)

    # Compute the occupancy probability, per bin
    occ_prob = time_mask / np.nansum(time_mask)

    # Calculate the spatial information, using a mask for nonzero values
    nz = np.nonzero(spatial_spiking)
    SI = np.nansum(occ_prob[nz] * spatial_spiking[nz] * np.log2(spatial_spiking[nz] / rate)) / rate

    # Gridness:
    # Technique 1: 

    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)
    mask_parameters = zip(starts, ends.tolist())

    # Uses my technique for computing gridness 
    gc = GridScorer(distractor_df, mask_parameters, nbins=nbins, smooth_width=smooth_width, min_max=False, folds=folds)
    ratemap, smoothmap = gc.calculate_ratemap()

    # Compute grid scores for all kind of modulations [6-fold, 10-fold, 8-fold, 4-fold] 
    gc_info = gc.get_scores(smoothmap)

    Gs1 = {f'{f}':[] for f in folds}
    for f in folds: 
        Gs1[f'{f}'].append(gc_info[0][f'{f}'])

    # Technique 2: note that this computes a correlation matrix after circularly rotating the map 6 degrees at at time. 
    correlationMatrix = calculate_correlation_matrix(smoothmap)
    rotations, correlations, Gs2, circularMatrix, threshold = calculate_spatial_periodicity(correlationMatrix, folds=folds)

    # Get rid of the -inf: 
    for f in folds: 
        if np.isneginf(Gs2[f'{f}']):
            Gs2[f'{f}'] = np.nan

    # Calculate the Fourier spectrum: 

    # Get the spiketrain

#         length = np.max(spikeTimes)
    length = np.ceil(distractor_df.session_length.unique()[0])
    fs = 100 # Hz 
    dt = 1/fs
    spike_train = np.zeros(int(length * fs) + 1).astype(int)
    inds = [int(ind * fs) for ind in spikeTimes if ind * fs <= spike_train.shape[-1]]
    spike_train[inds] = 1

    # Get the behavioral indices of every sample point 
    pos_y = np.nan * np.zeros_like(spike_train) 
    pos_x = np.nan * np.zeros_like(spike_train)

    ix = np.arange(len(spike_train))
    for trial, event in enumerate(ev_times): 
        ix_to_keep_event = np.where(np.logical_and((ix/fs)>=(event+post_stimulus[0]), 
                                                   (ix/fs)<=(event+post_stimulus[1])))[0].astype(int)

        pos_y[ix_to_keep_event] = distractor_df.loc[trial, 'y']
        pos_x[ix_to_keep_event] = distractor_df.loc[trial, 'x']

    adjustedRateMap = smoothmap - np.nanmean(smoothmap)
    meanFr = np.nanmean(ratemap)

    fourierSpectrogram = np.fft.fft2(adjustedRateMap, s=[256, 256])
    # normalize by environment size and mean firing rate
    fourierSpectrogram = fourierSpectrogram / (meanFr * math.sqrt(smoothmap.shape[0] * smoothmap.shape[1]))
    fourierSpectrogram = np.fft.fftshift(fourierSpectrogram)
    fourierSpectrogram = np.absolute(fourierSpectrogram)
    # normalize to 0-1 range: 
    fourierSpectrogram = (fourierSpectrogram - np.min(fourierSpectrogram)) / (np.max(fourierSpectrogram) - np.min(fourierSpectrogram))

    maxPower = np.max(fourierSpectrogram)

    # use a minimum of 3 second pushes to circularly shift the spike train 
    circ_shuffled_spikes = spike_utils.shuffle_circular(spikeTimes, shuffle_min=1250, n_shuffles=surr_num)

    # initialize space for the surrogate data 
    surr_Gs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    zGs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    pGs1 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}

    surr_Gs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    zGs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}
    pGs2 = {f'{x}': {f'{f}':[] for f in folds} for x in surr_types}

    surr_SIs = {f'{x}':[] for x in surr_types}
    zSIs = {f'{x}':[] for x in surr_types}
    pSIs = {f'{x}':[] for x in surr_types}

    surr_FPs = {f'{x}':[] for x in surr_types}
    zFPs = {f'{x}':[] for x in surr_types}
    pFPs = {f'{x}':[] for x in surr_types}

    for surr_type in surr_types: # 'circ'
        # iterate through types of surrogate
        for surr_i in range(surr_num): 
            # iterate through number of surrogates 
            surr_df = distractor_df.copy()
            if surr_type=='label':
                # Surrogate 1: Shuffle trial labels mismatching "space" and firing rate 
                surr_df['trial_fr_encoding'] = surr_df['trial_fr_encoding'].sample(frac=1).values
            elif surr_type=='circ': 
                 # Surrogate 2: Compute circ_trial_fr using the event times 
                distractor_fr = []
                for event in ev_times: 
                    st_to_keep_event = np.where(np.logical_and(circ_shuffled_spikes[surr_i, :]>=event+post_stimulus[0], 
                                                               circ_shuffled_spikes[surr_i, :]<=event+post_stimulus[1]))[0].astype(int)
                    distractor_fr.append(len(st_to_keep_event)/(post_stimulus[1]-post_stimulus[0]))

                surr_df['trial_fr_encoding'] = distractor_fr   

            surr_raw_map = surr_df.pivot_table(index='x_bins', 
                                    columns='y_bins', dropna=False,
                                    values='trial_fr_encoding', aggfunc=np.nanmean).values #.fillna(0).values
            surr_mask = surr_df.pivot_table(index='x_bins', 
                                   columns='y_bins', dropna=False,
                                   values='trial_fr_encoding', aggfunc='count')

            surr_time_mask = surr_mask * (post_stimulus[1]-post_stimulus[0])  # Turn counts into seconds 
            surr_time_mask = np.nan_to_num(surr_time_mask)

            # surr SI
            surr_rate = np.nansum(surr_raw_map * surr_time_mask) / np.nansum(surr_time_mask)

            # Compute the occupancy probability, per bin
            surr_occ_prob = surr_time_mask / np.nansum(surr_time_mask)

            # Calculate the spatial information, using a mask for nonzero values
            surr_nz = np.nonzero(surr_raw_map)
            surr_SI = np.nansum(surr_occ_prob[surr_nz] * surr_raw_map[surr_nz] * np.log2(surr_raw_map[surr_nz] / surr_rate)) / surr_rate
            surr_SIs[f'{surr_type}'].append(surr_SI)


            starts = [0.2] * 10 # This just establishes a mask around which to compute the SAC. 
            ends = np.linspace(0.4, 1.0, num=10)
            mask_parameters = zip(starts, ends.tolist())

            # Uses my technique for computing gridness 
            surr_gc = GridScorer(surr_df, 
                             mask_parameters, 
                             nbins=nbins, 
                             smooth_width=smooth_width, 
                             min_max=False, folds=folds)

            surr_ratemap, surr_smoothmap = surr_gc.calculate_ratemap()

            # Surrogate Fourier analysis 
            surr_spike_train = np.zeros(int(length * fs) + 1).astype(int)
            surr_inds = [int(ind * fs) for ind in circ_shuffled_spikes[surr_i, :] if ind * fs <= surr_spike_train.shape[-1]]
            surr_spike_train[surr_inds] = 1

            surr_adjustedRateMap = surr_smoothmap - np.nanmean(surr_smoothmap)
            surr_meanFr = np.nanmean(surr_ratemap)

            surr_fourierSpectrogram = np.fft.fft2(surr_adjustedRateMap, s=[256, 256])
            surr_fourierSpectrogram = surr_fourierSpectrogram / (surr_meanFr * math.sqrt(surr_smoothmap.shape[0] * surr_smoothmap.shape[1]))
            surr_fourierSpectrogram = np.fft.fftshift(surr_fourierSpectrogram)
            surr_fourierSpectrogram = np.absolute(surr_fourierSpectrogram)
            fourierSpectrogram = (surr_fourierSpectrogram - np.min(surr_fourierSpectrogram)) / (np.max(surr_fourierSpectrogram) - np.min(surr_fourierSpectrogram))


            surr_maxPower = np.max(surr_fourierSpectrogram)

            surr_FPs[f'{surr_type}'].append(surr_maxPower)


            # Compute grid scores for all kind of modulations [6-fold, 10-fold, 8-fold, 4-fold] 
            surr_gc_info = surr_gc.get_scores(surr_smoothmap)

            for f in folds: 
                surr_Gs1[f'{surr_type}'][f'{f}'].append(surr_gc_info[0][f'{f}'])

            # Technique 2: note that this computes a correlation matrix after circularly rotating the map 6 degrees at at time. 
            surr_correlationMatrix = calculate_correlation_matrix(surr_smoothmap)
            try:
                surr_rotations, surr_correlations, sGs2, surr_circularMatrix, surr_threshold = calculate_spatial_periodicity(surr_correlationMatrix, folds=folds)
            except:
                print('failed surrogate')
                continue

            # Get rid of the -inf: 
            for f in folds: 
                if np.isneginf(sGs2[f'{f}']):
                    sGs2[f'{f}'] = np.nan

            for f in folds:
                surr_Gs2[f'{surr_type}'][f'{f}'].append(sGs2[f'{f}'])


        # Compute the zscored true value compared to the surrogates of this type
        for k in zGs1[f'{surr_type}'].keys():
            zGs1[f'{surr_type}'][k].append(((Gs1[k][0] - np.nanmean(surr_Gs1[f'{surr_type}'][k])) / np.nanstd(surr_Gs1[f'{surr_type}'][k])))
            pGs1[f'{surr_type}'][k].append(sum(surr_Gs1[f'{surr_type}'][k] > Gs1[k][0]) / surr_num)

            zGs2[f'{surr_type}'][k].append(((Gs2[k] - np.nanmean(surr_Gs2[f'{surr_type}'][k])) / np.nanstd(surr_Gs2[f'{surr_type}'][k])))
            pGs2[f'{surr_type}'][k].append(sum(np.array(surr_Gs2[f'{surr_type}'][k]) > Gs2[k]) / surr_num)


        zSIs[f'{surr_type}'].append((SI - np.nanmean(surr_SIs[f'{surr_type}'])) / np.nanstd(surr_SIs[f'{surr_type}']))
        pSIs[f'{surr_type}'].append(sum(surr_SIs[f'{surr_type}']> SI) / surr_num)

        zFPs[f'{surr_type}'].append((maxPower - np.nanmean(surr_FPs[f'{surr_type}'])) / np.nanstd(surr_FPs[f'{surr_type}']))
        pFPs[f'{surr_type}'].append(sum(surr_FPs[f'{surr_type}'] > maxPower) / surr_num)

    Gs_df1_z = pd.DataFrame.from_dict(zGs1).reset_index().rename(columns={'index':'symmetry'})
    Gs_df1_p = pd.DataFrame.from_dict(pGs1).reset_index().rename(columns={'index':'symmetry'})
        
    Gs_df2_z = pd.DataFrame.from_dict(zGs2).reset_index().rename(columns={'index':'symmetry'})
    Gs_df2_p = pd.DataFrame.from_dict(pGs2).reset_index().rename(columns={'index':'symmetry'})
        
    SI_df1 = pd.DataFrame.from_dict(zSIs)
    SI_df2 = pd.DataFrame.from_dict(pSIs)
    
    FP_df1 = pd.DataFrame.from_dict(zFPs)
    FP_df2 = pd.DataFrame.from_dict(pFPs)
    
    for surr_type in surr_types:

        Gs_df1_z.rename(columns={f'{surr_type}': f'zG_{surr_type}_shift'}, inplace=True)
        Gs_df1_z[f'zG_{surr_type}_shift'] = Gs_df1_z[f'zG_{surr_type}_shift'].apply(lambda x: x[0])
        Gs_df1_p.rename(columns={f'{surr_type}': f'pG_{surr_type}_shift'}, inplace=True)
        Gs_df1_p[f'pG_{surr_type}_shift'] = Gs_df1_p[f'pG_{surr_type}_shift'].apply(lambda x: x[0])
        
        Gs_df2_z.rename(columns={f'{surr_type}': f'zG_{surr_type}_shift'}, inplace=True)
        Gs_df2_z[f'zG_{surr_type}_shift'] = Gs_df2_z[f'zG_{surr_type}_shift'].apply(lambda x: x[0])
        Gs_df2_p.rename(columns={f'{surr_type}': f'pG_{surr_type}_shift'}, inplace=True)
        Gs_df2_p[f'pG_{surr_type}_shift'] = Gs_df2_p[f'pG_{surr_type}_shift'].apply(lambda x: x[0])
        

        SI_df1.rename(columns={f'{surr_type}': f'zSI_{surr_type}_shift'}, inplace=True)
        SI_df2.rename(columns={f'{surr_type}': f'pSI_{surr_type}_shift'}, inplace=True)
        
        FP_df1.rename(columns={f'{surr_type}': f'zFP_{surr_type}_shift'}, inplace=True)
        FP_df2.rename(columns={f'{surr_type}': f'pFP_{surr_type}_shift'}, inplace=True)

    Gs_df1 = Gs_df1_z.merge(Gs_df1_p)
    Gs_df2 = Gs_df2_z.merge(Gs_df2_p)
    
    SI_df = SI_df1.merge(SI_df2, right_index=True, left_index=True)
    
    FP_df = FP_df1.merge(FP_df2, right_index=True, left_index=True)
    
    return Gs_df1, Gs_df2, SI_df, FP_df


## Phase precession: 

@numba.jit(nopython=True)
def pcorrelate(t, u, bins):
    """
    From : https://github.com/OpenSMFS/pycorrelate

    Compute correlation of two arrays of discrete events (Point-process).

    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.

    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.

    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    return G

def fast_acf(counts, width, bin_width, cut_peak=True):
        
    """
    Super fast ACF function relying on numba (above). 

    Parameters
    ----------
    cut_peak : bool
        Whether or not the largest central peak should be replaced for subsequent fitting
    counts : 1d array 
        Variable of interest (i.e. spike times or spike phases)
    width: float
        Time window for ACF
    bin_width: float
        Width of bins 

    Returns
    ----------
    acf: 1d array
        Counts for ACF
    bins: 1d array
        Lag bins for ACF

    Notes
    -----
    """
    def moving_average(a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    n_b = int(np.ceil(width / bin_width))  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    bins = np.linspace(-width, width, 2 * n_b, endpoint=True)
    temp = pcorrelate(counts, counts, np.split(bins, 2)[1])
    acf = np.ones(bins.shape[0] - 1)
    acf[0:temp.shape[0]] = np.flip(temp)
    acf[temp.shape[0]] = temp[0]
    acf[temp.shape[0] + 1:] = temp

    if cut_peak==True:
        acf[np.nanargmax(acf)] = np.sort(acf)[-2]
        
    bins = moving_average(bins, n=2)

    return acf, bins

def acf_power(acf, norm=True):
    
    """
    Compute the power spectrum of the signal by computing the FFT of the autocorrelation. 

    Parameters
    ----------
    acf: 1d array
        Counts for ACF

    norm: bool
        To normalize or not

    Returns
    ----------
    psd: 1d array
        Power spectrum 

    Notes
    -----
    """

    # Take the FFT
    fft = pyfftw.interfaces.numpy_fft.fft(acf, threads=multiprocessing.cpu_count())

    # Compute the power from the real component squared
    pow = (np.abs(fft) ** 2)
    
    # Account for nyquist
    psd = pow[0:round(pow.shape[0] / 2)]
    
    # normalize
    if norm:
        psd = psd / np.trapz(psd)

    return psd


def nonspatial_phase_precession(unwrapped_spike_phases, 
                                width=4 * 2 * np.pi, 
                                bin_width=np.pi/3, cut_peak=True, norm=True, psd_lims = [0.65, 1.55]):

    """
    Compute the nonspatial spike-LFP relationship modulation index.

    Parameters
    ----------
    unwrapped_spike_phases : 1d array 
        Spike phases that have been linearly unwrapped
    width: float
        Time window for ACF in cycles (default = 4 cycles)
    bin_width: float
        Width of bins in radians (default = 60 degrees)
    cut_peak : bool
        Whether or not the largest central peak should be replaced for subsequent fitting
    norm: bool
        To normalize the ACF or not
    Returns
    ----------
    max_freq: float
        Relative spike-LFP frequency of PSD peak

    MI: float
        Modulation index of non-spatial phase relationship 


    Notes
    -----
    """

    frequencies = (np.arange(2 * (width // bin_width) - 1)) * (2 * np.pi) / (
                2 * width - bin_width)    

    freqs_of_interest = np.intersect1d(np.where(frequencies> psd_lims[0]),
                                       np.where(frequencies<psd_lims[1]))

    acf, _ = fast_acf(unwrapped_spike_phases, width, bin_width, cut_peak=cut_peak)
    psd = acf_power(acf, norm=norm)
    all_peaks = find_peaks(psd[freqs_of_interest], None)[0]  # FIND ALL LOCAL MAXIMA IN WINDOW

    # make sure there is a peak.... .
    if ~np.any(all_peaks):
        return np.nan, np.nan

    max_peak = np.max(psd[freqs_of_interest][all_peaks])
    max_idx = [all_peaks[np.argmax(psd[freqs_of_interest][all_peaks])]]
    max_freq = frequencies[freqs_of_interest][max_idx]
    MI = max_peak / np.trapz(psd[freqs_of_interest])

    return max_freq, MI



from bycycle.cyclepoints import find_extrema, extrema_interpolated_phase, find_zerox
# Import filter function
from neurodsp.filt import filter_signal

def get_phase(lfp, freqs=(1, 11), fs=500, N_seconds=0.1, N_seconds_theta=1.5, threshold=25):
    """
    This function should use linear interpolation (bycycle) to get the phase of the lfp between a particular
    frequency band.

    :param N_seconds_theta: Not totally sure what this parameter is for (bycycle)
    :param fs: sampling rate (Hz)
    :param N_seconds: Not totally sure what this parameter is for (bycycle)
    :param lfp: The raw local field potential data. Typically loaded from binary file as int16 array
    :param freqs: tuple containing the frequency range for the phase
    :param threshold: if you want to return the indices where power exceeds a percentile threshold (i.e 25)
    :return: phase: the interpolated phase within the frequency selected
    :return: good_indices: bool array indicating where we have LFP > pow threshold
    """

    fast_hilbert = lambda x: scipy.signal.hilbert(x, next_fast_len(len(x)))[:len(x)]
    
    
    sos = butter(4, 30, btype="lowpass", output="sos", fs=fs)
    lfp_filtered = sosfiltfilt(sos, lfp)

#     # Lowpass filter < 30 Hz This is a fixed value based on the assumption I only care about the low freqeuency stuff
#     lfp_filtered = filter_signal(lfp, fs, 'lowpass', 30, n_seconds=N_seconds, remove_edges=False)
    # Band pass filter in the band I care about
    
    sos = butter(4, freqs, btype="bandpass", output="sos", fs=fs)
    lfp_narrowed = sosfiltfilt(sos, lfp)
    
#     lfp_narrowed = filter_signal(lfp, fs, 'bandpass', freqs, n_seconds=N_seconds_theta, remove_edges=False)

    sos = butter(4, (25, 80), btype="bandpass", output="sos", fs=fs)
    lfp_filt_artifact = sosfiltfilt(sos, lfp)
#     lfp_filt_artifact = filter_signal(lfp, fs, 'bandpass', (25, 80), n_cycles=3, remove_edges=False)

    # Compute the hilbert phase
    analytic_signal = fast_hilbert(lfp_narrowed)
    hilbert_phase = np.angle(analytic_signal)
    # unwrap phase
    hilbert_phase[~np.isnan(hilbert_phase)] = np.unwrap(hilbert_phase[~np.isnan(hilbert_phase)])

    # Compute the interpolated phase
    # Add buffer
    lfp_filt = np.insert(lfp_filtered, 0, lfp_filtered[0:int(N_seconds_theta * fs)])
    lfp_filt = np.append(lfp_filt, lfp_filt[-int(N_seconds_theta * fs):])
    lfp_narr = np.insert(lfp_narrowed, 0, lfp_narrowed[0:int(N_seconds_theta * fs)])
    lfp_narr = np.append(lfp_narr, lfp_narr[-int(N_seconds_theta * fs):])

    # Determine extrema
    lfp_filt = lfp_filt - np.nanmean(lfp_filt)  # demean so we have actual zero-crossings
    Ps, Ts = find_extrema(lfp_filt, fs, freqs, filter_kwargs={'n_seconds':N_seconds_theta})

    zeroxR, zeroxD = find_zerox(lfp_filt, Ps, Ts)

    # Do the linear interpolated phase:
    interp_phase = extrema_interpolated_phase(lfp_filt, Ps, Ts, zeroxR, zeroxD)

    # remove buffer
    interp_phase = interp_phase[int(N_seconds_theta * fs):-int(N_seconds_theta * fs)]

    # unwrap phase
    interp_phase[~np.isnan(interp_phase)] = np.unwrap(interp_phase[~np.isnan(interp_phase)])
    
    # find epileptic signal (from Gelinas paper)
    envelope = np.abs(fast_hilbert(lfp_filt_artifact))
    envelope_std = np.nanstd(envelope)
    envelope_mean = np.nanmean(envelope)
    epileptic_discharge = envelope > (envelope_mean + (4 * envelope_std))
    hilbert_phase[epileptic_discharge] = np.nan
    interp_phase[epileptic_discharge] = np.nan

    # find low power and high power phase estimates=
    if threshold > 0:
        lfp_power = np.abs(analytic_signal) ** 2
        pow_threshold = np.percentile(lfp_power, threshold)
        low_amp = lfp_power < pow_threshold
        high_amp = lfp_power > pow_threshold
    else:
        low_amp = np.nan
        high_amp = np.nan

    return {'interp_phase': interp_phase,
            'hilbert_phase': hilbert_phase,
            'low_amplitude': low_amp,
            'high_amplitude': high_amp}


def IED_detect_1Darray(signal, peak_thresh=4, closeness_thresh=0.25, width_thresh=0.2, fs=250):
    """
    Detect interictal epileptiform discharges (IEDs) in a 1D signal.
    
    Filters signal in beta-gamma band, detects peaks above threshold,
    and filters out IEDs that are too wide or too small.
    
    Parameters
    ----------
    signal : np.ndarray
        1D signal array to analyze.
    peak_thresh : float, default=4
        Z-score threshold for peak detection.
    closeness_thresh : float, default=0.25
        Minimum time (seconds) between IEDs.
    width_thresh : float, default=0.2
        Maximum width (seconds) for valid IEDs.
    fs : float, default=250
        Sampling frequency in Hz.
    
    Returns
    -------
    np.ndarray
        Array of IED timestamps in seconds.
    """
    
    
    fast_hilbert = lambda x: sp.signal.hilbert(x, next_fast_len(len(x)))[:len(x)]

#     n_times = len(signal)
    
    min_width = width_thresh * fs
    across_chan_threshold_samps = closeness_thresh * fs # This sets a threshold for detecting cross-channel IEDs 

    # filter data in beta-gamma band
    filt_ord = 4
    sos = butter(filt_ord, [25, 80], btype="bandpass", output="sos", fs=fs)
    filtered_data = sosfiltfilt(sos, signal)
    
#     filtered_data = mne_data.copy().filter(25, 80, n_jobs=-1)

#     n_fft = next_fast_len(n_times)

    # Hilbert bandpass amplitude 
    filtered_data = fast_hilbert(filtered_data)

    # Rectify: 
    filtered_data[filtered_data < 0] = 0 
#     filtered_data = np.abs(filtered_data)

    # Zscore
    filtered_data = zscore(filtered_data) 
#     filtered_data.apply_function(lambda x: zscore(x, axis=-1))

    # Find peaks 
    IED_samps, _ = sp.signal.find_peaks(filtered_data, 
                              height=peak_thresh, prominence=2, distance=closeness_thresh * fs)

    # Remove lame IEDs 
    widths = sp.signal.peak_widths(filtered_data, IED_samps, rel_height=0.5)
    wide_IEDs = np.where(widths[0] > min_width)[0]

    # 2. Too small 
    # Which IEDs are below 3 in z-scored unfiltered signal? 
    small_IEDs = np.where(zscore(signal)[IED_samps] < peak_thresh)[0]
    local_IEDs = [] 
    
    elim_IEDs = np.unique(np.hstack([small_IEDs, wide_IEDs]))
    revised_IED_samps = np.delete(IED_samps, elim_IEDs)
    IED_s = (revised_IED_samps / fs)
          
    return IED_s

def lfp_sta(spikes, lfp, bound):
    """
    Compute spike-triggered average (STA) of LFP signal.
    
    Averages LFP signal around each spike time to reveal phase locking
    or other spike-LFP relationships.
    
    Parameters
    ----------
    spikes : np.ndarray
        Spike times in milliseconds.
    lfp : np.ndarray
        LFP signal to average (can be filtered or unfiltered).
    bound : int
        Time window in milliseconds around each spike (+-bound).
    
    Returns
    -------
    tuple
        (sta, ste) where:
        - sta: Spike-triggered average
        - ste: Standard error of the mean
    """  

    num_spikes = len(spikes) 
    lfp_pre_avg = np.nan * np.ones([num_spikes,(2*bound)+1]) 
    for sidx in range(0,num_spikes):  
        if (math.ceil(spikes[sidx]) - bound <= 0):
            print('This spike occurs too close to the start') 
            sta = np.nanmean(lfp_pre_avg,0) 
            ste = np.nanstd(lfp_pre_avg,0) / np.sqrt(len(sta)) 
            # return sta, ste
            continue
        elif (math.floor(spikes[sidx]) + bound) >= len(lfp):
            print('This spike occurs too close to the end') 
            sta = np.nanmean(lfp_pre_avg,0) 
            ste = np.nanstd(lfp_pre_avg,0) / np.sqrt(len(sta)) 
            # return sta, ste
            continue
        idx1 = math.ceil(spikes[sidx]) - bound
        idx2 = math.floor(spikes[sidx]) + bound + 1
        lfp_pre_avg[sidx,:] = lfp[idx1:idx2] #  - nanmean(raw_lfp(idx1:idx2)); % subtract the mean of the signal 

    sta = np.nanmean(lfp_pre_avg,0) 
    ste = np.nanstd(lfp_pre_avg,0) / np.sqrt(len(sta)) 
    return sta, ste

from tqdm import tqdm
import numpy as np
import pandas as pd

def _distance(coord1, coord2):
    """
    Calculate Euclidean distance between two 2D coordinates.
    
    Parameters
    ----------
    coord1 : array-like
        First coordinate (x, y).
    coord2 : array-like
        Second coordinate (x, y).
    
    Returns
    -------
    float
        Euclidean distance between the two coordinates.
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def compute_path_efficiency(trajectory):
    """
    Compute path efficiency metrics for a trajectory.
    Returns dictionary with various efficiency measures.
    """
    if len(trajectory) < 2:
        return {
            'total_path_length': np.nan,
            'direct_distance': np.nan,
            'path_efficiency': np.nan,
            'normalized_path_length': np.nan
        }
    
    trajectory = np.array(trajectory)
    
    # Total path length (sum of consecutive distances)
    total_path_length = sum(_distance(trajectory[i], trajectory[i+1]) 
                           for i in range(len(trajectory)-1))
    
    # Direct distance (straight line from start to end)
    direct_distance = _distance(trajectory[0], trajectory[-1])
    
    # Path efficiency (direct distance / total path length)
    # Higher values = more efficient (straighter path)
    path_efficiency = direct_distance / total_path_length if total_path_length > 0 else np.nan
    
    # Normalized path length compared to random walk
    # Calculate average distance between all possible pairs in trajectory
    all_distances = [_distance(trajectory[i], trajectory[j]) 
                    for i in range(len(trajectory)) 
                    for j in range(i+1, len(trajectory))]
    avg_pairwise_distance = np.mean(all_distances) if all_distances else np.nan
    
    # Expected path length for random walk through these points
    expected_random_path = avg_pairwise_distance * (len(trajectory) - 1)
    
    # Normalized path length (actual / expected random)
    # Lower values = more efficient than random
    normalized_path_length = total_path_length / expected_random_path if expected_random_path > 0 else np.nan
    
    return {
        'total_path_length': total_path_length,
        'direct_distance': direct_distance,
        'path_efficiency': path_efficiency,
        'normalized_path_length': normalized_path_length
    }

def compute_null_distribution(trajectories, n_permutations=1000, random_state=None):
    """
    Compute null distribution of path efficiency metrics by permuting trajectories.
    
    Args:
        trajectories: List of trajectories (each is a list of coordinates)
        n_permutations: Number of random permutations to generate
        random_state: Random state for reproducibility
    
    Returns:
        Dictionary with null distributions for each metric
    """
    rng = np.random.default_rng(random_state)
    null_metrics = {
        'total_path_length': [],
        'direct_distance': [],
        'path_efficiency': [],
        'normalized_path_length': []
    }
    
    for _ in range(n_permutations):
        # Create random permutation of trajectory order
        shuffled_trajectories = rng.permutation(trajectories).tolist()
        
        # Compute metrics for shuffled trajectories
        for traj in shuffled_trajectories:
            metrics = compute_path_efficiency(traj)
            for key in null_metrics:
                null_metrics[key].append(metrics[key])
    
    return null_metrics

def compute_null_distribution(features, distance_func, n_permutations=1000):
    """
    Compute null distribution of clustering scores for a given feature space.
    
    Generates random permutations of features and computes clustering scores
    to establish a null distribution for statistical testing.
    
    Parameters
    ----------
    features : list
        List of feature values (positions for temporal, coordinates for spatial).
    distance_func : callable
        Function to compute distance between two features.
        Should take two arguments and return a scalar distance.
    n_permutations : int, default=1000
        Number of random permutations to generate.
    
    Returns
    -------
    np.ndarray
        Array of clustering scores under null hypothesis (random permutations).
    """
    null_scores = []
    
    for _ in range(n_permutations):
        # Create random permutation of the features
        shuffled_indices = np.random.permutation(len(features))
        shuffled_features = [features[i] for i in shuffled_indices]
        
        # Compute clustering scores for this random sequence
        available_indices = set(range(len(features)))
        transition_scores = []
        
        for ix in range(len(shuffled_features) - 1):
            current_feature = shuffled_features[ix]
            next_feature = shuffled_features[ix + 1]
            
            # Find and remove current item index
            current_item_idx = None
            for i, orig_feature in enumerate(features):
                if i in available_indices:
                    if distance_func == abs:  # Temporal case
                        if orig_feature == current_feature:
                            current_item_idx = i
                            break
                    else:  # Spatial cases
                        if np.allclose(orig_feature, current_feature):
                            current_item_idx = i
                            break
            
            if current_item_idx is not None:
                available_indices.discard(current_item_idx)
            
            # Calculate actual transition distance
            actual_distance = distance_func(current_feature, next_feature)
            
            # Calculate all possible distances to remaining items
            possible_distances = [distance_func(current_feature, features[i]) 
                                for i in available_indices]
            
            if len(possible_distances) > 0:
                # Rank the actual distance (0 = closest, 1 = furthest)
                rank = sum(1 for d in possible_distances if d < actual_distance) / len(possible_distances)
                transition_scores.append(rank)
        
        # Store mean clustering score for this permutation
        if transition_scores:
            null_scores.append(np.mean(transition_scores))
    
    return np.array(null_scores)


def compute_recall_clustering(all_encoding_df, all_retrieval_df, min_list_size=3, n_permutations=100):
    """
    Compute clustering metrics for memory recall sequences across multiple feature spaces.
    
    Analyzes how recalled items cluster in temporal, semantic, and emotional spaces
    compared to null distributions. Computes path efficiency and transition scores
    for different feature dimensions.
    
    Parameters
    ----------
    all_encoding_df : pd.DataFrame
        DataFrame containing encoding trial data with features (valence, arousal, PC1, PC2, etc.).
    all_retrieval_df : pd.DataFrame
        DataFrame containing retrieval trial data with recall order information.
    min_list_size : int, default=3
        Minimum number of unique recalled items required to analyze a list.
    n_permutations : int, default=100
        Number of permutations for null distribution computation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with clustering metrics for each subject/list/feature combination.
        Includes raw scores, z-scores, null distributions, and path efficiency metrics.
    """
    results = []

    for subj in tqdm(all_retrieval_df.subj.unique()): 
        subj_enc_df = all_encoding_df[all_encoding_df.subj==subj]
        subj_retr_df = all_retrieval_df[all_retrieval_df.subj==subj].sort_values(by=['list', 'recall_num'])
        subj_df = subj_retr_df.copy()
        subj_df = subj_df[['subj', 'list', 'image_name']].groupby(['subj','list']).agg(tuple).applymap(list).reset_index()
        recall_orders = subj_enc_df.merge(subj_retr_df, on=['subj', 'list', 'image_name']).drop_duplicates(subset=['image_name'], keep='first').groupby(['subj','list']).agg(tuple).applymap(list).image_num

        for list_ix in subj_df.list.unique():
            list_df = subj_retr_df[subj_retr_df.list==list_ix]
            recalls = subj_df[subj_df.list==list_ix].image_name.iloc[0]
            unique_recalls = list(set(filter(lambda x: x == x , recalls)))
            num_unique_recalls = len(unique_recalls)
            recall_performance = num_unique_recalls/12

            if num_unique_recalls < min_list_size:
                continue

            # Collect features for recalled items
            valence_for_recalls = []
            arousal_for_recalls = [] 
            arousal_valence_for_recalls = []
            pc1_pc2_for_recalls = []
            order_for_recalls = []
            time_to_recall = []

            for word in unique_recalls:
                time_to_recall.append(list_df.loc[list_df.image_name==word].time_between_recalls)
                valence_for_recalls.append(all_encoding_df.zValence[all_encoding_df.image_name==word].iloc[0])
                arousal_for_recalls.append(all_encoding_df.zArousal[all_encoding_df.image_name==word].iloc[0])
                arousal_valence_for_recalls.append((all_encoding_df.zValence[all_encoding_df.image_name==word].iloc[0],
                                                        all_encoding_df.zArousal[all_encoding_df.image_name==word].iloc[0]))
                pc1_pc2_for_recalls.append((all_encoding_df.zPC1[all_encoding_df.image_name==word].iloc[0],
                                                        all_encoding_df.zPC2[all_encoding_df.image_name==word].iloc[0]))
                order_for_recalls.append(subj_enc_df.image_num[(subj_enc_df.list==list_ix) & (subj_enc_df.image_name==word)].iloc[0])

            # Convert to arrays
            arousal_valence_for_recalls = np.array(arousal_valence_for_recalls)
            pc1_pc2_for_recalls = np.array(pc1_pc2_for_recalls)
            order_for_recalls = np.array(order_for_recalls)
            time_to_recall = np.array(time_to_recall)

            # Get encoding features for all items in this list
            list_ix_enc_df = all_encoding_df[(all_encoding_df.subj==subj) & (all_encoding_df.list==list_ix)]
            
            arousal_for_encoding = list_ix_enc_df.zArousal.tolist()
            valence_for_encoding = list_ix_enc_df.zValence.tolist()
            arousal_valence_for_encoding = [(x,y) for (x,y) in zip(list_ix_enc_df.zValence.tolist(), list_ix_enc_df.zArousal.tolist())]
            pc1_pc2_for_encoding = [(x,y) for (x,y) in zip(list_ix_enc_df.zPC1.tolist(), list_ix_enc_df.zPC2.tolist())]
            encoding_order = list_ix_enc_df.image_num.tolist()

            # Compute null distributions for each clustering type
            print(f"Computing null distributions for subj {subj}, list {list_ix}...")
            
            # Temporal null distribution
            temporal_null = compute_null_distribution(
                encoding_order, 
                lambda x, y: abs(x - y), 
                n_permutations
            )
            
            # Semantic null distribution  
            semantic_null = compute_null_distribution(
                pc1_pc2_for_encoding, 
                _distance, 
                n_permutations
            )
            
            # Emotional null distribution
            emotional_null = compute_null_distribution(
                arousal_valence_for_encoding, 
                _distance, 
                n_permutations
            )
            
            # Arousal null distribution (1D)
            arousal_null = compute_null_distribution(
                arousal_for_encoding, 
                lambda x, y: abs(x - y), 
                n_permutations
            )
            
            # Valence null distribution (1D)
            valence_null = compute_null_distribution(
                valence_for_encoding, 
                lambda x, y: abs(x - y), 
                n_permutations
            )

            # Compute path efficiency metrics for different trajectory types
            # 1. Temporal trajectory (order of recall in encoding sequence)
            temporal_trajectory = [(i, 0) for i in order_for_recalls]  # 1D temporal as 2D
            temporal_efficiency = compute_path_efficiency(temporal_trajectory)
            
            # 2. Semantic trajectory
            semantic_efficiency = compute_path_efficiency(pc1_pc2_for_recalls)
            
            # 3. Emotional trajectory  
            emotional_efficiency = compute_path_efficiency(arousal_valence_for_recalls)

            # Get the actual recall sequence (with positions)
            recalled_positions = recall_orders.loc[:, list_ix].values[0]
            
            # TEMPORAL CLUSTERING with sampling without replacement
            available_positions = set(encoding_order)
            temporal_transition_scores = []
            
            for ix in range(len(recalled_positions) - 1):
                current_pos = recalled_positions[ix]
                next_pos = recalled_positions[ix + 1]
                
                # Remove current position from available set
                available_positions.discard(current_pos)
                
                # Calculate actual transition distance
                actual_distance = abs(next_pos - current_pos)
                
                # Calculate all possible distances to remaining positions
                possible_distances = [abs(pos - current_pos) for pos in available_positions]
                
                if len(possible_distances) > 0:
                    # Rank the actual distance (0 = closest, 1 = furthest)
                    rank = sum(1 for d in possible_distances if d < actual_distance) / len(possible_distances)
                    temporal_transition_scores.append(rank)

            # SEMANTIC CLUSTERING with sampling without replacement
            available_indices = set(range(len(pc1_pc2_for_encoding)))
            semantic_transition_scores = []
            
            for ix in range(len(pc1_pc2_for_recalls) - 1):
                current_coord = pc1_pc2_for_recalls[ix]
                next_coord = pc1_pc2_for_recalls[ix + 1]
                
                # Find and remove current item index
                current_item_idx = None
                for i, enc_coord in enumerate(pc1_pc2_for_encoding):
                    if i in available_indices and np.allclose(enc_coord, current_coord):
                        current_item_idx = i
                        break
                
                if current_item_idx is not None:
                    available_indices.discard(current_item_idx)
                
                # Calculate actual transition distance
                actual_distance = _distance(current_coord, next_coord)
                
                # Calculate all possible distances to remaining items
                possible_distances = [_distance(current_coord, pc1_pc2_for_encoding[i]) 
                                    for i in available_indices]
                
                if len(possible_distances) > 0:
                    # Rank the actual distance (0 = closest, 1 = furthest)
                    rank = sum(1 for d in possible_distances if d < actual_distance) / len(possible_distances)
                    semantic_transition_scores.append(rank)

            # EMOTIONAL CLUSTERING with sampling without replacement
            available_indices = set(range(len(arousal_valence_for_encoding)))
            emotional_transition_scores = []
            arousal_transition_scores = []
            valence_transition_scores = []
            
            for ix in range(len(arousal_valence_for_recalls) - 1):
                current_coord = arousal_valence_for_recalls[ix]
                next_coord = arousal_valence_for_recalls[ix + 1]
                current_arousal = arousal_for_recalls[ix]
                next_arousal = arousal_for_recalls[ix + 1]
                current_valence = valence_for_recalls[ix]
                next_valence = valence_for_recalls[ix + 1]
                
                # Find and remove current item index
                current_item_idx = None
                for i, enc_coord in enumerate(arousal_valence_for_encoding):
                    if i in available_indices and np.allclose(enc_coord, current_coord):
                        current_item_idx = i
                        break
                
                if current_item_idx is not None:
                    available_indices.discard(current_item_idx)
                
                # Calculate actual distances
                actual_emotion_distance = _distance(current_coord, next_coord)
                actual_arousal_distance = abs(next_arousal - current_arousal)
                actual_valence_distance = abs(next_valence - current_valence)
                
                # Calculate all possible distances to remaining items
                possible_emotion_distances = [_distance(current_coord, arousal_valence_for_encoding[i]) 
                                            for i in available_indices]
                possible_arousal_distances = [abs(current_arousal - arousal_for_encoding[i]) 
                                            for i in available_indices]
                possible_valence_distances = [abs(current_valence - valence_for_encoding[i]) 
                                            for i in available_indices]
                
                if len(possible_emotion_distances) > 0:
                    # Rank the actual distances (0 = closest, 1 = furthest)
                    emotion_rank = sum(1 for d in possible_emotion_distances if d < actual_emotion_distance) / len(possible_emotion_distances)
                    arousal_rank = sum(1 for d in possible_arousal_distances if d < actual_arousal_distance) / len(possible_arousal_distances)
                    valence_rank = sum(1 for d in possible_valence_distances if d < actual_valence_distance) / len(possible_valence_distances)
                    
                    emotional_transition_scores.append(emotion_rank)
                    arousal_transition_scores.append(arousal_rank)
                    valence_transition_scores.append(valence_rank)

            # Calculate raw clustering scores
            raw_temporal_score = np.nanmean(temporal_transition_scores) if temporal_transition_scores else np.nan
            raw_semantic_score = np.nanmean(semantic_transition_scores) if semantic_transition_scores else np.nan
            raw_emotional_score = np.nanmean(emotional_transition_scores) if emotional_transition_scores else np.nan
            raw_arousal_score = np.nanmean(arousal_transition_scores) if arousal_transition_scores else np.nan
            raw_valence_score = np.nanmean(valence_transition_scores) if valence_transition_scores else np.nan
            
            # Calculate z-scores against null distributions
            temporal_zscore = (raw_temporal_score - np.mean(temporal_null)) / np.std(temporal_null) if not np.isnan(raw_temporal_score) and np.std(temporal_null) > 0 else np.nan
            semantic_zscore = (raw_semantic_score - np.mean(semantic_null)) / np.std(semantic_null) if not np.isnan(raw_semantic_score) and np.std(semantic_null) > 0 else np.nan
            emotional_zscore = (raw_emotional_score - np.mean(emotional_null)) / np.std(emotional_null) if not np.isnan(raw_emotional_score) and np.std(emotional_null) > 0 else np.nan
            arousal_zscore = (raw_arousal_score - np.mean(arousal_null)) / np.std(arousal_null) if not np.isnan(raw_arousal_score) and np.std(arousal_null) > 0 else np.nan
            valence_zscore = (raw_valence_score - np.mean(valence_null)) / np.std(valence_null) if not np.isnan(raw_valence_score) and np.std(valence_null) > 0 else np.nan

            # Create base row data
            base_row = {
                'subj': subj,
                'list': list_ix,
                'recall_performance': recall_performance,
                'temporal_trajectory': temporal_trajectory,
                'semantic_trajectory': pc1_pc2_for_recalls.tolist(),
                'emotional_trajectory': arousal_valence_for_recalls.tolist(),
                'time_to_recall': time_to_recall.tolist()
            }

            # Add rows for each clustering type in long format
            clustering_results = [
                {
                    'clustering_type': 'temporal',
                    'transition_scores': temporal_transition_scores,
                    'clustering_score_raw': raw_temporal_score,
                    'clustering_score_zscore': temporal_zscore,
                    'null_mean': np.mean(temporal_null),
                    'null_std': np.std(temporal_null),
                    'path_total_length': temporal_efficiency['total_path_length'],
                    'path_direct_distance': temporal_efficiency['direct_distance'],
                    'path_efficiency': temporal_efficiency['path_efficiency'],
                    'path_normalized_length': temporal_efficiency['normalized_path_length']
                },
                {
                    'clustering_type': 'semantic',
                    'transition_scores': semantic_transition_scores,
                    'clustering_score_raw': raw_semantic_score,
                    'clustering_score_zscore': semantic_zscore,
                    'null_mean': np.mean(semantic_null),
                    'null_std': np.std(semantic_null),
                    'path_total_length': semantic_efficiency['total_path_length'],
                    'path_direct_distance': semantic_efficiency['direct_distance'],
                    'path_efficiency': semantic_efficiency['path_efficiency'],
                    'path_normalized_length': semantic_efficiency['normalized_path_length']
                },
                {
                    'clustering_type': 'emotional',
                    'transition_scores': emotional_transition_scores,
                    'clustering_score_raw': raw_emotional_score,
                    'clustering_score_zscore': emotional_zscore,
                    'null_mean': np.mean(emotional_null),
                    'null_std': np.std(emotional_null),
                    'path_total_length': emotional_efficiency['total_path_length'],
                    'path_direct_distance': emotional_efficiency['direct_distance'],
                    'path_efficiency': emotional_efficiency['path_efficiency'],
                    'path_normalized_length': emotional_efficiency['normalized_path_length']
                },
                {
                    'clustering_type': 'arousal',
                    'transition_scores': arousal_transition_scores,
                    'clustering_score_raw': raw_arousal_score,
                    'clustering_score_zscore': arousal_zscore,
                    'null_mean': np.mean(arousal_null),
                    'null_std': np.std(arousal_null),
                    'path_total_length': np.nan,  # 1D, no meaningful path
                    'path_direct_distance': np.nan,
                    'path_efficiency': np.nan,
                    'path_normalized_length': np.nan
                },
                {
                    'clustering_type': 'valence',
                    'transition_scores': valence_transition_scores,
                    'clustering_score_raw': raw_valence_score,
                    'clustering_score_zscore': valence_zscore,
                    'null_mean': np.mean(valence_null),
                    'null_std': np.std(valence_null),
                    'path_total_length': np.nan,  # 1D, no meaningful path
                    'path_direct_distance': np.nan,
                    'path_efficiency': np.nan,
                    'path_normalized_length': np.nan
                }
            ]

            # Add each clustering type as a separate row
            for cluster_result in clustering_results:
                row = {**base_row, **cluster_result}
                results.append(row)

    return pd.DataFrame(results)





def convert_times_to_train(spikes, fs=1000, time_range=None):
    """Convert spike times into a binary spike train.

    Parameters
    ----------
    spikes : 1d array   
        Spike times, in seconds.
    fs : int, optional, default: 1000
        The sampling rate to use for the computed spike train, in Hz.
    time_range : list of [float, float], optional
        Expected time range of the spikes, used to infer the length of the output spike train.
        If not provided, the length is set as the observed time range of 'spikes'.

    Returns
    -------
    spike_train : 1d array
        Spike train.

    Examples
    --------
    Convert spike times into a corresponding binary spike train:

    >>> spikes = np.array([0.002, 0.250, 0.500, 0.750, 1.000, 1.250, 1.500, 2.000])
    >>> convert_times_to_train(spikes)
    array([0, 0, 1, ..., 0, 0, 1])
    """

    if not time_range:
        time_range = [np.floor(spikes[0]), np.ceil(spikes[-1])]

    length = time_range[1] - time_range[0]

    spike_train = np.zeros(int(length * fs) + 1).astype(int)
    inds = [int(ind * fs) for ind in spikes - time_range[0] if ind * fs <= spike_train.shape[-1]]
    spike_train[inds] = 1

    # Check that the spike times are fully encoded into the spike train
    msg = ("The spike times were not fully encoded into the spike train. " \
           "This probably means the spike sampling rate is too low to encode " \
           "spikes close together in time. Try increasing the sampling rate.")
    if not sum(spike_train) == len(spikes):
        raise ValueError(msg)

    return spike_train


def bin_data(dataframe, x_dim='valence', y_dim='arousal', metric='nspikes', nbins=None): 
    """
    Given a dataframe, bin the metric according to the x_dim and y_dim, with an equal number
    of nbins for each dimension.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to bin.
    x_dim : str
        The column name to bin along the x-axis.
    y_dim : str
        The column name to bin along the y-axis.
    metric : str
        The column name to bin.
    nbins : int
        The number of bins to use.

    Returns
    -------
    binned_data : pd.DataFrame
        A dataframe with the binned data.
    """

    x_bins = pd.cut(dataframe[x_dim], nbins)
    y_bins = pd.cut(dataframe[y_dim], nbins)

    dataframe[f'{x_dim}_binned'] = x_bins
    dataframe[f'{y_dim}_binned'] = y_bins

    # return two dataframes: one that applies sums the number of spikes (metric) 

    binned_data = dataframe.groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({metric: np.nansum}).reset_index()
    binned_data['count'] = dataframe.groupby([f'{x_dim}_binned', f'{y_dim}_binned']).size().reset_index()[0]

    return binned_data

def generate_maps(dataframe, x_dim='valence', y_dim='arousal', metric='nspikes', nbins=None, trial_time_s=None):
    """
    
    Return the firing rate map and the occupancy map.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to bin.
    x_dim : str
        The column name to bin along the x-axis.
    y_dim : str
        The column name to bin along the y-axis.
    metric : str
        The column name to bin.
    nbins : int
        The number of bins to use.
    trial_time_s : float
        The trial time in seconds.

    Returns
    -------
    spike_map : pd.DataFrame
        The spike map.
    occupancy_map : pd.DataFrame
        The occupancy map.

    """

    binned_data = bin_data(dataframe, x_dim=x_dim, y_dim=y_dim, metric=metric, nbins=nbins)

    spike_map = binned_data.pivot(f'{y_dim}_binned', f'{x_dim}_binned', metric) 
    occupancy_map  = binned_data.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 'count') * trial_time_s

    return spike_map.T, occupancy_map.T

def make_smoothing_window(nbins=None, ratio=0.325):
    """
    
    Make a 2D hamming window for smoothing.

    Parameters
    ----------
    nbins : int
        The number of bins.
    ratio : float
        The ratio of the hamming window.

    """
    
    nhamming = int(nbins * ratio)
    w = np.outer(np.hamming(nhamming), np.hamming(nhamming))
    w = w / np.sum(w)

    return w
    
def nanconv(a, k, *args):
    """

    Convolve two 2D arrays, with support for NaN values. 
    This function is a modified version of the nanconv function in MATLAB.

    https://www.mathworks.com/matlabcentral/fileexchange/41961-nanconv

    Parameters
    ----------
    a : np.array
        The first array.
    k : np.array
        The second array (filter).
    *args : list
        Additional arguments.
    
    Returns
    -------
    c : np.array
        The convolved array.
    
    """
    edge = True
    nanout = False
    is1D = False
    shape = 'same'

    if shape != 'same':
        raise NotImplementedError(f"Shape '{shape}' not implemented")

    sza = a.shape

    if is1D:
        if not (np.isscalar(a) or np.isscalar(k)):
            raise ValueError('A and B must be vectors.')
        a = np.atleast_2d(a).T
        k = np.atleast_2d(k).T

    o = np.ones_like(a)
    on = np.ones_like(a)
    n = np.isnan(a)

    a = np.where(n, 0, a)
    on = np.where(n, 0, on)

    if np.any(np.isnan(k)):
        raise ValueError('Filter (k) contains NaN values.')

    if np.any(n) or edge:
        flat = convolve2d(on, k, mode=shape)
    else:
        flat = o

    if np.any(n) and not edge:
        flat = flat / convolve2d(o, k, mode=shape)

    c = convolve2d(a, k, mode=shape) / flat

    if nanout:
        c[n] = np.nan

    if is1D and sza[0] == 1:
        c = c.T

    return c

def smooth_map(rate_map, sigma=1, truncate=2):

    """
    Smooth a spike map and occupancy map using a 2D hamming window.

    Parameters
    ----------
    spike_map : pd.DataFrame
        The spike map.
    occupancy_map : pd.DataFrame
        The occupancy map.
    nbins : int
        The number of bins.
    ratio : float  
        The ratio of the hamming window.
    min_occup : float
        The minimum occupancy.
    return_SI : bool
        Whether to return the spatial information.

    Returns
    -------
    fr_map_smoothed : np.array
        The smoothed firing rate map.
    info : float
        The spatial information.

    """
    
#     w = make_smoothing_window(nbins=nbins, ratio=ratio)
    
# #     # smooth the spiking
#     spike_map_smoothed = nanconv(spike_map, w)
# #     # smooth the occupancy
#     occupancy_map_smoothed = nanconv(occupancy_map, w)

#     # then compute the ratio
#     fr_map_smoothed = spike_map_smoothed / occupancy_map_smoothed

#     # fr_map_smoothed = spike_map / occupancy_map
#     # fr_map_smoothed = nanconv(fr_map_smoothed, w, 'edge')

    # rate_map = spike_map / occupancy_map
    V=rate_map.copy()
    V[np.isnan(rate_map)]=0
    VV=gaussian_filter(V, sigma=sigma, mode='constant', truncate=truncate)

    W=0*rate_map.copy()+1
    W[np.isnan(rate_map)]=0
    WW=gaussian_filter(W, sigma=sigma, mode='constant', truncate=truncate)

    fr_map_smoothed=VV/WW

    # # # #Require a minimum time spent in this region
    # fr_map_smoothed[occupancy_map<=min_occup] = np.nan
    # occupancy_map_smoothed[occupancy_map_smoothed<=min_occup] = np.nan

#     if return_SI:
        
# #         rate = np.nansum(rate_map * occupancy_map) / np.nansum(occupancy_map)

# #         # Compute the occupancy probability, per bin
# #         occ_prob = occupancy_map / np.nansum(occupancy_map)

# #         # Calculate the spatial information, using a mask for nonzero values
# #         nz = np.nonzero(rate_map)
# #         info = np.nansum(occ_prob[nz] * rate_map[nz] * np.log2(rate_map[nz] / rate)) / rate

    
#         # # Spatial information per spike (note: this is bits/spike, not bits per second like in Skaggs et al. 1993).
#         # f = ~np.isnan(fr_map_smoothed)
#         # h = spike_map_smoothed[f]
#         # t = occupancy_map_smoothed[f]
#         # lambdai = h/t
#         # p_i = t/np.sum(t)
#         # lambda_ = np.sum(p_i*lambdai)
#         # f = lambdai>0
#         # info = np.sum(p_i[f]*(lambdai[f]/lambda_)*np.log2(lambdai[f]/lambda_))

#         return fr_map_smoothed, info

#     else:

    return fr_map_smoothed

# Adapted from https://github.com/google-deepmind/grid-cells/tree/d1a2304d9a54e5ead676af577a38d4d87aa73041

def calculate_sac(fr_map_smoothed):
    """Calculating spatial autocorrelogram.
    
    Parameters
    ----------
    fr_map_smoothed : np.array
        The smoothed firing rate map.
    
    Returns
    -------
    x_coef : np.array
        The spatial autocorrelogram.
    
    """
    fr_map_smoothed_2 = fr_map_smoothed.copy()

    def filter2(b, x):
        stencil = np.rot90(b, 2)
        return convolve2d(x, stencil, mode='full')

    # fills NaNs with 0s
    fr_map_smoothed = np.nan_to_num(fr_map_smoothed)
    fr_map_smoothed_2 = np.nan_to_num(fr_map_smoothed_2)
    
    # make a map of ones, fill NaNs with 0s 
    ones_fr_map_smoothed = np.ones(fr_map_smoothed.shape)
    ones_fr_map_smoothed[np.isnan(fr_map_smoothed)] = 0
    ones_fr_map_smoothed_2 = np.ones(fr_map_smoothed_2.shape)
    ones_fr_map_smoothed_2[np.isnan(fr_map_smoothed_2)] = 0

    fr_map_smoothed[np.isnan(fr_map_smoothed)] = 0
    fr_map_smoothed_2[np.isnan(fr_map_smoothed_2)] = 0

    fr_map_smoothed_sq = np.square(fr_map_smoothed)
    fr_map_smoothed_2_sq = np.square(fr_map_smoothed_2)

    fr_map_smoothed_x_fr_map_smoothed_2 = filter2(fr_map_smoothed, fr_map_smoothed_2)
    sum_fr_map_smoothed = filter2(fr_map_smoothed, ones_fr_map_smoothed_2)
    sum_fr_map_smoothed_2 = filter2(ones_fr_map_smoothed, fr_map_smoothed_2)
    sum_fr_map_smoothed_sq = filter2(fr_map_smoothed_sq, ones_fr_map_smoothed_2)
    sum_fr_map_smoothed_2_sq = filter2(ones_fr_map_smoothed, fr_map_smoothed_2_sq)
    n_bins = filter2(ones_fr_map_smoothed, ones_fr_map_smoothed_2)
    n_bins_sq = np.square(n_bins)

    std_fr_map_smoothed = np.power(
        np.subtract(
            np.divide(sum_fr_map_smoothed_sq, n_bins),
            (np.divide(np.square(sum_fr_map_smoothed), n_bins_sq))), 0.5)
    std_fr_map_smoothed_2 = np.power(
        np.subtract(
            np.divide(sum_fr_map_smoothed_2_sq, n_bins),
            (np.divide(np.square(sum_fr_map_smoothed_2), n_bins_sq))), 0.5)
    covar = np.subtract(
        np.divide(fr_map_smoothed_x_fr_map_smoothed_2, n_bins),
        np.divide(np.multiply(sum_fr_map_smoothed, sum_fr_map_smoothed_2), n_bins_sq))
    x_coef = np.divide(covar, np.multiply(std_fr_map_smoothed, std_fr_map_smoothed_2))
    x_coef = np.real(x_coef)
    x_coef = np.nan_to_num(x_coef)
    
    return x_coef




class GridCell(object):
    """
    Class to calculate grid scores for a given rate map.
    
    Similar to GridScorer but designed for use with pre-computed ratemaps
    rather than DataFrames. Provides methods for computing grid scores
    across multiple fold symmetries.
    """

    def __init__(self, nmasks=10, mask_end=1.2, folds=np.arange(2,11), nbins=None):
        
        # Smallest annulus will always be [0.2, 0.4], but let user specify number of masks and mask end
        self.nmasks = nmasks
        self.mask_end = mask_end
        starts = [0.2] * self.nmasks
        ends = np.linspace(0.4, self.mask_end, num=self.nmasks)
        self.mask_parameters = zip(starts, ends.tolist())

        self._folds = folds
        self._nbins = nbins
        # peak_angles = [[*range(360//f, 360//f*(180//(360//f)), 360//f)] for f in folds] 
        # peak_angles = [item for sublist in peak_angles for item in sublist]
        # trough_angles = [[*range(360//f//2, 360//f*(180//(360//f)), 360//f)] for f in folds] 
        # trough_angles = [item for sublist in trough_angles for item in sublist]
        peak_angles = []
        trough_angles = []
        for f in folds:
            if f == 2:
                peak_angles += [180] 
                trough_angles += [90]
            elif f == 3:
                peak_angles += [120]
                trough_angles += [60]
            else: 
                peak_angles += [*range(360//f, 360//f*(180//(360//f)), 360//f)]
                trough_angles += [*range(360//f//2, 360//f*(180//(360//f)), 360//f)]
                
        # peak_angles = [angle
        #        for f in folds
        #        for angle in range(360 // f, 181, 360 // f)]
        # trough_angles = [angle
        #          for f in folds
        #          for angle in range((360 // f) // 2, 181, 360 // f)]

        corr_angles = np.sort(peak_angles + trough_angles)
        self._corr_angles = corr_angles
        # Create all masks
        self._masks = [(self._get_ring_mask(mask_min, mask_max), (mask_min,
                                                                  mask_max))
                       for mask_min, mask_max in self.mask_parameters]

    @staticmethod
    def _circle_mask(size, radius, in_val=1.0, out_val=0.0):
        """Calculating the grid scores with different radius."""
        sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
        x = np.linspace(-sz[0], sz[1], size[1])
        x = np.expand_dims(x, 0)
        x = x.repeat(size[0], 0)
        y = np.linspace(-sz[0], sz[1], size[1])
        y = np.expand_dims(y, 1)
        y = y.repeat(size[1], 1)
        z = np.sqrt(x**2 + y**2)
        z = np.less_equal(z, radius)
        vfunc = np.vectorize(lambda b: b and in_val or out_val)
        return vfunc(z)

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self._nbins * 2 - 1, self._nbins * 2 - 1]
        return (self._circle_mask(n_points, mask_max * self._nbins) *
                (1 - self._circle_mask(n_points, mask_min * self._nbins)))
    
    def grid_score(self, corr, folds=None):
        """
        Compute all symmetries. I wrote this in a sexy way!!! 
        """
        # Expected peak correlations of sinusoidal modulation
        peak_correlations = []
        # Expected trough correlations of sinusoidal modulation
        trough_correlations = []

        if folds == 2:
            peak_angles = [180] 
            trough_angles = [90]
        elif folds == 3:
            peak_angles = [120]
            trough_angles = [60]
        else: 
            peak_angles = [*range(360//folds, 360//folds*(180//(360//folds)), 360//folds)]
            trough_angles = [*range(360//folds//2, 360//folds*(180//(360//folds)), 360//folds)]

        for peak in peak_angles: 
            peak_correlations.append(corr[peak])
        for trough in trough_angles: 
            trough_correlations.append(corr[trough])

        gridScore = (np.nansum(peak_correlations) / len(peak_correlations)) - (np.nansum(trough_correlations) / len(trough_correlations))

        return gridScore

    def rotated_sacs(self, sac, angles):
        return [
            scipy.ndimage.interpolation.rotate(sac, angle, reshape=False)
            for angle in angles
        ]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        """Calculate Pearson correlations of area inside mask at corr_angles."""
        masked_sac = sac * mask
        ring_area = np.nansum(mask)
        # Calculate dc on the ring area
        masked_sac_mean = np.nansum(masked_sac) / ring_area
        # Center the sac values inside the ring
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.nansum(masked_sac_centered**2) / ring_area + 1e-5
        corrs = dict()
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            cross_prod = np.nansum(masked_sac_centered * masked_rotated_sac) / ring_area
            corrs[angle] = cross_prod / variance
            
        Gs = {f'{f}':np.nan for f in self._folds}
        for f in self._folds: 
            Gs[f'{f}'] = self.grid_score(corrs, folds=f)
        return Gs 

    def get_scores(self, rate_map):
        """Get summary of scrores for grid cells."""
        sac = calculate_sac(rate_map)
        rotated_sacs = self.rotated_sacs(sac, self._corr_angles)

        dd_GS = defaultdict(list)
        # iterate through each possible mask for the autocorrelation to calculate the grid score
        for mask, mask_params in self._masks:
            Gs = self.get_grid_scores_for_mask(sac, rotated_sacs, mask)
            for key, value in Gs.items():
                dd_GS[key].append(value)   
        
        # Pick the highest grid score for each of the masks 
        max_Gs = {f'{f}':np.nan for f in self._folds}
        max_masks = {f'{f}':np.nan for f in self._folds}
        
        for f in self._folds: 
            max_ind = np.argmax(dd_GS[f'{f}'])
            max_Gs[f'{f}'] = dd_GS[f'{f}'][max_ind]
            max_masks[f'{f}'] = self._masks[max_ind][1]

        return (max_Gs, max_masks, sac)
    
    def plot_sac(self,
               sac,
               mask_params=None,
               ax=None,
               title=None,
               *args,
               **kwargs):  # pylint: disable=keyword-arg-before-vararg
        """Plot spatial autocorrelogram."""
        if ax is None:
            ax = plt.gca()
        # Plot the sac
        plotting_sac_mask = self._circle_mask(
            [self._nbins * 2 - 1, self._nbins * 2 - 1],
            self._nbins,
            in_val=1.0,
            out_val=np.nan)
        
        useful_sac = sac * plotting_sac_mask
        ax.imshow(useful_sac, *args, **kwargs)
        # ax.pcolormesh(useful_sac, *args, **kwargs)
        # Plot a ring for the adequate mask
        if mask_params is not None:
            center = self._nbins - 1
            ax.add_artist(
            plt.Circle(
                  (center, center),
                  mask_params[0] * self._nbins,
                  # lw=bump_size,
                  fill=False,
                  edgecolor='k'))
            ax.add_artist(
              plt.Circle(
                  (center, center),
                  mask_params[1] * self._nbins,
                  # lw=bump_size,
                  fill=False,
                  edgecolor='k'))
        ax.axis('off')
        if title is not None:
            ax.set_title(title)

def scramble_matrix(matrix):
    """
    Randomly shuffle the values in a matrix while preserving its shape.
    
    Flattens the matrix, shuffles the values randomly, then reshapes to
    original dimensions. Useful for creating spatial surrogates.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix to scramble.
    
    Returns
    -------
    np.ndarray
        Scrambled matrix with same shape as input.
    """
    flat = matrix.flatten()
    np.random.shuffle(flat)

    return flat.reshape(matrix.shape)

def gelman_standardize(x):
    """
    Standardize data using Gelman's method (divide by 2 standard deviations).
    
    Gelman's standardization divides by 2 standard deviations instead of 1,
    which makes coefficients more comparable across different scales and
    is useful for Bayesian modeling.
    
    Parameters
    ----------
    x : np.ndarray
        Input array to standardize.
    
    Returns
    -------
    np.ndarray
        Standardized array with mean 0 and std 0.5.
    """
    return (x - np.nanmean(x)) / (2 * np.nanstd(x))

def surr_fft_map(fr_smoothed):
    """
    Create a surrogate ratemap by randomizing phase in frequency domain.
    
    Preserves the amplitude spectrum (power) while randomizing phases,
    creating a surrogate that maintains overall firing rate statistics
    but breaks spatial structure.
    
    Parameters
    ----------
    fr_smoothed : np.ndarray
        Smoothed firing rate map.
    
    Returns
    -------
    np.ndarray
        Surrogate ratemap with randomized phase but preserved amplitude spectrum.
    """
    # Compute 2D FFT of the rate map
    fft_map = np.fft.fft2(fr_smoothed)

    # Extract amplitude and phase
    amplitude = np.abs(fft_map)
    phase = np.angle(fft_map)

    # Generate random phase matrix with same shape
    random_phase = np.random.uniform(-np.pi, np.pi, size=fr_smoothed.shape)

    # Preserve DC component (frequency 0) phase = 0 to keep mean rate intact
    random_phase[0, 0] = 0

    # Construct new FFT with original amplitude and randomized phase
    shuffled_fft = amplitude * np.exp(1j * random_phase)

    # Inverse FFT to get surrogate map
    surr_fr_smoothed = np.fft.ifft2(shuffled_fft).real
    
    return surr_fr_smoothed

def poisson_generator(rate, duration, start_time=0):
    """Generator function for a Homogeneous Poisson distribution.

    Parameters
    ----------
    rate : float
        The average rate for the generator.
    duration : float
        Maximum duration. After this time, the generator will return.
    start_time: float, optional
        Timestamp of the start time for the generated sequence.

    Yields
    ------
    float
        A sample from the distribution.
        Sample is a relative value, based on `start_time`, in seconds.

    Examples
    --------
    Create a Poisson generator and sample from it:

    >>> gen = poisson_generator(20, duration=np.inf)
    >>> for ind in range(10):
    ...     sample = next(gen)
    """

    isi = 1. / rate

    cur_time = start_time
    end_time = start_time + duration

    while cur_time <= end_time:

        cur_time += isi * np.random.exponential()

        if cur_time > end_time:
            return

        yield cur_time

def simulate_spiking(st_s,
                     encoding_df,
                     encoding_time_in_neural_time, 
                     window_of_interest,
                     timeframe,
                     time_smooth_bin_s,
                     time_sigma, 
                     method='resample'):
    """
    Generate surrogate spike trains using various methods.
    
    Creates surrogate firing rate data by resampling, shuffling, or simulating
    spike trains. Used for statistical testing of spatial metrics.
    
    Parameters
    ----------
    st_s : np.ndarray
        Original spike times in seconds.
    encoding_df : pd.DataFrame
        DataFrame with encoding trial information.
    encoding_time_in_neural_time : list
        List of event times in neural time coordinates.
    window_of_interest : dict
        Dictionary mapping timeframe names to [start, end] time windows.
    timeframe : str
        Key for window_of_interest dictionary.
    time_smooth_bin_s : float
        Bin size in seconds for temporal smoothing.
    time_sigma : float
        Standard deviation for temporal Gaussian smoothing.
    method : str, default='resample'
        Surrogate method: 'resample', 'isi_shuffle', or 'poisson_sim'.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with surrogate firing rates and smoothed responses.
    """
    
    surr_df = encoding_df.copy()
    baseline_time = [-0.7, 0] 
    
    if method=='resample':
        # Sample 100% of spikes with replacement. 
        bs_st = st_s[np.random.choice(len(st_s), 
                                      len(st_s), replace=True)]
    elif method=='isi_shuffle':
        bs_st = np.cumsum(np.random.permutation(np.diff(st_s))).tolist()
        
    elif method=='poisson_sim': 
        rate = len(st_s) / encoding_df.session_length.mean()
        duration = encoding_df.session_length.mean()
        bs_st = np.array(list(poisson_generator(rate, duration))).tolist()
        
    bs_st = np.sort(bs_st)

    encoding_fr = []

    for event in encoding_time_in_neural_time: 
        st_to_keep_event = np.where(np.logical_and(bs_st>=event+window_of_interest[timeframe][0], 
                                                   bs_st<=event+window_of_interest[timeframe][1]))[0].astype(int)
        encoding_fr.append(len(st_to_keep_event)/(window_of_interest[timeframe][1]-window_of_interest[timeframe][0]))

    surr_df[f'trial_fr_encoding_{timeframe}'] = encoding_fr

    surr_df[f'trial_nspikes_encoding_{timeframe}'] = surr_df[f'trial_fr_encoding_{timeframe}'] * (window_of_interest[timeframe][1]-window_of_interest[timeframe][0])

    # smooth firing and baseline
    zscored_fr_ev = smooth_firing_rate_ev(bs_st, encoding_time_in_neural_time, 
                                          t_range=(-0.7, 2.7), # take the entire encoding event, essentially
                                          bin_size=time_smooth_bin_s, 
                                          sigma=time_sigma, 
                                          baseline='base')
    
    zscored_fr_ev = zscored_fr_ev[:, int((baseline_time[1]-baseline_time[0])/time_smooth_bin_s):]

    evoked_time_start = int(window_of_interest[timeframe][0]/time_smooth_bin_s)
    evoked_time_end = int(window_of_interest[timeframe][1]/time_smooth_bin_s)                                

    surr_df[f'evoked_fr_{timeframe}'] = np.nanmean(zscored_fr_ev[:, evoked_time_start:evoked_time_end], axis=1)
    
    return surr_df
    


def grid_score_parallel(x, 
                        nbins=15,
                        spiking_metrics = ['evoked_fr'], # ['trial_fr_encoding', 'evoked_fr', 'cat_resid', 'evoked_cat_resid']
                        return_SI=True,
                        surr_types=['florian_shuffle', 'label'], #'label', 'resample', 'isi_shuffle', 'poisson_sim', 'scramble', 'circular', 'fft'
                        time_smooth_bin_s = 0.01,
                        time_sigma = 0.05,
                        seed=2, # 1
                        surr_num=1000,
                        save_epoched_fr=False,
                        x_dim = 'valence',  # 'valence', 'arousal'
                        y_dim = 'arousal',
                        use_real_spikes = True):
    
    # start things off here: 
    random_state = np.random.seed(seed=seed)
    data_dir = "/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids"

    sync_data_original = pd.read_csv('/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/Micros/microwire_behavior_sync.csv')
    sync_data_replication = pd.read_csv('/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/Micros_New/microwire_behavior_sync_replication.csv')

    sync_data = pd.concat([sync_data_original, sync_data_replication])

    save_dir = f'{data_dir}/Results/scratch'
    
    # LOAD ALL ENCODING AND RETRIEVAL BEHAVIORAL DATA
    all_encoding_df = load_encoding_data()
    all_retrieval_df = load_retrieval_data()
    subj_data = load_subj_data()
    # merged_beh_df = all_encoding_df.merge(all_retrieval_df, on=['subj', 'list', 'image_name']).sort_values(['subj', 'list', 'image_num'])

    # rename the zscored features for simplicity
    image_features = ['arousal', 'valence', 'beauty', 'brightness', 'contrast', 'texture', 'color',
                      'spatial_frequency', 'complexity', 'sin_orientation', 'cos_orientation', 'pc1', 'pc2']
    
    
    mapping_dict_z = {
    'arousal': 'zArousal',
    'valence': 'zValence',
    'beauty': 'zBeauty',
    'brightness': 'zBrightness',
    'color': 'zColor',
    'contrast': 'zContrast',
    'texture': 'zTexture',
    'spatial_frequency': 'zSpatialFrequency',
    'complexity': 'zComplexity',
    'sin_orientation': 'zSinOrientation',
    'cos_orientation': 'zCosOrientation',
    'pc1': 'zPC1',
    'pc2': 'zPC2'}
                                    
     
    if use_real_spikes==True:
        all_spikes = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spikes_3_6_25.npy", allow_pickle=True)
    else:
        all_spikes = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spikes_sim_8_6_25.npy", allow_pickle=True)

    all_subj = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_subj_3_6_25.npy", allow_pickle=True)
    all_cell_num = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_cell_num_3_6_25.npy", allow_pickle=True)
    all_channel_num = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_channel_num_3_6_25.npy", allow_pickle=True)
    all_cell_mni = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_cell_mni_3_6_25.npy", allow_pickle=True)
    all_channel_loc = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_channel_loc_3_6_25.npy", allow_pickle=True)
    

    window_of_interest = {'early_trunc': [0.2, 1]}
    # window_of_interest = {'full':[0, 2]}
 
    # window_of_interest = {'early': [0, 1],
    #                       'early_trunc': [0.2, 1],
    #                       'late': [1, 2],
    #                       'post_stim': [2.0, 2.7],
    #                       'full':[0, 2], 
    #                       'full_trunc': [0.2, 2]}

    st_s = all_spikes[x]/1000
    subj = all_subj[x]
    cell_count = all_cell_num[x]
    channel_num = all_channel_num[x]
    channel_loc = all_channel_loc[x]
    channel_mni = all_cell_mni[x]
    
#     # Check gender for "raw_valence"
#     if x_dim == 'raw_valence':
#         if subj_data.loc[subj_data.subj==subj, 'sex'].values[0] == 'F':
#             x_dim = 'raw_valence_women'
#             y_dim = 'raw_arousal_women'
#         elif subj_data.loc[subj_data.subj==subj, 'sex'].values[0] == 'M': 
#             x_dim = 'raw_valence_men'
#             y_dim = 'raw_arousal_men'            
    
    subject_df = all_encoding_df[all_encoding_df.subj==subj].sort_values(by=['time'])

    encoding_time_in_neural_time = [(x*sync_data.slope[sync_data.subj==subj] + sync_data.offset[sync_data.subj==subj]).values for x in subject_df.time]
    encoding_list_time_in_neural_time = [(x*sync_data.slope[sync_data.subj==subj] + sync_data.offset[sync_data.subj==subj]).values for x in subject_df.encoding_list_start]

    encoding_df = pd.DataFrame(columns=['subj', 
                                            'channel_num', 
                                            'region', 
                                            'cluster',
                                            'image_num',
                                            'arousal',
                                            'valence',
                                            'category',
                                            'recalled',
                                            'recall_num',
                                            'trial_fr_baseline',
                                            'trial_fr_encoding'])

    # pre-populate what I can on the trial-level
    encoding_df['image_num'] = subject_df.image_num.tolist()
    encoding_df['image_name'] = subject_df.image_name.tolist()
    encoding_df['category'] = subject_df.Category.tolist()
    encoding_df['list'] = subject_df.list.tolist()
    

    encoding_df['arousal'] = subject_df.zArousal.tolist()
    encoding_df['valence'] = subject_df.zValence.tolist()
    encoding_df['beauty'] = subject_df.zBeauty.tolist()
    encoding_df['brightness'] = subject_df.zBrightness.tolist()
    encoding_df['color'] = subject_df.zColor.tolist()
    encoding_df['contrast'] = subject_df.zContrast.tolist()
    encoding_df['texture'] =subject_df.zTexture.tolist()
    encoding_df['spatial_frequency'] = subject_df.zSpatialFrequency.tolist()
    encoding_df['complexity'] = subject_df.zComplexity.tolist()
    encoding_df['sin_orientation'] = subject_df.zSinOrientation.tolist()
    encoding_df['cos_orientation'] = subject_df.zCosOrientation.tolist()
    encoding_df['pc1'] = subject_df.zPC1.tolist()
    encoding_df['pc2'] = subject_df.zPC2.tolist()
    
        
    encoding_df['raw_arousal'] = subject_df.raw_arousal.tolist()
    encoding_df['raw_valence'] = subject_df.raw_valence.tolist()
    encoding_df['raw_beauty'] = subject_df.beauty_mean.tolist()

    
    encoding_df['raw_arousal_men'] = subject_df.raw_arousal_men.tolist()
    encoding_df['raw_valence_men'] = subject_df.raw_valence_men.tolist()
    encoding_df['raw_arousal_women'] = subject_df.raw_arousal_women.tolist()
    encoding_df['raw_valence_women'] = subject_df.raw_valence_women.tolist()
    
    # Add in Lukas' category coding: 
    categories = ['animal', 'human_face', 'human_noface', 'natural_object', 'manmade_object']
    cont_categories = [f'{x}_cont' for x in categories] + ['indoor_env_cont', 'outdoor_env_cont']
    bi_categories = [f'{x}_bi' for x in categories] + ['environment_bi']
    
    for c in cont_categories:
        encoding_df[f'{c}'] = subject_df[f'{c}'].tolist()
    for c in bi_categories:
        encoding_df[f'{c}'] = subject_df[f'{c}'].tolist()
    
    
    encoding_df['recalled'] = subject_df.recall.tolist()
    encoding_df['recall_num'] = subject_df.recall_num.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

    # bin the two dimensions
    x_bins = pd.cut(encoding_df[f'{x_dim}'], nbins)
    y_bins = pd.cut(encoding_df[f'{y_dim}'], nbins)

    encoding_df[f'{x_dim}_binned'] = x_bins
    encoding_df[f'{y_dim}_binned'] = y_bins
    
    # pre-populate what I can on higher levels 
    encoding_df['cluster'] = cell_count
    encoding_df['subj'] = subj
    encoding_df['channel_num'] = channel_num
    encoding_df['region'] = channel_loc
    encoding_df['mni_x'] = channel_mni[0]
    encoding_df['mni_y'] = channel_mni[1]
    encoding_df['mni_z'] = channel_mni[2]
    
    encoding_df['session_length'] = subject_df.session_length.unique()[0]

    # define a time window (s) to look around the stimulus:
    baseline_time = [-0.7, 0] 
    # Baseline firing rate:
    baseline_fr = []
    for event in encoding_time_in_neural_time: 
        st_to_keep_event = np.where(np.logical_and(st_s>=event+baseline_time[0], 
                                                   st_s<=event+baseline_time[1]))[0].astype(int)
        baseline_fr.append(len(st_to_keep_event)/(baseline_time[1]-baseline_time[0]))

    encoding_df['trial_fr_baseline'] = baseline_fr

    # Encoding firing rate across whole list
    encoding_fr_list = []
    for event in encoding_list_time_in_neural_time: 
        st_to_keep_event = np.where(np.logical_and(st_s>=event-1, 
                                                   st_s<=event+35))[0].astype(int)
        encoding_fr_list.append(len(st_to_keep_event)/(36))

    encoding_df['list_fr_encoding'] = encoding_fr_list
    
    # count number of spikes in each encoding period
    encoding_df['trial_nspikes_baseline'] = encoding_df['trial_fr_baseline'] * (baseline_time[1]-baseline_time[0])
    
    # smooth firing and baseline
    zscored_fr_ev = smooth_firing_rate_ev(st_s, encoding_time_in_neural_time, 
                          t_range=(-0.7, 2.7), # take the entire encoding event, essentially
                          bin_size=time_smooth_bin_s, 
                          sigma=time_sigma, 
                          baseline='base')
    
    if save_epoched_fr:
        # Save the smoothed data to pull out for plots and regressions: 
        data = zscored_fr_ev.copy()  
        n_trials, n_times = data.shape

        # Reshape the data to (n_epochs, n_channels, n_times)
        # Assuming one channel (region) per trial
        data = data[:, np.newaxis, :]  # Shape: (240, 1, 4000)

        # Sampling frequency (adjust as appropriate for your data)
        sfreq = 1/time_smooth_bin_s  # Hz

        # Create an event array
        # Event array is of shape (n_events, 3), where each row is [sample_index, 0, event_id]
        events = np.array([[i, 0, 1] for i in range(n_trials)])

        # Define channel info
        info = mne.create_info(ch_names=[f'{cell_count}'], sfreq=sfreq)  

        # Define metadata using a pandas DataFrame
        metadata = pd.DataFrame({
            "subj": [subj] * n_trials,  # Modify as needed
            "region": [channel_loc] * n_trials,   # Modify as needed
            "channel_num": [channel_num] * n_trials,
            "cluster": [cell_count] * n_trials, 
        })

        # Set the starting time (tmin)
        tmin = baseline_time[0]  # Time in seconds where your data starts

        # Create the EpochsArray
        epochs = mne.EpochsArray(data, info, events=events, tmin=tmin)

        # Add metadata to the epochs
        epochs.metadata = metadata

        epochs.save(f'/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Results/scratch/cluster{x}_epochFR.fif', overwrite=True)

    # cut-off the baseline period seen above in t_range: 
    zscored_fr_ev = zscored_fr_ev[:, int((baseline_time[1]-baseline_time[0])/time_smooth_bin_s):]

    for timeframe in window_of_interest:
        
        for spiking_metric in spiking_metrics:

            encoding_fr = []
            
            for event in encoding_time_in_neural_time: 
                st_to_keep_event = np.where(np.logical_and(st_s>=event+window_of_interest[timeframe][0], 
                                                           st_s<=event+window_of_interest[timeframe][1]))[0].astype(int)
                encoding_fr.append(len(st_to_keep_event)/(window_of_interest[timeframe][1]-window_of_interest[timeframe][0]))

            encoding_df[f'trial_fr_encoding_{timeframe}'] = encoding_fr

            # encoding_df[f'unsmoothed_fr_diff_{timeframe}'] =  np.array(encoding_fr)- np.array(baseline_fr)

            encoding_df[f'unsmoothed_zscored_fr_{timeframe}'] =  (np.array(encoding_fr)- np.nanmean(baseline_fr)) / (np.nanstd(baseline_fr))

            encoding_df[f'trial_nspikes_encoding_{timeframe}'] = encoding_df[f'trial_fr_encoding_{timeframe}'] * (window_of_interest[timeframe][1]-window_of_interest[timeframe][0])

            evoked_time_start = int(window_of_interest[timeframe][0]/time_smooth_bin_s)
            evoked_time_end = int(window_of_interest[timeframe][1]/time_smooth_bin_s)
            encoding_df[f'evoked_fr_{timeframe}'] = np.nanmean(zscored_fr_ev[:, evoked_time_start:evoked_time_end], axis=1)

            # category residuals
            if 'environment_bi' in bi_categories:
                bi_categories.remove('environment_bi')
            predictors = " + ".join(bi_categories)  # Exclude dependent variable


            formula = f"trial_fr_encoding_{timeframe} ~ 1 + {predictors} + C(environment_bi, Treatment(0))"
            res = smf.ols(data=encoding_df, 
                       formula=formula).fit()
            encoding_df[f'cat_resid_{timeframe}'] = res.resid

            formula = f"evoked_fr_{timeframe} ~ 1 + {predictors} + C(environment_bi, Treatment(0))"
            res = smf.ols(data=encoding_df, 
                       formula=formula).fit()
            encoding_df[f'evoked_cat_resid_{timeframe}'] = res.resid


            formula = f"unsmoothed_zscored_fr_{timeframe} ~ 1 + {predictors} + C(environment_bi, Treatment(0))"
            res = smf.ols(data=encoding_df, 
                       formula=formula).fit()
            encoding_df[f'unsmoothed_cat_resid_{timeframe}'] = res.resid   

            # semantic residuals
            formula = f'trial_fr_encoding_{timeframe} ~ 1 + pc1 + pc2'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'pc_resid_{timeframe}'] = res.resid

            formula = f'evoked_fr_{timeframe} ~ 1 + pc1 + pc2'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'evoked_pc_resid_{timeframe}'] = res.resid

            formula = f'unsmoothed_zscored_fr_{timeframe} ~ 1 + pc1 + pc2'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'unsmoothed_pc_resid_{timeframe}'] = res.resid  


            # visual residuals
            formula = f'trial_fr_encoding_{timeframe} ~ 1 + brightness + contrast + spatial_frequency + complexity + sin_orientation + cos_orientation'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'viz_resid_{timeframe}'] = res.resid

            formula = f'evoked_fr_{timeframe} ~ 1 + brightness + contrast + spatial_frequency + complexity + sin_orientation + cos_orientation'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'evoked_viz_resid_{timeframe}'] = res.resid

            formula = f'unsmoothed_zscored_fr_{timeframe} ~ 1 + brightness + contrast + spatial_frequency + complexity + sin_orientation + cos_orientation'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'unsmoothed_viz_resid_{timeframe}'] = res.resid   

            # memory residuals
            formula = f'trial_fr_encoding_{timeframe} ~ 1 + recalled'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'mem_resid_{timeframe}'] = res.resid

            formula = f'evoked_fr_{timeframe} ~ 1 + recalled'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'evoked_mem_resid_{timeframe}'] = res.resid

            formula = f'unsmoothed_zscored_fr_{timeframe} ~ 1 + recalled'
            res = smf.ols(formula=formula, data=encoding_df).fit()
            encoding_df[f'unsmoothed_mem_resid_{timeframe}'] = res.resid 

            spike_map, occupancy_map = generate_maps(encoding_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                     nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                     trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])

            rate = np.nansum(spike_map) / np.nansum(occupancy_map)
            # Compute the occupancy probability, per bin
            occ_prob = occupancy_map.values / np.nansum(occupancy_map)
            # Calculate the spatial information, using a mask for nonzero values
            rm = spike_map.values/occupancy_map.values
            nz = np.nonzero(rm)
            SI = np.nansum(occ_prob[nz] * rm[nz] * np.log2(rm[nz] / rate)) / rate

            # compute the rate maps depending on the metric being used. 
            if spiking_metric == 'trial_fr_encoding':
                rate_map = spike_map/occupancy_map
                
                # 8/25/2025: 
                # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                half1_df = encoding_df.iloc[0:150]
                half2_df = encoding_df.iloc[-150:]
                
                # Get the split-half reliability using even-odd trial split
                even1_df = encoding_df.iloc[0::2]
                even2_df = encoding_df.iloc[1::2]

                spike_map1, occupancy_map1 = generate_maps(half1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                           nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                           trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                rate_map_half1 = spike_map1/occupancy_map1
                spike_map2, occupancy_map2 = generate_maps(half2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                           nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                           trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                rate_map_half2 = spike_map2/occupancy_map2
                
                spike_map3, occupancy_map3 = generate_maps(even1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                           nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                           trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                rate_map_even1 = spike_map3/occupancy_map3
                spike_map4, occupancy_map4 = generate_maps(even2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                           nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                           trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                rate_map_even2 = spike_map4/occupancy_map4
                
            else: 
                
                binned_data = encoding_df.groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()

                rate_map = binned_data.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                             f'{spiking_metric}_{timeframe}').T

                # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                binned_data_half1 = encoding_df.iloc[0:150].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                binned_data_half2 = encoding_df.iloc[-150:].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()  
                
                # Get the split-half reliability using even-odd trial split
                binned_data_even1 = encoding_df.iloc[0::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                binned_data_even2 = encoding_df.iloc[1::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()                

                rate_map_half1 = binned_data_half1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                               f'{spiking_metric}_{timeframe}').T
                rate_map_half2 = binned_data_half2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                               f'{spiking_metric}_{timeframe}').T
                                
                rate_map_even1 = binned_data_even1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                               f'{spiking_metric}_{timeframe}').T
                rate_map_even2 = binned_data_even2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                               f'{spiking_metric}_{timeframe}').T

            # Use the rate maps to compute metrics:
            fr_smoothed = smooth_map(rate_map)
            # get the grid score
            grid_scorer = GridCell(nbins=nbins)
            Gs, _, _ = grid_scorer.get_scores(fr_smoothed)
            
            # compute grid score across halves: 
            
            fr_smoothed_half1 = smooth_map(rate_map_half1)
            fr_smoothed_half2 = smooth_map(rate_map_half2) 
            
            Gs_half1, _, _ = grid_scorer.get_scores(fr_smoothed_half1)
            Gs_half2, _, _ = grid_scorer.get_scores(fr_smoothed_half2)

            # compute split-half reliability
                        
            fr_smoothed_even1 = smooth_map(rate_map_even1)
            fr_smoothed_even2 = smooth_map(rate_map_even2) 
            
            flat1 = fr_smoothed_even1.flatten()
            nan_mask1 = np.where(np.isnan(flat1))[0]
            flat2 = fr_smoothed_even2.flatten()
            nan_mask2 = np.where(np.isnan(flat2))[0]
            total_nan_mask = np.unique(np.concatenate([nan_mask1, nan_mask2]))
            flat1 = np.delete(flat1, total_nan_mask)
            flat2 = np.delete(flat2, total_nan_mask)

            Stability, _ = pearsonr(flat1, flat2)

            surr_Gs = {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            zGs =  {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            pGs =  {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            
            surr_Gs1 = {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            zGs1 =  {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            surr_Gs2 = {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            zGs2 =  {f'{x}': {key: [] for key in Gs.keys()} for x in surr_types}
            
            
            surr_SIs = {f'{x}':[] for x in surr_types}
            zSIs = {f'{x}':[] for x in surr_types}
            pSIs = {f'{x}':[] for x in surr_types}

            surr_Stability = {f'{x}':[] for x in surr_types}
            zStability = {f'{x}':[] for x in surr_types}
            pStability = {f'{x}':[] for x in surr_types}

            cell_res = pd.DataFrame(columns=['subj', 'region', 'cluster', 'symmetry', 'Gs', 'SI', 
                                             'Stability', 'zStability_label', 
                                             'pStability_label'] + 
                                    [f'zGs_{surr}' for surr in surr_types] + 
                                    [f'pGs_{surr}' for surr in surr_types] + 
                                    [f'sGs_mean_{surr}' for surr in surr_types] + 
                                    [f'sGs_std_{surr}' for surr in surr_types] + 
                                    [f'Gs_snr_{surr}' for surr in surr_types] + 
                                    [f'zSI_{surr}' for surr in surr_types] + 
                                    [f'pSI_{surr}' for surr in surr_types])

            cell_res['symmetry'] = np.arange(2, 11)

            for surr_type in surr_types:
                for surr in range(surr_num):
                    if surr_type in ['florian_shuffle', 'label', 'resample', 'isi_shuffle', 'poisson_sim', 'scramble', 'simulate']: 
                        # Construct surrogate specific dataframe:    
                        if surr_type in ['resample', 'isi_shuffle', 'poisson_sim']:
                            
                            surr_df = simulate_spiking(st_s, 
                                                       encoding_df,
                                                       encoding_time_in_neural_time, 
                                                       window_of_interest,
                                                       timeframe,
                                                       time_smooth_bin_s,
                                                       time_sigma,
                                                       method=surr_type)
                
                        elif surr_type == 'florian_shuffle':
                        
                            # Surrogate 10/10/2025: Florian wants to independently shuffle valence and arousal: 
                            surr_df = encoding_df.copy()
                            
                            surr_df[[f'{y_dim}']] = surr_df[[f'{y_dim}']].sample(frac=1).values
                            surr_df[[f'{x_dim}']] = surr_df[[f'{x_dim}']].sample(frac=1).values
                            
                        elif surr_type == 'label':
                            
                            # Surrogate 1: Shuffle trial labels mismatching "space" and firing rate 
                            # This is a random sample of EXISTING spatial sampling
                            
                            surr_df = encoding_df.copy()
                            surr_df[[f'{y_dim}', f'{x_dim}']] = surr_df[[f'{y_dim}', f'{x_dim}']].sample(frac=1).values
                    
                        elif surr_type == 'scramble':
                            
                            surr_df = encoding_df.copy()

                            # Determine the bounds of the space 
                            y_min = np.nanmin(all_encoding_df[f'{mapping_dict_z[y_dim]}'])
                            y_max = np.nanmax(all_encoding_df[f'{mapping_dict_z[y_dim]}'])
                            x_min = np.nanmin(all_encoding_df[f'{mapping_dict_z[x_dim]}'])
                            x_max = np.nanmax(all_encoding_df[f'{mapping_dict_z[x_dim]}'])

                            # Randomly sample from ALL of the POSSIBLE spatial sampling

                            surr_df[f'{y_dim}'] = np.random.uniform(y_min, y_max, len(encoding_df))
                            surr_df[f'{x_dim}'] = np.random.uniform(x_min, x_max, len(encoding_df))
                            
                        # Use surrogate-specific dataframe to do the rest of the steps
                        surr_spike_map, surr_occupancy_map = generate_maps(surr_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                           nbins=nbins, x_dim=x_dim, y_dim=y_dim,
                                                                           trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])

                        rate = np.nansum(surr_spike_map) / np.nansum(surr_occupancy_map)
                        # Compute the occupancy probability, per bin
                        occ_prob = surr_occupancy_map.values / np.nansum(surr_occupancy_map)
                        # Calculate the spatial information, using a mask for nonzero values
                        rm = surr_spike_map.values/surr_occupancy_map.values
                        nz = np.nonzero(rm)
                        sSI = np.nansum(occ_prob[nz] * rm[nz] * np.log2(rm[nz] / rate)) / rate

                        x_bins = pd.cut(surr_df[f'{x_dim}'], nbins)
                        y_bins = pd.cut(surr_df[f'{y_dim}'], nbins)

                        surr_df[f'{x_dim}_binned'] = x_bins
                        surr_df[f'{y_dim}_binned'] = y_bins

                        # compute the rate map depending on the metric being used. 
                        if spiking_metric == 'trial_fr_encoding':
                            surr_rate_map = surr_spike_map/surr_occupancy_map

                            # 8/25/2025: 
                            # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                            surr_half1_df = surr_df.iloc[0:150]
                            surr_half2_df = surr_df.iloc[-150:]

                            # Get the split-half reliability using even-odd trial split
                            surr_even1_df = surr_df.iloc[0::2]
                            surr_even2_df = surr_df.iloc[1::2]

                            surr_spike_map1, surr_occupancy_map1 = generate_maps(surr_half1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_half1 = surr_spike_map1/surr_occupancy_map1
                            surr_spike_map2, surr_occupancy_map2 = generate_maps(surr_half2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_half2 = surr_spike_map2/surr_occupancy_map2

                            surr_spike_map3, surr_occupancy_map3 = generate_maps(surr_even1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_even1 = surr_spike_map3/surr_occupancy_map3
                            surr_spike_map4, surr_occupancy_map4 = generate_maps(surr_even2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_even2 = surr_spike_map4/surr_occupancy_map4

                        else: 

                            surr_binned_data = surr_df.groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()

                            surr_rate_map = surr_binned_data.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                         f'{spiking_metric}_{timeframe}').T

                            # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                            surr_binned_data_half1 = surr_df.iloc[0:150].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                            surr_binned_data_half2 = surr_df.iloc[-150:].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()  

                            # Get the split-half reliability using even-odd trial split
                            surr_binned_data_even1 = surr_df.iloc[0::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                            surr_binned_data_even2 = surr_df.iloc[1::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()                

                            surr_rate_map_half1 = surr_binned_data_half1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T
                            surr_rate_map_half2 = surr_binned_data_half2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T

                            surr_rate_map_even1 = surr_binned_data_even1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T
                            surr_rate_map_even2 = surr_binned_data_even2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T


                        surr_fr_smoothed = smooth_map(surr_rate_map)
                        surr_fr_smoothed1 = smooth_map(surr_rate_map_half1)
                        surr_fr_smoothed2 = smooth_map(surr_rate_map_half2)

                        # get the grid score (typically some errors occur on bad shuffles):
                        try:

                            surr_flat1 = surr_fr_smoothed1.flatten()
                            surr_nan_mask1 = np.where(np.isnan(surr_flat1))[0]
                            surr_flat2 = surr_fr_smoothed2.flatten()
                            surr_nan_mask2 = np.where(np.isnan(surr_flat2))[0]

                            surr_total_nan_mask = np.unique(np.concatenate([surr_nan_mask1, surr_nan_mask2]))

                            surr_flat1 = np.delete(surr_flat1, surr_total_nan_mask)
                            surr_flat2 = np.delete(surr_flat2, surr_total_nan_mask)

                            sStability, _ = pearsonr(surr_flat1, surr_flat2)
                        except: 
                            sStability = np.nan

                        surr_Stability[surr_type].append(sStability)

                        # Gather the data 
                        surr_SIs[surr_type].append(sSI)

                        sGs, _, _ = grid_scorer.get_scores(surr_fr_smoothed)

                        for key, value in sGs.items():
                            surr_Gs[surr_type][key].append(value)
                            
                        # compute grid score across halves: 
                        surr_fr_smoothed_half1 = smooth_map(surr_rate_map_half1)
                        surr_fr_smoothed_half2 = smooth_map(surr_rate_map_half2) 

                        sGs_half1, _, _ = grid_scorer.get_scores(surr_fr_smoothed1)
                        sGs_half2, _, _ = grid_scorer.get_scores(surr_fr_smoothed2)
 
                        for key, value in sGs_half1.items():
                            surr_Gs1[surr_type][key].append(value)
                        for key, value in sGs_half2.items():
                            surr_Gs2[surr_type][key].append(value)                        
                        
                        
                    if surr_type == 'rotation':
                        shift_x = np.random.randint(1, nbins)   # avoid 0 shift
                        shift_y = np.random.randint(1, nbins)

                        surr_fr_smoothed = np.roll(np.roll(fr_smoothed, shift_y, axis=0), shift_x, axis=1)

                        surr_Stability[surr_type].append(np.nan)

                        # Gather the data 
                        surr_SIs[surr_type].append(np.nan)

                        sGs, _, _ = grid_scorer.get_scores(surr_fr_smoothed)

                        for key, value in sGs.items():
                            surr_Gs[surr_type][key].append(value)

                    elif surr_type=='fft':
                        
                        surr_fr_smoothed = surr_fft_map(fr_smoothed)
                        
                        surr_Stability[surr_type].append(np.nan)

                        # Gather the data 
                        surr_SIs[surr_type].append(np.nan)

                        sGs, _, _ = grid_scorer.get_scores(surr_fr_smoothed)

                        for key, value in sGs.items():
                            surr_Gs[surr_type][key].append(value)

                        # Optional: set negative values to zero (rate cannot be negative)
                        # surr_fr_smoothed[surr_fr_smoothed < 0] = 0

                    elif surr_type == 'field':
                        # This doesn't make sense for all cells. It is specifically something I should implement 
                        # for grid cells only, posthoc. 
                        pass

                if surr_type == 'circular': 
                    # In this method, we are going to circularly shift the firing rates. This can only be done ntrials - 1 unique times 
                    ntrials = len(encoding_df)
                    
                    shuffles = np.arange(1, ntrials)
                    
                    for shuffle in shuffles:
                        surr_df = encoding_df.copy()                        
                        surr_df[[f'{y_dim}', f'{x_dim}']] = np.roll(surr_df[[f'{y_dim}', f'{x_dim}']], shuffle)
                        
                        surr_spike_map, surr_occupancy_map = generate_maps(surr_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                   nbins=nbins, x_dim=x_dim, y_dim=y_dim,
                                   trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])

                        rate = np.nansum(surr_spike_map) / np.nansum(surr_occupancy_map)
                        # Compute the occupancy probability, per bin
                        occ_prob = surr_occupancy_map.values / np.nansum(surr_occupancy_map)
                        # Calculate the spatial information, using a mask for nonzero values
                        rm = surr_spike_map.values/surr_occupancy_map.values
                        nz = np.nonzero(rm)
                        sSI = np.nansum(occ_prob[nz] * rm[nz] * np.log2(rm[nz] / rate)) / rate

                        x_bins = pd.cut(surr_df[f'{x_dim}'], nbins)
                        y_bins = pd.cut(surr_df[f'{y_dim}'], nbins)

                        surr_df[f'{x_dim}_binned'] = x_bins
                        surr_df[f'{y_dim}_binned'] = y_bins

                        # compute the rate map depending on the metric being used. 
                        if spiking_metric == 'trial_fr_encoding':
                            surr_rate_map = surr_spike_map/surr_occupancy_map

                            # 8/25/2025: 
                            # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                            surr_half1_df = surr_df.iloc[0:150]
                            surr_half2_df = surr_df.iloc[-150:]

                            # Get the split-half reliability using even-odd trial split
                            surr_even1_df = surr_df.iloc[0::2]
                            surr_even2_df = surr_df.iloc[1::2]

                            surr_spike_map1, surr_occupancy_map1 = generate_maps(surr_half1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_half1 = surr_spike_map1/surr_occupancy_map1
                            surr_spike_map2, surr_occupancy_map2 = generate_maps(surr_half2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_half2 = surr_spike_map2/surr_occupancy_map2

                            surr_spike_map3, surr_occupancy_map3 = generate_maps(surr_even1_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_even1 = surr_spike_map3/surr_occupancy_map3
                            surr_spike_map4, surr_occupancy_map4 = generate_maps(surr_even2_df, metric=f'trial_nspikes_encoding_{timeframe}', 
                                                                       nbins=nbins, x_dim = x_dim, y_dim = y_dim,
                                                                       trial_time_s = window_of_interest[timeframe][1]-window_of_interest[timeframe][0])
                            surr_rate_map_even2 = surr_spike_map4/surr_occupancy_map4

                        else: 

                            surr_binned_data = surr_df.groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()

                            surr_rate_map = surr_binned_data.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                         f'{spiking_metric}_{timeframe}').T

                            # Compare the grid scores for the first 150 vs the last 150 trials of the data 
                            surr_binned_data_half1 = surr_df.iloc[0:150].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                            surr_binned_data_half2 = surr_df.iloc[-150:].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()  

                            # Get the split-half reliability using even-odd trial split
                            surr_binned_data_even1 = surr_df.iloc[0::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()
                            surr_binned_data_even2 = surr_df.iloc[1::2].groupby([f'{x_dim}_binned', f'{y_dim}_binned']).agg({f'{spiking_metric}_{timeframe}': np.nanmean}).reset_index()                

                            surr_rate_map_half1 = surr_binned_data_half1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T
                            surr_rate_map_half2 = surr_binned_data_half2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T

                            surr_rate_map_even1 = surr_binned_data_even1.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T
                            surr_rate_map_even2 = surr_binned_data_even2.pivot(f'{y_dim}_binned', f'{x_dim}_binned', 
                                                           f'{spiking_metric}_{timeframe}').T


                        surr_fr_smoothed = smooth_map(surr_rate_map)
                        surr_fr_smoothed1 = smooth_map(surr_rate_map_half1)
                        surr_fr_smoothed2 = smooth_map(surr_rate_map_half2)

                        # get the grid score (typically some errors occur on bad shuffles):
                        try:

                            surr_flat1 = surr_fr_smoothed1.flatten()
                            surr_nan_mask1 = np.where(np.isnan(surr_flat1))[0]
                            surr_flat2 = surr_fr_smoothed2.flatten()
                            surr_nan_mask2 = np.where(np.isnan(surr_flat2))[0]

                            surr_total_nan_mask = np.unique(np.concatenate([surr_nan_mask1, surr_nan_mask2]))

                            surr_flat1 = np.delete(surr_flat1, surr_total_nan_mask)
                            surr_flat2 = np.delete(surr_flat2, surr_total_nan_mask)

                            sStability, _ = pearsonr(surr_flat1, surr_flat2)
                        except: 
                            sStability = np.nan

                        surr_Stability[surr_type].append(sStability)

                        # Gather the data 
                        surr_SIs[surr_type].append(sSI)

                        sGs, _, _ = grid_scorer.get_scores(surr_fr_smoothed)

                        for key, value in sGs.items():
                            surr_Gs[surr_type][key].append(value)
                            
                        # compute grid score across halves: 
                        surr_fr_smoothed_half1 = smooth_map(surr_rate_map_half1)
                        surr_fr_smoothed_half2 = smooth_map(surr_rate_map_half2) 

                        sGs_half1, _, _ = grid_scorer.get_scores(surr_fr_smoothed1)
                        sGs_half2, _, _ = grid_scorer.get_scores(surr_fr_smoothed2)
 
                        for key, value in sGs_half1.items():
                            surr_Gs1[surr_type][key].append(value)
                        for key, value in sGs_half2.items():
                            surr_Gs2[surr_type][key].append(value) 

                # Compute the zscored true value compared to the surrogates of this type
                for k in zGs[surr_type].keys():
                    zGs[surr_type][k].append(((Gs[k] - np.nanmean(surr_Gs[surr_type][k])) / np.nanstd(surr_Gs[surr_type][k])))
                    zGs1[surr_type][k].append(((Gs_half1[k] - np.nanmean(surr_Gs1[surr_type][k])) / np.nanstd(surr_Gs1[surr_type][k])))
                    zGs2[surr_type][k].append(((Gs_half2[k] - np.nanmean(surr_Gs2[surr_type][k])) / np.nanstd(surr_Gs2[surr_type][k])))

                    pGs[surr_type][k].append(sum(surr_Gs[surr_type][k] > Gs[k]) / surr_num)

                zSIs[surr_type].append((SI - np.nanmean(surr_SIs[surr_type])) / np.nanstd(surr_SIs[surr_type]))
                pSIs[surr_type].append(sum(surr_SIs[surr_type] > SI) / surr_num)

                zStability[surr_type].append((Stability - np.nanmean(surr_Stability[surr_type])) / np.nanstd(surr_Stability[surr_type]))
                pStability[surr_type].append(sum(surr_Stability[surr_type] > Stability) / surr_num)

                for key in ['subj', 'region', 'cluster']:
                    cell_res[key] = encoding_df[key].unique()[0]

                cell_res[f'Stability'] = Stability
                cell_res[f'zStability_{surr_type}'] = zStability[surr_type][0]
                cell_res[f'pStability_{surr_type}'] = pStability[surr_type][0]

                cell_res[f'SI'] = SI
                cell_res[f'zSI_{surr_type}'] = zSIs[surr_type][0]
                cell_res[f'pSI_{surr_type}'] = pSIs[surr_type][0]


                for sym in cell_res.symmetry:
                    cell_res.loc[cell_res.symmetry==sym, f'Gs'] = Gs[str(sym)]
                    cell_res.loc[cell_res.symmetry==sym, f'zGs_{surr_type}'] = float(zGs[surr_type][str(sym)][0])
                    cell_res.loc[cell_res.symmetry==sym, f'pGs_{surr_type}'] = float(pGs[surr_type][str(sym)][0])
                    
                    cell_res.loc[cell_res.symmetry==sym, f'Gs_half1'] = Gs_half1[str(sym)]
                    cell_res.loc[cell_res.symmetry==sym, f'zGs1_{surr_type}'] = float(zGs1[surr_type][str(sym)][0])
                    cell_res.loc[cell_res.symmetry==sym, f'Gs_half2'] = Gs_half2[str(sym)]
                    cell_res.loc[cell_res.symmetry==sym, f'zGs2_{surr_type}'] = float(zGs2[surr_type][str(sym)][0])
                    



            cell_res['spiking_metric'] = spiking_metric
            cell_res['encoding_window'] = timeframe

            cell_res.to_pickle(f'/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Results/scratch/cluster{x}_res_{timeframe}_{nbins}_{spiking_metric}_{x_dim}_{y_dim}_8_26.pkl')
            
            # merge in the retrieval data:
            encoding_df = encoding_df.merge(all_retrieval_df[all_retrieval_df.subj==subj], on=['subj', 'list', 'image_name'], how='left').sort_values(['subj', 'list', 'image_num'])
            encoding_df.to_pickle(f'/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Results/scratch/cluster{x}_trial_data_{timeframe}_{nbins}_{spiking_metric}_{x_dim}_{y_dim}_8_26.pkl')


def detect_peaks_and_fields(rate_map,
                            z_thr=1,
                            min_area=3,
                            min_peak_rate_frac=0.3,
                            footprint_size=1,
                            watershed_promote=True):
    # Normalize (z-score)
    mean = np.nanmean(rate_map)
    std = np.nanstd(rate_map)
    zmap = (rate_map - mean) / (std + 1e-12)
    
    # Threshold to get binary mask
    mask = zmap > z_thr
    mask = remove_small_objects(mask, min_size=min_area)
    mask = mask.astype(bool)
    
    # Find local peaks above peak fraction
    global_peak = np.max(rate_map)
    coords = peak_local_max(rate_map, min_distance=footprint_size)
    peaks = [(r, c) for r, c in coords if rate_map[r, c] >= min_peak_rate_frac * global_peak]

    # Create markers from peaks
    markers = np.zeros_like(rate_map, dtype=int)
    for i, (r, c) in enumerate(peaks, 1):
        markers[r, c] = i

    # Initial labeling of blobs
    labeled = label(mask)

    # Optionally refine with watershed segmentation using peaks as seeds
    if watershed_promote and len(peaks) > 0:
        elevation = -rate_map  # peaks are minima in elevation
        markers_mask = np.where(markers > 0, markers, 0)
        labeled = watershed(elevation, markers=markers_mask, mask=mask)
    
    # Extract coordinates of pixels for each field
    fields = []
    for lab in np.unique(labeled):
        if lab == 0:
            continue
        coords = np.argwhere(labeled == lab)
        fields.append(coords.tolist())

    fields = np.array(fields, dtype=object)  # matches your screenshot style
    
    return peaks, fields


def encoding_window_spike_phases(x,
                           nsurr=500,
                             seed=2):
    """
    Compute phase-locking metrics for spikes during encoding window.
    
    Calculates pairwise phase consistency (PPC) and preferred phase for spikes
    occurring during encoding windows. Includes surrogate testing for statistical
    significance.
    
    Parameters
    ----------
    x : int
        Cell index to analyze.
    nsurr : int, default=500
        Number of surrogates for statistical testing.
    seed : int, default=2
        Random seed for reproducibility.
    
    Returns
    -------
    None
        Saves phase metrics to CSV file in Results directory.
    """
    # start things off here: 
    random_state = np.random.seed(seed=seed)
    data_dir = "/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids"

    sync_data_original = pd.read_csv('/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/Micros/microwire_behavior_sync.csv')
    sync_data_replication = pd.read_csv('/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/Micros_New/microwire_behavior_sync_replication.csv')

    sync_data = pd.concat([sync_data_original, sync_data_replication])

    save_dir = f'{data_dir}/Results/scratch'
    
    # LOAD ALL ENCODING AND RETRIEVAL BEHAVIORAL DATA
    all_encoding_df = load_encoding_data()
    all_retrieval_df = load_retrieval_data()
    # merged_beh_df = all_encoding_df.merge(all_retrieval_df, on=['subj', 'list', 'image_name']).sort_values(['subj', 'list', 'image_num'])

    # rename the zscored features for simplicity
    image_features = ['arousal', 'valence', 'brightness', 'contrast', 'texture', 'color',
                      'spatial_frequency', 'complexity', 'sin_orientation', 'cos_orientation', 'pc1', 'pc2']
    
    
    mapping_dict_z = {
    'arousal': 'zArousal',
    'valence': 'zValence',
    'brightness': 'zBrightness',
    'color': 'zColor',
    'contrast': 'zContrast',
    'texture': 'zTexture',
    'spatial_frequency': 'zSpatialFrequency',
    'complexity': 'zComplexity',
    'sin_orientation': 'zSinOrientation',
    'cos_orientation': 'zCosOrientation',
    'pc1': 'zPC1',
    'pc2': 'zPC2'}
                                    
            
    all_spikes = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spikes_3_6_25.npy", allow_pickle=True)
    # all_spike_phases = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spike_phases_2_5Hz_6_16_25.npy", allow_pickle=True)
    # all_spike_phases = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spike_phases_4_9Hz_6_16_25.npy", allow_pickle=True)
    all_spike_phases = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_spike_phases_3_6_25.npy", allow_pickle=True)    
    all_subj = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_subj_3_6_25.npy", allow_pickle=True)
    all_cell_num = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_cell_num_3_6_25.npy", allow_pickle=True)
    all_channel_num = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_channel_num_3_6_25.npy", allow_pickle=True)
    all_cell_mni = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_cell_mni_3_6_25.npy", allow_pickle=True)
    all_channel_loc = np.load(f"/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Neural/all_channel_loc_3_6_25.npy", allow_pickle=True)
    
    
    # let's restrict this to early trunc since that's what works for everything else: 
    timeframe = 'early_trunc'
    window_of_interest = {'early_trunc':[0.2, 1]}
    spiking_metric = 'evoked_fr'

    st_s = all_spikes[x]/1000
    sp_s = all_spike_phases[x]
    subj = all_subj[x]
    cell_count = all_cell_num[x]
    channel_num = all_channel_num[x]
    channel_loc = all_channel_loc[x]
    channel_mni = all_cell_mni[x]
    
    subject_df = all_encoding_df[all_encoding_df.subj==subj].sort_values(by=['time'])

    encoding_time_in_neural_time = [(x*sync_data.slope[sync_data.subj==subj] + sync_data.offset[sync_data.subj==subj]).values for x in subject_df.time]
    
    all_sp = []
    for event in encoding_time_in_neural_time: 
        st_to_keep_event = np.where(np.logical_and(st_s>=event+window_of_interest[timeframe][0], 
                                                   st_s<=event+window_of_interest[timeframe][1]))[0].astype(int)
        sp_to_keep_event = sp_s[st_to_keep_event]
        sp_nonan  = sp_to_keep_event[~np.isnan(sp_to_keep_event)]
        all_sp.append(sp_nonan)
        
    sp_all_ev = np.hstack(all_sp)
    
    wrapped_phase = (sp_all_ev + np.pi) % (2 * np.pi) - np.pi
    
    # Number of spikes
    n = len(wrapped_phase)
    
    # Convert phases to complex unit vectors
    unit_vectors = np.exp(1j * wrapped_phase)
    
    preferred_phase = np.angle(np.mean(unit_vectors))

    # Compute resultant vector length (for verification)
    r = np.abs(np.sum(unit_vectors)) / n
    ppc = (r**2 * n - 1) / (n - 1)
    
    # initialize the dataframe    
    phase_df = pd.DataFrame(columns=['subj', 'region', 'cluster', 'zPPC'])
    phase_df['subj'] = [subj]
    phase_df['region'] = [channel_loc]
    phase_df['cluster'] = [cell_count]
    phase_df['PPC'] = [ppc]
    
    if nsurr>0:
        # Compute PPC using vectorized operations
        # PPC = (R²*n - 1)/(n - 1), where R is the mean resultant vector length
        # This is mathematically equivalent to averaging all pairwise dot products

        x = sp_all_ev / (2 * np.pi)  # get the cycles for shuffling
        cycle = np.floor(x)
        unique_cycles, left_indices = np.unique(cycle, return_index=True)
        left_indices = left_indices.tolist()
        right_indices = left_indices[1:]
        # now shuffle each cycle (fuck a loop!!)
        shuffle_num = np.random.uniform(0, 2 * np.pi, size=(nsurr, len(left_indices)))
        # take out the loop by using np.split and np.concatenate
        split_phase = np.split(sp_all_ev, right_indices)

        surr_ppcs = []
        for i in range(nsurr):
            shuffled_phase = [np.sort(np.mod(a + b, 2 * np.pi)) for a, b in
                              zip(shuffle_num[i].tolist(), split_phase)]  # sort there
            try:
                surr_phase = np.concatenate(
                    [(b + (a / (2 * np.pi))) * (2 * np.pi) for a, b in zip(shuffled_phase, unique_cycles)])
            except:
                continue

            surr_phase_nonan = surr_phase[~np.isnan(surr_phase)]
            wrapped_surr = (surr_phase_nonan + np.pi) % (2 * np.pi) - np.pi

            # Number of spikes
            n = len(wrapped_surr)

            # Convert phases to complex unit vectors
            unit_vectors = np.exp(1j * wrapped_surr)

            # Compute resultant vector length (for verification)
            r = np.abs(np.sum(unit_vectors)) / n
            surr_ppc = (r**2 * n - 1) / (n - 1)
            surr_ppcs.append(surr_ppc)

        zPPC = (ppc - np.nanmean(surr_ppcs)) / (np.nanstd(surr_ppcs))

        phase_df['zPPC'] = [zPPC]
        phase_df['pref_phase'] = preferred_phase

    phase_df.to_csv(f'/home1/salman.qasim/Salman_Project/FR_Emotion_Lukas/EmotionGrids/Results/phase_df_{cell_count}_2_10Hz', index=False)
    