import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def convert_times_to_train(spikes, fs=1000, length=None):
    """
    Convert spike times into a binary spike train.
    
    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    fs : int, optional, default: 1000
        The sampling rate to use for the computed spike train, in Hz.
    length : float, optional
        The total length of the spike train to create, in seconds.
        If not provided, the length is set at the maximum timestamp in the input spike times.
    Returns
    -------
    spike_train : 1d array
        Spike train.
    Examples
    --------
    Convert spike times into a corresponding binary spike train:
    >>> spikes = np.array([0.002, 0.250, 0.500, 0.750, 1.000, 1.250, 1.500])
    >>> convert_times_to_train(spikes)
    array([0, 0, 1, ..., 0, 0, 1])
    """

    if not length:
        length = np.max(spikes)

    spike_train = np.zeros(int(length * fs) + 1).astype(int)
    inds = [int(ind * fs) for ind in spikes if ind * fs <= spike_train.shape[-1]]
    spike_train[inds] = 1

    # Check that the spike times are fully encoded into the spike train
    msg = ("The spike times were not fully encoded into the spike train. " \
           "This probably means the spike sampling rate is too low to encode " \
           "spikes close together in time. Try increasing the sampling rate.")
    if not sum(spike_train) == len(spikes):
        raise ValueError(msg)

    return spike_train


def convert_train_to_times(spike_train, fs=1000):
    """
    Convert a spike train representation into spike times, in seconds.
    
    Parameters
    ----------
    spike_train : 1d array
        Spike train.
    fs : int, optional, default: 1000
        The sampling rate of the computed spike train, in Hz.
    Returns
    -------
    spikes : 1d array
        Spike times, in seconds.
    Examples
    --------
    Convert a spike train into spike times:
    >>> spike_train = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
    >>> convert_train_to_times(spike_train)
    array([0.004, 0.006, 0.009, 0.011, 0.012, 0.014])
    """

    spikes = np.where(spike_train)[0] + 1
    spikes = spikes * (1 / fs)

    return spikes

def shuffle_circular(spikes, shuffle_min=20000, n_shuffles=1000, fs=1000):
    """
    Shuffle spikes based on circularly shifting the spike train.
    
    Creates surrogate spike trains by circularly rotating the original spike
    train by random amounts. Preserves the overall spike rate and temporal
    structure while breaking relationships to external events.
    
    Parameters
    ----------
    spikes : 1d array
        Spike times, in seconds.
    shuffle_min : int
        The minimum amount to rotate data, in terms of units of the spike train.
    n_shuffles : int, optional, default: 1000
        The number of shuffles to create.
    Returns
    -------
    shuffled_spikes : 2d array
        Shuffled spike times.
    Notes
    -----
    The input shuffle_min should always be less than the maximum time in which a spike occurred.
    Examples
    --------
    Shuffle spike times using the circular method:
    >>> from spiketools.sim.times import sim_spiketimes
    >>> spikes = sim_spiketimes(5, 30, 'poisson')
    >>> shuffled_spikes = shuffle_circular(spikes, shuffle_min=10000, n_shuffles=5)
    """

        
    try:
        spike_train = convert_times_to_train(spikes, fs=fs)
    except:
        fs=3000
        spike_train = convert_times_to_train(spikes, fs=fs)
        

    shuffles = np.random.randint(low=shuffle_min,
                                     high=len(spike_train)-shuffle_min,
                                     size=n_shuffles)

    shuffled_spikes = np.zeros([n_shuffles, len(spikes)])

    for ind, shuffle in enumerate(shuffles):
        temp_train = np.roll(spike_train, shuffle)
        shuffled_spikes[ind, :] = convert_train_to_times(temp_train, fs=fs)

    return shuffled_spikes


def smooth_firing_rate_ev(spike_times,
                          event_times, 
                          t_range=(-0.5, 1.5), 
                          bin_size=0.01, 
                          sigma=0.05,
                          baseline='all',
                          raster=False,
                          color='gray'):
    """
    Compute smoothed and z-scored firing rate aligned to events.
    
    Extracts spikes around each event time, bins them, smooths with a Gaussian
    filter, and z-scores relative to a baseline period. Optionally creates
    raster and firing rate plots.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds.
    event_times : np.ndarray
        Array of event times in seconds to align spikes to.
    t_range : tuple, default=(-0.5, 1.5)
        Time window around each event (start, end) in seconds.
    bin_size : float, default=0.01
        Size of time bins in seconds for firing rate calculation.
    sigma : float, default=0.05
        Standard deviation of Gaussian smoothing kernel in seconds.
    baseline : str, default='all'
        Baseline period for z-scoring:
        - 'all': Use entire time range
        - 'base': Use baseline period (negative times if t_range[0] < 0, else positive)
        - None: Return raw smoothed rate without z-scoring
    raster : bool, default=False
        If True, create raster plot and firing rate plot.
    color : str, default='gray'
        Color for plots.
    
    Returns
    -------
    np.ndarray
        Z-scored (or raw) smoothed firing rate array of shape (n_events, n_bins).
        Each row is a trial, each column is a time bin.
    """
    spike_times_per_event = [] 
    for event in event_times: 
        st_to_keep_event = np.where(np.logical_and(spike_times>=event+t_range[0], 
                                                   spike_times<=event+t_range[1]))[0].astype(int)
        spike_times_per_event.append(spike_times[st_to_keep_event] - event)

    # Compute number of time bins and time points
    n_bins = int(np.diff(t_range)[0] / bin_size)
    t = np.linspace(t_range[0], t_range[1], n_bins + 1)[:-1]
    # Compute firing rates
    spike_counts = np.zeros((len(spike_times_per_event), n_bins))
    for i in range(len(spike_times_per_event)):
        spike_counts[i], _ = np.histogram(spike_times_per_event[i], bins=n_bins, range=t_range)
        
    firing_rate = spike_counts / bin_size
    
    # Smooth firing rate
    smoothed_rate = gaussian_filter1d(firing_rate, sigma / bin_size, mode='nearest')
    
        # Z-score firing rate across trials
    if np.abs(t_range[0])> np.abs(t_range[1]):
        baseline_ix = np.where(t>=0)[0]
    elif np.abs(t_range[0]) < np.abs(t_range[1]):
        baseline_ix = np.where(t<=0)[0]
        
    if baseline=='all': 
        bmean = np.nanmean(smoothed_rate)
        bstd = np.nanstd(np.nanmean(smoothed_rate, axis=1))
        zscored_rate = (smoothed_rate - bmean) / bstd
    elif baseline=='base':
        bmean = np.nanmean(smoothed_rate[:, baseline_ix])
        bstd = np.nanstd(np.nanmean(smoothed_rate[:, baseline_ix], axis=1))
        zscored_rate = (smoothed_rate - bmean) / bstd
    if baseline==None: 
        zscored_rate = smoothed_rate
        
        
    if raster:
        # Create raster plot
        fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)
        for i in range(len(spike_times_per_event)):
            axs[0].eventplot(spike_times_per_event[i], lineoffsets=i, color=color, linelengths=9)
        axs[0].set_xlim(t_range)
        axs[0].set_ylabel('Trial')
        axs[0].set_title('Raster plot')
        axs[0].vlines(0, axs[0].get_ylim()[0], axs[0].get_ylim()[1])

        # Create firing rate plot
        sns.lineplot(x=t, y=zscored_rate, ax=axs[1], color=color)
        axs[1].fill_between(t, zscored_rate - np.std(zscored_rate), zscored_rate + np.std(zscored_rate),
                            alpha=0.2, color=color)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Z-scored firing rate')
        axs[1].set_title('Z-scored smoothed firing rate')
        axs[1].vlines(0, axs[1].get_ylim()[0], axs[1].get_ylim()[1])

        plt.tight_layout()
        
    return zscored_rate
