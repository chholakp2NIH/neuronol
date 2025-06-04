import cv2
import numpy as np
import scipy as sp
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import pandas as pd
from PyEMD import EMD
from neuronol_power import featureFunc_bandAbsolutePower, featureFunc_bandRelativePower, \
                           featureFunc_centroidFrequencyBand, featureFunc_dominantFrequency

def getAllMarginalSpectrumFeatures(raw, freq_bands, T_max=None):
    '''
    Evaluate all features from marginal Hilbert spectrum for the given mne.Raw object.
    '''
    # Get all EEG channel names
    eeg_chs = raw.copy().pick_types(eeg=True).ch_names

    # Calculate marginal spectra of all channels in the given raw
    marg_power_specs, f_marg = hilbert_spectra_from_raw(raw, compute_power_spec=True,
                                                        T_max=T_max)

    ## Collect all features in dict `F`
    F = {}

    # M1 features
    F1_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M1-HighBeta'],
                                           freq_bands['M1-Range'])
    F['1'] = (F1_HHT, '')

    # M2 features
    F2_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M2-Beta1+Beta2'],
                                           norm_and_log_density=True)
    frontal_chs = np.array([i_ch for i_ch, ch in enumerate(eeg_chs) if 'F' in ch])
    F3_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M2-Beta3'],
                                           channels=frontal_chs, norm_and_log_density=True)
    F['2'] = (F2_HHT, 'TODO')
    F['3'] = (F3_HHT, 'TODO')

    # M3 features
    n_eeg_channels = len(eeg_chs)
    F4_HHT = []; F5_HHT = []; F6_HHT = []; F7_HHT = []; F8_HHT = []; F9_HHT = [];
    F10_HHT = []; F11_HHT = []; F12_HHT = []; F13_HHT = []; F14_HHT = [];
    for i_ch in range(n_eeg_channels):
        F4_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Delta'],
                                                    channels=[i_ch], compute_magnitude=True))
        F5_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Theta'],
                                                    channels=[i_ch], compute_magnitude=True))
        F6_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Alpha'],
                                                    channels=[i_ch], compute_magnitude=True))
        F7_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Beta1'],
                                                    channels=[i_ch], compute_magnitude=True))
        F8_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Beta3'],
                                                    channels=[i_ch], compute_magnitude=True))
        F9_HHT.append(featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M3-Delta'],
                                                    freq_bands['M3-Range'], channels=[i_ch]))
        F10_HHT.append(featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M3-Theta'],
                                                     freq_bands['M3-Range'], channels=[i_ch]))
        F11_HHT.append(featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M3-Alpha'],
                                                     freq_bands['M3-Range'], channels=[i_ch]))
        F12_HHT.append(featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M3-Beta1'],
                                                     freq_bands['M3-Range'], channels=[i_ch]))
        F13_HHT.append(featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M3-Beta3'],
                                                     freq_bands['M3-Range'], channels=[i_ch]))
        F14_HHT.append(featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M3-Range'],
                                                     channels=[i_ch]))
    F['4'] = (F4_HHT, 'TODO')
    F['5'] = (F5_HHT, 'TODO')
    F['6'] = (F6_HHT, 'TODO')
    F['7'] = (F7_HHT, 'TODO')
    F['8'] = (F8_HHT, 'TODO')
    F['9'] = (F9_HHT, '')
    F['10'] = (F10_HHT, '')
    F['11'] = (F11_HHT, '')
    F['12'] = (F12_HHT, '')
    F['13'] = (F13_HHT, '')
    F['14'] = (F14_HHT, 'TODO')

    # M4 features
    F15_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Range'])
    F16_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band1'])
    F17_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band2'])
    F18_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band3'])
    F19_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band4'])
    F20_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band5'])
    F21_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band6'])
    F22_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band7'])
    F23_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band8'])
    F24_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Band9'])
    F25_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Delta+Theta'])
    F26_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Alpha'])
    F27_HHT = featureFunc_bandAbsolutePower(marg_power_specs, f_marg, freq_bands['M4-Beta'])

    F28_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band1'],
                                            freq_bands['M4-Range'])
    F29_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band1'],
                                            freq_bands['M4-Range'])
    F30_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band2'],
                                            freq_bands['M4-Range'])
    F31_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band3'],
                                            freq_bands['M4-Range'])
    F32_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band4'],
                                            freq_bands['M4-Range'])
    F33_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band5'],
                                            freq_bands['M4-Range'])
    F34_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band6'],
                                            freq_bands['M4-Range'])
    F35_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band7'],
                                            freq_bands['M4-Range'])
    F36_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Band8'],
                                            freq_bands['M4-Range'])
    F37_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Delta+Theta'],
                                            freq_bands['M4-Range'])
    F38_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Alpha'],
                                            freq_bands['M4-Range'])
    F39_HHT = featureFunc_bandRelativePower(marg_power_specs, f_marg, freq_bands['M4-Beta'],
                                            freq_bands['M4-Range'])


    F40_HHT, F41_HHT, F42_HHT = featureFunc_dominantFrequency(marg_power_specs, f_marg,
                                                              freq_bands['M4-Range'])

    F43_HHT, F47_HHT = featureFunc_centroidFrequencyBand(marg_power_specs, f_marg,
                                                         freq_bands['M4-Delta+Theta'])
    F44_HHT, F48_HHT = featureFunc_centroidFrequencyBand(marg_power_specs, f_marg,
                                                         freq_bands['M4-Alpha'])
    F45_HHT, F49_HHT = featureFunc_centroidFrequencyBand(marg_power_specs, f_marg,
                                                         freq_bands['M4-Beta'])
    F46_HHT, F50_HHT = featureFunc_centroidFrequencyBand(marg_power_specs, f_marg,
                                                         freq_bands['M4-Range'])

    F['15'] = (F15_HHT, 'TODO')
    F['16'] = (F16_HHT, 'TODO')
    F['17'] = (F17_HHT, 'TODO')
    F['18'] = (F18_HHT, 'TODO')
    F['19'] = (F19_HHT, 'TODO')
    F['20'] = (F20_HHT, 'TODO')
    F['21'] = (F21_HHT, 'TODO')
    F['22'] = (F22_HHT, 'TODO')
    F['23'] = (F23_HHT, 'TODO')
    F['24'] = (F24_HHT, 'TODO')
    F['25'] = (F25_HHT, 'TODO')
    F['26'] = (F26_HHT, 'TODO')
    F['27'] = (F27_HHT, 'TODO')
    F['28'] = (F28_HHT, '')
    F['29'] = (F29_HHT, '')
    F['30'] = (F30_HHT, '')
    F['31'] = (F31_HHT, '')
    F['32'] = (F32_HHT, '')
    F['33'] = (F33_HHT, '')
    F['34'] = (F34_HHT, '')
    F['35'] = (F35_HHT, '')
    F['36'] = (F36_HHT, '')
    F['37'] = (F37_HHT, '')
    F['38'] = (F38_HHT, '')
    F['39'] = (F39_HHT, '')
    F['40'] = (F40_HHT, 'Hz')
    F['41'] = (F41_HHT, 'TODO')
    F['42'] = (F42_HHT, '')
    F['43'] = (F43_HHT, 'Hz')
    F['44'] = (F44_HHT, 'Hz')
    F['45'] = (F45_HHT, 'Hz')
    F['46'] = (F46_HHT, 'Hz')
    F['47'] = (F47_HHT, 'TODO')
    F['48'] = (F48_HHT, 'TODO')
    F['49'] = (F49_HHT, 'TODO')
    F['50'] = (F50_HHT, 'TODO')

    return F

def hilbert_spectra_from_raw(raw, compute_power_spec=None, T_max=None, verbose=None):
    '''
    Estimates (marginal) Hilbert spectra of all EEG channels in raw.
    '''
    # Extract parameters and raw data
    Fs = np.floor(raw.info['sfreq'])
    if T_max is None:
        T_max = raw.times[-1] - raw.times[0]
    t = np.arange(0, T_max, 1 / Fs) # time vector

    # Get raw data
    raw = raw.copy()
    raw_data = raw.get_data(picks='eeg') * 1e6 # in μV
    raw_data = raw_data[:, :int(Fs * T_max)]
    if verbose and verbose is not None:
        print('Shape of raw_data =', raw_data.shape)

    # Perform EMD-HHT on each channel separately
    n_chs = raw_data.shape[0]
    marg_specs = []
    for i_ch in range(n_chs):
        x = raw_data[i_ch, :]
        imfs = perform_EMD(x, plot_emd=False)
        if verbose and verbose is not None:
            print('Found a total of %02d IMFs' % len(imfs))
        C = imfs[:-1]
        _, _, _, f_hht_marg, marg_spec = calculate_hilbert_spectrum(C, t, Fs,
                                                    compute_power_spec=compute_power_spec)
        marg_specs.append(marg_spec)

    return np.array(marg_specs), f_hht_marg

def perform_EMD(x, t=None, plot_emd=None):
    '''
    Perform empirical mode decomposition on signal block 'x'.
    '''
    # EMD
    emd_decomp = EMD()
    imfs = emd_decomp(x)

    # Visualize EMD
    if plot_emd and plot_emd is not None:
        if t is None:
            t = np.arange(len(x))
        plt.figure(figsize=(12, 12))
        for i in range(len(imfs)-1):
            plt.subplot(len(imfs), 1, i+1)
            plt.plot(t, x, color='0.8')
            plt.plot(t, imfs[i], 'k')
            plt.xticks([])
            plt.xlim([np.min(t), np.max(t)])
            plt.ylabel('IMF ' + str(i + 1))
        plt.subplot(len(imfs), 1, i+2)
        plt.plot(t, x, color='0.8')
        plt.plot(t, imfs[-1], 'k')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylabel('Residual')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
    return imfs

def _downsample_rows(arr, k):
    '''
    Downsample a measurement matrix along its rows.
    '''
    res = np.cumsum(arr, 0)[k-1::k]
    res[1:] = res[1:] - res[:-1]
    return res / k

def calculate_hilbert_spectrum(imfs, t, fs, n=5, k_dwnsamp=3, k_gauss=15,
                               compute_power_spec=None,
                               smoothing_downsample_freq=None,
                               smoothing_gauss_filt=None,
                               plot_marginal_hilbert_spec=None,
                               plot_hilbert_spec=None, plot_inst_freq=None):
    '''
    Calculate hilbert amplitude spectrum from a given set of intrinsic mode functions.
    '''

    ## Create Hilbert spectrum
    delta_t = 1 / fs; T = t[-1] - t[0] + delta_t
    fmin = fres = 1 / T; fmax = 1 / (n * delta_t)
    N = int(T / (n * delta_t))
    bin_centres = np.arange(N) * fres + fmin
    bin_edges = np.arange(N + 1) * fres + (fmin - fres / 2)

    f_hht = bin_centres
    hhts = np.zeros((len(imfs), N, (len(t) - 2)))

    for j, imf in enumerate(imfs):
        Z = hilbert(imf)
        A = np.abs(Z)
        theta_inst = np.unwrap(np.angle(Z))
        f_inst = 0.5 * (np.angle(-Z[2:] * np.conj(Z[:-2])) + np.pi) / (2 * np.pi) * fs
        t_hht = t[1:-1]; A_hht = A[1:-1]; P_hht = A[1:-1] ** 2

        # Plot instantaneous frequency curves
        if plot_inst_freq and plot_inst_freq is not None:
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            ax0.plot(t, imf, label='signal')
            ax0.plot(t, A, label='envelope')
            ax0.set_xlabel("time (s)")
            ax0.set_ylabel("signal (units)")
            ax0.legend()
            ax1.plot(t_hht, f_inst)
            ax1.set_xlabel("time (s)")
            ax1.set_ylabel("frequency (Hz)")
            fig.tight_layout()
            plt.show()

        # Binning of frequency values
        binned_freq = pd.cut(f_inst, bin_edges)
        bin_inds = binned_freq.codes

        # Populate Hilbert spectrum matrix
        if compute_power_spec and compute_power_spec is not None:
            for i, bin_ind in enumerate(bin_inds):
                if bin_ind > 0:
                    hhts[j][bin_ind][i] = P_hht[i]
        else:
            for i, bin_ind in enumerate(bin_inds):
                if bin_ind > 0:
                    hhts[j][bin_ind][i] = A_hht[i]

    hht = np.sum(hhts, axis=0)

    # Calculate marginal Hilbert spectrum
    marginal_spec = np.mean(hht, axis=1)
    f_hht_marginal = f_hht

    # Smoothing - Downsample Frequency in HHT
    if smoothing_downsample_freq and smoothing_downsample_freq is not None:
        hht = _downsample_rows(hht, k_dwnsamp)
        f_hht = _downsample_rows(f_hht, k_dwnsamp)

    # Smoothing - Weighted Gaussian Filtering
    if smoothing_gauss_filt and smoothing_gauss_filt is not None:
        hht = cv2.GaussianBlur(hht, (k_gauss, k_gauss), 0)

    # Plot Hilbert spectrum for all IMFs
    if plot_hilbert_spec and plot_hilbert_spec is not None:
        plt.figure()
        plt.pcolormesh(t_hht, f_hht, hht, cmap='hot')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    # Plot marginal Hilbert spectrum
    if plot_marginal_hilbert_spec and plot_marginal_hilbert_spec is not None:
        plt.figure()
        plt.plot(f_hht_marginal, marginal_spec)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Marginal Hilbert Spectrum')
        plt.show()

    return hht, t_hht, f_hht, f_hht_marginal, marginal_spec

def instantaneous_band_power(hht, f_hht, band):
    '''
    From HHT power spectrum, calculate instantaneous total power in a given frequency band.
    '''
    # Isolate Hilbert spectrum of the given frequency band
    inds_range = [(np.abs(f_hht - band[0])).argmin(), (np.abs(f_hht - band[1])).argmin()]
    inds = np.arange(inds_range[0], inds_range[1]+1)
    hht_band, f_band = hht[inds, :], f_hht[inds]

    # Sum power over given frequency band
    power_band = np.sum(hht_band, axis=0)

    return power_band

def Appendix__imperfect_emd(x, t=None, tol_sd=0.2, max_IMFs=25, max_siftings=100, plot_emd=None):
    """
    Perform empirical mode decomposition on a signal 'x' as described in Huang et al. 1998.
    The decomposition terminates whence either the sifting process is unable to find local
    peaks or valleys in the residual signal or the max. no. of intended IMFs have already
    been extracted.

    Parameters
    ----------
    x : 1D array
        Signal of interest.
    t : 1D array
        Time (or space) at which elements of 'x' were measured.
    tol_sd : scalar
        Tolerance in standard deviation between two consecutive siftings. Used as a stopping
        criterion. See Eq. (5.5) in Huang et al. 1998 for more details.
    max_IMFs : scalar
        Max. no. of IMFs to be extracted.
    max_siftings : scalar
        Max. no. of siftings to be performed for extracting each IMF.

    Returns
    -------
    c : 2D array
        IMFs extracted from the EMD.
    r : 1D array
        Residual signal.
    """
    if t is None:
        t = np.array(range(len(x)))
        label_x = ''
    else:
        label_x = 'Time (s)'
    r = x
    c = np.zeros((max_IMFs, len(x)))
    i = 0; stop_emd = False
    while (i < max_IMFs) and not stop_emd:
        print(i + 1)
        h_km1 = r
        k = 1
        while k < max_siftings:
            # Find upper and lower extrema and include first and last point of the signal
            j_pks = signal.argrelmax(h_km1)[0]; j_pks = np.append(np.append(0, j_pks), -1)
            j_vls = signal.argrelmin(h_km1)[0]; j_vls = np.append(np.append(0, j_vls), -1)

            # Make upper and lower envelopes
            if len(j_pks) > 3:
                spl_up = sp.interpolate.InterpolatedUnivariateSpline(t[j_pks], h_km1[j_pks], k=3)
                envp_up = spl_up(t)
            elif len(j_pks) > 2:
                spl_up = sp.interpolate.InterpolatedUnivariateSpline(t[j_pks], h_km1[j_pks], k=2)
                envp_up = spl_up(t)
            else:
                print('No local peaks found! Stopping.')
                stop_emd = True
                break
            if len(j_vls) > 3:
                spl_lw = sp.interpolate.InterpolatedUnivariateSpline(t[j_vls], h_km1[j_vls], k=3)
                envp_lw = spl_lw(t)
            elif len(j_vls) > 2:
                spl_lw = sp.interpolate.InterpolatedUnivariateSpline(t[j_vls], h_km1[j_vls], k=2)
                envp_lw = spl_lw(t)
            else:
                print('No local valleys found! Stopping.')
                stop_emd = True
                break

            # Calculate mean envelope
            m_k = (envp_up + envp_lw) / 2

            # Find the next sifted signal
            h_k = h_km1 - m_k

            # Test for stopping criterion
            sd = np.sum((h_km1 - h_k) ** 2) / np.sum(h_km1 ** 2)
            print(sd)
            if sd < tol_sd:
                print('IMF found!')
                c[i] = h_k
                break
            else:
                k += 1
                h_km1 = h_k
        r = r - c[i]
        i += 1

    # Delete extra zero rows that haven't been populated
    c = np.delete(c, range(i-1, max_IMFs), axis=0)

    # Plot
    if plot_emd and plot_emd is not None:
        plt.figure(figsize=(12, 12))
        for i in range(len(c)):
            plt.subplot(len(c)+1, 1, i+1)
            plt.plot(t, x, color='0.8')
            plt.plot(t, c[i], 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.ylabel('IMF ' + str(i + 1))
        plt.subplot(len(c)+1, 1, i+2)
        plt.plot(t, x, color='0.8')
        plt.plot(t, r, 'k')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylabel('Residual')
        plt.xlabel(label_x)
        plt.tight_layout()
        plt.show()

    return c, r

def Appendix__derivative_central_secondorder(x, fs):
    '''
    Calculate derivative of signal 'x' sampled at frequency 'fs'
    using central difference method of second order.
    '''
    n = len(x)
    dxdt = np.zeros((n, 1))
    dxdt[0] = (x[1] - x[0]) * fs # 1 / dt = fs
    for i in range(1, n-1):
        dxdt[i] = (x[i+1] - x[i-1]) / 2 * fs
    dxdt[n-1] = (x[n-1] - x[n-2]) * fs
    return dxdt

def Appendix__derivative_central_fourthorder(x, fs):
    '''
    Calculate derivative of signal 'x' sampled at frequency 'fs'
    using central difference method of fourth order.
    '''
    n = len(x)
    dxdt = np.zeros((n, 1))
    dxdt[0] = (x[1] - x[0]) * fs # 1 / dt = fs
    dxdt[1] = (x[2] - x[0]) * fs / 2
    for i in range(2, n-2):
        dxdt[i] = (- x[i+2] + 8*x[i+1] - 8*x[i-1] + x[i-2]) * fs / 12
    dxdt[n-2] = (x[n-1] - x[n-3]) * fs / 2
    dxdt[n-1] = (x[n-1] - x[n-2]) * fs
    return dxdt
