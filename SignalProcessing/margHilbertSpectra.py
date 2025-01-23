import os
import mne
import time
import numpy as np
import pandas as pd

from neuronol_signalprocessing import *

## Given

dir_data = os.path.expanduser('~/data/wash-u/preprocessed/v2/')
dir_results = os.path.expanduser('~/research/results/wash-u/hht/')

## Calculate Hilbert spectra for all subjects

T_max = None

# Start clocking execution
start_time = time.time()

truncated_files = []
fnames = sorted([f for f in os.listdir(dir_data) if f[-4:] == '.fif'])
n_files = len(fnames)
for i_file, fname_raw in enumerate(fnames):
    print("_" * 50); print("\nRunning File-%d out of %d files...\n" % (i_file + 1, n_files))

    # Read preprocessed EEG data
    fpath_raw = os.path.join(dir_data, fname_raw)
    raw = mne.io.read_raw_fif(fpath_raw)

    # Find intervals for eyes-closed resting state
    ec_end = [raw.annotations.onset[i] for i, annt in
              enumerate(raw.annotations.description) if annt in ('1', '3')]
    ec_intvl = {}
    ec_intvl['EC1'] = np.floor([ec_end[0] - 63, ec_end[0] - 3])
    ec_intvl['EC2'] = np.floor([ec_end[1] - 63, ec_end[1] - 3])

    for ec in ec_intvl:
        print('\nEvaluating Condition-%s...' % ec)

        tmin, tmax = ec_intvl[ec]
        if tmin > 0:
            cropped_raw = raw.copy().crop(tmin=tmin, tmax=tmax)
            if T_max is None:
                T_max = cropped_raw.times[-1] - cropped_raw.times[0]

            # Get all EEG channel names
            eeg_chs = cropped_raw.copy().pick_types(eeg=True).ch_names

            # Calculate marginal spectra of all channels in the given raw
            marg_power_specs, f_marg = hilbert_spectra_from_raw(cropped_raw, compute_power_spec=True,
                                                                T_max=T_max)
            
            # Export spectra in CSV format as a DataFrame
            dict_marg_specs = {'Frequency': f_marg}
            for i_ch, ch_name in enumerate(eeg_chs):
                dict_marg_specs[ch_name] = marg_power_specs[i_ch]
            df_marg_specs = pd.DataFrame.from_dict(dict_marg_specs)
            fname_marg_specs = fname_raw[:15] + '_cond=%s_Tmax=%.3f_hht-marg-spectra.csv' % (ec, T_max)
            fpath_marg_specs = os.path.join(dir_results, 'MarginalSpectra', fname_marg_specs)
            df_marg_specs.to_csv(fpath_marg_specs, index=False)
        else:
            truncated_files.append((fname_raw, ec))
            
# End clocking execution and display result
execution_time = (time.time() - start_time)
print('Execution time in seconds:', execution_time)