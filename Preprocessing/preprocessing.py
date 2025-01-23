import mne
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

def preprocessEEGWashU_v2(fname_raw, data_path, res_path,
                          clean_cardiac_ssp=True, clean_ocular_ica=True,
                          clean_cardiac_ica=False, make_report=True):
    '''
    Preprocess EEG data from Wash-U dataset. Remove artifacts from horizontal
    ocular and cardiac activity using ICA and SSP respectively.
    '''
    ## Read sample data and instantiate Report

    # Setup
    fname_report = fname_raw[:-4] + '_ica_ssp.html'
    fname_reconst_raw = fname_raw[:-4] + '_ica_ssp_eeg.fif'
    fpath_raw = op.join(data_path, fname_raw)
    fpath_report = op.join(res_path, fname_report)
    fpath_reconst_raw = op.join(data_path, '../preprocessed/v2/', fname_reconst_raw)
    figs = []; titles = []

    # Read data and instantiate Report
    raw = mne.io.read_raw_cnt(fpath_raw, eog='auto', ecg=['EKG'])
    raw.drop_channels(['M2', 'VEOG']) # Drop 'M2' (ref) and 'VEOG' (blinks already corrected)
    if make_report:
        report = mne.Report(title='Ocular and cardiac artifact correction')
        report.add_raw(raw=raw, title='Raw', psd=False)

    # Load data into memory
    raw.load_data()

    ## Use ICA to remove ocular and/or cardiac artifacts

    if clean_cardiac_ica or clean_ocular_ica:
        # Filtering to remove slow drifts - Since filtering is a linear operator, the
        # ICA solution found from the filtered signal can be applied to the unfiltered
        # signal
        filt_raw_ica = raw.copy().filter(l_freq=0.5, h_freq=None)

        # Fitting the ICA solution
        ica = mne.preprocessing.ICA(n_components=30, max_iter='auto', random_state=97)
        ica.fit(filt_raw_ica)
        # ica

        eog_indices_ica = []; ecg_indices_ica = []

    if clean_ocular_ica:
        # Start with empty list of excluded ICs
        ica.exclude = []

        # Get evoked ocular activity
        eog_evoked_ica = mne.preprocessing.create_eog_epochs(raw).average()
        eog_evoked_ica.apply_baseline(baseline=(None, -0.2))
        figs.append(eog_evoked_ica.plot_joint())
        titles.append('Evoked EOG')

        # Find which ICs match the EOG pattern
        eog_indices_ica, eog_scores_ica = ica.find_bads_eog(raw)
        ica.exclude = eog_indices_ica

        # Add ICA to Report
        if make_report:
            report.add_ica(ica=ica,
                           title='ICA cleaning of ocular artifacts',
                           inst=raw,
                           eog_evoked=eog_evoked_ica,
                           eog_scores=eog_scores_ica,
                           n_jobs=4 # could be increased
                          )

        # Plot ICs applied to raw data with EOG matches highlighted
        if ica.n_components > 15:
            ica_inds = np.arange(ica.n_components)
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[:15]))
            titles.append('EOG: Plot ICs applied to raw data with EOG matches highlighted - Part 1')
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[15:]))
            titles.append('EOG: Plot ICs applied to raw data with EOG matches highlighted - Part 2')
        else:
            figs.append(ica.plot_sources(raw, show_scrollbars=False))
            titles.append('EOG: Plot ICs applied to raw data with EOG matches highlighted')

    if clean_cardiac_ica:
        # Start with empty list of excluded ICs
        ica.exclude = []

        # Get evoked cardiac activity
        ecg_evoked_ica = mne.preprocessing.create_ecg_epochs(raw).average()
        ecg_evoked_ica.apply_baseline(baseline=(None, -0.2))
        figs.append(ecg_evoked_ica.plot_joint())
        titles.append('Evoked ECG')

        # Find which ICs match the ECG pattern and select only the first ICA component
        ecg_indices_ica_all, ecg_scores_ica_all = ica.find_bads_ecg(raw)
        ecg_indices_ica = [ecg_indices_ica_all[0]]
        ecg_scores_ica = np.zeros((ica.n_components,))
        ecg_scores_ica[ecg_indices_ica] = ecg_scores_ica_all[ecg_indices_ica]
        ica.exclude = ecg_indices_ica

        # Add ICA to Report
        if make_report:
            report.add_ica(ica=ica,
                           title='ICA cleaning of cardiac artifacts',
                           inst=raw,
                           ecg_evoked=ecg_evoked_ica,
                           ecg_scores=ecg_scores_ica,
                           n_jobs=4 # could be increased
                          )

        # Plot ICs applied to raw data with ECG matches highlighted
        if ica.n_components > 15:
            ica_inds = np.arange(ica.n_components)
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[:15]))
            titles.append('ECG: Plot ICs applied to raw data with ECG matches highlighted - Part 1')
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[15:]))
            titles.append('ECG: Plot ICs applied to raw data with ECG matches highlighted - Part 2')
        else:
            figs.append(ica.plot_sources(raw, show_scrollbars=False))
            titles.append('ECG: Plot ICs applied to raw data with ECG matches highlighted')

    reconst_raw = raw.copy()
    if clean_cardiac_ica or clean_ocular_ica:
        # Combine ICs with ocular and cardiac artifacts
        ica.exclude = eog_indices_ica + ecg_indices_ica

        # Reconstruction of raw data after ICA exclusion
        ica.apply(reconst_raw)

    ## Use SSP to remove cardiac artifacts

    if clean_cardiac_ssp:
        # Visualizing the artifacts
        ecg_epochs_ssp = mne.preprocessing.create_ecg_epochs(raw)
        ecg_evoked_ssp = ecg_epochs_ssp.average()
        ecg_evoked_ssp.apply_baseline((None, None))
        figs.append(ecg_evoked_ssp.plot_joint())
        titles.append('Evoked ECG')

        # Calculate ECG-removal projection matrix
        ecg_projs_ssp, _ = mne.preprocessing.compute_proj_ecg(raw, n_eeg=2, reject=None)

        # Visualize the topographies and effects of removed ECG projections
        figs.append(mne.viz.plot_projs_joint(ecg_projs_ssp, ecg_evoked_ssp, picks_trace=None))
        titles.append('Removed projectors of cardiac activity')

        # Add SSP projections to the raw data and activate them
        reconst_raw.add_proj(ecg_projs_ssp)
        reconst_raw.apply_proj()

    ## Visualize repair, save reconstructed data, and generate diagnostic report

    # Visualize repair
    n_blocks = int(np.ceil(raw.times[-1] / 10))
    figs_raw = []; caps_raw = []; figs_reconst_raw = []; caps_reconst_raw = []
    for i_blk in range(n_blocks):
        start_time = 0 + i_blk * 10
        figs_raw.append(raw.plot(n_channels=len(raw), show_scrollbars=False,
                                 start=start_time, duration=10))
        caps_raw.append('Original raw: Slide-%02d' % (i_blk + 1))
        figs_reconst_raw.append(reconst_raw.plot(n_channels=len(raw), show_scrollbars=False,
                                                 start=start_time, duration=10))
        caps_reconst_raw.append('Reconstructed raw: Slide-%02d' % (i_blk + 1))

    # Export preprocessed data to a new .fif file
    reconst_raw.save(fpath_reconst_raw, overwrite=True)

    # Save final report
    if make_report:
        for fig, title in zip(figs, titles):
            report.add_figure(fig=fig, title=title)
        report.add_figure(fig=figs_raw, title='Original raw', caption=caps_raw)
        report.add_figure(fig=figs_reconst_raw, title='Reconstructed raw',
                          caption=caps_reconst_raw)
        report.save(fpath_report, overwrite=True, open_browser=False)
        plt.close('all')

    return reconst_raw

def preprocessEEGWashU_v1(fname_raw, data_path, res_path,
                          clean_ocular=True, clean_cardiac=True, make_report=True):
    '''
    Preprocess EEG data from Wash-U dataset. Remove artifacts from horizontal ocular
    and cardiac activity using ICA.
    '''
    # Setup
    fname_report = fname_raw[:-4] + '.html'
    fname_reconst_raw = fname_raw[:-4] + '_eeg.fif'
    fpath_raw = op.join(data_path, fname_raw)
    fpath_report = op.join(res_path, fname_report)
    fpath_reconst_raw = op.join(data_path, '../preprocessed/v1/', fname_reconst_raw)
    figs = []; captions = []

    # Read data and instantiate Report
    raw = mne.io.read_raw_cnt(fpath_raw, eog='auto', ecg=['EKG'])
    raw.drop_channels(['M2', 'VEOG']) # Drop 'M2' (ref) and 'VEOG' (blinks already corrected)
    if make_report:
        report = mne.Report(title='Ocular and cardiac artifact correction')
        report.add_raw(raw=raw, title='Raw', psd=False)

    # Load data into memory
    raw.load_data()

    # Filtering to remove slow drifts - Since filtering is a linear operator, the
    # ICA solution found from the filtered signal can be applied to the unfiltered
    # signal
    filt_raw_ica = raw.copy().filter(l_freq=0.5, h_freq=None)

    # Fitting the ICA solution
    ica = mne.preprocessing.ICA(n_components=30, max_iter='auto', random_state=97)
    ica.fit(filt_raw_ica)

    # Find and remove ICA components correlated to artifacts
    eog_indices = []; ecg_indices = []

    if clean_ocular:
        # Start with empty list of excluded ICs
        ica.exclude = []

        # Get evoked ocular activity
        eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
        eog_evoked.apply_baseline(baseline=(None, -0.2))
        figs.append(eog_evoked.plot_joint())
        captions.append('Evoked EOG')

        # Find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        ica.exclude = eog_indices

        # Add ICA to Report
        if make_report:
            report.add_ica(ica=ica,
                           title='ICA cleaning of ocular artifacts',
                           inst=raw,
                           eog_evoked=eog_evoked,
                           eog_scores=eog_scores,
                           n_jobs=4 # could be increased
                          )

        # Plot ICs applied to raw data with EOG matches highlighted
        if ica.n_components > 15:
            ica_inds = np.arange(ica.n_components)
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[:15]))
            captions.append('EOG: Plot ICs applied to raw data with EOG matches highlighted - Part 1')
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[15:]))
            captions.append('EOG: Plot ICs applied to raw data with EOG matches highlighted - Part 2')
        else:
            figs.append(ica.plot_sources(raw, show_scrollbars=False))
            captions.append('EOG: Plot ICs applied to raw data with EOG matches highlighted')

    if clean_cardiac:
        # Start with empty list of excluded ICs
        ica.exclude = []

        # Get evoked cardiac activity
        ecg_evoked = mne.preprocessing.create_ecg_epochs(raw).average()
        ecg_evoked.apply_baseline(baseline=(None, -0.2))
        figs.append(ecg_evoked.plot_joint())
        captions.append('Evoked ECG')

        # Find which ICs match the ECG pattern and select only the first ICA component
        ecg_indices_all, ecg_scores_all = ica.find_bads_ecg(raw)
        ecg_indices = [ecg_indices_all[0]]
        ecg_scores = np.zeros((ica.n_components,))
        ecg_scores[ecg_indices] = ecg_scores_all[ecg_indices]
        ica.exclude = ecg_indices

        # Add ICA to Report
        if make_report:
            report.add_ica(ica=ica,
                           title='ICA cleaning of cardiac artifacts',
                           inst=raw,
                           ecg_evoked=ecg_evoked,
                           ecg_scores=ecg_scores,
                           n_jobs=4 # could be increased
                          )

        # Plot ICs applied to raw data with ECG matches highlighted
        if ica.n_components > 15:
            ica_inds = np.arange(ica.n_components)
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[:15]))
            captions.append('ECG: Plot ICs applied to raw data with ECG matches highlighted - Part 1')
            figs.append(ica.plot_sources(raw, show_scrollbars=False, picks=ica_inds[15:]))
            captions.append('ECG: Plot ICs applied to raw data with ECG matches highlighted - Part 2')
        else:
            figs.append(ica.plot_sources(raw, show_scrollbars=False))
            captions.append('ECG: Plot ICs applied to raw data with ECG matches highlighted')

    # Combine ICs with ocular and cardiac artifacts
    ica.exclude = eog_indices + ecg_indices

    # Reconstruction of raw data after ICA exclusion
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    # Visualize repair
    figs.append(raw.plot(n_channels=len(raw), show_scrollbars=False))
    captions.append('Original raw')
    figs.append(reconst_raw.plot(n_channels=len(raw), show_scrollbars=False))
    captions.append('Reconstructed raw')

    # Export preprocessed data to a new .fif file
    reconst_raw.save(fpath_reconst_raw, overwrite=True)

    # Save final report
    if make_report:
        for fig, caption in zip(figs, captions):
            report.add_figure(fig=fig, title=caption)
        report.save(fpath_report, overwrite=True, open_browser=False)
        plt.close('all')

    return reconst_raw
