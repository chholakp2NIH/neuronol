import datetime as dt
import json
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import re
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat

class CustomError(Exception):
    """
    Exception raised for custom error scenarios.

    Attributes:
        message: explanation of the error.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        
def create_MNE_montage_from_BS_dig_data_new(fpath_dig):
    """
    Reads Brainstorm Digitizer data (mat-file) to generate a MNE montage object.
    Works on data collected via Windows MATLAB R2024b; BST Version: 06-May-2025.
    """
    # Load digitized data
    data = loadmat(fpath_dig)

    # Extract channel, headshape and fiducial locations
    #    Channel locations
    df_channel_locs = pd.DataFrame(data["Channel"].transpose().flatten())
    ch_pos = {u[0]: v.flatten() for u, v in zip(df_channel_locs["Name"], df_channel_locs["Loc"])}

    #    Headspace and fiducial locations
    df_headpoint_locs = pd.DataFrame(data["HeadPoints"].flatten())
    lbls_hp_arr = df_headpoint_locs['Label'].to_numpy()
    lbls_hp = [
        x.item() if isinstance(x, np.ndarray) and x.size == 1 else 'EXTRA'
        for x in lbls_hp_arr[0][0]
    ]
    locs_hp = [w for w in df_headpoint_locs["Loc"][0].transpose()]
    nasion = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "NAS"], axis=0)
    lpa = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "LPA"], axis=0)
    rpa = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "RPA"], axis=0)
    hsp = np.array([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "EXTRA"])

    # Create montage
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp)

    return dig

def create_MNE_montage_from_BS_dig_data_old(fpath_dig):
    """
    Reads Brainstorm Digitizer data (mat-file) to generate a MNE montage object.
    Works on data collected via MacOS MATLAB R2022b; BST Version: 04-Mar-2024.
    """
    # Load digitized data
    data = loadmat(fpath_dig)

    # Extract channel, headshape and fiducial locations
    #    Channel locations
    df_channel_locs = pd.DataFrame(data["Channel"].transpose().flatten())
    ch_pos = {u[0]: v.flatten() for u, v in zip(df_channel_locs["Name"], df_channel_locs["Loc"])}

    #    Headspace and fiducial locations
    df_headpoint_locs = pd.DataFrame(data["HeadPoints"].flatten())
    lbls_hp = [w[0][0] for w in df_headpoint_locs["Label"][0].transpose()]
    locs_hp = [w for w in df_headpoint_locs["Loc"][0].transpose()]
    nasion = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "NA"], axis=0)
    lpa = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "LPA"], axis=0)
    rpa = np.average([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "RPA"], axis=0)
    hsp = np.array([loc for loc, lbl in zip(locs_hp, lbls_hp) if lbl == "EXTRA"])

    # Create montage
    dig = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=nasion, lpa=lpa, rpa=rpa, hsp=hsp)

    return dig

def read_event_timings_from_imotions_data(df_imotions, event_col, positive_event_val, t_0):
    df_event = df_imotions.drop(df_imotions.index[df_imotions[event_col].isna()])  # drop rows without event data
    if len(df_event[event_col]) == 0: raise CustomError("No data found in %s column." % event_col)
    df_event.dropna(axis=1, how='all', inplace=True)                # drop columns that don't have any data left
    df_event = df_event.drop(df_event.index[df_event[event_col] != positive_event_val])    # drop rows with negative event value (e.g., '0')
    times = (df_event['Timestamp'].astype(float) - t_0) / 1000      # in seconds
    times.reset_index(drop=True, inplace=True)
    return times.to_list()

def read_event_markers_from_imotions_data(df_imotions, t_0, t_win_dupl_markers=1, marker_col='Description'):
    '''
    Reads LSL markers from raw dataframe based on iMotions' exported data.
    Drops trigs that occur sooner than `t_win_dupl_markers` time (in s) after an existing trig.
    '''
    df_triggers = df_imotions.drop(df_imotions.index[df_imotions[marker_col].isna()])  # drop rows without trigger data
    df_triggers.dropna(axis=1, how='all', inplace=True)  # drop columns that don't have any data left
    df_triggers = df_triggers[['Timestamp', marker_col]]    # keep only trigger vals and their times

    # Drop rows for "trials-start" triggers
    df_triggers = df_triggers.drop(df_triggers.index[df_triggers[marker_col] == 'trials-start'])
    
    # Drop rows with duplicate trigger entries
    df_triggers['Timestamp'] = df_triggers['Timestamp'].astype('float')
    df_triggers.reset_index(inplace=True, drop=True)
    times = df_triggers['Timestamp'] / 1000 # in seconds
    tdiff = np.diff(times)  # differences between subsequent time stamps
    keep_inds = [0] + \
        [(i + 1) for i, w in enumerate(tdiff) if w > t_win_dupl_markers] # jumps in tdiff
    df_triggers = df_triggers.loc[keep_inds]
    times_event = (df_triggers['Timestamp'].astype(float) - t_0) / 1000     # in seconds
    event_desc = df_triggers['Description']
    
    return times_event.to_list(), event_desc.to_list()

def read_imotions_data(fpath_imotions_data):
    """
    Read data exported from iMotions and convert to a
    compact/filtered dataframe and other variables.
    """
    
    # Read CSV file with exported data from iMotions
    df_imotions = pd.read_csv(fpath_imotions_data, low_memory=False)

    # Find recording date-time
    inx_rectime = df_imotions.index[
        df_imotions.iloc[:, 0] == '#Recording time'
    ].to_list()[0]
    for data in df_imotions.iloc[inx_rectime].astype(str):
        if "Date" in data:
            m = re.search(r'^Date: (\d{4}-\d{2}-\d{2})$', data)
            date_str = m.group(1)
        elif "Time" in data:
            m = re.search(r'^Time: (.+) .+$', data)
            time_str = m.group(1)
    datetime_str = date_str + ' ' + time_str
    recording_dt = dt.datetime.fromisoformat(datetime_str)

    # Drop general info at top
    inx_first_timepoint = df_imotions.index[df_imotions.iloc[:, 0] == '1'].to_list()[0]
    inx_header = inx_first_timepoint - 1
    df_imotions.drop(df_imotions.head(inx_header).index, inplace=True)

    # Change header
    header = df_imotions.iloc[0].values
    df_imotions = df_imotions[1:]
    df_imotions.columns = header

    return df_imotions, recording_dt

def extract_eeg_from_imotions_data(df_imotions, possible_unwanted_columns):
    """
    Extract EEG data from iMotions dataframe. Returns EEG data in Volts.
    """

    df_eeg = df_imotions.drop(df_imotions.index[df_imotions['Fp1'].isna()]) # drop rows without EEG data
    df_eeg.dropna(axis=1, how='all', inplace=True)  # drop columns that don't have any data left
    available_unwanted_columns = [
        c for c in possible_unwanted_columns if c in df_eeg.columns
    ]
    df_eeg.drop(columns=available_unwanted_columns, inplace=True)   # drop avail unwanted cols
    df_eeg = df_eeg.astype('float')
    df_eeg.iloc[:, 1:] /= 1e6 # convert EEG signal unit from μV to V
    df_eeg.reset_index(drop=True, inplace=True)
    
    return df_eeg

def add_interpolated_ecg_to_eeg(df_imotions, col_ecg, df_eeg, interp='linear'):
    """
    Extracts ECG data from iMotions data, then interpolates it
    using `interp` interpolation
    to match the timestamps for EEG data before adding ECG data
    to the EEG dataframe.

    Args:
        interp (str): Interpolation method. Can either be 'linear'
                      (default) or 'B-spline'.
    """

    # Extract ECG data
    df_ecg = df_imotions.drop(df_imotions.index[df_imotions[col_ecg].isna()])  # drop rows without ECG data
    df_ecg.dropna(axis=1, how='all', inplace=True)              # drop columns that don't have any data left
    df_ecg = df_ecg[['Timestamp', col_ecg]] # keep only primary ECG channel and the timestamps
    df_ecg = df_ecg.astype(float)
    df_ecg.iloc[:, 1:] /= 1e3 # convert ECG signal unit from mV to V
    df_ecg.reset_index(drop=True, inplace=True)

    # Interpolate ECG data to match EEG timestamps and add to EEG df
    if interp == 'linear':
        df_eeg[col_ecg] = np.interp(
            df_eeg['Timestamp'], df_ecg['Timestamp'], df_ecg[col_ecg]
        )
    elif interp == 'B-spline':
        bspl = make_interp_spline(df_ecg['Timestamp'], df_ecg[col_ecg], k=3)    # B-spline cubic
        df_eeg[col_ecg] = bspl(df_eeg['Timestamp'])
    
    return df_eeg

def read_event_times_from_task(
        fpath_stimtime_task, recording_dt
):
    '''
    Reads event timestamps from Excel file, converts them to date-times,
    and subtracts starting time of recording (`recording_dt`),
    to finally produce event times and descriptions similar to triggers.
    '''

    # Read event stim timestamps from Excel
    df_stimtime_task = pd.read_excel(fpath_stimtime_task)
    
    # Drop rows for "trials-start" events
    df_stimtime_task = df_stimtime_task.drop(
        df_stimtime_task.index[
            df_stimtime_task['EventName'] == 'trials-start'
        ]
    )
    
    # Extract event names
    task_event_names = df_stimtime_task['EventName'].to_list()

    # Convert event timestamps to datetime
    task_stims_dt = [dt.datetime.fromtimestamp(ts)
                    for ts in df_stimtime_task['EventTime']]
    
    # Subtract recording datetime to get relative times of event stims
    task_event_times = [(w - recording_dt).total_seconds()
                        for w in task_stims_dt]
    
    return task_event_times, task_event_names

def read_event_markers_from_event_files(
        event_files_dir, tasknames, recording_dt
):
    """
    Scans `event_files_dir` for event files corresponding
    to the list of tasks in `tasknames`, reads the event
    times and labels from those files, and then calculates trigger
    times by referencing the event times to the date-time of
    data recording (`recording_dt`).
    """

    event_times, event_labels = [], []
    event_files = [
        f.name for task in tasknames
               for f in os.scandir(event_files_dir)
               if f.is_file() and task + "_eventsstimtimes" in f.name
    ]
    for fname_event_file in event_files:
        fpath_event_file = os.path.join(event_files_dir, fname_event_file)
        task_event_times, task_event_labels = read_event_times_from_task(
            fpath_event_file, recording_dt
        )
        event_times += task_event_times
        event_labels += task_event_labels

    return event_times, event_labels

def convert_df_eeg_to_mne_raw_object(
        df_eeg, ideal_ch_names, ecg_data_imported, sfreq,
        report=None, create_mne_report=False
):
    """
    Converts filtered EEG/ECG dataframe to MNE Raw object.
    """

    # Convert dataframe to numpy array
    eeg_data = df_eeg.iloc[:, 1:].values.T  # first column contains timestamps

    # Create MNE info object for the data
    ch_names = df_eeg.columns.tolist()[1:]
    if ch_names != ideal_ch_names:  # rename ch_names if wrongly named
        ch_names = ideal_ch_names
        message = "Renamed EEG channel names to ideal names."
        print("\n!!", message, "\n")
        if create_mne_report: report.add_html(
            title="Incorrect EEG channel names in iMotions data",
            section="EEG data import",
            html=message,
        )
    if ecg_data_imported: ch_types = ['eeg'] * (len(ch_names) - 1) + ['ecg']
    else: ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

    # Create MNE raw object for the data
    raw = mne.io.RawArray(eeg_data, info)

    return raw

def get_dig2eeg_mapping(fpath_dig2eegmap):
    """
    Reads the JSON file with the digitization to EEG
    channel name mapping and return it as a dict.
    """
    with open(fpath_dig2eegmap, "r") as f:
        content = json.load(f)
    return content["DigitizedChannelsDict"]

def import_digitization_data(
        fpath_dig, raw, fpath_dig2eegmap, head_radius, report=None,
        create_plots=False, create_mne_report=False
):
    """
    Import digitization data and add it to Raw object.
    """

    # Start with an empty message
    message = ''

    # Read BS digitization data into an MNE montage
    try:
        dig = create_MNE_montage_from_BS_dig_data_old(fpath_dig)
    except:
        dig = create_MNE_montage_from_BS_dig_data_new(fpath_dig)

    # Rename digitized channels if named numerically
    # if "EEG" in dig.ch_names[0]:
    #     dig.ch_names = dig_ch_names
    if "EEG01" in dig.ch_names:
        dig2eeg = get_dig2eeg_mapping(fpath_dig2eegmap)
        eeg_ch_names = [dig2eeg[w] for w in dig.ch_names]
        dig.ch_names = eeg_ch_names
        message += "Renamed digitized channels to corresponding EEG channel names.\n"

    # Add GND channel to MNE raw object
    mne.add_reference_channels(raw, ref_channels=['GND'], copy=False)

    # Combine MNE montage with raw data
    raw.set_montage(dig)

    # Set message for MNE report and output log
    message += "Successfully imported digitization data."

    # Visualize digitized data
    if create_plots or create_mne_report:
        fig = dig.plot(sphere=head_radius)     # 2D
        if create_mne_report: report.add_figure(
            fig=fig,
            title="2D Matplotlib plot",
            section="EEG Channel and Headshape Digitization",
        )
        fig = dig.plot(kind="3d")               # 3D
        if create_mne_report: report.add_figure(
            fig=fig,
            title="3D Matplotlib plot",
            section="EEG Channel and Headshape Digitization",
        )
        fig = mne.viz.plot_alignment(
            raw.info,
            dig=True,
            eeg=True,
        )                                       # 3D PyVista
        fig.plotter.camera.elevation = 20
        fig.plotter.camera.azimuth = 45
        if create_mne_report: report.add_figure(
            fig=fig,
            title="3D PyVista plot",
            section="EEG Channel and Headshape Digitization",
        )

    # Drop ground channel (in future, we might use this; for now, we drop GND after merging dig and raw)
    raw.drop_channels('GND')

    return message

def get_head_radius(fpath_headcircum):
    """
    Reads head circumference (in meters) from file and returns head radius (in meters).
    """
    with open(fpath_headcircum, "r") as f:
        # content = f.read()  # head circumference in meters
        content = json.load(f)
    head_circum = float(content["Value"])
    head_radius = head_circum / (2 * np.pi) # head circumference / 2π
    return head_radius, head_circum
    
def make_mne_raw_from_imotions_and_bs_dig_data(
        sub, ses, data_dir, results_dir,
        possible_unwanted_columns, col_ecg, col_blink,
        col_trigs, t_win_dupl_markers, tasknames, sfreq,
        report=None, verbose=True,
        create_mne_report=False, create_plots=False
):
    """
    Build MNE Raw object from iMotions and Brainstorm digitization data.
    """

    # Set file names and paths
    fname_imotions_data = sub + "_task-all_eeg.csv"
    fname_exportedraw = fname_imotions_data[:-4] + ".fif"
    fname_dig = sub + "_task-all_acq-polhemus_headshape.mat"
    fpath_report = os.path.join(
        results_dir, "Logs/DataImportToMNE",
        "importreport_%s_ses-%s.html" % (sub, ses)
    )
    fname_headcircum = sub + "_desc-manual_headcircumference.json"
    fpath_headcircum = os.path.join(
        data_dir, sub, "ses-" + ses, "eeg", fname_headcircum
    )
    fpath_dig2eegmap = os.path.join(
        data_dir, "sourcedata", 'group_desc-digitized2eeg_mapping.json'
    )

    # Initialize MNE report (if needed)
    if create_mne_report: report = mne.Report(title=sub, verbose=False)

    # Get head radius
    head_radius, head_circum = get_head_radius(fpath_headcircum)
    message = "Head circumference: %0.1fcm; Head radius: %0.3fm." % (
        head_circum * 100, head_radius
    )
    if verbose: print("\n!!", message, "\n")
    if create_mne_report: report.add_html(title="Head radius import", html=message)

    # Import raw data in CSV file and convert to a formatted dataframe
    fpath_imotions_data = os.path.join(data_dir, sub, "ses-" + ses, "eeg", fname_imotions_data)
    df_imotions, recording_dt = read_imotions_data(fpath_imotions_data)
    if verbose: print("\nRecording time:", recording_dt)
    if verbose:
        try: display(df_imotions)
        except: print(df_imotions)

    # Extract EEG data from raw data df
    df_eeg = extract_eeg_from_imotions_data(df_imotions, possible_unwanted_columns)
    if verbose:
        try: display(df_eeg)
        except: print(df_eeg)

    # Extract ECG signal (Channel: "ECG LL-RA CAL"; in millivolts)
    if col_ecg in df_imotions.columns:
        try:
            df_eeg = add_interpolated_ecg_to_eeg(df_imotions, col_ecg, df_eeg)
            ecg_data_imported = True
            message = "Successfully imported ECG data."
        except Exception as e:
            ecg_data_imported = False
            message = e
    else:
        ecg_data_imported = False
        message = "ECG column ('%s') not found in raw data from iMotions." % col_ecg
    if verbose: print("\n!!", message, "\n")
    if create_mne_report: report.add_html(title="ECG data import", html=message)

    # Extract blink data
    if col_blink in df_imotions.columns:
        try:
            blink_times = read_event_timings_from_imotions_data(
                df_imotions, event_col=col_blink, positive_event_val='1',
                t_0=df_eeg['Timestamp'][0])
            blink_data_imported = True
            message = "Successfully imported blink data."
        except Exception as e:
            blink_data_imported = False
            message = e
    else:
        blink_data_imported = False
        message = "Blink column ('%s') not found in raw data from iMotions." % col_blink
    if verbose: print("\n!!", message, "\n")
    if create_mne_report: report.add_html(title="Blink data import", html=message)

    # Extract event triggers
    if col_trigs in df_imotions.columns:
        try:
            trigger_times, trigger_desc = read_event_markers_from_imotions_data(
                df_imotions, df_eeg['Timestamp'][0],
                t_win_dupl_markers=t_win_dupl_markers, marker_col=col_trigs
            )
            message = "Successfully imported event data from iMotions data."
            trig_data_imported = True
        except Exception as e:
            trig_data_imported = False
            message = e
    else:
        message = "Trigger column ('%s') not found in raw data from iMotions." % col_trigs
        try:
            event_files_dir = os.path.join(data_dir, "%s/ses-%s/eeg" % (sub, ses))
            trigger_times, trigger_desc = read_event_markers_from_event_files(
                event_files_dir, tasknames, recording_dt
            )
            message += '\nSuccessfully imported event data from event file(s).'
            trig_data_imported = True
        except Exception as e:
            message += '\n' + e
            trig_data_imported = False
    if verbose: print("\n!!", message, "\n")
    if create_mne_report: report.add_html(title="Event data import", html=message)

    # Make MNE Raw object for EEG/ECG data
    eeg_ch_names = list(get_dig2eeg_mapping(fpath_dig2eegmap).values())
    ideal_ch_names = eeg_ch_names[:-1] + [col_ecg]
    raw = convert_df_eeg_to_mne_raw_object(
        df_eeg, ideal_ch_names, ecg_data_imported, sfreq,
        report=report, create_mne_report=create_mne_report
    )

    # Add annotation for blinks and/or event triggers
    if blink_data_imported or trig_data_imported:
        onsets = []; descriptions = []
        if blink_data_imported:
            onsets += blink_times
            descriptions += ['blink'] * len(blink_times)
        if trig_data_imported:
            onsets += trigger_times
            descriptions += trigger_desc

        #   Create MNE annotations
        annots = mne.Annotations(
            onset = onsets,
            duration = 0,
            description = descriptions,
        )
        raw.set_annotations(annots)
        # print(raw.annotations)

        # Convert MNE annotations to MNE events
        events, event_dict = mne.events_from_annotations(raw)
        if create_plots:
            plt.ion()
            fig = mne.viz.plot_events(
                events, sfreq=raw.info["sfreq"],
                first_samp=raw.first_samp, event_id=event_dict
            )   # plot events
        if create_mne_report: report.add_events(
            title="Events from blink and/or event triggers",
            events=events,
            sfreq=raw.info["sfreq"],
            first_samp=raw.first_samp,
            event_id=event_dict
        )

    # Import BS digitizer data to MNE montage
    fpath_dig = os.path.join(data_dir, sub, "ses-" + ses, "eeg", fname_dig)
    if os.path.exists(fpath_dig):
        try:
            message = import_digitization_data(
                fpath_dig, raw, fpath_dig2eegmap, head_radius, report=report,
                create_plots=create_plots, create_mne_report=create_mne_report
            )
        except Exception as e:
            message = e
    else:
        message = "Digitization file ('%s') not found." % fpath_dig
    if verbose: print("\n!!", message, "\n")
    if create_mne_report: report.add_html(
        title="Digitization data import",
        section="EEG Channel and Headshape Digitization",
        html=message,
    )

    # Add MNE raw object to MNE report
    if create_mne_report: report.add_raw(raw=raw, title="Raw", psd=True)

    # Save MNE report (if created)
    if create_mne_report: report.save(fpath_report, overwrite=True)

    # Close all figures (if only created for the MNE report)
    if create_mne_report and not create_plots: plt.close('all') # closes all plt figures

    # Export raw data in FIF format
    fpath_exportedraw = os.path.join(data_dir, sub, "ses-" + ses, "eeg", fname_exportedraw)
    raw.save(fpath_exportedraw, overwrite=True)

    return raw