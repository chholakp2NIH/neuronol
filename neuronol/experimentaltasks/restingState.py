"""
Version 0.8:
1. Added head radius to starting prompt.
"""

# ==============
# Import modules
# ==============

#import psychtoolbox as ptb
from psychopy import core, gui, data, visual, sound
from psychopy.hardware import keyboard

import json
import os
import numpy as np
import pandas as pd
from pylsl import StreamInfo, StreamOutlet
import time

# =====================
# Set experiment params
# =====================

project_name = 'EEGAcamp'
task_name = 'RestingState'
session_names = [
    "Practice",
    "StudyVisit2",
    "StudyVisit3",
    "StudyVisit4",
    "FollowUp1M",
    "FollowUp3M",
    "FollowUp6M",
]
t_stim = 60     # stim time (in seconds)

# ============================
# Create LSL stream and outlet
# ============================

# Create a new stream info
unique_stream_name = "marker_stream"
stream_info = StreamInfo('ExpMarkers', 'Markers', 1, 0, 'string', unique_stream_name)

# Make an outlet
stream_outlet = StreamOutlet(stream_info)

# ======================================
# Create a GUI dialog box for user input
# ======================================

# Get subject filename and other information through a dialog box
exp_info = {
    'SubID': 'sub-00',  # subject's study ID
    'SessionName': session_names,
    'DataDir': os.path.join(os.path.expanduser('~/data/bids'), project_name),
    'HeadRadius (in cm)': 100,
}
dlg = gui.DlgFromDict(dictionary=exp_info, title=task_name)

# If 'Cancel' is pressed, quit experiment
if not dlg.OK:
    print('\n===>\tUser quit. Exiting.')
    core.quit()

# Adjust stim time for practice visit
if exp_info["SessionName"] == "Practice": t_stim = 5

# ================================================
# Data storage: basic information, filename & path
# ================================================

# Set date, experiment name, and data path
exp_info['date'] = data.getDateStr()    # get date and time
exp_info['exp_name'] = task_name
exp_info['data_path'] = os.path.join(exp_info['DataDir'], exp_info['SubID'],
                                     'ses-%s' % exp_info['SessionName'].lower(),
                                     'eeg')

# Check if chosen data path exists; if not, create it
if not os.path.isdir(exp_info['data_path']):
    os.makedirs(exp_info['data_path'])

# Create filname/filepath for the experiment data
fname_data = exp_info['SubID'] + '_task-' + task_name.lower() + '_events'
fname_data_head_radius = exp_info['SubID'] + '_desc-manual_headcircumference.json'
fpath_data = os.path.join(exp_info['data_path'], fname_data)
fpath_data_head_radius = os.path.join(exp_info['data_path'], fname_data_head_radius)

# print(exp_info['HeadRadius (in cm)'])

# ================================
# Creation of windows and messages
# ================================

# Open a window

win = visual.Window(monitor='NeuroPhysLabStim', size=(1920, 1080), color='black', 
                    units='pix', fullscr=True, screen=1)

#win = visual.Window(monitor='IrishTwins', size=(1920, 1080), color='black', 
#                     units='pix', fullscr=True, screen=1)

#win = visual.Window(monitor="testMonitor", size=(1920, 1080), color='black',
#                   units='pix', fullscr=True, screen=0)

# Define welcome page
page_welcome = visual.TextStim(win,
    text="""Welcome to the experiment!
    \nPress SPACEBAR to continue.
    """,
    color='white', height=40)

# Define end page
page_end = visual.TextStim(win,
    text="""You have reached the end of this part of the experiment.
    \nThank you for participating.
    \nPress SPACEBAR to exit.
    """,
    color='white', height=40)

# =========================
# Generate auditory stimuli
# =========================

snd_oe = sound.backend_ptb.SoundPTB(value=500, secs=0.05, stereo=True,
         volume=1.0, loops=0, sampleRate=48000,
         blockSize=128, preBuffer=-1, hamming=False,
         startTime=0, stopTime=-1, name='',
         autoLog=True, syncToWin=None)

snd_ce = sound.backend_ptb.SoundPTB(value=1000, secs=0.05, stereo=True,
         volume=1.0, loops=0, sampleRate=48000,
         blockSize=128, preBuffer=-1, hamming=False,
         startTime=0, stopTime=-1, name='',
         autoLog=True, syncToWin=None)

# =========================
# Define the trial sequence
# =========================

# Create a list of triggers corresponding to each stim type
stim_list = ["open", "closed", "open", "closed"]

# if exp_info["SessionName"] == "Practice": stim_list = stim_list[:10]
print("Order of stimuli:", stim_list)

# Convert stim order list to list of dicts with key ('Stimulus') - value pairing
stim_order = [{'Stimulus': stim} for stim in stim_list]

# Use stim order list of dicts to create trials object via data.TrialHandler,
# specifying further settings
trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential')

# ============
# Run the task
# ============

# Set up keyboard object
kb = keyboard.Keyboard()

# Display welcome message
page_welcome.draw() # draw page to buffer screen
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])   # wait for spacebar key press
# before advancing

# Display instruction page
visual.TextStim(win,
                text="In this part of the experiment, you will be resting with your eyes opened or closed.",
                pos=[0, 400], color="white", height=40).draw()
visual.TextStim(win,
                text="When opened, gaze (not stare) at the cross.",
                pos=[0, 200], color="white", height=40).draw()
visual.ShapeStim(win, size=45, ori=45, vertices="cross",
                 pos=[0, 0], lineColor="white", fillColor="white").draw()
visual.TextStim(win,
                text="You will hear a low-pitch beep for open eyes and a high-pitch beep for closed eyes.",
                pos=[0, -200], color="white", height=40).draw()
visual.TextStim(win,
                text="Press SPACEBAR when you are ready.",
                pos=[0, -400], color="white", height=40).draw()
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])   # wait for spacebar key press
# before advancing

# Show blank screen for 2 seconds
win.flip()
core.wait(2)

# Run through the trials, stimulus by stimulus
stimuli = []; stim_times = []
stream_outlet.push_sample(['trials-start']) # send start trials marker
start_time = time.time()    # time to start trials
stim_times.append(('trials-start', start_time))
for i_trial, trial in enumerate(trials):

    # Select stim to be drawn/played
    if trial['Stimulus'] == 'open':
        msg = "Eyes OPEN"
        snd = snd_oe
    elif trial['Stimulus'] == 'closed':
        msg = "Eyes CLOSED"
        snd = snd_ce
    else:
        raise ValueError("trial['Stimulus'] can only be 'open or 'closed'.")
    
    # Draw the trial stimulus
    page_trl = visual.TextStim(win, text=msg, color='white', height=40)
    page_trl.draw() # draw page to buffer screen
    win.flip()  # flip screen to front

    # Play the tone
    snd.play()

    # Send stim onset marker and note stim onset time
    stream_outlet.push_sample([trial['Stimulus']])
    stim_time = time.time()    # stim onset time
    stim_times.append((trial['Stimulus'], stim_time))

    # Show stim for stimulation time (e.g., 60 s) and then remove
    core.wait(t_stim)

    # Collect trial data separately to create pd.DataFrame object
    stimuli.append(trial['Stimulus'])

    # If participant presses 'escape', stop the experiment
    keys = kb.getKeys(waitRelease=True)
    key_names = [key.name for key in keys]
    print("\nPressed key(s):", key_names)
    if 'escape' in key_names:
        print('\n===>\tUser quit. Exiting.')
        core.quit()

# Mark trial ending
stream_outlet.push_sample(['trials-end'])   # send end trials marker
end_time = time.time()  # time trials end
stim_times.append(('trials-end', end_time))

# =====================
# End of the experiment
# =====================

# Find total execution time for all trials
execution_time = end_time - start_time  # calculate exec time
print('\nAll trials executed in %0.3f seconds.' % execution_time)

# Save all data using trials object
trials.saveAsWideText(fpath_data + 'default.csv', delim=',')

# Save trial data as DataFrame
trialData = {
    'Stimulus': stimuli,
}
df_TrialData = pd.DataFrame(trialData)
df_TrialData.to_excel(fpath_data + 'custom.xlsx', index=False)

# Save stim times as DataFrame
df_stimtimes = pd.DataFrame(stim_times, columns=['EventName', 'EventTime'])
df_stimtimes.to_excel(fpath_data + 'stimtimes.xlsx', index=False)

# Export the head radius as JSON file
data_head_radius = {
    "Value": exp_info['HeadRadius (in cm)'] / 100,  # in meters
    "Unit": "meters",
    "MeasurementMethod": "Measured with tape around head."
}
with open(fpath_data_head_radius, 'w') as f:
    json.dump(data_head_radius, f, indent=4)

# Play the tone
snd_oe.play()

# Display end message
page_end.draw() # draw page to buffer screen
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])   # wait for spacebar key press
# before advancing

# Shut down the experiment
win.close()
core.quit()