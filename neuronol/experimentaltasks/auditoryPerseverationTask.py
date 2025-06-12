"""
Version 0.7:
1. Added event time stamps logger.
"""

# ====================
# PsychoPy audio setup
# ====================

from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB']
prefs.hardware['audioLatencyMode'] = '4'

# ==============
# Import modules
# ==============

import psychtoolbox as ptb
from psychopy import sound, core, gui, data, visual
from psychopy.hardware import keyboard

import os
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pylsl import StreamInfo, StreamOutlet
import time

# ================
# Helper functions
# ================

# Function to generate trial sequence
def generateRandomStimSeqAPT_FishbowlSampling(rand_seed=375364):
    '''
    Use fishbowl sampling to generate a sequence of stimuli to be used in
    the Auditory Perseveration Task.
    '''
    # Generate a list of candidates
    fishbowl_stims = ['dev'] * 40 + ['trg_low'] * 80 + ['trg_high'] * 80

    # Set seed for RNG
    np.random.seed(rand_seed)

    # Start loop to draw stimuli from fishbowl randomly
    stim_seq = []; n_tars = 0; n_devs = 0
    n_inter_tones = 0
    while (n_tars < 160) and (n_devs < 40):

        # Shuffle fishbowl and then draw
        np.random.shuffle(fishbowl_stims)
        draw = fishbowl_stims[0]
        
        if n_inter_tones >= 4:  # If suff no. of interm. target tones are present, draw any
            stim_seq.append(draw)
            fishbowl_stims.pop(0)
            if draw == 'dev':
                n_devs += 1
                n_inter_tones = 0
            else:
                n_tars += 1
                n_inter_tones += 1
        else:                   # If suff no. of interm. target tones aren't present, draw either '1'/'2'
            if draw == 'dev':
                pass
            else:
                stim_seq.append(draw)
                fishbowl_stims.pop(0)
                n_tars += 1
                n_inter_tones += 1
  
    return stim_seq, n_devs

# =====================
# Set experiment params
# =====================

project_name = 'EEGAcamp'
task_name = 'AuditoryPerseverationTask'
rand_seed_dict = {
    "Practice": 375364,     # 33 devs
    "StudyVisit2": 655531,  # 33 devs
    "StudyVisit3": 722816,  # 33 devs
    "StudyVisit4": 803652,  # 33 devs
    "FollowUp1M": 852283,   # 33 devs
    "FollowUp3M": 1137081,  # 33 devs
    "FollowUp6M": 1833957,  # 33 devs
}   # dict to select random seed to use throughout for RNGs
t_stim = 0.05   # stim time
rewardPerCorrectResp = 0.25  # amount in dollars earned per correct response
rkey = 'rctrl'; lkey = 'lctrl'  # right and left keys

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
#data_dir = os.path.expanduser('~/data/experiments/')
exp_info = {
    'SubID': 'sub-00',  # subject's study ID
    'SessionName': list(rand_seed_dict.keys()),
    'DataDir': os.path.join(os.path.expanduser('~/data/bids/'), project_name),
}
dlg = gui.DlgFromDict(dictionary=exp_info, title=task_name)

# If 'Cancel' is pressed, quit experiment
if not dlg.OK:
    print('\n===>\tUser quit. Exiting.')
    core.quit()

# ================================================
# Data storage: basic information, filename & path
# ================================================

# Set rand seed based on study visit
rand_seed = rand_seed_dict[exp_info['SessionName']]

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
fpath_data = os.path.join(exp_info['data_path'], fname_data)

# ================================
# Creation of windows and messages
# ================================

# Open a window

win = visual.Window(monitor='NeuroPhysLabStim', size=(1920, 1080), color='black', 
                    units='pix', fullscr=True, screen=1)

# win = visual.Window(monitor='IrishTwins', size=(1920, 1080), color='black', 
#                     units='pix', fullscr=True, screen=1)

#win = visual.Window(monitor="testMonitor", size=(1920, 1080), color='black',
#                   units='pix', fullscr=True, screen=0)

# Define welcome page
page_welcome = visual.TextStim(win,
    text="""Welcome to the task!
    \nPress SPACEBAR to continue.
    """,
    color='white', height=40)

# Define instructions page
page_instructions = visual.TextStim(win,
    text="""In this task, you will hear three different sounds.
    \nMake sure your EYES ARE OPEN for this task.
    \nPress the correct button corresponding to each sound as instructed.
    \nPress SPACEBAR when you are ready.
    """,
    color='white', height=40)

# Define fixation page
page_fixation = visual.Rect(win, size=(15, 15), lineColor='white', fillColor='white')

# Define task completion page
page_task_completion = visual.TextStim(win,
    text="You have successfully finished the task.",
    pos=[0, 120], color='white', height=40)

# Define end page
page_end = visual.TextStim(win,
    text="""You have reached the end of this task.
    \nThank you for participating.
    \nPress SPACEBAR to exit.
    """,
    color='white', height=40)

# ==============================
# Generate/read auditory stimuli
# ==============================

snd_dev = sound.backend_ptb.SoundPTB(value='./Tones/WhiteNoise.wav',
    stereo=True, volume=1.0, loops=0, sampleRate=48000,
    blockSize=128, preBuffer=-1, hamming=False,
    startTime=0, stopTime=-1, name='',
    autoLog=True, syncToWin=None)

snd_trg_low = sound.backend_ptb.SoundPTB(value=500, secs=t_stim, stereo=True,
    volume=1.0, loops=0, sampleRate=48000,
    blockSize=128, preBuffer=-1, hamming=False,
    startTime=0, stopTime=-1, name='',
    autoLog=True, syncToWin=None)

snd_trg_high = sound.backend_ptb.SoundPTB(value=1000, secs=t_stim, stereo=True,
    volume=1.0, loops=0, sampleRate=48000,
    blockSize=128, preBuffer=-1, hamming=False,
    startTime=0, stopTime=-1, name='',
    autoLog=True, syncToWin=None)

# =========================
# Define the trial sequence
# =========================

# Create a list of triggers corresponding to each tone
stim_list, _ = generateRandomStimSeqAPT_FishbowlSampling(rand_seed)
if exp_info["SessionName"] == "Practice": stim_list = stim_list[:10]
stim_list += ['blank']  # add a blank stim at the end which is never read
print("Order of stimuli:", stim_list)

# Convert stim order list to list of dicts with key ('stimulus') - value pairing
stim_order = [{'Stimulus': stim} for stim in stim_list]
n_trials = len(stim_order)

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
keys = kb.waitKeys(keyList=['space']) # wait for spacebar key press
# before advancing

# Display instruction page
page_instructions.draw()    # draw page to buffer screen
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space']) # wait for spacebar key press
# before advancing

# Display fixation page
page_fixation.draw()    # draw page to buffer screen
win.flip()  # flip screen to front

# Initialize correct response
correctResp_high = 1
correctResp_low = -1
totalCorrectRespsRegistered = 0

# Run through the trials, stimulus by stimulus
stimuli = []; stim_times = []; resps = []
pressed_keys = []; resp_assessments = []; rts = []
# last_trial_stim = ''
rng = default_rng(rand_seed)    # initiate RNG with seed
stream_outlet.push_sample(['trials-start']) # send start trials marker
start_time = time.time()    # time to start trials
stim_times.append(('trials-start', start_time))
for i_trial, trial in enumerate(trials):

    if i_trial < n_trials - 1:

        # Select tone to play in next trial
        if trial['Stimulus'] == 'trg_high':
            snd = snd_trg_high
        elif trial['Stimulus'] == 'trg_low':
            snd = snd_trg_low
        elif trial['Stimulus'] == 'dev':
            snd = snd_dev
        else:
            raise ValueError(
                "trial['Stimulus'] can only be 'dev', 'trg_low', or 'trg_high'.")
    
        # Preschedule sound playback
        t_isi = 2.5 + 0.5 * rng.uniform()
        now = ptb.GetSecs()
        snd.play(when=now+t_isi)

        # Pause for interstimulus interval time
        core.wait(t_isi)

        # Send stim onset marker and note stim onset time
        stream_outlet.push_sample([trial['Stimulus']])
        stim_time = time.time()    # stim onset time
        stim_times.append((trial['Stimulus'], stim_time))

    if i_trial == n_trials - 1:
        # Pause for interstimulus interval time
        t_isi = 2.5 + 0.5 * rng.uniform()
        core.wait(t_isi)

    if i_trial > 0:    

        # Get key presses and response time
        resp = None # intialize response
        rt = None   # initialize response time
        keys = kb.getKeys(waitRelease=True)
        key_names = [key.name for key in keys]
        print("\nPressed key(s):", key_names)
        keypresses = [(key.name, key.rt) for key in keys]   # keypress details
        times_allowable_resps = [key.rt for key in keys if key.name in (lkey, rkey)]
        if len(times_allowable_resps) > 0: rt = times_allowable_resps[0]
        # print(keypresses)
        # print("Response time:", rt)

        # Assign response to pressed key(s)
        if (lkey in key_names) and (rkey in key_names): resp = 0
        elif lkey in key_names: resp = -1
        elif rkey in key_names: resp = 1
        else: print('No response registered!')
        # print('Assessed response =', resp)

        # Assess whether response is correct
        if last_trial_stim == 'dev':    # response to last trial stimulus is registered/assessed in this iteration
            markerCorrectTrialResp = None
        else:
            if resp == correctTrialResp:
                markerCorrectTrialResp = True
                totalCorrectRespsRegistered += 1
                print('Correct response!')
            else:
                markerCorrectTrialResp = False
                print('Incorrect response.')

        # Add trial data to trials object
        trials.addData('Response', resp)
        trials.addData('ResponseTime', rt)
        trials.addData('ResponseAssessment', markerCorrectTrialResp)

        # Collect trial data separately to create pd.DataFrame object
        stimuli.append(last_trial_stim)
        resps.append(resp)
        rts.append(rt)
        resp_assessments.append(markerCorrectTrialResp)
        pressed_keys.append(keypresses)
        
    if i_trial < n_trials - 1:

        # Select correct resp to next trial
        if trial['Stimulus'] == 'trg_high':
            correctTrialResp = correctResp_high
        elif trial['Stimulus'] == 'trg_low':
            correctTrialResp = correctResp_low
        elif trial['Stimulus'] == 'dev':
            correctResp_buffer = correctResp_high
            correctResp_high = correctResp_low
            correctResp_low = correctResp_buffer
        else:
            raise ValueError(
                "trial['Stimulus'] can only be 'dev', 'trg_low', or 'trg_high'.")
        
        # Clear keypress buffer
        kb.getKeys(waitRelease=True)

        # Keep a copy of this trial's stim for next trial
        last_trial_stim = trial['Stimulus']

        # Start keypress timer
        kb.clock.reset()

        # Pause for stimulation time
        core.wait(t_stim)

# Mark trial ending
stream_outlet.push_sample(['trials-end'])   # send end trials marker
end_time = time.time()  # time trials end
stim_times.append(('trials-end', end_time))

# =====================
# End of the experiment
# =====================

# Find total execution time for all trials
execution_time = end_time - start_time  # calculate exec time
print('\nAll trials executed in %0.1f seconds.' % execution_time)

# Calculate total money earned
totalRewardEarned = rewardPerCorrectResp * totalCorrectRespsRegistered

# Save all data using trials object
trials.saveAsWideText(fpath_data + 'default.csv', delim=',')

# Save trial data as DataFrame
trialData = {
    'Stimulus': stimuli,
    'Response': resps,
    'ResponseTime': rts,
    'ResponseAssessment': resp_assessments,
    'PressedKeys': pressed_keys,
}
df_TrialData = pd.DataFrame(trialData)
df_TrialData.to_excel(fpath_data + 'custom.xlsx', index=False)

# Save stim times as DataFrame
df_stimtimes = pd.DataFrame(stim_times, columns=['EventName', 'EventTime'])
df_stimtimes.to_excel(fpath_data + 'stimtimes.xlsx', index=False)

# Display task completion page
page_task_completion.draw() # draw initial page to buffer screen
visual.TextStim(win, 
                text="You made a total of $%0.2f from this task." % totalRewardEarned,
                pos=[0, 0], color='white', bold=True,
                height=40).draw()   # show money earned
visual.TextStim(win, 
                text="Correct response(s): %d" % totalCorrectRespsRegistered,
                pos=[0, -120], color='white',
                height=40).draw()   # show no. of correct/incorrect resps
visual.TextStim(win, text="Press SPACEBAR to continue.",
                pos=[0, -240], color='white',
                height=40).draw()   # add spacebar message
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])    # wait for spacebar key press
# before advancing

# Print out total correct responses that were registered and total money earned
print("\nTotal correct responses =", totalCorrectRespsRegistered)
print("Total reward earned = $%0.2f" % totalRewardEarned)

# Display end message
page_end.draw() # draw page to buffer screen
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])    # wait for spacebar key press
# before advancing

# Shut down the experiment
win.close()
core.quit()