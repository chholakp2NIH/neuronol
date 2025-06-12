"""
Version 0.7:
1. Added event time stamps logger.
"""

# ==============
# Import modules
# ==============

# import psychtoolbox as ptb
from psychopy import core, gui, data, visual
from psychopy.hardware import keyboard

import os
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pylsl import StreamInfo, StreamOutlet
import time

# =====================
# Set experiment params
# =====================

project_name = 'EEGAcamp'
task_name = 'VisualOddballTask'
rand_seed_dict = {
    "Practice": 0,
    "StudyVisit2": 1,
    "StudyVisit3": 2,
    "StudyVisit4": 3,
    "FollowUp1M": 4,
    "FollowUp3M": 5,
    "FollowUp6M": 6,
}   # dict to select random seed to use throughout for RNGs
t_stim = 0.2    # stim time
rewardPerCorrectResp = 0.5  # amount in dollars earned per correct response

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
#exp_name = 'VisualOddballTask'
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

# ================================
# Define visual stimuli parameters
# ================================

# Define parameters of standard stimulus
verts_std = [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]]
lineColor_std = 'white'; fillColor_std = False
ori_std = 0

# Define parameters of target stimulus
verts_tar = 'cross'
lineColor_tar = 'white'; fillColor_tar = 'white'
ori_tar = 45

# Define parameters of deviant stimulus
verts_dev = [[-0.5, -0.5], [0, 0.5], [0.5, -0.5]]
colors_dev = ['deeppink', 'aqua', 'slateblue', 'palegreen', 'salmon']
oris_dev = [theta for theta in range(45, 360, 45)]  # 7 angles between [45, 315]

# =========================
# Define the trial sequence
# =========================

# Create a list of triggers corresponding to each stim type
stim_list = ['std'] * 210 + ['tar'] * 35 + ['dev'] * 35
np.random.seed(rand_seed); np.random.shuffle(stim_list)

if exp_info["SessionName"] == "Practice": stim_list = stim_list[:10]
print("Order of stimuli:", stim_list)

# Convert stim order list to list of dicts with key ('stimulus') - value pairing
stim_order = [{'Stimulus': stim} for stim in stim_list]

# Use stim order list of dicts to create trials object via data.TrialHandler,
# specifying further settings
trials = data.TrialHandler(stim_order, nReps=1, extraInfo=exp_info,
                           method='sequential')

# Create color-ori pair combinations for 'dev' stim
devColsOris = [(c, o) for c in colors_dev for o in oris_dev]
np.random.shuffle(devColsOris)  # randomly shuffle the list

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
                text="In this task, you will see different shaped figures.",
                pos=[0, 400], color="white", height=40).draw()
visual.TextStim(win,
                text="Press the DOWN ARROW key under your index finger when you see a cross",
                pos=[0, 200], color="white", height=40).draw()
visual.ShapeStim(win, size=45, ori=45, vertices="cross",
                 lineColor="white", fillColor="white").draw()
visual.TextStim(win,
                text="Otherwise, DO NOT press any key.",
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
pressed_keys = []; assessments = []; rts = []
rng = default_rng(rand_seed)    # initiate RNG with seed
stream_outlet.push_sample(['trials-start']) # send start trials marker
start_time = time.time()    # time to start trials
stim_times.append(('trials-start', start_time))
for i_trial, trial in enumerate(trials):

    # Select stim to be drawn
    if trial['Stimulus'] == 'std':
        verts = verts_std
        lineColor = lineColor_std
        fillColor = fillColor_std
        ori = ori_std
    elif trial['Stimulus'] == 'tar':
        verts = verts_tar
        lineColor = lineColor_tar
        fillColor = fillColor_tar
        ori = ori_tar
    elif trial['Stimulus'] == 'dev':
        verts = verts_dev
        color_dev, ori_dev = devColsOris[0]
        devColsOris.pop(0)
        lineColor = color_dev
        fillColor = color_dev
        ori = ori_dev
    else:
        raise ValueError("trial['Stimulus'] can only be 'std', 'tar', or 'dev'.")
    
    # Draw the trial stimulus
    page_trl = visual.ShapeStim(win, size=45, ori=ori, vertices=verts,
                                lineColor=lineColor, fillColor=fillColor)
    page_trl.draw() # draw page to buffer screen
    win.flip()  # flip screen to front

    # Send stim onset marker and note stim onset time
    stream_outlet.push_sample([trial['Stimulus']])
    stim_time = time.time()    # stim onset time
    stim_times.append((trial['Stimulus'], stim_time))

    # Start keypress timer
    kb.clock.reset()

    # Show stim for stimulation time (e.g., 200 ms) and then remove
    core.wait(t_stim)
    win.flip()

    # Pause for 1.6-2 secs
    t_pause = 1.6 + 0.4 * rng.uniform()
    core.wait(t_pause)

    # Get pressed keys and assess response
    assessment = None # initialize assessment
    rt = None   # initialize response time
    keys = kb.getKeys(waitRelease=True)
    key_names = [key.name for key in keys]
    print("\nPressed key(s):", key_names)
    keypresses = [(key.name, key.rt) for key in keys]
    if trial['Stimulus'] == 'tar':
        if 'down' in key_names:
            # Set response assessment
            assessment = True
            print("Correct response!")

            # Find time for first correct response
            times_correct_resp = [key.rt for key in keys
                                  if key.name == 'down']
            rt = times_correct_resp[0] # get response time
        else:
            # Set response assessment
            assessment = False
            print("Incorrect response.")
    elif (trial['Stimulus'] == 'std') or (trial['Stimulus'] == 'dev'):
        if 'down' not in key_names:
            # Set response assessment
            assessment = True
            print("Correct response!")
        else:
            # Set response assessment
            assessment = False
            print("Incorrect response.")
    else:
        raise ValueError("trial['Stimulus'] can only be 'std', 'tar', or 'dev'.")
    
    # Add trial data to trials object
    trials.addData('Assessment', assessment)
    trials.addData('ResponseTime', rt)

    # Collect trial data separately to create pd.DataFrame object
    stimuli.append(trial['Stimulus'])
    pressed_keys.append(keypresses)
    assessments.append(assessment)
    rts.append(rt)

    # If participant presses 'escape', stop the experiment
    if 'escape' in key_names:
        print('\n===>\tUser quit. Exiting.')
        core.quit()

# Mark trial ending
stream_outlet.push_sample(['trials-end'])   # send end trials marker
end_time = time.time()  # time trials end
stim_times.append(('trials-end', end_time))

# Verify all trials had a response recorded
assert None not in assessments, "Found 'None' in `assessments`"

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
    'PressedKeys': pressed_keys,
    'Assessment': assessments,
    'ResponseTime': rts,
}
df_TrialData = pd.DataFrame(trialData)
df_TrialData.to_excel(fpath_data + 'custom.xlsx', index=False)

# Save stim times as DataFrame
df_stimtimes = pd.DataFrame(stim_times, columns=['EventName', 'EventTime'])
df_stimtimes.to_excel(fpath_data + 'stimtimes.xlsx', index=False)

# Calculate total money earned
n_right = df_TrialData[
    df_TrialData['Stimulus'] == 'tar']['Assessment'].sum()  # total correct resps for 'tar' stims
n_wrong = (df_TrialData['Assessment'] == False).sum()   # total wrong resps for all stims
netCorrectResps = n_right - n_wrong
totalRewardEarned = rewardPerCorrectResp * netCorrectResps
if totalRewardEarned < 0: totalRewardEarned = 0   # if net money earned is negative, give 0

# Display task completion page
page_task_completion.draw() # draw initial page to buffer screen
visual.TextStim(win, 
                text="You made a total of $%0.2f from this task." % totalRewardEarned,
                pos=[0, 0], color='white', bold=True,
                height=40).draw()   # show money earned
visual.TextStim(win, 
                text="Correct response(s): %d\nIncorrect response(s): %d" % (n_right, n_wrong),
                pos=[0, -120], color='white',
                height=40).draw()   # show no. of correct/incorrect resps
visual.TextStim(win, text="Press SPACEBAR to continue.",
                pos=[0, -240], color='white',
                height=40).draw()   # add spacebar message
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])   # wait for spacebar key press
# before advancing

# Print out total correct responses and total money earned
print("\nNet correct responses =", netCorrectResps)
print("Total reward earned = $%0.2f" % totalRewardEarned)

# Display end message
page_end.draw() # draw page to buffer screen
win.flip()  # flip screen to front
keys = kb.waitKeys(keyList=['space'])   # wait for spacebar key press
# before advancing

# Shut down the experiment
win.close()
core.quit()