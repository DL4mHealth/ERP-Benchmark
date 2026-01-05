Dataset raw name: EEG: 3-Stim Auditory Oddball and Rest in Parkinson's
Dataset Link: https://openneuro.org/datasets/ds003490/versions/1.1.0

Short name: AOPD

I. EEG Preprocessing Steps

1. Select common EEG channels shared across all subjects and reorder them.
2. Set the EEG montage to 'standard_1020'.
3. Apply a 60 Hz notch filter before bandpass filtering.
4. Apply bandpass filtering with default frequency range: 0.5–40 Hz.
5. Interpolate bad channels if `do_bad_interp` is set to True.
6. Re-reference EEG signals to the average reference.
7. Perform ICA on a 1 Hz high-pass filtered copy:
   - If `mne_icalabel` is available, automatically exclude components related to eye blinks, muscle activity, heartbeats, etc.
   - Subject-specific ICA settings are used for known IDs.
8. Downsample EEG data to 200 Hz.

II. Data Segmentation Steps

1. Extract stimulus events from the corresponding `events.tsv` file.
2. Retain only events starting with 'S' followed by a number (e.g., S 1, S 2, S 128).
3. Map event codes to stimulus types:
   - S200 → Target tone → stimulus = 0
   - S201 → Standard tone → stimulus = 1
   - S202 → Novel tone → stimulus = 2
4. Epoch the data around stimulus onset from -0.2 to 0.8 seconds.
5. Apply baseline correction using the interval from -0.2 to 0 seconds.
6. Reshape EEG data from (N, C, T) to (N, T, C) for model input.

III. Output Format

Each subject's processed data is saved as:

- `feature_XXX.npy`: EEG data with shape (N, T, C)
- `label_XXX.npy`: Corresponding labels with shape (N, 5)

Where:
- N: number of trials
- T: number of time points per trial (e.g., 250 for 1 seconds at 250 Hz)
- C: number of EEG channels

Label Structure (`y`):

Each label row contains:

[task_id, stimulus_type, subject_id, session_id, disease_id]

- `task_id`: always 1 (Oddball task)
- `stimulus_type`: 0 = Target, 1 = Standard, 2 = novel
- `subject_id`: numeric identifier from 'sub-XXX'
- `disease_id`:  0: CTL, 1: PD

