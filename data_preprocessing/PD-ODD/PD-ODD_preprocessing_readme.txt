Dataset raw Name: Cross-Modal Oddball Task
Dataset Link: https://openneuro.org/datasets/ds004574/versions/1.0.0
Paper Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592174/

I. EEG Preprocessing Steps

1. Select common EEG channels across all subjects and reorder them consistently.
2. Set the EEG montage to 'standard_1020'.
3. Apply a 60 Hz notch filter before bandpass filtering.
4. Apply bandpass filtering with a default range of 0.5–40 Hz.
5. Interpolate bad channels if any are marked and `do_bad_interp` is True.
6. Re-reference the EEG signals to the average reference.
7. Perform ICA on a 1 Hz high-pass filtered copy:
   - For subject sub-042, ICA uses `n_components=None`; for all others, `n_components=0.999`.
   - If `mne_icalabel` is available, automatically exclude components related to eye blinks, muscle activity, heartbeat, line noise, and channel noise based on predefined probability thresholds.
8. Downsample all EEG data to 200 Hz.

II. Data Segmentation Steps

1. Extract events from the `events.tsv` file.
2. Select events with value code `S 2`(go cue) as time-lock  for segmentation.
3. Match these events with behavioral data from `beh.tsv`, using the number of trials in both files.
4. Epoch EEG data from -0.5 to 1.0 seconds around the selected event onset.
5. Apply baseline correction using the interval from -0.3 to -0.2 seconds.
6. Reshape data into shape (N, T, C), where:
   - N = number of trials,
   - T = number of time points,
   - C = number of EEG channels.

III. Output Format

Each subject’s processed data is saved as:

- `feature_XXX.npy`: EEG data array with shape (N, T, C)
- `label_XXX.npy`: Corresponding labels with shape (N, 5)

Label Structure (`y`):

Each row in the label file contains:

[task_id, trial_type, accuracy, subject_id, disease_id]

- `task_id`: always 1 (Cross-Modal Oddball task)
- `trial_type`: 
   - 0 = standard trial
   - 1 = auditory oddball
   - 2 = visual oddball
- `accuracy`: response correctness, 1 = correct, 0 = incorrect
- `subject_id`: numeric ID from the folder name (e.g., `sub-001`)
- `disease_id`: 0 = healthy control, 1 = Parkinson’s disease (from `participants.tsv`)

Behavioral labels are extracted from `beh.tsv`, using fields "Odd: Audio", "Odd: Visual", and "Acc".