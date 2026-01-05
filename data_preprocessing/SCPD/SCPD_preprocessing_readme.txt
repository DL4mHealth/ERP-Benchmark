Dataset raw name: EEG: Simon Conflict in Parkinson's
Dataset Link: https://openneuro.org/datasets/ds003509/versions/1.1.0
Paper Link: https://www.sciencedirect.com/science/article/pii/S0028393218302185

Short name: SCPD

I. Preprocess the EEG Data

Preprocessing Steps:**
1. Select common EEG channels and reorder them (excluding unreliable electrodes: FT9, FT10, TP9, TP10).
2. Set the standard 10–20 montage (`standard_1020`).
3. Apply a **60 Hz notch filter** before band-pass filtering.
4. Apply a **band-pass filter** (default: 0.5–40 Hz).
5. Interpolate bad channels if `do_bad_interp=True`.
6. Re-reference to average.
7. Perform ICA on a 1 Hz high-pass copy of the data and automatically remove eye movement/muscle components using **mne-icalabel** thresholds:
   - Eye blink ≥ 0.7
   - Muscle artifact ≥ 0.6
   - Heartbeat ≥ 0.5
   - Line noise ≥ 0.8
   - Channel noise ≥ 0.9
8. Downsample to **250 Hz** if necessary.

**Notes:**
- Subject `sub-026` is automatically skipped due to a known ICA issue.
- The pipeline supports loading `.set` files directly with fallback to epoch reading if needed.

II. Data Segmentation Steps

1. Extract **cue-lock events** from the `task-Simon_events.tsv` file using `trial_type` entries beginning with “Trn Stim”.
2. Use the **stimulus onset times** as the lock point for segmenting.
3. Epoch the data from **-0.5 s to +1.0 s** around the cue event.
4. Baseline-correct the epochs using the interval **(-0.3 s to -0.2 s)**.
5. Extract **accuracy** directly from the events file:
   - Accuracy = 0 (correct), 1 (incorrect), 99 (no response found)

III. Output Data Structure

After preprocessing and segmentation, the pipeline saves:

- **X**: Preprocessed EEG data with shape `(N, T, C)`
  - N = number of trials
  - T = number of time points
  - C = number of channels
- **y**: Labels with shape `(N, 4)` in the format:
  `[task_id, accuracy, subject_id, disease_id]`

Where:
- `task_id` = always 1 for Simon Conflict
- `accuracy` = 0 (correct), 1 (incorrect), 99 (no response found)
- `subject_id` = ID of the subject
- `disease_id` = 0 (Control), 1 (Parkinson’s Disease)

Each subject’s data is saved separately:
- Features: `feature_001.npy`
- Labels: `label_001.npy`

