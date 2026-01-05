Dataset raw name: Simon-conflict Task
Dataset Link: https://openneuro.org/datasets/ds004580/versions/1.0.0
Paper Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592174/

Short name: PD-SIM

## Manually delete  "1182.9060000000	1.0000000000	591453	S  3" in sub-33_task-Simon_events.tsv
## No S 3 event in the EEG data, should be a mistake.

I. Preprocess the EEG data as following steps:
    Preprocessing steps ：
      1) choose common channels and reorder
      2) Set Montage
      3) 60 Hz Notch（before band pass）
      4) bandpass filter（default 0.5–40 Hz）
      5) interpolate bad channels（if do_bad_interp is True）
      6) re-reference to average
      7) ICA（在 1 Hz 高通的副本上拟合，自动剔除眼动/肌电等分量，需 mne-icalabel）
      8) downsample to 200 Hz


II. Data segmentation steps:
    1) Extract the events from the task-Simon_events.tsv file.
    2) Use the S1 - Visual Cue event as the lock for segmenting.
    3) Epoch the data from -500 ms to 1000 ms around the S1 - Visual Cue event.
    4) Baseline correct the epochs using the pre-stimulus interval (-300 ms to -200 ms).


III. Finally we get X, y with shape (N, T, C) and (N, 5) for each subject, respectively.
N is the number of trials, T is the number of time points, C is the number of channels,
each subject is saved in a separate file. X is the EEG data, y is the labels.
e.g, for subject 1, X is feature_001.npy, y is label_001.npy.
y : [task_id, congruentcondition, accuracy, subject_id, disease_id]
task_id is always 1 for simon conflict,
congruentcondition for congruent and incongruent condition of Simon conflict task, 1 for congruent, 2 for incongruent
accuracy for response correctness, 1 for correct, 2 for incorrect, 99 for no response,
subject_id is the ID of the subject,
disease_id is the ID of the disease (0 for healthy, 1 for pakinson's disease).


IV. paper link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10592174/
    paper abstract link:  https://jnnp.bmj.com/content/94/11/945.abstract
