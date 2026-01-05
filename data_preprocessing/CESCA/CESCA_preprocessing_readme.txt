Dataset raw name: Cognitive Electrophysiology in Socioeconomic Context in Adulthood: An EEG dataset
Dataset Link: https://openneuro.org/datasets/ds006018/versions/1.2.2
Paper Link: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0307406

Short name: CESCA
Sub-task dataset short name: CESCA-AODD, CESCA-VODD, CESCA-FLANKER


This data processing file only processes the auditory oddball task and visual oddball task of the
Cognitive Electrophysiology in Socioeconomic Context in Adulthood: An EEG dataset.
The visual oddball data with subject_id 55 and 127 had a buffer overflow problem during acquisition, so they were skipped.

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
    1) Extract the events from the task-auditoryoddball_events.tsv or task-visualoddball_events.tsv.
    2) Remove invalid events at the beginning - boundary, S202, S  1
    3) Epoch the data from -200ms to 800ms around stimulation and response events.
    4) Baseline correct the epochs using the pre-stimulus interval (-200 ms to 0 ms).


III. Finally we get X, y with shape (N, T, C) and (N, 4) for auditory and visual task, respectively,
named CESCA-AODD and CESCA-VODD.
N is the number of trials, T is the number of time points, C is the number of channels, for this dataset, C = 26
each subject is saved in a separate file. X is the EEG data, y is the labels.
e.g, for subject 1, X is feature_001.npy, y is label_001.npy.
y : [task_id, stimulation type, subject_id, parent_degree]
task id is 0 for  auditory oddball task, 1 for visual oddball task.
For auditory oddball task, stimulation type is -1 for Pre-standard stimulus, 0 for Standard stimulus, and 1 for Deviant stimulus.
For visual oddball task, stimulation type is 0 for Standard stimulation, 1 for Oddball stimulation
For flanker task, stimulation type is 0 for Congruent stimulus, 1 for Incongruent stimulus
subject_id is the ID of the subject.
For parent_degree, ['High School or Less', 'Some College or Associate Degree', 'Bachelor's Degree or Higher'] -> (0, 1, 2).