Data	 : NUI_Maynooth_EEG 
Creator : Kevin Sweeney (ksweeney'at'eeng'dot'nuim'dot'ie)

Data recorded using the methodology outlined by Sweeney et. al. in 
"Methodology for Validating Artifact Removal Techniques for Physiological Signals". 
- (http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6236173)
Data used in "The use of Ensemble Empirical Mode Decomposition with Canonical Correlation Analysis as a Novel Artifact Removal Technique". 
- (http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=06332491)

------------------------------------------------------ 
23 Trials of EEG recorded 
Each trial consists of 2 channels of highly correlated electroencephalography (EEG) data recorded from the pre-frontal cortex.
One channel is contaminated with motion artifact. 
Second channel is free from motion artifact.

------------------------------------------------------ 
Trigger information
- Single trigger sampled individually by the EEG and accelerometer recording systems. 
- Initial rise represents the start of the experiment 
- Final fall represents the end of the experiment 
- Interviening low values represent the epochs where motion artifact was produced

------------------------------------------------------ 
Sampling Frequencies
- EEG : 2048 Hz 
- Accelerometer : 200 Hz

------------------------------------------------------ 
NUI_Maynooth_EEG 
Columns in the TrialX.txt file represent:
Column 1: Sample index 
Column 2: Channel 1 : Raw EEG : sampled @ 2048 Hz 
Column 3: Channel 2 : Raw EEG : sampled @ 2048 Hz 
Column 4: Trigger data for EEG data : sampled @ 2048 Hz 
Column 5: Accelerometer 1 : X-axis : sampled @ 200 Hz 
Column 6: Accelerometer 1 : Y-axis : sampled @ 200 Hz 
Column 7: Accelerometer 1 : Z-axis : sampled @ 200 Hz 
Column 8: Accelerometer 2 : X-axis : sampled @ 200 Hz 
Column 9: Accelerometer 2 : Y-axis : sampled @ 200 Hz 
Column 10: Accelerometer 2 : Z-axis : sampled @ 200 Hz 
Column 11: Trigger data for accelerometer data : sampled @ 200 Hz
Therefore, for example, the first second of recording is represented by the first 2048 samples of column 2-4 and the first 200 samples of columns 5-11.