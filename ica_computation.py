import mne
from mne.preprocessing import ICA
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel

# --- Parameters ---

subject = "003"
session = 1
run = 1
raw_file = f"calibration_files/calibration_sub{subject}_session{session}_run{run}.fif"
ica_file = f"ica_files/calibration_sub{subject}_session{session}_run{run}-ica.fif"
output_file = f"ica_files/calibration_sub{subject}_session{session}_run{run}_cleaned.fif"

# --- Load raw EEG ---
raw = mne.io.read_raw_fif(raw_file, preload=True)
raw.filter(l_freq=1.0, h_freq=None)

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# --- Create synthetic EOG (used only for correlation) ---
frontal_chs = ["Fp1", "Fp2"]
picks = mne.pick_channels(raw.info["ch_names"], include=frontal_chs)
synthetic_eog = raw.get_data(picks).mean(axis=0)

info = mne.create_info(["EOG"], raw.info["sfreq"], ch_types=["eog"])
synthetic_eog_raw = mne.io.RawArray(synthetic_eog[np.newaxis, :], info)

raw_with_eog = raw.copy().add_channels([synthetic_eog_raw], force_update_info=True)

# --- Fit ICA only on EEG channels (so it matches your streamed data) ---
ica = ICA(n_components=None, method="picard", random_state=42)
ica.fit(raw, picks="eeg", reject_by_annotation=True)  # <-- no synthetic channel here

# --- Detect blink components using synthetic EOG ---
eog_inds, scores = ica.find_bads_eog(raw_with_eog, ch_name="EOG")
print("Blink-related ICA components:", eog_inds)

# --- Plot inspection before excluding ---
ica.plot_components()                  # topographies of all components
ica.plot_sources(raw_with_eog)         # time series of ICA sources
ica.plot_scores(scores)                # correlation scores with EOG


# --- Exclude components manually or with threshold ---
ica.exclude = []  # reset
print(scores)
ica.exclude = [i for i, s in enumerate(scores) if abs(s) > 0.5]
print("Excluding ICA components:", ica.exclude)
ica.plot_sources(raw_with_eog, picks=ica.exclude)   # only suspected blink components
ica.plot_components(picks=ica.exclude)    # topographies of suspected blink components

# --- Apply ICA to raw EEG ---
raw_clean = ica.apply(raw.copy())


# --- Save ICA solution and cleaned EEG ---
ica.save(ica_file)  # <-- use this ICA in neurofeedback
raw_clean.save(output_file, overwrite=True)

print(f"ICA solution saved to: {ica_file}")
print(f"Cleaned EEG saved to: {output_file}")
