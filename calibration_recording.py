
import time
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop

subject = "003"
session = 1
run = 1
speed_factor = 1.0

ch_names = ["Fp1","F7","F3","FC5","T7","C3","CP5","P7","P3","O1","AFz","Fz","FC1","FC2","Cz","CP1","CP2","Pz","PO3","POz","PO4","Oz","Fp2","F8","F4","FC6","C4","T8","CP6","P4","P8","O2"]

def record_stream_to_fif(out_filename, timeout = 2):
    print("[CLIENT] Looking for Explore_8547_ExG stream ...")
    streams = resolve_byprop('name','Explore_8547_ExG',timeout = 10)
    if len(streams) == 0:
        print("[CLIENT] Can't find Explore_8547_ExG stream.")
        return
    
    inlet = StreamInlet(streams[0])
    fs = int(inlet.info().nominal_srate())
    n_channels = inlet.info().channel_count()
    print(f"[CLIENT] Found EEG stream with {n_channels} channels at {fs}Hz")

    # # Get channel names
    # ch_elem = inlet.info().desc().child('channels').first_child()
    # ch_names = []
    # for _ in range (n_channels):
    #     ch_names.append(ch_elem.child_value('label'))
    #     ch_elem = ch_elem.next_sibling()

    print(f"[CLIENT] Channel names: {ch_names}")
    print(f"[CLIENT] Recording ...")

    # Storage
    all_samples = []
    all_timestamps = []
    start_time = time.time()

    while True:
        sample, timestamp = inlet.pull_sample(timeout=timeout)
        if sample is None:
            print("[CLIENT] Stream ended (no new samples).")
            break
        all_samples.append(sample)
        all_timestamps.append(timestamp)
        # Print progress every 10 seconds
        current_time = time.time()
        if (current_time - start_time) >= 10: 
            print(f"Recorded {len(all_samples)/fs:.1f} seconds of data...")
            start_time = time.time()
    
    if not all_samples:
        print("[CLIENT] No data recorded.")
        return        

    data = np.array(all_samples).T  

    info = mne.create_info(ch_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)

    raw.save(out_filename, overwrite=True)
    print(f"[CLIENT] Saved {raw.n_times/fs:.1f} seconds of data to {out_filename}")


if __name__ == "__main__":
    print("Waiting for the EEG stream to start...")
    # Make sure the EEG is streaming before running this

    record_stream_to_fif(f"calibration_files/calibration_sub{subject}_session{session}_run{run}.fif")
