import time
import mne
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import threading
import tkinter as tk
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Ellipse
from PIL import Image, ImageTk
from PyQt5.QtWidgets import QApplication, QLabel

# ------------------ Parameters ------------------
subject = "003"
session = 1
run = 2
ica_filename = f"ica_files/calibration_sub{subject}_session{session}_run1-ica.fif"  
occ_channels = ["PO3", "PO4", "POz", "O1", "O2", "Oz"]
buffer_seconds = 1

# We'll compute PSD over this frequency range
psd_fmin, psd_fmax = 4.0, 25.0

# thresholds for mapping slope -> attention (tune these)
slope_low_threshold = -0.50   # below this (more negative) considered low attention
slope_high_threshold = -0.25  # above this (less negative) considered high attention


ch_names = ["Fp1","F7","F3","FC5","T7","C3","CP5","P7","P3","O1","AFz","Fz","FC1","FC2","Cz","CP1","CP2","Pz","PO3","POz","PO4","Oz","Fp2","F8","F4","FC6","C4","T8","CP6","P4","P8","O2"]


# ------------------ Flower GUI ------------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class FlowerNeurofeedback(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#E0F7FA")
        self.master = master
        self.started = False
        self.attention_step = 0
        self.attention_level = 0
        self.max_attention = 0
        self.remaining_time = 300
        self.finished_time = 0
        
        # --- TBR Plot components (initialized later in a Toplevel window) ---
        self.tbr_window = None
        self.tbr_fig = None
        self.tbr_ax = None
        self.tbr_line = None
        self.tbr_canvas = None # Added for clarity
        self.slope_data_x = []
        self.slope_data_y = []
        self.saved_eeg = []      # will store [timestamp, ch1, ch2, ..., chN]
        self.saved_slope = []     # will store [timestamp, avg_slope]
        self.channel_names = []

        
        # --- Flower Plot setup ---
        self.fig, self.ax = self._setup_flower_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, pady=10, anchor="center")

        # Initialize flower parts
        self._initialize_flower_parts()
        self.draw_flower(0) # Draw initial state

        # Slider and TBR Label Frame
        bottom_frame = tk.Frame(self, bg="#E0F7FA")
        bottom_frame.pack(pady=10, anchor="center")
        
        # Slider
        self.slider = ctk.CTkSlider(bottom_frame, from_=0, to=100,
                                            number_of_steps=100,
                                            command=lambda val: self.update_from_slider(val),
                                            width=400, height=20)
        self.slider.pack(pady=5)
        self.slider_value_label = tk.Label(bottom_frame, text="0%", font=("Arial", 16), bg="#E0F7FA")
        self.slider_value_label.pack(pady=5)
        
        # TBR Value Label (Stays on main screen for easy view)
        self.slope_value_label = tk.Label(bottom_frame, text="Slope: N/A", font=("Arial", 16), bg="#E0F7FA")
        self.slope_value_label.pack(pady=5)

        # Timer label
        self.timer_label = tk.Label(self, text="Remaining time : 02:00",
                                            font=("Arial", 16, "bold"),
                                            fg="white", bg="#555555", pady=5)
        self.timer_label.pack(pady=10)

        # Results page
        self.page_result = tk.Frame(self.master, bg="#E0F7FA")
        # Ensure image is loaded properly (assuming you have "star_full.png" and "star_empty.png")
        try:
            self.star_full_small = ImageTk.PhotoImage(Image.open("star_full.png").resize((40, 40)))
            self.star_empty_small = ImageTk.PhotoImage(Image.open("star_empty.png").resize((40, 40)))
        except FileNotFoundError:
             # Placeholder for stars if files are missing
             self.star_full_small = None
             self.star_empty_small = None
        
        # Bind the close event of the main window to also close the TBR window
        self.master.protocol("WM_DELETE_WINDOW", self._on_main_close)

    def _on_main_close(self):
        """Handles closing both the main window and the TBR window."""
        if self.tbr_window:
            self.tbr_window.destroy()
        self.master.quit()
        self.master.destroy()

    def _setup_flower_plot(self):
        # Helper to set up the plot environment
        fig, ax = plt.subplots(facecolor='#E0F7FA')
        #ax.set_title("Flower Growth (Concentration Level)")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 8)
        ax.set_aspect("equal")
        ax.axis("off")
        # Soil
        soil = Ellipse((0, -0.5), width=8, height=2,
                       facecolor="#A0522D", edgecolor="#A0522D", zorder=0)
        ax.add_patch(soil)
        return fig, ax
    
    def _setup_tbr_plot(self, tbr_win):
        """Sets up the Slope plot within the Toplevel window."""
        fig, ax = plt.subplots(facecolor='#E0F7FA', figsize=(6, 4))
        ax.set_title("Real-time Average Spectral Slope")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Average Spectral Slope")
        ax.set_xlim(0, 300) # Initial time limit
        ax.set_ylim(-4, 0) # Initial Slope limit
        ax.grid(True)
        # Add horizontal lines for thresholds

        ax.axhline(y=slope_low_threshold, color = 'r', linestyle='--', label=f'slope = {slope_low_threshold}')
        ax.axhline(y=slope_high_threshold, color = 'orange', linestyle=':', label=f'slope = {slope_high_threshold}')

        ax.legend(loc='upper right', fontsize='small')
        line, = ax.plot([], [], 'b-') # Initialize an empty line plot
        
        # Embed the plot into the Toplevel window
        canvas = FigureCanvasTkAgg(fig, master=tbr_win)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        return fig, ax, line, canvas

    def _setup_tbr_window(self):
        """Initializes the TBR plot window and components."""
        if self.tbr_window and self.tbr_window.winfo_exists():
             self.tbr_window.destroy()
        
        # --- 1. Create the Toplevel window for the Slope plot ---
        self.tbr_window = tk.Toplevel(self.master)
        self.tbr_window.title("Real-time Spectral Slope Plot")
        self.tbr_window.geometry("600x400")
        
        # Prevent the user from killing the main window by closing this one
        self.tbr_window.protocol("WM_DELETE_WINDOW", lambda: self.tbr_window.withdraw()) 
        
        # --- 2. Initialize the plot inside the new window ---
        self.slope_data_x.clear()
        self.slope_data_y.clear()
        self.tbr_fig, self.tbr_ax, self.tbr_line, self.tbr_canvas = self._setup_tbr_plot(self.tbr_window)

        # Bring the TBR window to the front
        self.tbr_window.lift()
        self.tbr_window.attributes('-topmost', True)
        self.tbr_window.attributes('-topmost', False) # Remove 'always on top' property
        
        # Ensure it is displayed
        self.tbr_window.deiconify()


    def _initialize_flower_parts(self):
        # Flower parts for the main game plot
        self.stem_line, = self.ax.plot([], [], color="#4CAF50", lw=6)
        self.petals = []
        self.leaf = None
        self.leaf_2 = None
        self.center = plt.Circle((0, 6), 0, color="#FFF176", zorder=5)
        self.ax.add_patch(self.center)
    
    # Flower drawing (KEPT AS IS)
    def draw_flower(self, g, ax=None, petals_list=None, leaf_ref=None, leaf_2_ref=None, center_ref=None):
        current_ax = ax if ax is not None else self.ax
        current_petals = petals_list if petals_list is not None else self.petals
        current_center = center_ref if center_ref is not None else self.center
        
        if ax is None:
            if self.leaf: self.leaf.remove(); self.leaf=None
            if self.leaf_2: self.leaf_2.remove(); self.leaf_2=None

        # --- Remove any existing "seed" patches ---
        for artist in getattr(self, "_seed_patches", []):
            artist.remove()
        self._seed_patches = []

        stem_height = 6 * g

        if ax is None:
            self.stem_line.set_data([0, 0], [0, stem_height])
        else:
            current_ax.plot([0, 0], [0, stem_height], color="#4CAF50", lw=6)

        # --- Draw a seed when g == 0 ---
        if g <= 0.001:
            seed = plt.Circle((0, -0.2), 0.2, color="#5D4037", zorder=3, ec="black")
            sprout = Ellipse((0, 0.0), width=0.05, height=0.2,
                            facecolor="#4CAF50", edgecolor="#2E7D32", alpha=0.8, zorder=2)
            current_ax.add_patch(seed)
            current_ax.add_patch(sprout)
            if ax is None:
                self._seed_patches = [seed, sprout]
            if ax is None:
                self.canvas.draw_idle()
            return  # Stop here — don’t draw stem or petals yet

        

        stem_height = 6 * g
        
        if ax is None:
            self.stem_line.set_data([0,0],[0,stem_height])
        else:
            current_ax.plot([0,0],[0,stem_height], color="#4CAF50", lw=6) 

        if g > 0.3:
            leaf_size = g
            leaf = Ellipse((leaf_size/2, stem_height/2),
                           width=leaf_size*1.2, height=leaf_size*0.6, angle=30,
                           facecolor="#4CAF50", edgecolor="#4CAF50", alpha=0.8, zorder=1)
            current_ax.add_patch(leaf)
            leaf_2 = Ellipse((-leaf_size/2, stem_height/2-0.5),
                              width=leaf_size*1.2, height=leaf_size*0.6, angle=-30,
                              facecolor="#4CAF50", edgecolor="#4CAF50", alpha=0.8, zorder=1)
            current_ax.add_patch(leaf_2)
            if ax is None:
                self.leaf = leaf
                self.leaf_2 = leaf_2

        for p in current_petals: p.remove()
        current_petals.clear()

        if g > 0.5:
            n_petals=30
            petal_length=1.5*(g-0.5)*2+0.5
            petal_width=0.3
            angles = np.linspace(0,360,n_petals,endpoint=False)
            for angle in angles:
                x_center=petal_length/2*np.cos(np.deg2rad(angle))
                y_center=stem_height+petal_length/2*np.sin(np.deg2rad(angle))
                petal = Ellipse((x_center,y_center), width=petal_width, height=petal_length,
                                 angle=angle+90, facecolor="#9B59B6", edgecolor="#7D3C98",
                                 alpha=0.5, zorder=2)
                current_ax.add_patch(petal)
                current_petals.append(petal)
        
        if ax is None:
            current_center.radius = g*0.3 if g>0.5 else 0
            current_center.center = (0, stem_height) if g>0.5 else (0,6)
            self.canvas.draw_idle()
        else:
            if g > 0.5:
                 new_center = plt.Circle((0, stem_height), g*0.3, color="#FFF176", zorder=5)
                 current_ax.add_patch(new_center)
            # The canvas drawing will be handled by the display_final_flower method

    def display_final_flower(self, max_attention):
        # Set up a new Matplotlib plot for the results page
        result_fig, result_ax = self._setup_flower_plot()
        
        self.draw_flower(max_attention/100, ax=result_ax, petals_list=[], leaf_ref=None, leaf_2_ref=None, center_ref=None)
        
        result_canvas = FigureCanvasTkAgg(result_fig, master=self.page_result)
        result_canvas_widget = result_canvas.get_tk_widget()
        result_canvas_widget.pack(fill="both", expand=True, pady=5)
        
        result_canvas.draw()
        
        self._result_canvas = result_canvas
        self._result_fig = result_fig

    def update_tbr_plot(self, avg_slope):
        """Updates the TBR plot in the separate window."""
        if not self.tbr_window or not self.tbr_window.winfo_exists():
            # If the window was closed manually, stop trying to update
            return
            
        current_time = 300 - self.remaining_time # Time elapsed
        
        # 1. Update data lists
        self.slope_data_x.append(current_time)
        self.slope_data_y.append(avg_slope)
        
        # 2. Update plot line data
        self.tbr_line.set_data(self.slope_data_x, self.slope_data_y)
        
        # 3. Adjust x-axis limit
        # Ensure plot always shows a window of time, dynamically expanding
        x_max = max(10, current_time + 5)
        self.tbr_ax.set_xlim(0, x_max)
        
        # 4. Adjust y-axis limit
        y_min = np.min(self.slope_data_y)
        y_max = np.max(self.slope_data_y)

        padding = 0.1*(y_max-y_min if y_max != y_min else 1)

        self.tbr_ax.set_ylim(y_min-padding, y_max+padding)
        
        # 5. Redraw the canvas
        self.tbr_canvas.draw_idle()

        # 6. Update the TBR label on the main screen
        self.slope_value_label.config(text=f"Slope: {avg_slope:.2f}")
        
    def update_from_slider(self, val):
        """Allows manual control of the flower for testing/demonstration if needed."""
        g = float(val) / 100.0
        self.draw_flower(g)

    def start_feedback(self):
        """Starts the main neurofeedback loop thread and timer."""
        if not self.started:
            self.started=True
            
            # TBR window setup is handled in main loop before go_to_game call
            
            threading.Thread(target=self.run_neurofeedback, daemon=True).start()
            self.update_timer()

    # ------------------ Neurofeedback + Flower Update ------------------
    def run_neurofeedback(self):
        # --- Load ICA ---
        try:
            ica = mne.preprocessing.read_ica(ica_filename)
            unmixing_matrix = ica.unmixing_matrix_
            mixing_matrix = ica.mixing_matrix_
            blink_ics = ica.exclude
            mixing_matrix_clean = mixing_matrix.copy()
            for ic in blink_ics:
                mixing_matrix_clean[:, ic] = 0
        except FileNotFoundError:
            print(f"ICA file not found: {ica_filename}")
            return

        # --- Connect to LSL ---
        streams = None
        print("Searching for LSL Explore_8547_ExG stream...")
        while streams is None or len(streams)==0:
            streams = resolve_byprop('name','Explore_8547_ExG',timeout=2)
            time.sleep(1)
        
        print("LSL stream found. Starting acquisition.")
        inlet = StreamInlet(streams[0])
        fs = int(inlet.info().nominal_srate())
        n_channels = inlet.info().channel_count()

        # Get channel names properly
        # ch_names=[]
        # ch_elem = inlet.info().desc().child("channels").first_child()
        # for _ in range(n_channels):
        #     ch_names.append(ch_elem.child_value("label"))
        #     ch_elem = ch_elem.next_sibling()
        occ_indices = [i for i,ch in enumerate(ch_names) if ch in occ_channels]

        if len(occ_indices)==0:
            print("ERROR: No occipital channels found in stream.")
            return
        
        self.channel_names = ch_names

        samples_in_buffer = int(fs*buffer_seconds)
        data_buffer=[np.zeros(samples_in_buffer) for _ in range(n_channels)]
        ptr=0

        while self.remaining_time>0:
            sample,timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            self.saved_eeg.append([timestamp] + sample)
            for j in range(n_channels):
                data_buffer[j][ptr % samples_in_buffer]=sample[j]
            ptr+=1

            if ptr % samples_in_buffer == 0:
                # buffer ready: shape (n_times, n_channels)
                raw_buffer_np = np.array(data_buffer).T  # shape (samples, channels)
                # Apply ICA unmixing and remove blink ICs
                ica_sources = np.dot(raw_buffer_np, unmixing_matrix.T)
                clean_data = np.dot(ica_sources, mixing_matrix_clean.T)  # shape (samples, channels)
                clean_occ_data = clean_data[:, occ_indices]             # shape (samples, n_occ)

                # ---- compute PSD for occ channels ----
                try:
                    # psd_array_multitaper expects shape (n_channels, n_times)
                    psd, freqs = mne.time_frequency.psd_array_multitaper(
                        clean_occ_data.T, sfreq=fs, fmin=psd_fmin, fmax=psd_fmax, verbose=False
                    )
                    # psd shape: (n_occ, n_freqs)
                except Exception as e:
                    print("PSD computation failed:", e)
                    continue

                slopes = []
                for ch_psd in psd:
                    valid_mask = ch_psd > 0
                    if np.sum(valid_mask) < 2:
                        slopes.append(np.nan)
                        continue
                    log_freqs = np.log10(freqs[valid_mask])
                    log_psd = np.log10(ch_psd[valid_mask])
                    # Linear fit in log-log: slope = coef of log_freqs
                    try:
                        slope, intercept = np.polyfit(log_freqs, log_psd, 1)
                        slopes.append(slope)
                    except Exception as e:
                        slopes.append(np.nan)

                avg_slope = float(np.nanmean(slopes)) if len(slopes)>0 else np.nan

                current_time = 300 - self.remaining_time
                self.saved_slope.append([current_time, avg_slope])


                # --- Safely update the Slope plot in the separate window ---
                # This must be scheduled on the main thread via self.master.after()
                self.master.after(0, self.update_tbr_plot, avg_slope)
                # ---------------------------------------------------------

                # ---------------- Flower logic based on slope ----------------
                # NOTE: slopes are typically negative (1/f). Tune thresholds above as needed.

                if not np.isnan(avg_slope):
                    if avg_slope > slope_high_threshold:
                        # relatively flat spectrum --> "higher attention" (example mapping)
                        self.attention_step += 5
                    elif avg_slope > slope_low_threshold:
                        self.attention_step += 2.5
                    # else: no reward for very negative slope
                self.attention_step = min(self.attention_step, 100)
                self.attention_level = self.attention_step
                self.max_attention = max(self.max_attention, self.attention_level)
                g = self.attention_level/100

                # Flower updates also scheduled on the main thread
                self.master.after(0, self.draw_flower, g)
                self.master.after(0, self.slider.set, self.attention_level)
                self.master.after(0, self.slider_value_label.config, dict(text=f"{int(self.attention_level)}%"))

                if self.attention_level >= 100:
                    self.finished_time = 300 - self.remaining_time
                    self.remaining_time = 0
                    return
                

    def update_timer(self):
        if self.remaining_time<=0:
            self.timer_label.config(text="🛑 Time elapsed !")
            if self.finished_time == 0:
                self.finished_time = 300
            self.master.after(1500, self.display_result)
            return
        minutes = self.remaining_time//60
        seconds = self.remaining_time%60
        self.timer_label.config(text=f"⏳ Time remaining : {minutes:02d}:{seconds:02d}")
        self.remaining_time -=1
        self.after(1000,self.update_timer)


    def display_result(self):
        if self.tbr_window and self.tbr_window.winfo_exists():
            self.tbr_window.withdraw() # Hide the TBR plot when showing results

        self.pack_forget()
        self.page_result.pack(fill="both", expand=True)
        tk.Label(self.page_result,text="Your Result",font=("Arial",20,"bold"),bg="#E0F7FA").pack(pady=10)
        self.display_final_flower(self.max_attention)
        tk.Label(self.page_result,text=f"{int(self.max_attention)}%", font=("Arial",20,"bold"),bg="#E0F7FA").pack(pady=5)

        # Assuming star images are correctly loaded for the results page
        if self.star_full_small and self.star_empty_small:
            # Re-load or resize to a larger image for the results page display
            # Note: The original code reloads with a different size, but we'll use a placeholder variable for safety.
            try:
                star_full_large = ImageTk.PhotoImage(Image.open("star_full.png").resize((80, 80))) 
                star_empty_large = ImageTk.PhotoImage(Image.open("star_empty.png").resize((80, 80))) 
            except FileNotFoundError:
                 star_full_large = self.star_full_small # Use small if large fails
                 star_empty_large = self.star_empty_small # Use small if large fails
        else:
             star_full_large = None
             star_empty_large = None

        stars_frame=tk.Frame(self.page_result,bg="#E0F7FA")
        stars_frame.pack(pady=10)
        stars_labels=[]
        for i in range(5):
            label=tk.Label(stars_frame,image=star_empty_large,bg="#E0F7FA")
            label.pack(side="left", padx=5)
            stars_labels.append(label)
        
        nb_stars = min(5, max(1, int(self.max_attention)//20))
        if star_full_large:
            for i in range(nb_stars):
                stars_labels[i].config(image=star_full_large)

        self.save_results()
        self.save_as_fif()

            
        tk.Label(self.page_result,text=f"Completed in : {int(self.finished_time)} seconds",
                         font=("Arial",16,"bold"),bg="#E0F7FA").pack(pady=10)
        tk.Button(self.page_result,text="Leave",command=self.master.quit,
                         bg="gray",fg="white",font=("Arial",14),padx=20,pady=10).pack(pady=10)
        tk.Button(self.page_result,text="Restart",command=self.restart_test,
                         bg="#4caf50",fg="white",font=("Arial",14),padx=20,pady=10).pack(pady=10)

        self.save_results_numeric(f"results_slope/numeric_results_sub{subject}_session{session}_run{run}.txt")
        
    def save_results_numeric(self, filename):

        with open(filename, "w") as f:
            f.write(f"flower growth : {str(self.max_attention)}, time taken : {str(self.finished_time)} seconds\n")
        print(f"Results saved in {filename}")


    def save_as_fif(self):
        from mne import create_info, io

        if not self.saved_eeg or not self.channel_names:
            print("No EEG data or channel names to save.")
            return

        # Extract sampling rate from buffer length
        eeg_data = np.array(self.saved_eeg)  # shape: (N_samples, N_channels+1)
        timestamps = eeg_data[:, 0]
        data_only = eeg_data[:, 1:].T  # shape: (n_channels, n_times)

        # Estimate sampling rate from timestamps
        if len(timestamps) < 2:
            print("Not enough data to calculate sampling rate.")
            return

        dt = np.diff(timestamps)
        sfreq = 1 / np.median(dt)

        # Create MNE Raw object
        info = create_info(ch_names=self.channel_names, sfreq=sfreq, ch_types='eeg')
        raw = io.RawArray(data_only, info)

        # Save the file
        out_file = f"results_slope/eeg_data_sub{subject}_session{session}_run{run}.fif"
        raw.save(out_file, overwrite=True)
        print(f"Saved EEG as FIF: {out_file}")

            
    def save_results(self):
        os.makedirs("results_slope", exist_ok=True)

        # --- Save EEG ---
        eeg_file = f"results_slope/eeg_data_sub{subject}_session{session}_run{run}.csv"
        with open(eeg_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + self.channel_names)   # header row
            writer.writerows(self.saved_eeg)

        # --- Save Slope ---
        slope_file = f"results_slope/slope_values_sub{subject}_session{session}_run{run}.csv"
        with open(slope_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_elapsed_sec", "avg_slope"])
            writer.writerows(self.saved_slope)

        print(f"Saved EEG to {eeg_file} and Slope to {slope_file}")

    def restart_test(self):
        if self.tbr_window and self.tbr_window.winfo_exists():
            # Re-initialize the window and its plot
            self._setup_tbr_window()
            self.tbr_window.deiconify() # Show the TBR plot again

        self.page_result.pack_forget()
        self.pack(fill="both",expand=True)
        self.started=False
        self.remaining_time=300
        self.max_attention=0
        self.attention_level=0
        self.attention_step=0
        
        # Reset flower display
        self.draw_flower(0)
        self.slider.set(0)
        self.slider_value_label.config(text="0%")
        
        # Reset TBR display
        self.slope_value_label.config(text="TBR: N/A")
        
        self.update_timer()
        self.start_feedback()

# ------------------ Main ------------------
root=tk.Tk()
root.title("Neurofeedback Concentration Training")
root.state('zoomed')
root.configure(bg="#E0F7FA")

# Welcome page
welcome_page=tk.Frame(root,bg="#E0F7FA")
welcome_page.pack(fill="both",expand=True)
tk.Label(welcome_page,text="Welcome to Neurofeedback\nConcentration Training",
             font=("Helvetica",28,"bold"),bg="#E0F7FA",pady=20).pack(pady=40)
tk.Label(welcome_page,text="This training will help you practice focusing.\n\n"
             "1. **ENSURE YOUR EEG DEVICE IS STREAMING TO LSL (type 'EEG').**\n" # <-- NEW INSTRUCTION
             "2. Press Start to begin.\n"
             "3. Concentrate and watch the flower grow.\n"
             "4. Try to keep your focus as high as possible!",
             font=("Arial",16),bg="#E0F7FA",justify="center").pack(pady=20)


test_page = FlowerNeurofeedback(root)

def go_to_game():
    """Function to transition from welcome screen to game and START the feedback loop."""
    welcome_page.pack_forget()
    test_page.pack(fill="both",expand=True)
    # The TBR window is already set up and running, we just need to start the feedback loop/timer
    test_page.start_feedback() # This starts the LSL reading and timer

tk.Button(welcome_page,text="Start Training",command=go_to_game,
           font=("Arial",20,"bold"),bg="#4CAF50",fg="white",padx=40,pady=20).pack(pady=40)



# --- 2. Set up the TBR window IMMEDIATELY ---
# This creates the Toplevel window and initializes the plot, making it visible.
# This must be scheduled with 'after' to ensure the main window (root) has initialized.
root.after(100, test_page._setup_tbr_window)

root.mainloop()