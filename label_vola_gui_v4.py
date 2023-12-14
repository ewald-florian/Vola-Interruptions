import os
import time
import sys
import random
random.seed(42)

import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import Entry
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Get directory of this script to work with relative paths.
script_dir = os.path.dirname(os.path.realpath(__file__))


class SampleLabelerGUI(tk.Tk):
    def __init__(self,
                 sample_batch_number=1,
                 shuffle_vola_batch=True,
                 time_frame_seconds=240,
                 display_news=True,
                 news_buffer=10,
                 reset_result_file=False,
                 verbose=True):

        super().__init__()

        self.data_dir = os.path.join(script_dir, "vola_data_midpoints_new_new")
        self.target_dir = os.path.join(script_dir, "labeled_data")
        self.news_dir = os.path.join(script_dir, "news", "news_reduced.csv.gz")
        self.verbose = verbose
        self.shuffle_vola_batch = shuffle_vola_batch
        self.sample_batch_number = sample_batch_number
        self.display_news = display_news
        self.time_frame_seconds = time_frame_seconds
        self.news_buffer = news_buffer
        self.isin_mapping = pd.read_csv("utils/DAX40_ISIN_NAME.csv",
                                        index_col=["ISIN"])
        self.news_df = pd.read_csv(self.news_dir,
                                   parse_dates=["TIMESTAMP_UTC"])

        if reset_result_file:
            self._reset_result_file()
        self.result_df = self._load_result_df()

        # -- TK Inter
        self.window = tk.Tk()
        self.window.title("Vola Sample Label")

        # Create a frame for buttons and entry field
        self.button_frame = tk.Frame(self.window)
        # Put buttons above plot
        self.button_frame.pack(side="top", padx=10, pady=10)

        # Create a canvas for the Matplotlib fig
        self.canvas = None
        self.canvas_widget = None

    def _reset_result_file(self):
        """
        Reset result file "label_result_file.csv" by changing its filename.
        The old file is not deleted but just renamed to:
        "_labeled_samples_" + <current time> + "csv"
        Hence, no data will be lost by the rest.
        """
        result_file = os.path.join(self.target_dir, "label_result_file.csv")
        current_time_str = time.strftime("%Y-%m-%d_%H-%M-%S",
                                         time.localtime())
        filename = "_labeled_samples_" + current_time_str + ".csv"
        filepath = os.path.join(self.target_dir, filename)
        os.rename(result_file, filepath)

    def _load_result_df(self):
        """Load the existing label results into result_df and store the old
        file under a different filename.
        After the labeling process is complete, all old and new entries will
        be stores to "label_result_file.csv".
        """
        result_file = os.path.join(self.target_dir, "label_result_file.csv")
        # Load Existing result file.
        if os.path.exists(result_file):

            # Load results from label_results_file.csv
            result_df = pd.read_csv(result_file)

            # Save the old version.
            self._reset_result_file()

            return result_df

        else:
            # If the file doesn't exist, create a new DataFrame
            return pd.DataFrame(columns=['Filename', 'Label', 'Comment'])

    def _load_vola_df(self, vola_filename):
        """
        Load individual vola events.
        :param vola_filename: name of csv file
        :return vola_df: pd.DataFrame of vola event
        """
        self.vola_filename = vola_filename
        self.current_isin = vola_filename.split("_")[0]
        filepath = os.path.join(self.data_dir, vola_filename)
        vola_df = pd.read_csv(filepath)
        return vola_df

    def _plot_vola_old(self, df, display_slopes=True):
        """
        Create Plot of vola event.
        :param df: vola dataframe
        :return fig: matplotlib plot of midpoint time series
        """
        # Assume that vola start is in the middle of the sample.
        # TODO: This is NOT Vola start!

        # Deducto one since pd df index starts with 0.
        vola_start = (self.time_frame_seconds / 2) - 1

        pre_vola = df[df.index < vola_start]
        post_vola = df[df.index >= vola_start]

        # Fit a line through before_vola and after_vola
        coeff_before = np.polyfit(pre_vola.index, pre_vola["Midpoint_Norm"],1)
        coeff_after = np.polyfit(post_vola.index, post_vola["Midpoint_Norm"],1)

        line_before = np.polyval(coeff_before, pre_vola.index)
        line_after = np.polyval(coeff_after, post_vola.index)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Midpoint_Norm"], label="Midpoint Norm", color="b")
        ax.axvline(x=vola_start, color="r", linestyle=":",
                   linewidth=1, label=f"Vola Start")

        if display_slopes:
            ax.plot(pre_vola.index, line_before, linestyle="--", color="g",
                    linewidth=1, label="Pre Vola Regression")
            ax.plot(post_vola.index, line_after, linestyle="--", color="y",
                    linewidth=1, label="Post Vola Regression")

        ax.set_title("Vola Interruption", fontsize=14)
        ax.legend()

        return fig

    def _plot_vola(self, df):
        """Create plot of vola event."""

        time_frame_seconds = 240
        vola_start = (time_frame_seconds / 2) - 1

        insert_index = 120
        num_rows_to_insert = 10

        # Add NaN rows to create a gap during the vola interruption.
        new_rows = pd.DataFrame(np.nan, index=range(insert_index,
                        insert_index + num_rows_to_insert), columns=df.columns)

        df = pd.concat([df.iloc[:insert_index], new_rows,
                        df.iloc[insert_index:]]).reset_index(drop=True)

        pre_vola = df[df.index <= 119]
        empty = df[(df.index > 119) & (df.index < 130)]
        post_vola = df[df.index >= 130]

        # Fit a line through before_vola and after_vola
        coeff_before = np.polyfit(pre_vola.index, pre_vola["Midpoint_Norm"], 1)
        coeff_after = np.polyfit(post_vola.index, post_vola["Midpoint_Norm"],
                                 1)

        line_before = np.polyval(coeff_before, pre_vola.index)
        line_after = np.polyval(coeff_after, post_vola.index)

        plt.plot(df["Midpoint_Norm"], label="Midpoint Norm", color="b")

        plt.axvline(x=vola_start, color="r", linestyle=":",
                    linewidth=1, label=f"Vola Start")

        plt.plot(pre_vola.index, line_before, linestyle="--", color="g",
                 linewidth=1, label="Pre Vola Regression")

        plt.plot(post_vola.index, line_after, linestyle="--", color="y",
                 linewidth=1, label="Post Vola Regression")

        plt.title("Vola Interruption", fontsize=14)
        plt.legend()

        return plt.gcf()

    def _get_vola_info(self, df):
        """
        Get stock name and vola start timestamp.
        :param df: vola_df
        :return info_text: info dict
        """
        isin = self.current_isin
        name = self.isin_mapping.loc[isin].NAME
        # deduct 1 since pd df index starts with 0.
        vola_start_time = str(df.loc[(self.time_frame_seconds/2) - 1,
                                    "Date_Time"]).split(".")[0]
        info_dict = {"Datetime": vola_start_time,
                     "Company": name}
        return info_dict

    def _get_news_info(self, df, number_of_news_displayed):
        """
        Get infos from news around the vola event.
        :param vola_start_time:
        :param news_buffer:
        :param number_of_news_displayed:
        :return:
        """

        news_dict = {}

        name = self.isin_mapping.loc[self.current_isin].NAME
        vola_start_time = str(df.loc[(self.time_frame_seconds / 2) - 1,
                            "Date_Time"]).split(".")[0]

        vola_start_time = pd.to_datetime(vola_start_time)
        news_start_time = vola_start_time - pd.Timedelta(self.news_buffer, "m")
        news_end_time = vola_start_time + pd.Timedelta(self.news_buffer, "m")

        # filter news df
        name_mask = (self.news_df["ENTITY_NAME"] == name)
        start_mask = (self.news_df["TIMESTAMP_UTC"] > news_start_time)
        end_mask = (self.news_df["TIMESTAMP_UTC"] < news_end_time)
        combined_masks = name_mask & start_mask & end_mask
        filtered_news = self.news_df.loc[combined_masks]

        # Count news.
        news_dict["news count total"] = len(filtered_news)
        news_dict["news count pre"] = len(filtered_news.loc[
                            (self.news_df["TIMESTAMP_UTC"] < news_start_time)])
        news_dict["news count post"] = len(filtered_news.loc[
                        (self.news_df["TIMESTAMP_UTC"] >= news_start_time)])

        # Mean values.
        news_dict["avg event sentiment"] = round(
            filtered_news["EVENT_SENTIMENT_SCORE"].mean(), 2)
        news_dict["avg similarity days"] = round(
            filtered_news["EVENT_SIMILARITY_DAYS"].mean(), 2)
        news_dict["avg event relevance"] = round(
            filtered_news["EVENT_RELEVANCE"].mean(), 2)

        # Individual news.
        filtered_news = filtered_news.sort_values(by="EVENT_RELEVANCE",
                                                  ascending=False)
        # Only keep unique event texts and drop NaN
        unique_event_texts = list(
            filtered_news["EVENT_TEXT"].dropna().unique())
        for i, event_text in enumerate(
                unique_event_texts[:number_of_news_displayed]):
            news_dict[f"EVENT_TEXT: {i + 1}"] = event_text

        return news_dict

    def _store_input(self, label: int, comment: str, exit: bool):
        """
        Take input from tk widget and store to results_df.
        :param label: int (Wanted / Unwanted)
        :param comment: str
        :param exit: bool, exit program if True
        :return: None
        """
        results = [self.vola_filename, label, comment]
        self.result_df.loc[len(self.result_df)] = results

        if exit:
            self._exit_labeling()

    def _save_results_to_csv(self):
        """
        Store result df to csv file.
        :return: None
        """
        # Overwrite the file "label_result_file.csv".
        filepath = os.path.join(self.target_dir, "label_result_file.csv")
        self.result_df.to_csv(filepath, index=False)
        if self.verbose:
            print(f"Results stored to: {filepath}")

    def _exit_labeling(self):
        """
        Store results and end program.
        :return: None
        """
        self._save_results_to_csv()
        if self.verbose:
            print("Labelling Process Ended via Exit.")
        # Stop the programm.
        sys.exit()

    def _create_widget(self, fig, info_text):
        """
        Create tk widget for the current sample.
        :param fig: plt.figure
        :param info_text: info dict
        :return label_value: bool, label (wanted / unwanted)
        :return comment_value: str, comment
        :return exit_value: bool
        """
        label_value = None
        comment_value = None
        exit_value = None

        # Only create canvas if it doesn't exist yet
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(side="top", fill="both", expand=True)

        def on_wanted_click():
            nonlocal label_value
            label_value = 0
            wanted_button.config(state=tk.DISABLED)
            unwanted_button.config(state=tk.NORMAL)

        def on_unwanted_click():
            nonlocal label_value
            label_value = 1
            unwanted_button.config(state=tk.DISABLED)
            wanted_button.config(state=tk.NORMAL)

        def on_next_click():
            nonlocal comment_value, exit_value
            comment_value = comment_entry.get()
            exit_value = False
            canvas_widget.destroy()
            fig.clf()
            plt.close(fig)
            window.destroy()

        def on_exit_click():
            nonlocal exit_value
            exit_value = True
            canvas_widget.destroy()
            fig.clf()
            plt.close(fig)
            window.destroy()

        # Create the main window
        window = tk.Tk()
        # window = tk.Toplevel()  # Use Toplevel instead of Tk
        window.title("Vola Sample Label")

        window.geometry("1600x800")

        # Display key-value pairs from the info_text dictionary
        for key, value in info_text.items():
            info_label = tk.Label(window, text=f"{key}: {value}", anchor="w",
                                  padx=10)
            info_label.pack(side="top", fill="both")

        # Create a frame for buttons and entry field
        button_frame = tk.Frame(window)
        # Put buttons above plotl
        button_frame.pack(side="top", padx=10, pady=10)

        # Create a canvas for the Matplotlib fig
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side="top", fill="both", expand=True)

        # Create buttons and entry field inside the frame
        wanted_button = tk.Button(button_frame, text="Wanted",
                                  command=on_wanted_click)
        wanted_button.pack(side="left", padx=10)

        unwanted_button = tk.Button(button_frame, text="Unwanted",
                                    command=on_unwanted_click)
        unwanted_button.pack(side="left", padx=10)

        comment_label = tk.Label(button_frame, text="Comment:")
        comment_label.pack(side="left", padx=10)

        comment_entry = Entry(button_frame)
        comment_entry.pack(side="left", padx=10)

        next_button = tk.Button(button_frame, text="Next",
                                command=on_next_click)
        next_button.pack(side="left", padx=10)

        exit_button = tk.Button(button_frame, text="Exit",
                                command=on_exit_click)
        exit_button.pack(side="left", padx=10)

        # Bind the window close event to on_exit_click
        window.protocol("WM_DELETE_WINDOW", on_exit_click)

        # Run the main loop
        window.mainloop()

        return label_value, comment_value, exit_value

    def _get_vola_sample_batch(self):
        """
        Split all vola samples into several batches.
        Return the selecte batch of samples.
        :return vola_batch, list
        """
        NUM_BATCHES = 10
        all_vola_samples = [f for f in os.listdir(self.data_dir) if
                            f.endswith('.csv.gz')]
        avg_len = len(all_vola_samples) // NUM_BATCHES
        remainder = len(all_vola_samples) % NUM_BATCHES
        split_dict = {}
        start = 0
        for i in range(NUM_BATCHES):
            end = start + avg_len + (1 if i < remainder else 0)
            split_dict[i + 1] = all_vola_samples[start:end]
            start = end
        vola_batch = split_dict[self.sample_batch_number]

        if self.shuffle_vola_batch:
            random.shuffle(vola_batch)

        return vola_batch

    def run(self):
        """
        Run program.
        """
        vola_batch = self._get_vola_sample_batch()

        for vola in vola_batch:

            # Skip vola samples which are already labeled.
            if vola in list(self.result_df["Filename"]):
                continue

            try:
                vola_df = self._load_vola_df(vola)
                fig = self._plot_vola(vola_df)
                display_dict = self._get_vola_info(vola_df)
                # Add news info to display dict.
                if self.display_news:
                    news_dict = self._get_news_info(vola_df, 10)
                    display_dict = {**display_dict, **news_dict}
                label, comment, exit = self._create_widget(fig=fig,
                                                        info_text=display_dict)
                self._store_input(label, comment, exit)

                # Store results directly to csv.
                self._save_results_to_csv()
                # Load results from csv.
                result_file = os.path.join(self.target_dir,
                                           "label_result_file.csv")
                self.result_df = pd.read_csv(result_file)

            except Exception as e:
                print(f"Error processing {vola}: {e}")

        self._save_results_to_csv()

    def run_from_list(self, file_name_list):
        """
        :param file_name_list, list of files to be labeled.
        Run program.
        """

        for vola in file_name_list:

            try:
                vola_df = self._load_vola_df(vola)
                fig = self._plot_vola(vola_df)
                display_dict = self._get_vola_info(vola_df)
                # Add news info to display dict.
                if self.display_news:
                    news_dict = self._get_news_info(vola_df, 10)
                    display_dict = {**display_dict, **news_dict}
                label, comment, exit = self._create_widget(fig=fig, info_text=display_dict)
                self._store_input(label, comment, exit)

                # Store results directly to csv.
                self._save_results_to_csv()
                # Load results from csv.
                result_file = os.path.join(self.target_dir,
                                           "label_result_file.csv")
                self.result_df = pd.read_csv(result_file)

            except Exception as e:
                print(f"Error processing {vola}: {e}")

        self._save_results_to_csv()

    def __iter__(self):
        pass

    def __next__(self):
        pass



# Test Code
if __name__ == '__main__':
    sample_labeller_gui = SampleLabelerGUI(sample_batch_number=9)
    sample_labeller_gui.run()
