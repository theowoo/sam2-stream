#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

import tkinter as tk
from tkinter import ttk

import cv2
import h5py
import matplotlib.colors
import matplotlib.pyplot as plt

import numpy as np
import rich_click as click

from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tqdm.rich import tqdm

from sam2.utils.io import load_sam2_model, refine_sam2_model
from sam2.utils.misc import overlay_prediction, remove_lines

ANNOTATION_SUFFIX = ".annotation"


class VideoBrowser(object):
    def __init__(self):
        self.positions = np.array([[0, 0]])
        self.is_target = np.array([1])
        self.frames = np.ones(len(self.positions)) * 0
        self.ann_frames = np.ones((len(self.positions), 100, 100, 3)) * 127
        self.ann_frame_idx = np.ones(len(self.positions)) * 0
        self.labels = np.ones(len(self.positions)) * 0
        self.target_var = None
        self.is_target_mode = True
        self.label_listbox = None
        self.label_var = None
        self.label_curr_var = None
        self.label_curr_idx = 0
        self.output = None

    def init_annotation(self, input, tmp_output):
        rel_video_path = os.path.relpath(input, os.path.dirname(tmp_output))
        with h5py.File(tmp_output, "w") as hf:
            hf.attrs["video"] = rel_video_path
            hf.create_dataset("frames", data=self.frames, dtype="int32")
            hf.create_dataset("ann_frames", data=self.ann_frames, dtype="uint8")
            hf.create_dataset("ann_frame_idx", data=self.ann_frame_idx, dtype="int32")
            hf.create_dataset("labels", data=self.labels, dtype="int32")
            hf.create_dataset("positions", data=self.positions, dtype="float32")
            hf.create_dataset("is_target", data=self.is_target, dtype="int32")

        self.tmp_output = tmp_output

        self.positions = np.empty((0, 2))
        self.is_target = np.empty((0, 1))
        self.ann_frames = np.empty(0)
        self.frames = np.empty(0)
        self.ann_frame_idx = np.empty(0)
        self.labels = np.empty(0)

    def load_annotation(self):
        with h5py.File(self.output, "r") as hf:
            self.frames = hf["frames"][:]
            self.ann_frames = hf["ann_frames"][:]
            self.ann_frame_idx = hf["ann_frame_idx"][:]
            self.labels = hf["labels"][:]
            self.positions = hf["positions"][:]
            self.is_target = hf["is_target"][:]

        click.secho("Exisitng annotations loaded.", fg="green")

    def init_model(self, model, model_config):
        model_instance = load_sam2_model(model, model_config)
        model_instance = refine_sam2_model(model_instance, self.tmp_output)
        click.secho(f"{model} loaded", fg="green")
        self.model = model_instance

    def add_points(self, position, is_target, frame, frame_idx, label):
        self.positions = np.append(self.positions, position, axis=0)
        self.ann_frame_idx = np.append(self.ann_frame_idx, [frame_idx])
        self.is_target = np.append(self.is_target, np.array([is_target], dtype=int))
        self.frames = np.append(self.frames, [frame])
        self.labels = np.append(self.labels, [label])

    def write_annotation(self):
        with h5py.File(self.tmp_output, "r+") as hf:
            del hf["frames"]
            del hf["ann_frames"]
            del hf["ann_frame_idx"]
            del hf["labels"]
            del hf["positions"]
            del hf["is_target"]
            hf.create_dataset("frames", data=self.frames, dtype="int32")
            hf.create_dataset("ann_frames", data=self.ann_frames, dtype="uint8")
            hf.create_dataset("ann_frame_idx", data=self.ann_frame_idx, dtype="int32")
            hf.create_dataset("labels", data=self.labels, dtype="int32")
            hf.create_dataset("positions", data=self.positions, dtype="float32")
            hf.create_dataset("is_target", data=self.is_target, dtype="int32")
        click.secho("Temp annotation file written.", fg="blue")

    def init_canvas(self):
        n_cols = min(self.n_cols, len(self.ann_frames))
        n_rows = int(np.ceil(len(self.ann_frames) / n_cols))

        self.fig, self.ax = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
        self.ax = self.ax.reshape(-1)

        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)  # A tk.DrawingArea.
        remove_lines()

        self.canvas.draw()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def update_canvas(self):
        if len(self.ann_frames) > len(self.ax):
            for axis in self.ax:
                axis.remove()

            n_cols = min(self.n_cols, len(self.ann_frames))
            n_rows = int(np.ceil(len(self.ann_frames) / n_cols))

            self.ax = []

            for i in range(len(self.ann_frames)):
                if i == 0:
                    axis = self.fig.add_subplot(n_rows, n_cols, i + 1)
                else:
                    axis = self.fig.add_subplot(
                        n_rows, n_cols, i + 1, sharex=self.ax[0], sharey=self.ax[0]
                    )

                self.ax.append(axis)

            remove_lines()

        self.fig.canvas.draw()

    def on_click(self, event):
        if (
            (event.button is MouseButton.LEFT)
            and event.inaxes
            and (self.toolbar.mode != "zoom rect")
            and (self.toolbar.mode != "pan/zoom")
        ):
            click.secho("Left click", fg="blue")

            ax_idx = np.where(
                [event.inaxes == self.ax[i] for i in range(len(self.ax))]
            )[0][0]

            if ax_idx < len(self.ann_frames):
                new_position = np.array([[event.xdata, event.ydata]])
                frame = self.frames_samples[ax_idx]
                self.add_points(
                    new_position,
                    self.is_target_mode,
                    frame,
                    ax_idx,
                    self.label_curr_idx,
                )
                self.write_annotation()
                self.update_model()
                self.update_predictions()
                self.update_plots()
                self.canvas.draw()
                click.secho(
                    f"position: {new_position} "
                    f"is_target: {self.is_target_mode} "
                    f"frame: {frame} "
                    f"ax_idx: {ax_idx} "
                    f"label_curr_idx: {self.label_curr_idx}",
                    fg="blue",
                )

            else:
                click.secho("Empty subplot. Ignored.", fg="yellow")

    def load_frames(self):
        images = []

        for i, f in enumerate(tqdm(self.frames_samples, desc="Reading frames:")):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            _, im = self.cap.read()
            images.append(im)

        self.ann_frames = images

    def init_plot_handles(self):
        im_plots = []
        pos_plots = []
        neg_plots = []

        pbar = tqdm(total=len(self.frames_samples), desc="Reading frames:")
        for i, (f, im) in enumerate(zip(self.frames_samples, self.ann_frames)):
            # Plot with segmentation overlay
            im_viz = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im_handle = self.ax[i].imshow(im_viz)
            im_plots.append(im_handle)

            # Plot annotations
            pos = self.ax[i].scatter([], [], s=20, fc="k")
            pos_plots.append(pos)
            neg = self.ax[i].scatter([], [], s=20, ec="k", fc="None")
            neg_plots.append(neg)

            self.ax[i].set_title(f)

            pbar.update(1)
            pbar.refresh()

        self.fig.canvas.draw()

        self.im_plots = im_plots
        self.pos_plots = pos_plots
        self.neg_plots = neg_plots

    def update_model(self):
        self.model = refine_sam2_model(self.model, self.tmp_output)
        click.secho("Re-prompted SAM2.", fg="blue")

    def update_predictions(self):
        overlay_list = []
        for i, im in enumerate(tqdm(self.ann_frames, desc="Predicting masks:")):
            overlay, _ = overlay_prediction(im, self.model)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay_list.append(overlay)
        self.predictions = overlay_list

    def update_plots(self):
        pbar = tqdm(total=len(self.frames_samples), desc="Updating plots:")
        for f, imax, im, pos, neg in zip(
            self.frames_samples,
            self.im_plots,
            self.predictions,
            self.pos_plots,
            self.neg_plots,
        ):
            # Update overlay prediction
            imax.set_data(im)

            # Update positive points
            labels = self.labels[(self.frames == f) & (self.is_target == 1)]
            hue = (labels + 1) / (len(np.unique(self.labels)) + 1)
            color_hsv = np.ones((len(hue), 3))
            color_hsv[..., 0] = hue
            xx, yy = self.positions[(self.frames == f) & (self.is_target == 1)].T
            if len(xx) > 0:
                pos.set_offsets(np.vstack([xx, yy]).T)
                pos.set_facecolors(matplotlib.colors.hsv_to_rgb(color_hsv))

            # Update negative points
            labels = self.labels[(self.frames == f) & (self.is_target == 0)]
            hue = (labels + 1) / (len(np.unique(self.labels)) + 1)
            color_hsv = np.ones((len(hue), 3))
            color_hsv[..., 0] = hue
            xx, yy = self.positions[(self.frames == f) & (self.is_target == 0)].T
            if len(xx) > 0:
                neg.set_offsets(np.vstack([xx, yy]).T)
                neg.set_edgecolors(matplotlib.colors.hsv_to_rgb(color_hsv))

            pbar.update(1)
            pbar.refresh()

        # Update count
        self.count_text.configure(text=f"Prompt count: {len(self.positions)}")

    def target_func(self) -> None:
        self.is_target_mode = self.target_var.get()
        click.secho(f"is_target: {self.is_target_mode}", fg="blue")

    def label_func(self, event) -> None:
        selected_indices = self.label_listbox.curselection()
        if len(selected_indices) > 0:
            i = selected_indices[0]
            labels = self.label_var.get()
            click.secho(f"label selected: {i}", fg="blue")
            self.label_curr_var.set(labels[i])
            self.label_curr_idx = i
        else:
            click.secho("No label selected.", fg="yellow")

    def update_label(self, event):
        selected_indices = self.label_listbox.curselection()
        original_variables = self.label_var.get()
        new_variables = list(original_variables)
        i = selected_indices[0]
        new_label = self.label_curr_var.get()
        click.secho(f"label renamed: {i} - {new_label}", fg="blue")
        new_variables[i] = new_label
        self.label_var.set(new_variables)

    def add_frame(self, event):
        frame_to_add = self.frame_entry_var.get()

        if frame_to_add in self.frames_samples:
            click.secho("Frame already loaded. Add another.", fg="red")
            return

        n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        assert frame_to_add < n_frames, f"Frame number must be smaller than {n_frames}."

        self.frames_samples = np.append(self.frames_samples, [frame_to_add])

        # load new frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_add)
        _, im = self.cap.read()
        self.ann_frames = np.append(self.ann_frames, [im], axis=0)

        self.write_annotation()
        self.update_canvas()
        self.init_plot_handles()
        self.update_model()
        self.update_predictions()
        self.update_plots()
        self.canvas.draw()

    def close_func(self):
        os.rename(self.tmp_output, self.output)
        click.secho(f"Annotation renamed: {self.output}", fg="green")
        self.root.quit()


def annotate_sam2(
    input: str,
    model: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    n_samples: int = 3,
    n_cols: int = 4,
    output: str = None,
):
    # Set up paths
    output_dirname = os.path.dirname(output)
    output_basename = os.path.basename(output)
    tmp_output = os.path.join(output_dirname, "." + output_basename)

    # Set up object
    browser = VideoBrowser()
    browser.n_cols = n_cols
    browser.output = output

    # Set up video
    browser.cap = cv2.VideoCapture(input)
    n_frames = browser.cap.get(cv2.CAP_PROP_FRAME_COUNT)
    browser.frames_samples = np.linspace(
        0, n_frames, n_samples, endpoint=False, dtype=int
    )

    # Initialise model
    browser.init_annotation(input, tmp_output)
    browser.init_model(model, model_config)

    # Set up TK
    browser.root = tk.Tk()
    browser.root.geometry("1200x600")
    browser.root.title("Menu")

    # Load annotations if exists
    if os.path.isfile(output):
        browser.load_annotation()
        if not np.all(np.isin(browser.frames, browser.frames_samples)):
            for f in np.unique(
                browser.frames[~np.isin(browser.frames, browser.frames_samples)]
            ):
                browser.frames_samples = np.append(browser.frames_samples, [f])
                new_idx = len(browser.frames_samples) - 1
                browser.ann_frame_idx[browser.frames == f] = new_idx
                click.secho(f"Append frame {f} as index {new_idx}", fg="yellow")

    # Set up plot
    browser.load_frames()
    browser.predictions = browser.ann_frames
    browser.init_canvas()
    browser.init_plot_handles()

    # Set up count
    browser.count_text = tk.Label(
        browser.root, text=f"Prompt count: {len(browser.positions)}"
    )

    # Update predictions if loaded from annotation
    if os.path.isfile(output):
        browser.write_annotation()
        browser.update_model()
        browser.update_predictions()
        browser.update_plots()

    # Set up menu
    browser.target_var = tk.StringVar(value=True)
    radio1 = tk.Radiobutton(
        browser.root,
        text="Positive",
        value=True,
        variable=browser.target_var,
        command=browser.target_func,
    )
    radio2 = tk.Radiobutton(
        browser.root,
        text="Negative",
        value=False,
        variable=browser.target_var,
        command=browser.target_func,
    )

    # Set up label list
    browser.label_var = tk.Variable(value=["label1", "label2"])
    browser.label_listbox = tk.Listbox(
        browser.root,
        listvariable=browser.label_var,
        height=3,
    )
    browser.label_listbox.select_set(0)
    browser.label_listbox.bind("<<ListboxSelect>>", browser.label_func)

    # Set up label entry
    browser.label_curr_var = tk.StringVar()
    label_entry = tk.Entry(browser.root, textvariable=browser.label_curr_var, width=20)
    label_entry.bind("<Return>", browser.update_label)

    # Set up add frame
    browser.frame_entry_var = tk.IntVar()
    frame_entry = tk.Entry(browser.root, textvariable=browser.frame_entry_var, width=20)
    frame_entry.bind("<Return>", browser.add_frame)

    # Saet up close button
    button = ttk.Button(browser.root, text="Save & Exit", command=browser.close_func)

    # Default matplotlib interactive toolbar
    browser.toolbar = NavigationToolbar2Tk(
        browser.canvas, browser.root, pack_toolbar=False
    )
    browser.toolbar.update()

    # Packing order is critical
    browser.count_text.pack(side=tk.LEFT, padx=5, pady=5)
    radio1.pack(side=tk.LEFT, padx=5, pady=5)
    radio2.pack(side=tk.LEFT, padx=5, pady=5)
    browser.label_listbox.pack(side=tk.LEFT)
    label_entry.pack(side=tk.LEFT)
    ttk.Label(browser.root, text="Add frame:").pack(side=tk.LEFT)
    frame_entry.pack(side=tk.LEFT)
    button.pack(side=tk.LEFT, padx=5, pady=5)
    browser.toolbar.pack(side=tk.BOTTOM, fill=tk.X)
    browser.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    tk.mainloop()


@click.command()
@click.argument("input", required=True)
@click.option("--model", "-m", required=True, help="Path to SAM2 model checkpoint.")
@click.option(
    "--model-config",
    "-mc",
    required=True,
    help="Name of YAML model config, aka model base.",
    default="configs/sam2.1/sam2.1_hiera_s.yaml",
)
@click.option(
    "--n-samples",
    "-n",
    required=False,
    type=int,
    default=3,
    help="Number of sample frames.",
)
@click.option(
    "--n-cols",
    "-c",
    required=False,
    type=int,
    default=4,
    help="Number of columns.",
)
@click.option("--output", "-o", required=False, help="Output annotation.")
def annotate_sam2_cmd(
    input: str,
    model: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    n_samples: int = 3,
    n_cols: int = 4,
    output: str = None,
):
    if output is None:
        output = os.path.splitext(input)[0] + ANNOTATION_SUFFIX

    annotate_sam2(
        input=input,
        model=model,
        model_config=model_config,
        n_samples=n_samples,
        n_cols=n_cols,
        output=output,
    )


if __name__ == "__main__":
    annotate_sam2_cmd()
