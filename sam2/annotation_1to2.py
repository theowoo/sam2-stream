#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os

import h5py
import numpy as np
import rich_click as click


def convert(input, output):
    old_dset = h5py.File(input, "r")

    with h5py.File(output, "w") as hf:
        hf.attrs["videos"] = [old_dset.attrs["video"]]
        hf.attrs["videos_samples"] = np.zeros(len(old_dset["ann_frames"][:]), dtype=int)
        ann_frame_idx = old_dset["ann_frame_idx"][:]
        frame_samples = []
        for i in range(len(ann_frame_idx)):
            if i in ann_frame_idx:
                f = old_dset["frames"][ann_frame_idx == i][0]
            else:
                f = np.nan
            frame_samples.append(f)
        hf.attrs["frames_samples"] = frame_samples
        hf.attrs["label_names"] = old_dset.attrs["label_names"]

        hf.create_dataset("ann_frames", data=old_dset["ann_frames"][:], dtype="uint8")
        hf.create_dataset("ann_frame_idx", data=ann_frame_idx, dtype="int32")
        hf.create_dataset("labels", data=old_dset["labels"][:], dtype="int32")
        hf.create_dataset("positions", data=old_dset["positions"][:], dtype="float32")
        hf.create_dataset("is_target", data=old_dset["is_target"][:], dtype="int32")


@click.command(context_settings={"show_default": True})
@click.argument("input", required=True)
@click.option("--output", "-o", help="Output annotation file")
def convert_cmd(input, output):
    if os.path.isfile(output):
        click.confirm(f"File exists: {output}, overwrite?", abort=True)

    convert(input, output)


if __name__ == "__main__":
    convert_cmd()
