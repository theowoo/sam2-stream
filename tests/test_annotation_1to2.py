import h5py
import numpy as np
import pytest
from sam2.annotation_1to2 import convert


def test_convert(tmpdir):

    old_file = tmpdir.join("old.annotation")
    new_file = tmpdir.join("new.annotation")

    with h5py.File(old_file, "w") as hf:
        hf.attrs["video"] = "video.mp4"
        hf.attrs["label_names"] = ["label0", "label1"]
        hf.create_dataset("frames", data=[0, 0, 5], dtype="int32")
        hf.create_dataset(
            "ann_frames",
            data=np.stack(
                [np.ones((3, 2, 4)) * 0, np.ones((3, 2, 4)) * 1, np.ones((3, 2, 4)) * 2]
            ),
            dtype="uint8",
        )
        hf.create_dataset("ann_frame_idx", data=[0, 0, 1], dtype="int32")
        hf.create_dataset("labels", data=[0, 0], dtype="int32")
        hf.create_dataset("positions", data=[[1, 1], [0, 0]], dtype="float32")
        hf.create_dataset("is_target", data=[1, 1], dtype="int32")

    convert(old_file, new_file)

    old_dset = h5py.File(old_file, "r")
    new_dset = h5py.File(new_file, "r")

    assert np.all(new_dset["ann_frames"][:] == old_dset["ann_frames"][:])
    assert np.all(new_dset["ann_frame_idx"][:] == old_dset["ann_frame_idx"][:])
    assert np.all(new_dset["labels"][:] == old_dset["labels"][:])
    assert np.all(new_dset["positions"][:] == old_dset["positions"][:])
    assert np.all(new_dset["is_target"][:] == old_dset["is_target"][:])
    assert np.all(new_dset.attrs["label_names"] == old_dset.attrs["label_names"])
    assert np.all(new_dset.attrs["videos"] == ["video.mp4"])
    assert np.all(new_dset.attrs["videos_samples"] == [0, 0, 0])
    assert np.all(new_dset.attrs["frames_samples"][:2] == [0, 5])
    assert np.isnan(new_dset.attrs["frames_samples"][2])
