#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys

import cv2
import h5py
import numpy as np
import rich_click as click
from tqdm.rich import tqdm

from sam2.utils.io import load_sam2_model, refine_sam2_model
from sam2.utils.misc import overlay_prediction

ANNOTATION_SUFFIX = ".annotation"


def run_sam2(
    input: str,
    model: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    annotation: str = None,
    output_video: str = None,
    output: str = None,
    sampling_rate: float = 1,
    start: float = 0,
    end: float = None,
    rerun: bool = False,
    codec: str = "VP90",
):

    if output is not None and os.path.isfile(output) and not rerun:
        raise RuntimeError("Output already exists. Use flag --rerun to overwrite.")

    model_instance = load_sam2_model(
        model,
        model_config,
    )

    model_instance = refine_sam2_model(model_instance, annotation)

    # Set up video info
    cap = cv2.VideoCapture(input)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_video is not None:
        if codec == "H264":
            fourcc = 0x21  # https://stackoverflow.com/q/34024041
        elif cv2.__version__.split(".")[0] >= "3":
            fourcc = cv2.VideoWriter_fourcc(*codec)
        else:
            fourcc = cv2.cv.CV_FOURCC(*codec)
    else:
        fourcc = None

    if start is None:
        frame_start = 0
    else:
        frame_start = start * fps

    if end is None:
        frame_end = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    else:
        frame_end = end * fps

    total_samples = int(np.ceil((frame_end - frame_start) / sampling_rate))

    # Set up tmp output video file name

    if output_video is None and output is None:
        play_video = True
    else:
        play_video = False

    if output_video is not None:
        output_dirname = os.path.dirname(output_video)
        if output_dirname != "" and not os.path.isdir(output_dirname):
            os.mkdir(output_dirname)

        output_video_basename = os.path.basename(output_video)
        tmp_output_video_basename = "." + output_video_basename
        tmp_output_video = os.path.join(output_dirname, tmp_output_video_basename)

        wr = cv2.VideoWriter(tmp_output_video, fourcc, fps, (w, h))

    if output is not None:
        output_dirname = os.path.dirname(output)
        if output_dirname != "" and not os.path.isdir(output_dirname):
            os.mkdir(output_dirname)

        output_basename = os.path.basename(output)
        tmp_output_basename = "." + output_basename
        tmp_output = os.path.join(output_dirname, tmp_output_basename)

        rel_input_path = os.path.relpath(input, os.path.dirname(output))
        rel_model_path = os.path.relpath(model, os.path.dirname(output))
        rel_annotation_path = os.path.relpath(annotation, os.path.dirname(output))

        with h5py.File(tmp_output, "w") as hf:
            hf.create_dataset(
                "segmentation",
                (0, h, w),
                dtype="uint8",
                maxshape=(total_samples, h, w),
            )
            hf.attrs["input"] = rel_input_path
            hf.attrs["model"] = rel_model_path
            hf.attrs["model_config"] = model_config
            hf.attrs["annotation"] = rel_annotation_path
            hf.attrs["sampling_rate"] = sampling_rate
            hf.attrs["frame_start"] = frame_start
            hf.attrs["frame_end"] = frame_end
            hf.create_dataset("frame_numbers", data=total_samples, dtype="int32")

    PBAR = tqdm(total=total_samples, file=sys.stdout)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    for i in np.arange(total_samples):
        # Read frame and get mask
        _, im = cap.read()

        frame, mask = overlay_prediction(im, model_instance)

        frame_number = (
            frame_start + i * sampling_rate
        )  # Calculate the actual frame number
        cv2.putText(
            frame,
            str(frame_number),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Show on screen
        if play_video:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("frame", frame)

        if output is not None:
            with h5py.File(tmp_output, "r+") as hf:
                length = hf["segmentation"].shape[0]
                hf["segmentation"].resize(length + 1, axis=0)
                hf["segmentation"][-1:] = mask

        if output_video is not None:
            wr.write(frame)
        PBAR.update(1)

        # Skip frames between samples
        for _ in range(sampling_rate - 1):
            cap.grab()

        if play_video and cv2.waitKey(1) and 0xFF == ord("q"):
            break

    cap.release()

    if output_video is not None:
        wr.release()
        os.rename(tmp_output_video, output_video)

    if output is not None:
        os.rename(tmp_output, output)


@click.command(context_settings={"show_default": True})
@click.argument("input", required=True)
@click.option("--model", "-m", required=True, help="Path to SAM2 model checkpoint.")
@click.option(
    "--model-config",
    "-mc",
    help="Name of YAML model config, aka model base.",
    default="configs/sam2.1/sam2.1_hiera_s.yaml",
)
@click.option(
    "--annotation",
    type=str,
    help="Path to annotation file containing point prompts for refining a sam2 cuttlefish model.",
)
@click.option("--output-video", "-ov", help="Output video file")
@click.option("--output", "-o", help="Output segmentation file")
@click.option(
    "--sampling-rate", default=1, type=int, help="Sampling rate (every n frames)"
)
@click.option("--start", type=float, help="Start video at (seconds)")
@click.option("--end", type=float, help="End video at (seconds)")
@click.option("--rerun", is_flag=True, help="Overwrite existing output")
@click.option("--codec", default="VP90", type=str, help="Alternative: H264")
def run_sam2_cmd(
    input: str,
    model: str,
    model_config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
    annotation: str = None,
    output_video: str = None,
    output: str = None,
    sampling_rate: float = 1,
    start: float = 0,
    end: float = None,
    rerun: bool = False,
    codec: str = "VP90",
):
    "Plot mask overlay."
    if annotation is None:
        annotation = os.path.splitext(input)[0] + ANNOTATION_SUFFIX

    run_sam2(
        input=input,
        model=model,
        model_config=model_config,
        annotation=annotation,
        output_video=output_video,
        output=output,
        sampling_rate=sampling_rate,
        start=start,
        end=end,
        rerun=rerun,
        codec=codec,
    )


if __name__ == "__main__":
    run_sam2_cmd()
