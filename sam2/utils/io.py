"""Input-output utilities."""


def load_sam2_model(sam2_checkpoint, model_cfg):
    """
    Load a SAM2 model from a checkpoint.

    Parameters
    ----------
    sam2_checkpoint : str
        Path to saved model checkpoint
    model_cfg : str
        Config for the model (installed with sam-2)

    Returns
    -------
    predictor :
        Default predictor
    """

    import torch

    # use bfloat16 for the entire script
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        # (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from sam2.build_sam import build_sam2_camera_predictor

    predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

    return predictor


def refine_sam2_model(predictor, annotation_path):
    """
    Refine (prompt) a SAM2 model using an annotation file.

    Parameters
    ----------
    predictor :
        Default predictor from load_sam2_model function
    annotation_path : str
        Path to annotation file

    Returns
    -------
    refined_predictor :
        Refined predictor after applying annotations
    """

    import h5py
    import numpy as np

    with h5py.File(annotation_path, "r") as hf:
        ann_frames = hf["ann_frames"][:]
        ann_frame_idx = hf["ann_frame_idx"][:]
        labels = hf["labels"][:]
        positions = hf["positions"][:]
        is_target = hf["is_target"][:]

    for i, im in enumerate(ann_frames):
        if i == 0:
            predictor.load_first_frame(im)
        else:
            predictor.add_conditioning_frame(im)

        for ann_obj_id in np.unique(labels[ann_frame_idx == i]):
            curr_filter = (ann_frame_idx == i) & (labels == ann_obj_id)
            predictor.add_new_prompt(
                frame_idx=i,
                obj_id=ann_obj_id,
                points=positions[curr_filter],
                labels=is_target[curr_filter],
            )
    return predictor
