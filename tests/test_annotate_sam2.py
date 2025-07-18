import numpy as np
from sam2.annotate_sam2 import VideoBrowser


def test_combine_multiple_sources():

    browser = VideoBrowser()

    browser.videos_samples = np.array([0, 0, 0])
    browser.videos = ["video_current.mp4"]

    browser.existing_videos_samples = np.array([0, 1])
    browser.existing_videos = ["video0.mp4", "video1.mp4"]

    browser.combine_multiple_sources()

    assert np.all(browser.videos == ["video0.mp4", "video1.mp4", "video_current.mp4"])
    assert np.all(browser.videos_samples == np.array([2, 2, 2]))
    assert np.all(browser.existing_videos_samples == np.array([0, 1]))

    browser = VideoBrowser()

    browser.videos_samples = np.array([0, 0, 0])
    browser.videos = ["video_current.mp4"]

    browser.existing_videos_samples = np.array([0, 1])
    browser.existing_videos = ["video_current.mp4", "video0.mp4"]

    browser.combine_multiple_sources()

    assert np.all(browser.videos == ["video0.mp4", "video_current.mp4"])
    assert np.all(browser.videos_samples == np.array([1, 1, 1]))
    assert np.all(browser.existing_videos_samples == np.array([1, 0]))


def test_append_existing_ann_frames():

    ## Case 1: all new

    browser = VideoBrowser()

    # New samples
    browser.videos_samples = np.array([0, 0, 0])
    browser.frames_samples = np.array([0, 5, 10])
    browser.ann_frames = np.stack(
        [np.ones((3, 2, 4)) * 0, np.ones((3, 2, 4)) * 1, np.ones((3, 2, 4)) * 2]
    )

    # Loaded from file
    browser.existing_videos_samples = np.array([0, 0, 1])
    browser.existing_frames_samples = np.array([0, 4, 8])
    browser.existing_ann_frames = np.stack(
        [np.ones((3, 2, 4)) * 3, np.ones((3, 2, 4)) * 4, np.ones((3, 2, 4)) * 5]
    )
    browser.ann_frame_idx = np.array([1, 1, 2])

    browser.append_existing_ann_frames()

    assert np.all(
        browser.ann_frames
        == np.stack(
            [
                np.ones((3, 2, 4)) * 0,
                np.ones((3, 2, 4)) * 1,
                np.ones((3, 2, 4)) * 2,
                np.ones((3, 2, 4)) * 4,
                np.ones((3, 2, 4)) * 5,
            ]
        )
    )
    assert np.all(browser.ann_frame_idx == np.array([3, 3, 4]))

    ## Case 2: mixed old and new

    browser = VideoBrowser()

    # New samples
    browser.videos_samples = np.array([0, 0, 0])
    browser.frames_samples = np.array([0, 5, 10])
    browser.ann_frames = np.stack(
        [np.ones((3, 2, 4)) * 0, np.ones((3, 2, 4)) * 1, np.ones((3, 2, 4)) * 2]
    )

    # Loaded from file
    browser.existing_videos_samples = np.array([0, 0, 1])
    browser.existing_frames_samples = np.array([0, 4, 8])
    browser.existing_ann_frames = np.stack(
        [np.ones((3, 2, 4)) * 3, np.ones((3, 2, 4)) * 4, np.ones((3, 2, 4)) * 5]
    )
    browser.ann_frame_idx = np.array([0, 0, 2])

    browser.append_existing_ann_frames()

    assert np.all(
        browser.ann_frames
        == np.stack(
            [
                np.ones((3, 2, 4)) * 0,
                np.ones((3, 2, 4)) * 1,
                np.ones((3, 2, 4)) * 2,
                np.ones((3, 2, 4)) * 5,
            ]
        )
    )
    assert np.all(browser.ann_frame_idx == np.array([0, 0, 3]))
