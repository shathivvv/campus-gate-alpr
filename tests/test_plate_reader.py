import numpy as np

from plate_reader import crop_plate, read_plate


class FakeOCR:
    """Stands in for PaddleOCR so tests run instantly without the model."""

    def __init__(self, texts, scores):
        self.result = {"rec_texts": texts, "rec_scores": scores}

    def predict(self, _image):
        return [self.result]


def test_reads_single_line_plate():
    assert read_plate(FakeOCR(["WA 1234 B"], [0.95]), None) == ("WA1234B", 0.95)


def test_joins_two_line_plate():
    assert read_plate(FakeOCR(["BJT", "8558"], [0.9, 0.8]), None) == ("BJT8558", 0.8)


def test_skips_noise_line():
    assert read_plate(FakeOCR(["TOYOTA", "VCD123"], [0.99, 0.7]), None) == ("VCD123", 0.7)


def test_rejects_non_plate_text():
    assert read_plate(FakeOCR(["HELLO WORLD"], [0.99]), None) is None
    assert read_plate(FakeOCR([], []), None) is None


def test_crop_is_clipped_to_frame_and_keeps_aspect_ratio():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_plate(frame, (600, 440, 700, 500))  # box runs off the bottom-right corner
    assert crop is not None
    assert crop.shape[1] == 256


def test_tiny_box_is_ignored():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert crop_plate(frame, (10, 10, 12, 12)) is None
