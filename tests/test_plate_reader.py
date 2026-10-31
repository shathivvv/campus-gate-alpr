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


class HalvesOCR:
    """Fake OCR that returns a different line for the top and bottom half of a crop."""

    def __init__(self, top, bottom):
        self.top, self.bottom = top, bottom

    def predict(self, image):
        if image[0, 0, 0] == 1 and image[-1, 0, 0] == 2:
            return [{"rec_texts": ["CDA"], "rec_scores": [0.5]}]  # whole plate: one line only
        text = self.top if image[0, 0, 0] == 1 else self.bottom
        return [{"rec_texts": [text], "rec_scores": [0.9]}]


def test_two_line_plate_is_read_in_halves():
    crop = np.zeros((200, 240, 3), dtype=np.uint8)  # near-square, like a two-line plate
    crop[:10] = 1    # mark the top and bottom edges so the fake OCR can tell which part it got
    crop[-10:] = 2
    assert read_plate(HalvesOCR("QM", "8812"), crop) == ("QM8812", 0.9)


def test_wide_crop_is_not_split():
    crop = np.zeros((80, 256, 3), dtype=np.uint8)
    assert read_plate(FakeOCR(["WA1234B"], [0.9]), crop) == ("WA1234B", 0.9)
