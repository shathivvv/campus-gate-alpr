
"""Confirms a plate only when several frames agree on the same reading.

A single OCR read is noisy (motion blur, glare, '0' read as 'O'), so one bad frame
shouldn't create a log entry. A car stays in view for several frames, so we wait
until the same text has been read MIN_VOTES times within a short window.
"""
from collections import deque


def is_near_duplicate(a, b):
    """True if one plate is contained in the other or they differ by one character."""
    if a in b or b in a:
        return True
    if abs(len(a) - len(b)) > 1:
        return False

    # Edit distance <= 1 (one substitution, insertion or deletion)
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1

    if len(a) == len(b):
        return a[i + 1:] == b[i + 1:]

    shorter, longer = sorted((a, b), key=len)
    return shorter[i:] == longer[i + 1:]


class PlateVoter:
    def __init__(self, min_votes=3, window=4.0, cooldown=15.0):
        self.min_votes = min_votes
        self.window = window
        self.cooldown = cooldown
        self.reads = deque()  # (time, text, score)
        self.last_logged = {}  # text -> time it was last confirmed

    def add(self, text, score, now):
        """Records one read. Returns (text, mean_score) when the plate is confirmed, else None."""

        self.reads.append((now, text, score))

        # Remove reads that are outside the voting time window.
        while self.reads and now - self.reads[0][0] > self.window:
            self.reads.popleft()

        # Remove expired entries to prevent last_logged from growing forever.
        self.last_logged = {
            plate: logged_at
            for plate, logged_at in self.last_logged.items()
            if now - logged_at < self.cooldown
        }

        # A partial misread of a car we just logged (WA1234B -> A1234B)
        # is treated as the same car.
        if any(
            now - logged_at < self.cooldown and is_near_duplicate(text, plate)
            for plate, logged_at in self.last_logged.items()
        ):
            return None

        # Collect confidence scores for matching plate reads.
        scores = [s for _, t, s in self.reads if t == text]

        if len(scores) < self.min_votes:
            return None

        # Remember when this plate was confirmed.
        self.last_logged[text] = now

        # Clear old reads so they cannot confirm a misread variant.
        self.reads.clear()

        return text, sum(scores) / len(scores)
