"""Confirms a plate only when several frames agree on the same reading.

A single OCR read is noisy (motion blur, glare, '0' read as 'O'), so one bad frame
shouldn't create a log entry. A car stays in view for several frames, so we wait
until the same text has been read MIN_VOTES times within a short window.
"""
from collections import deque


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
        while self.reads and now - self.reads[0][0] > self.window:
            self.reads.popleft()

        if now - self.last_logged.get(text, float("-inf")) < self.cooldown:
            return None

        scores = [s for _, t, s in self.reads if t == text]
        if len(scores) < self.min_votes:
            return None

        self.last_logged[text] = now
        # Start fresh so leftover reads of this car can't confirm a misread variant of it.
        self.reads.clear()
        return text, sum(scores) / len(scores)
