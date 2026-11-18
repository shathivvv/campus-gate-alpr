from voter import PlateVoter, is_near_duplicate


def test_confirms_after_enough_matching_reads():
    voter = PlateVoter(min_votes=3, window=4.0, cooldown=15.0)
    assert voter.add("WA1234B", 0.9, now=0.0) is None
    assert voter.add("WA1234B", 0.8, now=0.5) is None
    text, score = voter.add("WA1234B", 0.7, now=1.0)
    assert text == "WA1234B"
    assert abs(score - 0.8) < 1e-9


def test_one_off_misread_is_never_logged():
    voter = PlateVoter(min_votes=3)
    voter.add("WA1234B", 0.9, now=0.0)
    assert voter.add("WA1Z34B", 0.9, now=0.2) is None  # misread
    voter.add("WA1234B", 0.9, now=0.4)
    assert voter.add("WA1234B", 0.9, now=0.6)[0] == "WA1234B"


def test_old_reads_expire():
    voter = PlateVoter(min_votes=3, window=4.0)
    voter.add("WA1234B", 0.9, now=0.0)
    voter.add("WA1234B", 0.9, now=1.0)
    assert voter.add("WA1234B", 0.9, now=10.0) is None  # first two reads are too old


def test_same_plate_not_logged_again_during_cooldown():
    voter = PlateVoter(min_votes=1, cooldown=15.0)
    assert voter.add("WA1234B", 0.9, now=0.0) is not None
    assert voter.add("WA1234B", 0.9, now=5.0) is None
    assert voter.add("WA1234B", 0.9, now=20.0) is not None


def test_partial_misread_of_logged_plate_is_ignored():
    voter = PlateVoter(min_votes=1, cooldown=15.0)
    assert voter.add("WA1234B", 0.9, now=0.0) is not None
    assert voter.add("A1234B", 0.9, now=1.0) is None   # missing first letter
    assert voter.add("NA1234B", 0.9, now=2.0) is None  # one letter wrong
    assert voter.add("VCD123", 0.9, now=3.0) is not None  # a different car


def test_near_duplicate():
    assert is_near_duplicate("WA1234B", "A1234B")
    assert is_near_duplicate("BKL5678C", "BKI5678C")
    assert is_near_duplicate("JQR905", "JQR9105")
    assert not is_near_duplicate("BKL5678C", "PLM4321")
    assert not is_near_duplicate("ABC1234", "ABC9934")

def test_expired_plates_are_removed_from_last_logged():
    voter = PlateVoter(min_votes=1, cooldown=15.0)

    # Confirm the first car
    voter.add("WA1234B", 0.9, now=0.0)

    # A different car arrives after the cooldown
    voter.add("VCD123", 0.9, now=20.0)

    # The old plate should no longer be stored
    assert "WA1234B" not in voter.last_logged