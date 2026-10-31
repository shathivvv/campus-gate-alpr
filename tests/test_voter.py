from voter import PlateVoter


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
