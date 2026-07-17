from src.whisper.sent_timeline import SentTimeline


def test_sent_timeline_identity_when_no_drop() -> None:
    tl = SentTimeline()
    
    # Simulate sending 10 chunks of 0.5s with no drops
    for i in range(10):
        tl.record_sent(sent_duration=i * 0.5, real_start=i * 0.5)
        
    # Check mapping
    assert tl.map_to_real(0.0) == 0.0
    assert tl.map_to_real(2.5) == 2.5
    assert tl.map_to_real(4.9) == 4.9


def test_sent_timeline_compensates_for_drops() -> None:
    tl = SentTimeline()
    
    # 0.0 - 1.0 (2 chunks) sent normally
    tl.record_sent(0.0, 0.0)
    tl.record_sent(0.5, 0.5)
    
    # 1.0 - 4.0 dropped (3 seconds dropped)
    # Next chunk sent has real_start=4.0, but sent_duration is still 1.0
    tl.record_sent(1.0, 4.0)
    tl.record_sent(1.5, 4.5)
    
    # Next chunk dropped: 5.0 - 6.0 (1 second dropped)
    # Next chunk sent has real_start=6.0, sent_duration is 2.0
    tl.record_sent(2.0, 6.0)
    
    # Test mapping
    assert tl.map_to_real(0.0) == 0.0  # Before drop
    assert tl.map_to_real(0.5) == 0.5  # Before drop
    assert tl.map_to_real(1.0) == 4.0  # After first drop (offset +3.0)
    assert tl.map_to_real(1.5) == 4.5  # After first drop (offset +3.0)
    assert tl.map_to_real(2.0) == 6.0  # After second drop (offset +4.0)
    assert tl.map_to_real(2.5) == 6.5  # After second drop (offset +4.0)

def test_sent_timeline_reset() -> None:
    tl = SentTimeline()
    tl.record_sent(0.0, 5.0) # offset 5.0
    assert tl.map_to_real(1.0) == 6.0
    
    tl.reset()
    assert tl.map_to_real(1.0) == 1.0 # identity after reset
    
    tl.record_sent(0.0, 0.0) # new epoch from 0
    assert tl.map_to_real(0.5) == 0.5

def test_sent_timeline_empty() -> None:
    tl = SentTimeline()
    assert tl.map_to_real(10.5) == 10.5
