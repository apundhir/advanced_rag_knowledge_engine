from __future__ import annotations

from app.ingestion.sentence_window import split_into_sentence_windows


def test_sentence_window_edges():
    text = "A. B. C. D."
    windows = split_into_sentence_windows(text, window_size=1)
    assert len(windows) == 4
    # First sentence window should include only first two sentences
    w0, m0 = windows[0]
    assert m0["s_idx"] == "0" and m0["e_idx"] == "1"
    assert "A." in w0 and "B." in w0
    # Middle window includes three sentences
    w1, m1 = windows[1]
    assert m1["s_idx"] == "0" and m1["e_idx"] == "2"
    assert "A." in w1 and "B." in w1 and "C." in w1
