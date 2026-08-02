from pipeline.batch import BatchInput, format_elapsed, run_batch


def test_format_elapsed_uses_korean_clock_text():
    assert format_elapsed(0) == "0분 00초"
    assert format_elapsed(65.9) == "1분 05초"
    assert format_elapsed(3661) == "1시간 1분 01초"


def test_run_batch_preserves_order_and_maps_file_progress_to_overall_progress():
    inputs = [
        BatchInput(item_id="a", filename="a.png", data=b"a"),
        BatchInput(item_id="b", filename="b.png", data=b"b"),
    ]
    events = []

    def analyze(item, report):
        report(0.25, "prepare")
        report(1.0, "done")
        return item.filename.upper()

    outcomes = run_batch(inputs, analyze, on_progress=events.append)

    assert [o.filename for o in outcomes] == ["a.png", "b.png"]
    assert [o.result for o in outcomes] == ["A.PNG", "B.PNG"]
    assert all(o.error is None for o in outcomes)
    assert events[0].overall_fraction == 0.0
    assert any(abs(e.overall_fraction - 0.125) < 1e-9 for e in events)
    assert any(abs(e.overall_fraction - 0.5) < 1e-9 for e in events)
    assert events[-1].overall_fraction == 1.0
    assert [e.overall_fraction for e in events] == sorted(e.overall_fraction for e in events)


def test_run_batch_keeps_other_results_when_one_file_fails():
    inputs = [
        BatchInput(item_id="ok", filename="ok.png", data=b"ok"),
        BatchInput(item_id="bad", filename="bad.png", data=b"bad"),
        BatchInput(item_id="ok2", filename="ok2.png", data=b"ok2"),
    ]

    def analyze(item, report):
        report(0.5, "running")
        if item.item_id == "bad":
            raise ValueError("broken image")
        report(1.0, "done")
        return item.item_id

    outcomes = run_batch(inputs, analyze)

    assert outcomes[0].result == "ok"
    assert outcomes[1].result is None
    assert "broken image" in outcomes[1].error
    assert outcomes[2].result == "ok2"
