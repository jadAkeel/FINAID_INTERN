import json

from forecast_select.pretrained import local_pretrained_preflight


def test_pretrained_preflight_is_local_and_explicit(tmp_path):
    path = local_pretrained_preflight(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["mode"] == "local_only_no_external_data"
    assert all(item["status"] == "blocked" for item in report["candidates"].values())
    assert all(item["smoke_test"] == "not_run" for item in report["candidates"].values())
