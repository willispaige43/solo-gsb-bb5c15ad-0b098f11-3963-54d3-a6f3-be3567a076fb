"""GET /jobs/{id}/series/{series_uid}/ct-voltage：CT 管电压逐帧分布与冲突明细。"""
from __future__ import annotations

import io
import os
import zipfile

import pytest
from fastapi.testclient import TestClient
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from app.crypto import derive_uid
from tests.sample_data import _write_bytes

SECRET = "ct-voltage-secret-1234"

STUDY = "3.1"
SERIES = "2.1"
FRAME = "4.1"


# --------------------------------------------------------------------------- #
# 合成 DICOM
# --------------------------------------------------------------------------- #
def _ct_exposure(kvp):
    item = Dataset()
    item.KVP = kvp
    return item


def _image_bytes(
    sop_uid: str,
    *,
    series_uid: str = SERIES,
    study_uid: str = STUDY,
    modality: str = "CT",
    top_kvp=None,
    number_of_frames: int | None = None,
    per_frame_kvps=None,
    shared_kvp=None,
    shared_present: bool = True,
    per_frame_length: int | None = None,
):
    """构造 CT 文件。

    top_kvp: 顶层 KVP（None 表示不写）。
    per_frame_kvps: 每帧 CTExposureSequence 的 KVP 列表（元素 None 表示该帧
        功能组不含 CTExposureSequence / KVP）。
    shared_kvp: SharedFunctionalGroupsSequence 内共享 KVP（None 表示不写该序列）。
    per_frame_length: 强制把 PerFrameFunctionalGroupsSequence 写成指定长度
        （与 number_of_frames 不一致，用于构造 frame_count_mismatch）。
    bad_per_frame_kvp: 写入无法解析的 KVP 文本。
    """
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = FRAME
    ds.PatientID = "P-1"
    ds.PatientName = "Zhang^San"
    ds.Modality = modality
    if number_of_frames is not None:
        ds.NumberOfFrames = number_of_frames
    if top_kvp is not None:
        ds.KVP = top_kvp
    if shared_kvp is not None:
        shared_item = Dataset()
        shared_item.CTExposureSequence = [_ct_exposure(shared_kvp)]
        ds.SharedFunctionalGroupsSequence = [shared_item]
    elif not shared_present:
        pass

    if per_frame_kvps is not None or per_frame_length is not None:
        n = per_frame_length if per_frame_length is not None else len(per_frame_kvps)
        source = per_frame_kvps or []
        items = []
        for i in range(n):
            item = Dataset()
            kvp = source[i] if i < len(source) else None
            if kvp is not None:
                item.CTExposureSequence = [_ct_exposure(kvp)]
            items.append(item)
        ds.PerFrameFunctionalGroupsSequence = items
    return _write_bytes(ds)


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in files.items():
            zf.writestr(name, blob)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DICOM_ANON_WORK_DIR", str(tmp_path / "jobs"))
    import importlib
    import app.config
    import app.jobs
    importlib.reload(app.config)
    importlib.reload(app.jobs)
    import app.main
    importlib.reload(app.main)
    with TestClient(app.main.app) as c:
        yield c


def _upload(client, files: dict[str, bytes]):
    r = client.post(
        "/jobs",
        files={"file": ("studies.zip", _zip_bytes(files), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    assert r.status_code == 201, r.text
    return r.json()


def _kvp(client, job_id: str, series_uid: str = SERIES):
    return client.get(
        f"/jobs/{job_id}/series/{derive_uid(SECRET, series_uid)}/ct-voltage"
    )


def _anon(original: str) -> str:
    return derive_uid(SECRET, original)


# --------------------------------------------------------------------------- #
# 成功路径
# --------------------------------------------------------------------------- #
def test_single_frame_top_level_kvp(client):
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "b.dcm": _image_bytes("1.1.1", top_kvp=120),
        "c.dcm": _image_bytes("1.1.2", top_kvp=80),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["series_instance_uid"] == _anon(SERIES)
    assert data["instance_count"] == 3
    assert data["frame_count"] == 3
    assert data["missing_kvp_frame_count"] == 0
    assert data["distinct_kvp_count"] == 2
    assert data["mixed_voltage"] is True
    assert data["kvp_distribution"] == [
        {"kvp": 80.0, "frame_count": 1},
        {"kvp": 120.0, "frame_count": 2},
    ]

    instances = data["instances"]
    assert [i["sop_instance_uid"] for i in instances] == sorted(
        i["sop_instance_uid"] for i in instances
    )
    by_sop = {i["sop_instance_uid"]: i for i in instances}
    inst = by_sop[_anon("1.1.0")]
    assert inst["modality"] == "CT"
    assert inst["declared_frame_count"] == 1
    assert inst["frame_count"] == 1
    assert inst["kvp_values"] == [120.0]
    assert inst["kvp_sources"] == ["top_level"]
    assert inst["kvp_distribution"] == [{"kvp": 120.0, "frame_count": 1}]
    assert inst["missing_kvp_frame_count"] == 0
    assert inst["output_file"].endswith(".dcm")


def test_uniform_voltage_not_mixed(client):
    files = {f"s{i}.dcm": _image_bytes(f"1.1.{i}", top_kvp=120) for i in range(3)}
    job = _upload(client, files)
    data = _kvp(client, job["id"]).json()
    assert data["mixed_voltage"] is False
    assert data["distinct_kvp_count"] == 1
    assert data["kvp_distribution"] == [{"kvp": 120.0, "frame_count": 3}]


def test_missing_top_level_kvp_counted_not_conflict(client):
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "b.dcm": _image_bytes("1.1.1"),  # 无 KVP
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frame_count"] == 2
    assert data["missing_kvp_frame_count"] == 1
    assert data["kvp_distribution"] == [{"kvp": 120.0, "frame_count": 1}]
    missing_inst = next(
        i for i in data["instances"] if i["sop_instance_uid"] == _anon("1.1.1")
    )
    assert missing_inst["kvp_values"] == [None]
    assert missing_inst["kvp_sources"] == [None]
    assert missing_inst["missing_kvp_frame_indices"] == [0]


def test_per_frame_functional_groups_kvp_takes_precedence(client):
    # 多帧：逐帧 CTExposureSequence 优先于共享与顶层
    files = {
        "mf.dcm": _image_bytes(
            "1.1.0",
            top_kvp=70,
            number_of_frames=3,
            shared_kvp=100,
            per_frame_kvps=[120, 140, None],
        ),
    }
    job = _upload(client, files)
    data = _kvp(client, job["id"]).json()
    inst = data["instances"][0]
    assert inst["frame_count"] == 3
    assert inst["declared_frame_count"] == 3
    assert inst["kvp_values"] == [120.0, 140.0, 100.0]
    assert inst["kvp_sources"] == [
        "per_frame_functional_groups",
        "per_frame_functional_groups",
        "shared_functional_groups",
    ]
    assert data["kvp_distribution"] == [
        {"kvp": 100.0, "frame_count": 1},
        {"kvp": 120.0, "frame_count": 1},
        {"kvp": 140.0, "frame_count": 1},
    ]
    assert data["missing_kvp_frame_count"] == 0
    assert data["mixed_voltage"] is True


def test_per_frame_missing_falls_back_to_top_level(client):
    # 该帧无 CTExposureSequence、无共享组 -> 回退顶层 KVP
    files = {
        "mf.dcm": _image_bytes(
            "1.1.0",
            top_kvp=120,
            number_of_frames=2,
            shared_present=False,
            per_frame_kvps=[80, None],
        ),
    }
    job = _upload(client, files)
    inst = _kvp(client, job["id"]).json()["instances"][0]
    assert inst["kvp_values"] == [80.0, 120.0]
    assert inst["kvp_sources"] == [
        "per_frame_functional_groups",
        "top_level",
    ]


def test_shared_only_multiframe(client):
    # 多帧只有共享 CTExposureSequence（无逐帧序列）：每帧取共享值
    files = {
        "mf.dcm": _image_bytes(
            "1.1.0", number_of_frames=4, shared_kvp=110, shared_present=True
        ),
    }
    job = _upload(client, files)
    data = _kvp(client, job["id"]).json()
    inst = data["instances"][0]
    assert inst["frame_count"] == 4
    assert inst["kvp_values"] == [110.0] * 4
    assert set(inst["kvp_sources"]) == {"shared_functional_groups"}
    assert data["kvp_distribution"] == [{"kvp": 110.0, "frame_count": 4}]


def test_multiframe_mixed_kvp_aggregation(client):
    files = {
        "a.dcm": _image_bytes(
            "1.1.0", number_of_frames=2, shared_present=False,
            per_frame_kvps=[120, 120],
        ),
        "b.dcm": _image_bytes(
            "1.1.1", number_of_frames=3, shared_present=False,
            per_frame_kvps=[80, 80, None],
        ),
    }
    job = _upload(client, files)
    data = _kvp(client, job["id"]).json()
    assert data["frame_count"] == 5
    assert data["missing_kvp_frame_count"] == 1
    assert data["kvp_distribution"] == [
        {"kvp": 80.0, "frame_count": 2},
        {"kvp": 120.0, "frame_count": 2},
    ]
    inst_b = next(i for i in data["instances"]
                  if i["sop_instance_uid"] == _anon("1.1.1"))
    assert inst_b["missing_kvp_frame_indices"] == [2]
    assert inst_b["kvp_distribution"] == [
        {"kvp": 80.0, "frame_count": 2},
    ]


def test_invalid_kvp_value_treated_as_missing(client):
    # 空文本 / 非正 KVP 均计入缺失，不报冲突；无法解析的文本由 _coerce_kvp 单测覆盖
    # （pydicom 写出阶段会拒绝构造非法 DS 值）
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=""),
        "c.dcm": _image_bytes("1.1.2", top_kvp=0),
        "d.dcm": _image_bytes("1.1.3", top_kvp=-120),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["frame_count"] == 3
    assert data["missing_kvp_frame_count"] == 3
    assert data["kvp_distribution"] == []
    assert data["distinct_kvp_count"] == 0
    assert data["mixed_voltage"] is False


def test_coerce_kvp_unit():
    from app.ct_voltage import _coerce_kvp

    assert _coerce_kvp(None) is None
    assert _coerce_kvp("") is None
    assert _coerce_kvp("   ") is None
    assert _coerce_kvp("120abc") is None
    assert _coerce_kvp("abc") is None
    assert _coerce_kvp(0) is None
    assert _coerce_kvp(-120) is None
    assert _coerce_kvp(float("nan")) is None
    assert _coerce_kvp(float("inf")) is None
    assert _coerce_kvp("120") == 120.0
    assert _coerce_kvp("80.0") == 80.0
    assert _coerce_kvp(110) == 110.0


def test_number_of_frames_one_treated_as_single_frame(client):
    # NumberOfFrames=1 且无逐帧序列：单帧，走顶层 KVP
    files = {"a.dcm": _image_bytes("1.1.0", number_of_frames=1, top_kvp=120)}
    job = _upload(client, files)
    inst = _kvp(client, job["id"]).json()["instances"][0]
    assert inst["frame_count"] == 1
    assert inst["kvp_values"] == [120.0]
    assert inst["kvp_sources"] == ["top_level"]


def test_instances_sorted_by_anonymous_sop_uid(client):
    files = {
        "z.dcm": _image_bytes("1.1.9", top_kvp=120),
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "m.dcm": _image_bytes("1.1.4", top_kvp=120),
    }
    job = _upload(client, files)
    uids = [i["sop_instance_uid"] for i in _kvp(client, job["id"]).json()["instances"]]
    assert uids == sorted(uids)


def test_only_requested_series_aggregated(client):
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "other.dcm": _image_bytes("1.9.0", series_uid="2.9", top_kvp=80),
    }
    job = _upload(client, files)
    data = _kvp(client, job["id"]).json()
    assert data["instance_count"] == 1
    assert data["instances"][0]["sop_instance_uid"] == _anon("1.1.0")


def test_reads_anonymized_output_only(client):
    """统计只读取匿名输出；明细中不回显原始 UID。"""
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "b.dcm": _image_bytes("1.1.1", top_kvp=140),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 200
    raw = r.text
    assert "1.1.0" not in raw
    assert SERIES not in raw
    assert _anon("1.1.0") in raw


# --------------------------------------------------------------------------- #
# 404 / 409
# --------------------------------------------------------------------------- #
def test_job_not_found_404(client):
    r = client.get(f"/jobs/deadbeef/series/{_anon(SERIES)}/ct-voltage")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert detail["code"] == "JOB_NOT_FOUND"
    assert detail["series_instance_uid"] == _anon(SERIES)


def test_series_not_found_404(client):
    files = {"a.dcm": _image_bytes("1.1.0", top_kvp=120)}
    job = _upload(client, files)
    r = _kvp(client, job["id"], series_uid="2.999")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "SERIES_NOT_FOUND"


def test_series_without_successful_outputs_404(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bad.dcm", b"\x00\x01not dicom")
    r = client.post(
        "/jobs",
        files={"file": ("b.zip", buf.getvalue(), "application/zip")},
        data={"secret": SECRET, "shift_dates": "false"},
    )
    job = r.json()
    resp = client.get("/jobs/%s/series/2.25.whatever/ct-voltage" % job["id"])
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "SERIES_NOT_FOUND"


def test_failed_job_409(client):
    r = client.post(
        "/jobs",
        files={"file": ("broken.zip", b"PK\x03\x04 garbage", "application/zip")},
        data={"secret": SECRET},
    )
    job = r.json()
    assert job["status"] == "failed"
    resp = client.get(f"/jobs/{job['id']}/series/x/ct-voltage")
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "JOB_NOT_READY"
    assert detail["series_instance_uid"] == "x"


# --------------------------------------------------------------------------- #
# 409：冲突明细
# --------------------------------------------------------------------------- #
def test_conflict_frame_count_mismatch_too_few(client):
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "bad.dcm": _image_bytes(
            "1.1.1",
            number_of_frames=4,
            per_frame_kvps=[80, 80],
            per_frame_length=2,
        ),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "CT_VOLTAGE_CONFLICT"
    assert detail["series_instance_uid"] == _anon(SERIES)
    problems = [p for p in detail["problems"]
                if p["reason"] == "frame_count_mismatch"]
    assert len(problems) == 1
    p = problems[0]
    assert p["sop_instance_uid"] == _anon("1.1.1")
    assert p["declared_frame_count"] == 4
    assert p["per_frame_sequence_length"] == 2
    assert p["number_of_frames"] == "4"
    assert p["output_file"].endswith(".dcm")


def test_conflict_frame_count_mismatch_too_many(client):
    files = {
        "bad.dcm": _image_bytes(
            "1.1.0",
            number_of_frames=2,
            per_frame_kvps=[120, 120, 120],
            per_frame_length=3,
        ),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    p = next(p for p in r.json()["detail"]["problems"]
             if p["reason"] == "frame_count_mismatch")
    assert p["declared_frame_count"] == 2
    assert p["per_frame_sequence_length"] == 3


def test_conflict_frame_count_mismatch_single_claim_multi_items(client):
    # 声明 1 帧却带 3 个逐帧数据项：结构不自洽，同样拒绝给统计
    files = {
        "bad.dcm": _image_bytes(
            "1.1.0",
            number_of_frames=1,
            per_frame_kvps=[120, 120, 120],
            per_frame_length=3,
        ),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    p = next(p for p in r.json()["detail"]["problems"]
             if p["reason"] == "frame_count_mismatch")
    assert p["declared_frame_count"] == 1
    assert p["per_frame_sequence_length"] == 3


def test_conflict_ambiguous_sop_output(client):
    # 两个同 SOP UID 的文件 -> <uid>.dcm / <uid>-2.dcm
    files = {
        "dup_a.dcm": _image_bytes("1.1.77", top_kvp=120),
        "dup_b.dcm": _image_bytes("1.1.77", top_kvp=80),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    problems = [p for p in r.json()["detail"]["problems"]
                if p["reason"] == "ambiguous_output"]
    assert len(problems) == 1
    p = problems[0]
    assert p["sop_instance_uid"] == _anon("1.1.77")
    assert p["output_files"] == sorted(
        [f"{_anon('1.1.77')}.dcm", f"{_anon('1.1.77')}-2.dcm"]
    )


def test_conflict_unreadable_output(client):
    files = {
        "a.dcm": _image_bytes("1.1.0", top_kvp=120),
        "b.dcm": _image_bytes("1.1.1", top_kvp=140),
    }
    job = _upload(client, files)
    from app.main import registry
    job_obj = registry.get(job["id"])
    target = os.path.join(job_obj.result.output_dir, f"{_anon('1.1.1')}.dcm")
    with open(target, "wb") as fh:
        fh.write(b"\x00\x01\x02not a dicom anymore")
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    problems = [p for p in r.json()["detail"]["problems"]
                if p["reason"] == "unreadable"]
    assert len(problems) == 1
    assert problems[0]["sop_instance_uid"] == _anon("1.1.1")


def test_conflict_output_missing(client):
    files = {"a.dcm": _image_bytes("1.1.0", top_kvp=120)}
    job = _upload(client, files)
    from app.main import registry
    job_obj = registry.get(job["id"])
    os.remove(os.path.join(job_obj.result.output_dir, f"{_anon('1.1.0')}.dcm"))
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    problems = [p for p in r.json()["detail"]["problems"]
                if p["reason"] == "output_missing"]
    assert len(problems) == 1
    assert problems[0]["sop_instance_uid"] == _anon("1.1.0")


def test_conflict_collects_multiple_problems_at_once(client):
    # 一个帧数不符 + 一个输出歧义：同一次 409 全部给出，不返回统计
    files = {
        "ok.dcm": _image_bytes("1.1.0", top_kvp=120),
        "mismatch.dcm": _image_bytes(
            "1.1.1", number_of_frames=3, per_frame_kvps=[80], per_frame_length=1
        ),
        "dup_a.dcm": _image_bytes("1.1.77", top_kvp=120),
        "dup_b.dcm": _image_bytes("1.1.77", top_kvp=80),
    }
    job = _upload(client, files)
    r = _kvp(client, job["id"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    reasons = {p["reason"] for p in detail["problems"]}
    assert reasons == {"frame_count_mismatch", "ambiguous_output"}
    assert detail["conflict_count"] == len(detail["problems"]) == 2
    # 冲突响应不携带分布统计
    assert "kvp_distribution" not in detail


def test_conflict_problems_contain_only_anonymized_uids(client):
    files = {
        "bad.dcm": _image_bytes(
            "1.1.1", number_of_frames=9, per_frame_kvps=[80], per_frame_length=1
        ),
    }
    job = _upload(client, files)
    detail = _kvp(client, job["id"]).json()["detail"]
    raw = str(detail)
    assert SERIES not in raw
    assert "1.1.1" not in raw
    assert _anon(SERIES) in raw
    assert _anon("1.1.1") in raw


# --------------------------------------------------------------------------- #
# OpenAPI
# --------------------------------------------------------------------------- #
def test_ct_voltage_endpoint_in_openapi(client):
    r = client.get("/openapi.json")
    paths = r.json()["paths"]
    key = "/jobs/{job_id}/series/{series_instance_uid}/ct-voltage"
    assert key in paths
    assert "get" in paths[key]
    schema = paths[key]["get"]
    assert "SeriesCtVoltageOut" in str(schema["responses"]["200"])
    endpoints = client.get("/").json()["endpoints"]
    assert "GET /jobs/{job_id}/series/{series_instance_uid}/ct-voltage" in endpoints
