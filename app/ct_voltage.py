"""CT 管电压（KVP，(0018,0060)）逐帧分布分析。

数据来源严格限定为“已完成任务 output 目录中的匿名 DICOM 文件”，
**不重新解压或处理原始 ZIP**，并且只纳入**成功写出的实例**
（状态非 error、存在扁平输出文件且记录了匿名 SOP UID）。

对系列内每个实例逐帧解析管电压，取值优先级（每帧独立）：

1. 该帧在 ``PerFrameFunctionalGroupsSequence`` 中对应数据项的
   ``CTExposureSequence[].KVP``（Enhanced CT Image IOD）；
2. ``SharedFunctionalGroupsSequence`` 内 ``CTExposureSequence[].KVP``
   （所有帧共享）；
3. 顶层 ``KVP``（Classic CT Image IOD）。

单帧文件（``NumberOfFrames`` 缺省或为 1）按一帧处理；找不到任何有效
KVP 的帧计入 ``missing_kvp_frame_count``，但**不是冲突**——
缺失值本身就是需要暴露的分布信息。

存在以下任一结构性问题时返回 ``409 CT_VOLTAGE_CONFLICT``，在
``problems[]`` 中一次性给出含匿名 SOP UID 的全部明细，**不返回
不完整统计**：

* ``frame_count_mismatch``：``NumberOfFrames`` 声明值与
  ``PerFrameFunctionalGroupsSequence`` 长度不符（含声明 1 帧却给出多个
  逐帧数据项；空逐帧序列视同未提供，按单帧回退处理）；
* ``ambiguous_output``：同一匿名 SOPInstanceUID 对应多份物理输出；
* ``unreadable`` / ``output_missing``：匿名输出无法读取或磁盘文件缺失。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import pydicom
from pydicom.errors import InvalidDicomError

from .anonymizer import FileReport, JobResult

# detail.problems[].reason 取值（机器可读）
REASON_UNREADABLE = "unreadable"
REASON_OUTPUT_MISSING = "output_missing"
REASON_AMBIGUOUS_OUTPUT = "ambiguous_output"
REASON_FRAME_COUNT_MISMATCH = "frame_count_mismatch"

# KVP 在各层级的点号路径，便于冲突/明细中标注取值来源
SOURCE_PER_FRAME = "per_frame_functional_groups"
SOURCE_SHARED = "shared_functional_groups"
SOURCE_TOP_LEVEL = "top_level"


class CtVoltageError(Exception):
    """CT 管电压分析失败；http_status 由路由层映射为 HTTPException。"""

    def __init__(
        self,
        code: str,
        http_status: int,
        message: str,
        extra: Optional[dict] = None,
        problems: Optional[list[dict]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status
        detail: dict = {"code": code, "message": message}
        if extra:
            detail.update(extra)
        if problems:
            detail["problems"] = problems
        self.detail = detail


@dataclass
class _FrameKvp:
    """单帧解析结果；kvp/source 为 None 表示该帧缺失管电压。"""
    index: int
    kvp: Optional[float]
    source: Optional[str]  # per_frame_functional_groups | shared_functional_groups | top_level | None(缺失)


# --------------------------------------------------------------------------- #
# 输出收集
# --------------------------------------------------------------------------- #
def _successful_reports(result: JobResult) -> list[FileReport]:
    """任务内真正写出匿名文件、且记录了匿名 SOP UID 的报告。"""
    return [
        report
        for report in result.files
        if report.status != "error" and report.output_name and report.sop_instance_uid
    ]


def _series_reports(result: JobResult, series_uid: str) -> list[FileReport]:
    return [
        report
        for report in _successful_reports(result)
        if report.series_instance_uid == series_uid
    ]


def _detect_ambiguous_outputs(reports: list[FileReport]) -> list[dict]:
    """同一匿名 SOPInstanceUID 对应多份物理输出时逐个 UID 报歧义。"""
    by_sop: dict[str, list[str]] = {}
    for report in reports:
        by_sop.setdefault(report.sop_instance_uid, []).append(report.output_name)

    problems: list[dict] = []
    for sop_uid in sorted(by_sop):
        names = sorted(by_sop[sop_uid])
        if len(names) > 1:
            problems.append({
                "sop_instance_uid": sop_uid,
                "output_file": names[0],
                "output_files": names,
                "reason": REASON_AMBIGUOUS_OUTPUT,
                "message": (
                    "同一匿名 SOPInstanceUID 对应多份输出文件，"
                    "无法确定用于管电压分析的唯一实例"
                ),
            })
    return problems


# --------------------------------------------------------------------------- #
# KVP 解析
# --------------------------------------------------------------------------- #
def _coerce_kvp(value: object) -> Optional[float]:
    """把 DS/IS 元素值转成正有限浮点；空值/不可解析/非正/非有限均视为缺失。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        kvp = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(kvp) or kvp <= 0:
        return None
    return kvp


def _first_ct_exposure_kvp(item) -> Optional[float]:
    """从单个功能组数据项的 CTExposureSequence 中取第一个有效 KVP。"""
    seq = getattr(item, "CTExposureSequence", None)
    if not seq:
        return None
    for exposure in seq:
        kvp = _coerce_kvp(getattr(exposure, "KVP", None))
        if kvp is not None:
            return kvp
    return None


def _shared_kvp(ds) -> Optional[float]:
    """SharedFunctionalGroupsSequence 内共享的 CTExposureSequence[].KVP。"""
    shared = getattr(ds, "SharedFunctionalGroupsSequence", None)
    if not shared:
        return None
    for item in shared:
        kvp = _first_ct_exposure_kvp(item)
        if kvp is not None:
            return kvp
    return None


def _declared_frame_count(ds) -> tuple[int, Optional[str]]:
    """解析 NumberOfFrames；缺省/空值/非法/非正均按单帧处理。

    返回 (声明帧数, 原始文本或 None)。
    """
    raw = getattr(ds, "NumberOfFrames", None)
    if raw is None or not str(raw).strip():
        return 1, None
    text = str(raw).strip()
    try:
        frames = int(text)
    except (TypeError, ValueError):
        return 1, text
    return (frames, text) if frames > 0 else (1, text)


def _frame_kvps(ds) -> tuple[list[_FrameKvp], int, Optional[str]]:
    """逐帧解析 KVP，返回 (逐帧结果, 声明帧数, 原始 NumberOfFrames 文本)。

    ``frame_count_mismatch`` 由调用方依据声明帧数与逐帧序列长度判定。
    """
    declared, frames_text = _declared_frame_count(ds)
    per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    shared_kvp = _shared_kvp(ds)
    top_kvp = _coerce_kvp(getattr(ds, "KVP", None))

    if per_frame:
        frames: list[_FrameKvp] = []
        for index, item in enumerate(per_frame):
            kvp = _first_ct_exposure_kvp(item)
            if kvp is not None:
                frames.append(_FrameKvp(index, kvp, SOURCE_PER_FRAME))
            elif shared_kvp is not None:
                frames.append(_FrameKvp(index, shared_kvp, SOURCE_SHARED))
            elif top_kvp is not None:
                frames.append(_FrameKvp(index, top_kvp, SOURCE_TOP_LEVEL))
            else:
                frames.append(_FrameKvp(index, None, None))
        return frames, declared, frames_text

    # 无逐帧功能组：声明帧数全部按共享/顶层值处理（声明多帧时调用方报冲突）
    frames = []
    for index in range(declared):
        if shared_kvp is not None:
            frames.append(_FrameKvp(index, shared_kvp, SOURCE_SHARED))
        elif top_kvp is not None:
            frames.append(_FrameKvp(index, top_kvp, SOURCE_TOP_LEVEL))
        else:
            frames.append(_FrameKvp(index, None, None))
    return frames, declared, frames_text


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def analyze_series_ct_voltage(job, series_uid: str) -> dict:
    """读取系列成功输出并逐帧统计 KVP 分布；任何结构性错误均抛 CtVoltageError。"""
    result: JobResult = job.result
    reports = _series_reports(result, series_uid)
    if not reports:
        raise CtVoltageError(
            "SERIES_NOT_FOUND",
            404,
            "该匿名 SeriesInstanceUID 在本任务的成功输出中不存在",
            extra={"series_instance_uid": series_uid},
        )

    problems: list[dict] = _detect_ambiguous_outputs(reports)
    ambiguous_sops = {p["sop_instance_uid"] for p in problems}

    instances: list[dict] = []
    # 全系列聚合：电压（kV）-> 帧数
    distribution: dict[float, int] = {}
    total_frames = 0
    total_missing = 0

    for report in sorted(reports, key=lambda r: r.sop_instance_uid):
        if report.sop_instance_uid in ambiguous_sops:
            continue
        path = os.path.join(result.output_dir, report.output_name)
        if not os.path.isfile(path):
            problems.append({
                "sop_instance_uid": report.sop_instance_uid,
                "output_file": report.output_name,
                "reason": REASON_OUTPUT_MISSING,
                "message": "匿名输出文件在任务目录中不存在（可能已被清理）",
            })
            continue

        try:
            ds = pydicom.dcmread(path, force=False)
        except (InvalidDicomError, OSError, EOFError) as exc:
            problems.append({
                "sop_instance_uid": report.sop_instance_uid,
                "output_file": report.output_name,
                "reason": REASON_UNREADABLE,
                "message": f"匿名输出不是可读取的 DICOM 文件：{exc}",
            })
            continue
        except Exception as exc:  # 标签编码损坏等
            problems.append({
                "sop_instance_uid": report.sop_instance_uid,
                "output_file": report.output_name,
                "reason": REASON_UNREADABLE,
                "message": f"读取匿名输出失败：{exc}",
            })
            continue

        frames, declared, frames_text = _frame_kvps(ds)
        per_frame = getattr(ds, "PerFrameFunctionalGroupsSequence", None)

        # 多帧声明数与逐帧序列长度不符：明确冲突，不给不完整统计。
        # 含“声明 1 帧却给出多个逐帧数据项”这种不自洽情况；
        # 空逐帧序列视同未提供，按单帧回退逻辑处理。
        if per_frame and len(per_frame) != declared:
            problems.append({
                "sop_instance_uid": report.sop_instance_uid,
                "output_file": report.output_name,
                "reason": REASON_FRAME_COUNT_MISMATCH,
                "declared_frame_count": declared,
                "per_frame_sequence_length": len(per_frame),
                "number_of_frames": frames_text,
                "message": (
                    f"NumberOfFrames 声明 {declared} 帧，但 "
                    f"PerFrameFunctionalGroupsSequence 含 {len(per_frame)} 个数据项，"
                    "无法逐帧确定管电压"
                ),
            })
            continue

        kvp_values: list[Optional[float]] = []
        kvp_sources: list[Optional[str]] = []
        missing_indices: list[int] = []
        per_instance: dict[float, int] = {}
        for frame in frames:
            kvp_values.append(frame.kvp)
            kvp_sources.append(frame.source)
            total_frames += 1
            if frame.kvp is None:
                total_missing += 1
                missing_indices.append(frame.index)
            else:
                distribution[frame.kvp] = distribution.get(frame.kvp, 0) + 1
                per_instance[frame.kvp] = per_instance.get(frame.kvp, 0) + 1

        modality = str(getattr(ds, "Modality", "") or "").strip() or None
        sop_uid = str(getattr(ds, "SOPInstanceUID", "") or "").strip() \
            or report.sop_instance_uid

        instances.append({
            "sop_instance_uid": sop_uid,
            "output_file": report.output_name,
            "modality": modality,
            "declared_frame_count": declared,
            "frame_count": len(frames),
            "missing_kvp_frame_count": len(missing_indices),
            "missing_kvp_frame_indices": missing_indices,
            "kvp_values": kvp_values,
            "kvp_sources": kvp_sources,
            "kvp_distribution": [
                {"kvp": kvp, "frame_count": count}
                for kvp, count in sorted(per_instance.items())
            ],
        })

    if problems:
        raise CtVoltageError(
            "CT_VOLTAGE_CONFLICT",
            409,
            (
                "匿名输出存在帧数声明不一致、SOP 输出歧义或文件不可读，"
                "无法给出完整可靠的管电压分布统计"
            ),
            extra={
                "series_instance_uid": series_uid,
                "instance_count": len(instances),
                "conflict_count": len(problems),
            },
            problems=sorted(
                problems,
                key=lambda p: (p.get("output_file") or "",
                               p.get("sop_instance_uid") or ""),
            ),
        )

    distinct = [
        {"kvp": kvp, "frame_count": count}
        for kvp, count in sorted(distribution.items())
    ]
    present_voltages = sorted(distribution)
    return {
        "series_instance_uid": series_uid,
        "instance_count": len(instances),
        "frame_count": total_frames,
        "missing_kvp_frame_count": total_missing,
        "distinct_kvp_count": len(present_voltages),
        "mixed_voltage": len(present_voltages) > 1,
        "kvp_distribution": distinct,
        "instances": instances,
    }
