import logging

from methods.LightMem.src.lightmem.memory.utils import (
    convert_extraction_results_to_memory_entries,
    normalize_source_id,
)


def test_corrected_source_id_is_used_for_timestamp_and_speaker() -> None:
    entries = convert_extraction_results_to_memory_entries(
        extracted_results=[
            {"cleaned_result": [[{"source_id": 5, "fact": "corrected fact"}]]}
        ],
        timestamps_list=[f"2026-01-01T00:00:0{i}" for i in range(5)],
        weekday_list=["Thursday"] * 5,
        speaker_list=[
            {"speaker_id": f"speaker-{i}", "speaker_name": f"Speaker {i}"}
            for i in range(5)
        ],
        topic_id_map={4: 99},
        max_source_ids=[2],
        logger=logging.getLogger("test_lightmem_source_id"),
    )

    assert len(entries) == 1
    assert entries[0].time_stamp == "2026-01-01T00:00:04"
    assert entries[0].speaker_id == "speaker-4"
    assert entries[0].topic_id == 99


def test_decorated_source_id_is_recovered_without_aborting_memory_build() -> None:
    entries = convert_extraction_results_to_memory_entries(
        extracted_results=[
            {"cleaned_result": [[{"source_id": "0.医生", "fact": "medical fact"}]]}
        ],
        timestamps_list=["2026-01-01T00:00:00"],
        weekday_list=["Thursday"],
        speaker_list=[{"speaker_id": "doctor", "speaker_name": "Doctor"}],
        topic_id_map={0: 7},
        max_source_ids=[0],
        logger=logging.getLogger("test_lightmem_source_id"),
    )

    assert len(entries) == 1
    assert entries[0].time_stamp == "2026-01-01T00:00:00"
    assert entries[0].speaker_id == "doctor"
    assert entries[0].topic_id == 7


def test_source_id_normalization_is_bounded() -> None:
    assert normalize_source_id(-3, max_valid_sid=4) == 0
    assert normalize_source_id("99 note", max_valid_sid=4) == 4
    assert normalize_source_id("not-an-id", max_valid_sid=4) == 0
