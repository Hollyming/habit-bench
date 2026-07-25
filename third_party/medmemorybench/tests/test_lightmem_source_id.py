import logging

from methods.LightMem.src.lightmem.memory.utils import (
    convert_extraction_results_to_memory_entries,
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
