from scripts.rescore_locomo_common_reader import (
    build_bounded_evidence,
    build_question_prompt,
    build_user_prompt,
    load_model_snapshot,
    official_score,
    parse_reader_response,
    summarize,
)


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(token) for token in tokens)


def test_common_reader_evidence_uses_ranked_prefix_and_exact_budget():
    evidence, used, included = build_bounded_evidence(
        [{"memory": "first"}, {"memory": "second"}],
        _CharTokenizer(),
        30,
    )
    assert used == 30
    assert included == 2
    assert evidence.startswith("[Memory 1]\nfirst")
    assert evidence.endswith("[Memory 2]\nsec")


def test_common_reader_uses_fixed_type_prompt_and_official_metric():
    row = {
        "query_type": "temporal",
        "question": "When?",
        "expected_answer": "7 May 2023",
        "evaluation_details": {"category": 2},
    }
    prompt = build_question_prompt(row)
    assert "ABSOLUTE dates" in prompt
    assert "When?" in prompt
    assert official_score(row, "7 May 2023") == 1.0
    user_prompt = build_user_prompt(row, "ranked evidence")
    assert user_prompt.startswith("Memory evidence:\nranked evidence")
    assert "When?" in user_prompt


def test_common_reader_records_frozen_model_snapshot(tmp_path):
    manifest = tmp_path / "model.ready"
    manifest.write_text(
        '{"status":"ready","repo":"Qwen/Qwen3-8B","revision":"abc",'
        '"small_files":{"config.json":{"sha256":"cfg"},'
        '"tokenizer.json":{"sha256":"tok"}}}',
        encoding="utf-8",
    )
    snapshot = load_model_snapshot(manifest)
    assert snapshot["revision"] == "abc"
    assert snapshot["config_sha256"] == "cfg"
    assert snapshot["tokenizer_sha256"] == "tok"
    assert len(snapshot["manifest_sha256"]) == 64


def test_common_reader_summary_reports_official_f1_and_threshold_accuracy():
    rows = [
        {
            "query_type": "single_hop",
            "common_reader_official_score": 1.0,
            "common_reader_is_correct": True,
            "common_reader_evidence_tokens": 10,
        },
        {
            "query_type": "single_hop",
            "common_reader_official_score": 0.25,
            "common_reader_is_correct": False,
            "common_reader_evidence_tokens": 20,
        },
    ]
    result = summarize(rows)
    assert result["mean_official_score"] == 0.625
    assert result["threshold_0_5_accuracy"] == 0.5
    assert result["mean_evidence_tokens"] == 15.0


def test_common_reader_parses_one_schema_bounded_answer():
    assert parse_reader_response('{"answer":"  pottery  "}') == "pottery"
