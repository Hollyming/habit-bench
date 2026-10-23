"""Bounded, unordered I/O execution; callers alone persist/score results."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def configured_answer_workers(env):
    """Validate before building memory; both HTTP backends support bounded I/O."""
    workers = int(env.get("HABITBENCH_ANSWER_WORKERS", "1"))
    if workers < 1:
        raise ValueError("HABITBENCH_ANSWER_WORKERS must be positive")
    backend = env.get("HABITBENCH_INFERENCE_BACKEND", "local-vllm")
    if workers > 1 and backend not in {"local-vllm", "external-openai-compatible"}:
        raise ValueError(f"Concurrent answers require an OpenAI-compatible HTTP backend: {backend}")
    return workers


def answer_results(probes, contexts, answerer, workers=1):
    if workers < 1:
        raise ValueError("Answer workers must be positive")
    def invoke(probe):
        try:
            return probe, answerer.answer(probe, contexts[probe["probe_id"]]["memory_context"]), None
        except Exception as error:
            return probe, None, error
    if workers == 1:
        for probe in probes:
            yield invoke(probe)
        return
    iterator = iter(probes)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="http-answer") as pool:
        pending = set()
        for _ in range(workers):
            probe = next(iterator, None)
            if probe is not None:
                pending.add(pool.submit(invoke, probe))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                probe = next(iterator, None)
                if probe is not None:
                    pending.add(pool.submit(invoke, probe))
