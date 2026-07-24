#!/usr/bin/env python3
"""Count exact local Qwen tokens for JSON texts received on stdin."""

import json
import sys
from transformers import AutoTokenizer


model_path = sys.argv[1]
payload = json.load(sys.stdin)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in payload["texts"]]
json.dump({"counts": counts, "total": sum(counts)}, sys.stdout)
