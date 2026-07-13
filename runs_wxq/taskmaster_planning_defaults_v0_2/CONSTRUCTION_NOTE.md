# Construction Note

v0.2 revises v0.1 after review found many short-session artifacts and answer leakage.

## Main Changes From v0.1

- Sessions and probes are generated with `gpt-5.5` `reasoning_effort=xhigh` from Taskmaster-2 flights/hotels seed scenarios.
- The final LLM-generated slice uses a 12-template planning-default bank instead of a single repeated hidden preference.
- Each generated session must pass a length floor of 1500 characters and 12 messages.
- Probe wording avoids obvious phrases such as `habit`, `gold`, and `established preference`.
- Distractor choices are written as plausible travel-planning tradeoffs rather than obvious nonanswers.
- The directory keeps the same public/private/review/reports split used by the reference runs.

## Generation Recipe

1. Start from the v0.1 filtered Taskmaster travel seeds.
2. Provide balanced flights/hotels seed scenarios to the generator.
3. Generate each synthetic user's support, boundary, exception, and distractor sessions.
4. Generate one probe of each type per user.
5. Normalize, validate length/choice/evidence contracts, then write public/private/review artifacts.

## Known Limitation

The habit evidence is synthetic and model-generated from Taskmaster seed scenarios. Human review should still check whether each probe is answerable only after reading the user's history and whether the distractors are sufficiently competitive.
