# Changelog from v0.4

- Replaced raw-dialogue splicing with task-event extraction and persona-conditioned rewriting.
- Increased history from 71 to 240 sessions per user.
- Reduced pseudo-users from 40 to 24 so each lifeline can be substantially longer and denser.
- Increased active habits from 5 to 6 per user, with two temporal updates and a tentative one-off signal.
- Replaced 10 probes/user with 18 harder probes/user.
- Added three-habit composition, mixed boundary/support, mixed exception/support, drift composition, drift exception, conditional scope, and insufficient-evidence probes.
- Added identity, difficulty, evidence-distance, token-length, and source-cluster audits.
