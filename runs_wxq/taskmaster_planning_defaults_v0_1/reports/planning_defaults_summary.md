# Taskmaster-2 Planning Defaults v0.1

- Created: 2026-06-30T04:48:28.884827+00:00
- Status: taskmaster_planning_defaults_auto_validated_pending_human_audit
- Seed source: `google-research-datasets/Taskmaster-2`
- Source domains: `flights`, `hotels`
- HABIT-Bench family: `planning_defaults`
- Representative domain: `travel`
- Hidden habit: business travel prefers early arrivals and about a 90-minute meeting buffer

## Counts

- Users: 30
- Sessions: 1080
- Probes: 120
- Support sessions: 150
- Boundary sessions: 60
- Exception sessions: 30
- Distractor sessions: 840

## Filter Summary

- `flights`: raw=2481, accepted_before_cap=1903, selected=800
- `hotels`: raw=2357, accepted_before_cap=1784, selected=800

## Human Review Handoff

Review `review/planning_defaults_review_queue_sample.csv` first, then audit the full queue.
Use `accept`, `revise`, or `reject` in `reviewer_decision` and keep notes short and concrete.

Primary accept criteria: Taskmaster seed is travel-domain coherent, repeated support establishes the business-travel buffer default, boundary/exception probes avoid false personalization, and the gold choice is unique.
