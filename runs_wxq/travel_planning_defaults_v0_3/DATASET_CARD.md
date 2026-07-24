# Travel Planning Defaults v0.3

Status: candidate dataset; model review and human audit still required.

This version targets travel long-horizon user-agent planning conversations.
It intentionally uses variable numbers of scoped user defaults instead of one habit per user.

- Users: 36
- Sessions: 2612
- Probes: 201
- Habits: 108
- Habit count distribution: {3: 14, 4: 9, 5: 3, 2: 5, 1: 5}
- Signal counts: {'support': 465, 'distractor': 1846, 'exception': 93, 'mixed_support': 68, 'boundary_counterexample': 140}
- Probe counts: {'direct_use': 36, 'boundary': 36, 'exception': 36, 'explicit_retrieval': 36, 'cross_habit_selection': 31, 'conflict_current_override': 26}
- Habit domains: {'hotels': 19, 'flights': 31, 'booking_policy': 6, 'baggage': 7, 'loyalty': 7, 'itinerary_pacing': 7, 'family_travel': 9, 'ground_transport': 16, 'accessibility': 6}
- Avg chars/session: 1693.79
- Avg messages/session: 15.23

Evaluation note: the intended formal path is memory retrieval top-k followed by a Qwen RAG answer head, not the old lexical choice scorer.
