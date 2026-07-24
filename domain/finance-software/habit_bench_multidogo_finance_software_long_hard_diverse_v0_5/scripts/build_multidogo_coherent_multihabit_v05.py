#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import random
import re
import shutil
import statistics
import sys
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
LIB_PATH = HERE / "lib_v04_generation.py"
spec = importlib.util.spec_from_file_location("habit_v04_lib", LIB_PATH)
lib = importlib.util.module_from_spec(spec)
sys.modules["habit_v04_lib"] = lib
assert spec.loader is not None
spec.loader.exec_module(lib)

SEED = 20260720
SOURCE_DATASET = "awslabs/multi-domain-goal-oriented-dialogues-dataset"
DATASET_ID = "habit_bench_multidogo_finance_software_long_hard_diverse_v0_5"
DOMAINS = ("finance", "software")
FINANCE_USER_COUNT = 36
SOFTWARE_USER_COUNT = 9
TOTAL_SESSIONS_PER_USER = 320  # includes one identity/continuity anchor
PROBES_PER_USER = 18
CLUSTERS_PER_DOMAIN = 30

# Remove the weakly source-grounded license/subscription habit. The remaining
# habits are retained only when source audit meets minimum support thresholds.
RETAINED_HABIT_IDS = [
    "finance_confirm_money_movement",
    "finance_confirm_card_account_changes",
    "finance_minimal_pii_secure_verification",
    "finance_fraud_lost_card_urgent_escalation",
    "finance_balance_statement_summary_first",
    "finance_fee_dispute_evidence_then_case",
    "finance_credit_loan_cautious_no_commitment",
    "finance_payment_status_latest_check",
    "software_collect_diagnostics_before_fix",
    "software_docs_lookup_for_update_install",
    "software_one_try_then_escalate",
    "software_secure_login_password_flow",
    "software_platform_specific_steps",
    "software_backup_before_risky_change",
    "software_ticket_receipt_summary",
]
HABITS = [dict(h) for h in lib.HABITS if h["habit_id"] in RETAINED_HABIT_IDS]
HABIT_BY_ID = {h["habit_id"]: h for h in HABITS}
HABITS_BY_DOMAIN = {d: [h for h in HABITS if h["domain"] == d] for d in DOMAINS}


# Four counterbalanced, safe policy variants per habit. The benchmark does not
# treat one generic best-practice wording as universally correct: each pseudo-
# user repeatedly establishes one of these plausible workflows, and the same
# variant is a gold answer for some users and a distractor for others.
POLICY_VARIANTS: dict[str, list[dict[str, str]]] = {
    "finance_confirm_money_movement": [
        {"variant_id":"itemized_final_yes","label":"itemized readback then one final yes","action":"read back amount, source, destination or payee, and timing, then wait for one final yes","body":"I’ll read back the amount, source, destination or payee, and timing shown in the request, then wait for one final yes before anything is submitted."},
        {"variant_id":"draft_transaction_card","label":"unsubmitted transaction card","action":"prepare an unsubmitted transaction card and use its single approval control","body":"I’ll prepare an unsubmitted transaction card with the amount, source, destination or payee, and timing, and use the card’s single approval control before submission."},
        {"variant_id":"two_checkpoint_confirmation","label":"two-checkpoint confirmation","action":"confirm the destination first and the amount and timing second","body":"I’ll use two short checkpoints: first confirm the destination or payee, then confirm the amount and timing before the transaction is submitted."},
        {"variant_id":"secure_preview_confirmation","label":"secure preview confirmation","action":"open a secure preview and require approval inside that preview","body":"I’ll open a secure preview containing the exact transaction details and require approval inside that preview rather than treating the chat request as authorization."},
    ],
    "finance_confirm_card_account_changes": [
        {"variant_id":"precise_change_readback","label":"precise change readback","action":"restate the exact setting and consequence, then obtain one final approval","body":"I’ll restate the exact card or account setting to be changed and its consequence, then obtain one final approval before applying it."},
        {"variant_id":"before_after_comparison","label":"before-and-after comparison","action":"show a before-and-after comparison and wait for approval","body":"I’ll show a concise before-and-after comparison for the requested card or account change and wait for approval on that comparison before applying it."},
        {"variant_id":"secure_change_preview","label":"secure in-app change preview","action":"create a secure in-app change preview and require device approval","body":"I’ll create a secure in-app preview of the requested change and require device approval there before the account state is updated."},
        {"variant_id":"change_checklist_callback","label":"change checklist plus verified callback","action":"complete a change-impact checklist and confirm through an authenticated callback","body":"I’ll complete a short impact checklist for the requested change and confirm it through an authenticated callback initiated from the account before applying it."},
    ],
    "finance_minimal_pii_secure_verification": [
        {"variant_id":"minimal_secure_panel","label":"minimal secure panel","action":"request only minimum fields in an in-app secure panel","body":"I’ll request only the minimum fields needed in an in-app secure panel and will not ask for full account numbers, passwords, PINs, or one-time codes in chat."},
        {"variant_id":"authenticated_app_callback","label":"authenticated app callback","action":"start a verified callback from the authenticated app","body":"I’ll start a verified callback from the authenticated app and keep sensitive identifiers out of the chat transcript."},
        {"variant_id":"device_passkey_approval","label":"device or passkey approval","action":"use a device or passkey approval with minimal fallback information","body":"I’ll use a device or passkey approval and request only minimal fallback information through the secure surface, never full secrets in chat."},
        {"variant_id":"secure_message_center","label":"secure message-center verification","action":"move verification to the secure message center and use only masked identifiers in chat","body":"I’ll move verification to the secure message center and use only masked identifiers in this conversation, without repeating or storing sensitive values."},
    ],
    "finance_fraud_lost_card_urgent_escalation": [
        {"variant_id":"lock_then_fraud_team","label":"immediate lock then fraud team","action":"lock the affected card immediately and route the incident to the fraud team","body":"I’ll immediately lock the affected card, check its current transaction state, and route the incident to the fraud team with a written case summary."},
        {"variant_id":"verify_incident_then_lock","label":"verify incident details then lock","action":"verify the last known legitimate activity, lock the card, and escalate","body":"I’ll verify the last known legitimate activity, lock the affected card, and then escalate the incident with the suspicious details preserved."},
        {"variant_id":"temporary_freeze_review","label":"temporary freeze and secure review","action":"place a temporary freeze, perform a secure review, then decide replacement","body":"I’ll place a temporary freeze, run a secure review of the suspicious activity, and then confirm whether replacement or a longer block is needed."},
        {"variant_id":"card_controls_timeline","label":"card controls plus incident timeline","action":"activate card controls and send a concise incident timeline to fraud support","body":"I’ll activate the relevant card controls and send a concise incident timeline to fraud support so the protective action and next checkpoint are both visible."},
    ],
    "finance_balance_statement_summary_first": [
        {"variant_id":"headline_then_evidence","label":"headline then evidence","action":"lead with the account-level finding and then list supporting entries","body":"I’ll lead with the account-level finding, then list the small set of entries and reconciliation details that support it."},
        {"variant_id":"exceptions_first_table","label":"exceptions-first table","action":"show unusual entries first in a compact table and close with the overall finding","body":"I’ll show unusual entries first in a compact table and close with the overall account-level finding and remaining uncertainty."},
        {"variant_id":"chronological_timeline","label":"chronological review timeline","action":"present a chronological transaction timeline followed by the conclusion","body":"I’ll present a concise chronological timeline of the relevant entries and then state the overall conclusion at the end."},
        {"variant_id":"merchant_category_rollup","label":"merchant and category rollup","action":"group relevant entries by merchant or category and connect each group to the conclusion","body":"I’ll group the relevant entries by merchant or category and connect each group to the final account-level conclusion."},
    ],
    "finance_fee_dispute_evidence_then_case": [
        {"variant_id":"evidence_pack_then_approval","label":"evidence pack then approval","action":"assemble the evidence pack, summarize it, and ask before filing","body":"I’ll assemble the relevant statement line, receipt, merchant, date, and amount into an evidence pack, summarize it, and ask before filing a case."},
        {"variant_id":"unsubmitted_case_draft","label":"unsubmitted case draft","action":"prepare an unsubmitted dispute draft and request approval on the draft","body":"I’ll prepare an unsubmitted dispute draft with the available evidence and request approval on that draft before it becomes a case."},
        {"variant_id":"merchant_reconciliation_first","label":"merchant reconciliation first","action":"reconcile the merchant record first and file only if the mismatch remains","body":"I’ll reconcile the merchant and statement records first, document any remaining mismatch, and ask before filing a dispute if it is still unresolved."},
        {"variant_id":"charge_timeline_threshold","label":"charge timeline and escalation threshold","action":"build a charge timeline and use an agreed threshold before opening a case","body":"I’ll build a concise charge timeline, identify the unresolved evidence gap, and use the agreed escalation threshold before opening a case."},
    ],
    "finance_credit_loan_cautious_no_commitment": [
        {"variant_id":"requirements_and_ranges","label":"requirements and scenario ranges","action":"explain requirements and scenario ranges without predicting approval","body":"I’ll explain the documented requirements and show scenario ranges without predicting approval or starting an application."},
        {"variant_id":"budget_stress_test","label":"budget stress test first","action":"run a budget stress test and list documentation without submitting","body":"I’ll run a budget stress test using the stated assumptions, list the documentation that would be needed, and avoid submitting or implying approval."},
        {"variant_id":"lender_question_checklist","label":"lender-question checklist","action":"prepare a lender-question checklist and mark unknowns","body":"I’ll prepare a lender-question checklist, mark the facts that are still unknown, and keep it separate from any application or approval prediction."},
        {"variant_id":"side_by_side_scenarios","label":"side-by-side scenarios","action":"compare several financing scenarios and leave the decision unsubmitted","body":"I’ll compare several financing scenarios side by side, state the assumptions and trade-offs, and leave every application or commitment unsubmitted."},
    ],
    "finance_payment_status_latest_check": [
        {"variant_id":"live_lookup_timestamp","label":"live lookup with timestamp","action":"check the live account state and label it with an as-of time","body":"I’ll check the live account or payment state, distinguish queued, pending, and posted items, and label the result with an as-of time."},
        {"variant_id":"secure_status_page","label":"secure status-page interpretation","action":"use the secure status page and explain the current state with a timestamp","body":"I’ll use the secure status page, explain the current processing state, and include the time at which that page was checked."},
        {"variant_id":"ledger_notification_crosscheck","label":"ledger and notification cross-check","action":"cross-check the official ledger with account notifications","body":"I’ll cross-check the official ledger with account notifications, resolve any mismatch, and state the freshest verified status and time."},
        {"variant_id":"processing_timeline_refresh","label":"processing timeline plus refresh point","action":"show the current processing stage and the next refresh checkpoint","body":"I’ll show the current processing stage, the most recent verified update, and the next refresh checkpoint rather than relying on a generic timing estimate."},
    ],
    "software_collect_diagnostics_before_fix": [
        {"variant_id":"environment_error_repro","label":"environment, error, reproduction","action":"collect version, operating system, exact error, and reproduction path first","body":"I’ll first collect the product version, operating system, exact displayed error, and reproduction path, then use those details for a focused diagnosis."},
        {"variant_id":"clean_profile_reproduction","label":"clean-profile reproduction","action":"reproduce in a clean profile before changing the main setup","body":"I’ll first reproduce the issue in a clean profile, compare it with the main setup, and then target the difference instead of changing the production environment blindly."},
        {"variant_id":"consented_diagnostic_bundle","label":"consented minimal diagnostic bundle","action":"collect a minimal diagnostic bundle with consent","body":"I’ll request consent for a minimal diagnostic bundle containing the build, error, and relevant logs, then diagnose from that bundle without collecting unrelated data."},
        {"variant_id":"last_known_good_comparison","label":"last-known-good comparison","action":"compare current behavior with the last known good state and relevant event logs","body":"I’ll compare the current behavior with the last known good state and the relevant event logs, then isolate the smallest changed condition."},
    ],
    "software_docs_lookup_for_update_install": [
        {"variant_id":"official_docs_current_version","label":"official current-version documentation","action":"verify official documentation for the installed version","body":"I’ll verify the official documentation for the installed version, cite the applicable support article, and separate confirmed instructions from assumptions."},
        {"variant_id":"built_in_help_then_online","label":"built-in help then online lookup","action":"check built-in help and release notes first, then go online if they do not match","body":"I’ll check the installed product’s built-in help and release notes first, then use the official online documentation if the local information does not match."},
        {"variant_id":"kb_changelog_crosscheck","label":"knowledge-base and changelog cross-check","action":"cross-check the vendor knowledge base with the changelog","body":"I’ll cross-check the vendor knowledge base with the version changelog and use only steps that apply to the current build and platform."},
        {"variant_id":"compatibility_known_issues","label":"compatibility matrix and known issues","action":"verify the compatibility matrix and current known-issues page","body":"I’ll verify the compatibility matrix and current known-issues page, then provide the platform-specific path that matches the installed version."},
    ],
    "software_one_try_then_escalate": [
        {"variant_id":"one_attempt_then_ticket","label":"one focused attempt then ticket","action":"try one focused next step and then escalate with the result","body":"I’ll try one focused next step; if it fails, I’ll preserve the result and escalate the same issue with the evidence rather than cycling through broad fixes."},
        {"variant_id":"two_distinct_attempts_then_ticket","label":"two distinct attempts then ticket","action":"allow two non-redundant focused attempts before escalation","body":"I’ll allow two non-redundant focused attempts, record both results, and then escalate with the evidence if the issue remains."},
        {"variant_id":"reproducible_blocker_immediate_escalation","label":"immediate escalation for reproducible blockers","action":"escalate immediately once the blocker is reproducible and evidence is complete","body":"Once the blocker is reproducible and the evidence bundle is complete, I’ll escalate immediately instead of requiring another self-service attempt."},
        {"variant_id":"timeboxed_attempt_then_handoff","label":"time-boxed attempt then handoff","action":"time-box one diagnostic attempt and hand off when the limit is reached","body":"I’ll time-box one diagnostic attempt, record the outcome, and hand the case to support when that limit is reached rather than extending the loop."},
    ],
    "software_secure_login_password_flow": [
        {"variant_id":"in_app_secure_reset","label":"in-app secure reset","action":"use the in-app reset and never request credentials in chat","body":"I’ll use the in-app secure reset flow and will never request a password, PIN, or one-time code in chat."},
        {"variant_id":"verified_browser_device_approval","label":"verified browser plus device approval","action":"use the verified browser portal and device approval","body":"I’ll use the verified browser recovery portal and require device approval, keeping every credential and one-time code out of the conversation."},
        {"variant_id":"authenticated_app_callback","label":"authenticated-app callback","action":"initiate a recovery callback from the authenticated app","body":"I’ll initiate a recovery callback from the authenticated app and use only masked account references in the chat."},
        {"variant_id":"passkey_security_key_recovery","label":"passkey or security-key recovery","action":"try passkey recovery first and use the security-key fallback","body":"I’ll try passkey recovery first and use the registered security-key fallback if needed, without collecting reusable secrets in chat."},
    ],
    "software_platform_specific_steps": [
        {"variant_id":"numbered_menu_steps","label":"numbered menu steps","action":"give short numbered menu steps with checkpoints","body":"I’ll give short numbered steps tailored to the current platform, naming the menus and adding a checkpoint after each major stage."},
        {"variant_id":"phase_checklist","label":"compact phase checklist","action":"organize the setup into a compact phase checklist","body":"I’ll organize the setup into a compact phase checklist for preparation, installation, configuration, and verification."},
        {"variant_id":"step_result_rollback_table","label":"step/result/rollback table","action":"use a table with step, expected result, and rollback column","body":"I’ll use a concise table with each step, its expected result, and the rollback action if that checkpoint fails."},
        {"variant_id":"gui_command_pairing","label":"GUI and command pairing","action":"pair each GUI path with an equivalent command where available","body":"I’ll pair each platform-specific GUI path with an equivalent command where one is available, and include a verification checkpoint for both."},
    ],
    "software_backup_before_risky_change": [
        {"variant_id":"full_backup_restore_point","label":"full backup and restore point","action":"verify a full backup or restore point before approval","body":"I’ll verify a full backup or restore point, summarize the change risk, and obtain final approval before the risky change begins."},
        {"variant_id":"config_export_restore_test","label":"configuration export and restore test","action":"export configuration and test that it can be restored","body":"I’ll export the application configuration, test that the export can be restored, and then request approval for the change."},
        {"variant_id":"sandbox_dry_run_snapshot","label":"sandbox dry run and snapshot","action":"run the change in a sandbox and take a snapshot before production","body":"I’ll run the change in a sandbox, capture a verified snapshot, and obtain approval before applying the same operation to the main environment."},
        {"variant_id":"staged_rollout_checkpoint","label":"staged rollout with rollback checkpoint","action":"use a staged rollout with an explicit rollback checkpoint","body":"I’ll use a staged rollout, define the rollback checkpoint and success criteria, and wait for approval before moving beyond the first stage."},
    ],
    "software_ticket_receipt_summary": [
        {"variant_id":"concise_prose_receipt","label":"concise prose receipt","action":"provide a concise prose receipt with issue, evidence, owner, and next update","body":"I’ll provide a concise prose receipt containing the issue, reproduction evidence, priority, reference, owner, and next expected update."},
        {"variant_id":"structured_receipt_table","label":"structured receipt table","action":"provide a structured table for the submitted case","body":"I’ll provide a structured receipt table with the issue, environment, reproduction steps, attachments, priority, reference, owner, and next update."},
        {"variant_id":"timeline_handoff_receipt","label":"timeline and handoff receipt","action":"provide a short timeline followed by the handoff details","body":"I’ll provide a short incident timeline followed by the submitted-case reference, current owner, attachments, and the next handoff checkpoint."},
        {"variant_id":"machine_readable_plus_note","label":"machine-readable block plus human note","action":"provide a compact machine-readable case block and a short human summary","body":"I’ll provide a compact machine-readable case block for reuse, followed by a short human summary of the issue, owner, and next update."},
    ],
}

ACTIVE_VARIANT_ASSIGNMENTS: dict[tuple[str, str], int] = {}
OLD_VARIANT_ASSIGNMENTS: dict[tuple[str, str], int] = {}


def build_variant_assignments(personas: list[Any], assignments: dict[str, list[str]]) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int]]:
    active_map: dict[tuple[str, str], int] = {}
    old_map: dict[tuple[str, str], int] = {}
    for hid in RETAINED_HABIT_IDS:
        users = sorted(p.user_id for p in personas if hid in assignments[p.user_id])
        offset = stable_index(hid, "variant-offset", mod=4)
        for rank, uid in enumerate(users):
            active = (rank + offset) % 4
            active_map[(uid, hid)] = active
            # Rotate among the other three variants; this is balanced and never
            # identical to the active workflow.
            old_map[(uid, hid)] = (active + 1 + stable_index(uid, hid, "old-variant", mod=3)) % 4
            if old_map[(uid, hid)] == active:
                old_map[(uid, hid)] = (active + 1) % 4
    return active_map, old_map


def active_variant_index(profile: Any, hid: str) -> int:
    return ACTIVE_VARIANT_ASSIGNMENTS[(profile.user_id, hid)]


def old_variant_index(profile: Any, hid: str) -> int:
    return OLD_VARIANT_ASSIGNMENTS.get((profile.user_id, hid), (active_variant_index(profile, hid) + 1) % 4)


def variant_record(profile: Any, hid: str, mode: str = "active") -> dict[str, str]:
    active = active_variant_index(profile, hid)
    if mode in {"active", "default", "new"}:
        idx = active
    elif mode == "old":
        idx = old_variant_index(profile, hid)
    elif mode == "partial":
        idx = next(i for i in range(4) if i not in {active, old_variant_index(profile, hid)})
    elif mode == "unsafe":
        used = {active, old_variant_index(profile, hid), next(i for i in range(4) if i not in {active, old_variant_index(profile, hid)})}
        idx = next(i for i in range(4) if i not in used)
    else:
        idx = active
    return POLICY_VARIANTS[hid][idx]


def variant_summary(profile: Any, hid: str, mode: str = "active") -> str:
    return variant_record(profile, hid, mode)["action"]


def user_priority_order(profile: Any, hids: list[str]) -> list[str]:
    # Priority probes only use same-family workflows, so ordering can be a
    # genuine, safe user-specific habit rather than a universal safety answer.
    return sorted(hids, key=lambda h: stable_index(profile.user_id, h, "workflow-order", mod=10**9))


def priority_group(active: list[str], user_id: str, n: int) -> list[str]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for hid in active:
        by_family[HABIT_BY_ID[hid]["family"]].append(hid)
    pools = [v for v in by_family.values() if len(v) >= n and HABIT_BY_ID[v[0]]["family"] != "privacy_consent"]
    if not pools:
        non_privacy = [h for h in active if HABIT_BY_ID[h]["family"] != "privacy_consent"]
        pools = [non_privacy if len(non_privacy) >= n else active]
    pool = sorted(pools, key=lambda vals: stable_index(user_id, "priority-pool", *sorted(vals), mod=10**9))[0]
    start = stable_index(user_id, f"priority-{n}", mod=len(pool))
    return [pool[(start+i) % len(pool)] for i in range(n)]
# Broaden the recurring-issue pattern to match the actual source dialogues.
HABIT_BY_ID["software_one_try_then_escalate"]["pattern"] = (
    r"ticket|agent|human|representative|escalat|supervisor|support team|"
    r"still.{0,30}(?:not working|failing|broken)|(?:not working|error|issue).{0,35}again|"
    r"recurr|repeated|multiple times|keeps (?:failing|freezing|crashing)"
)

PAIR_CANDIDATES = {
    "finance": [
        ("finance_payment_status_latest_check", "finance_confirm_money_movement"),
        ("finance_minimal_pii_secure_verification", "finance_fraud_lost_card_urgent_escalation"),
        ("finance_minimal_pii_secure_verification", "finance_confirm_card_account_changes"),
        ("finance_fee_dispute_evidence_then_case", "finance_minimal_pii_secure_verification"),
        ("finance_credit_loan_cautious_no_commitment", "finance_minimal_pii_secure_verification"),
        ("finance_balance_statement_summary_first", "finance_payment_status_latest_check"),
        ("finance_fee_dispute_evidence_then_case", "finance_payment_status_latest_check"),
    ],
    "software": [
        ("software_docs_lookup_for_update_install", "software_backup_before_risky_change"),
        ("software_secure_login_password_flow", "software_one_try_then_escalate"),
        ("software_collect_diagnostics_before_fix", "software_one_try_then_escalate"),
        ("software_platform_specific_steps", "software_docs_lookup_for_update_install"),
        ("software_secure_login_password_flow", "software_collect_diagnostics_before_fix"),
        ("software_ticket_receipt_summary", "software_one_try_then_escalate"),
        ("software_backup_before_risky_change", "software_collect_diagnostics_before_fix"),
    ],
}

# ----------------------------------------------------------------------------
# Persona profiles and stable continuity facts.
# ----------------------------------------------------------------------------

@dataclass
class Persona:
    user_id: str
    domain: str
    name: str
    first_name: str
    pronouns: str
    city: str
    state: str
    role: str
    company: str
    email: str
    phone: str
    voice_style: str
    checking_last4: str = ""
    savings_last4: str = ""
    card_last4: str = ""
    account_last4: str = ""
    os: str = ""
    browser: str = ""
    desktop_app: str = ""
    mail_app: str = ""
    meeting_app: str = ""
    plan: str = ""
    recurring_context: str = ""
    monthly_anchor: str = ""
    organization_size: str = ""

NAMES = [
    "Maya Chen", "Jordan Brooks", "Elena Ruiz", "Noah Patel", "Avery Thompson", "Lena Okafor",
    "Daniel Kim", "Sofia Martinez", "Ethan Walker", "Priya Shah", "Marcus Lee", "Nina Alvarez",
    "Owen Foster", "Camila Santos", "Theo Nguyen", "Leila Hassan", "Julian Reed", "Amara Johnson",
    "Miles Carter", "Zoe Bennett", "Iris Park", "Caleb Morgan", "Fatima Rahman", "Leo Castillo",
    "Harper Evans", "Mateo Rivera", "Grace Liu", "Samuel Adeyemi", "Chloe Dubois", "Elias Turner",
    "Aisha Bello", "Rowan Murphy", "Isabel Costa", "Henry Zhao", "Mei Tan", "Layla Williams",
    "Finn O'Connor", "Mei Lin", "Adrian Flores", "Sara Ibrahim", "Ben Cooper", "Yara Khalil",
    "Lucas Martin", "Eva Novak", "Rina Das",
]
CITIES = [
    ("Portland", "OR"), ("Austin", "TX"), ("Raleigh", "NC"), ("Denver", "CO"),
    ("Madison", "WI"), ("Sacramento", "CA"), ("Columbus", "OH"), ("Richmond", "VA"),
    ("Pittsburgh", "PA"), ("Tucson", "AZ"), ("Boise", "ID"), ("Minneapolis", "MN"),
]
FIN_ROLES = [
    "freelance designer", "high-school teacher", "clinic coordinator", "small-business owner",
    "graduate student", "operations consultant", "nonprofit manager", "retired librarian",
    "restaurant manager", "research assistant", "construction estimator", "event planner",
]
SOF_ROLES = [
    "product manager", "QA analyst", "marketing coordinator", "data analyst", "software developer",
    "customer-success lead", "graphic designer", "researcher", "project coordinator", "IT generalist",
    "content editor", "operations manager",
]
COMPANIES = [
    "Cedar Loop Studio", "Northstar Analytics", "Juniper Works", "Blue Harbor Labs",
    "Paper Kite Media", "Willow Ridge Health", "Copper Finch Design", "Orchard Field Group",
    "Lighthouse Learning", "Mosaic River Consulting", "Brightline Research", "Crescent Oak Co.",
]
VOICE_STYLES = ["concise", "conversational", "careful", "direct", "detail-aware"]
FIN_CONTEXTS = [
    "a quarterly household budget close", "a freelance tax reserve review", "a tuition payment plan",
    "a clinic reimbursement reconciliation", "a small-business cash-flow check", "an annual insurance review",
    "a nonprofit grant-spending review", "a retirement income reconciliation", "a vendor-payment cycle",
    "a research travel reimbursement close", "a renovation budget", "an event deposit schedule",
]
SOF_CONTEXTS = [
    "a client onboarding rollout", "a release-candidate test", "a reporting migration", "an accessibility review",
    "a remote-work setup", "a customer-demo environment", "a design-system update", "a research data handoff",
    "a new-hire setup", "a quarterly archive migration", "an editorial workflow change", "a support dashboard rollout",
]

def make_personas() -> list[Persona]:
    out: list[Persona] = []
    for i in range(FINANCE_USER_COUNT):
        name = NAMES[i]
        first, last = name.split()[0], name.split()[-1]
        city, state = CITIES[i % len(CITIES)]
        out.append(Persona(
            user_id=f"mdgo_v05_fin_user_{i:04d}", domain="finance", name=name, first_name=first,
            pronouns=["they/them", "she/her", "he/him"][i % 3], city=city, state=state,
            role=FIN_ROLES[i % len(FIN_ROLES)], company="", email=f"{re.sub(r'[^a-z0-9]', '', first.lower())}.{re.sub(r'[^a-z0-9]', '', last.lower())}@persona.example",
            phone=f"+1-202-555-{1100+i:04d}", voice_style=VOICE_STYLES[i % len(VOICE_STYLES)],
            checking_last4=f"{4120+i*17:04d}"[-4:], savings_last4=f"{7310+i*19:04d}"[-4:],
            card_last4=f"{8610+i*23:04d}"[-4:], account_last4=f"{4120+i*17:04d}"[-4:],
            recurring_context=FIN_CONTEXTS[i % len(FIN_CONTEXTS)], monthly_anchor=["the 3rd", "the 8th", "the 15th", "the last business day"][i % 4],
        ))
    for i in range(SOFTWARE_USER_COUNT):
        name = NAMES[FINANCE_USER_COUNT+i]
        first, last = name.split()[0], name.split()[-1]
        city, state = CITIES[i % len(CITIES)]
        os_name = ["Windows 11", "macOS 15", "Ubuntu 24.04"][i % 3]
        out.append(Persona(
            user_id=f"mdgo_v05_sof_user_{i:04d}", domain="software", name=name, first_name=first,
            pronouns=["they/them", "she/her", "he/him"][i % 3], city=city, state=state,
            role=SOF_ROLES[i % len(SOF_ROLES)], company=COMPANIES[i % len(COMPANIES)], email=f"{re.sub(r'[^a-z0-9]', '', first.lower())}.{re.sub(r'[^a-z0-9]', '', last.lower())}@persona.example",
            phone=f"+1-202-555-{1200+i:04d}", voice_style=VOICE_STYLES[(i+2) % len(VOICE_STYLES)],
            account_last4=f"{5210+i*29:04d}"[-4:], os=os_name,
            browser=["Chrome", "Firefox", "Edge", "Safari"][i % 4],
            desktop_app=["AsterDesk", "Nimbus Workbench", "Orbit Office"][i % 3],
            mail_app=["Aster Mail", "Nimbus Mail", "Orbit Inbox"][i % 3],
            meeting_app=["Aster Meet", "Nimbus Call", "Orbit Rooms"][i % 3],
            plan=["individual", "professional", "team"][i % 3],
            recurring_context=SOF_CONTEXTS[i % len(SOF_CONTEXTS)], organization_size=["8-person", "14-person", "22-person", "35-person"][i % 4],
        ))
    return out

# ----------------------------------------------------------------------------
# Source loading, task-event extraction, clustering and retention audit.
# ----------------------------------------------------------------------------

GENERIC = re.compile(r"^(?:hi|hello|hey|yes|no|ok|okay|thanks|thank you|bye|good morning|good evening|nothing else)[.! ,:-]*$", re.I)
PII = re.compile(r"ssn|social security|account number|card number|password|otp|pin|full name|my name|address|phone|email|company name", re.I)
SOFTWARE_CORE = re.compile(
    r"\b(?:software|application|app|install|uninstall|update|upgrade|version|patch|browser|windows|mac|ubuntu|linux|"
    r"error|bug|crash|freeze|login|password|file|folder|backup|restore|rollback|reset|server|sync|ticket|configuration|"
    r"network|computer|laptop|desktop|cloud|outlook|email|mail|skype|video|meeting|camera|microphone)\b", re.I)
SOFTWARE_NOISE = re.compile(
    r"musical instrument|music center|keyboard order|guitar|piano|drum|flute|saxophone|yamaha|\bpsr[- ]?[a-z0-9]*|"
    r"food|pizza|burger|flight|airline|hotel|restaurant|room booking|car rental|reimburse|travel expense|"
    r"checking account|savings account|bank statement|mortgage|personal loan|credit score|cash withdrawal|fund transfer|"
    r"debit card|credit card|overdraft", re.I)


def stable_index(*parts: str, mod: int) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:16], 16) % mod


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_conversations(path: Path, domain: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by[row["conversationId"]].append({
                "conversation_id": row["conversationId"],
                "turn": int(row.get("turnNumber") or 0),
                "role": "user" if row.get("authorRole") == "customer" else "assistant",
                "text": clean(row.get("utterance") or ""),
                "domain": domain,
            })
    for rows in by.values():
        rows.sort(key=lambda x: x["turn"])
    return dict(by)


def source_task_text(rows: list[dict[str, Any]]) -> str:
    vals: list[tuple[int, str]] = []
    for r in rows:
        if r["role"] != "user":
            continue
        t = clean(r["text"])
        if not t or GENERIC.match(t) or PII.search(t):
            continue
        words = re.findall(r"[A-Za-z]+", t)
        if len(words) < 3:
            continue
        score = len(words)
        if re.search(r"need|want|help|can|could|how|why|what|lost|error|issue|change|check|transfer|balance|update|install|password|login|charge|fee|loan|statement|refund|server", t, re.I):
            score += 12
        vals.append((score, t))
    vals.sort(reverse=True)
    chosen = [v for _, v in vals[:3]]
    text = " ".join(chosen)
    text = re.sub(r"\b\d{4,18}\b", "[number]", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email]", text)
    return clean(text)[:700]


def classify_tags(text: str, domain: str) -> list[str]:
    tags = []
    for h in HABITS_BY_DOMAIN[domain]:
        if re.search(h["pattern"], text, re.I):
            tags.append(h["habit_id"])
    return tags


def source_quality(rows: list[dict[str, Any]], domain: str, task_text: str, tags: list[str]) -> float:
    user_turns = [r for r in rows if r["role"] == "user"]
    n_words = len(re.findall(r"[A-Za-z]+", task_text))
    turn_score = min(len(rows), 24) / 24
    word_score = min(n_words, 80) / 80
    tag_score = min(len(tags), 3) / 3
    q = 0.35 * turn_score + 0.35 * word_score + 0.30 * tag_score
    if len(user_turns) < 2:
        q -= .12
    return round(max(0.0, q), 4)


def build_source_records(path: Path, domain: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = load_conversations(path, domain)
    records: dict[str, dict[str, Any]] = {}
    for cid, rows in raw.items():
        if not 5 <= len(rows) <= 45:
            continue
        full = " ".join(r["text"] for r in rows)
        if domain == "software":
            if len(SOFTWARE_CORE.findall(full)) < 2 or SOFTWARE_NOISE.search(full):
                continue
        task = source_task_text(rows)
        tags = classify_tags(task + " " + full[:1500], domain)
        if not task or not tags:
            continue
        q = source_quality(rows, domain, task, tags)
        if q < .24:
            continue
        records[cid] = {
            "conversation_id": cid,
            "domain": domain,
            "task_text": task,
            "tags": tags,
            "turn_count": len(rows),
            "quality_score": q,
        }
    # Deterministic lexical-hash buckets diversify source allocation without
    # expensive clustering. Buckets are not claimed to be semantic clusters;
    # they are reproducible source strata used only for coverage auditing.
    n_clusters = CLUSTERS_PER_DOMAIN
    bucket_terms: dict[int, Counter[str]] = defaultdict(Counter)
    counts: Counter[int] = Counter()
    stop = {"the","and","for","that","this","with","from","have","has","was","were","are","you","your","can","could","would","please","need","want","help","about","into","when","what","how","why","my","our","but","not"}
    for cid, rec in records.items():
        normalized = normalize_text(rec["task_text"])
        primary_tag = rec["tags"][0] if rec["tags"] else "untagged"
        lexical_key = " ".join(normalized.split()[:18])
        bucket = stable_index(domain, primary_tag, lexical_key, mod=n_clusters)
        rec["cluster_id"] = int(bucket)
        counts[bucket] += 1
        for token in re.findall(r"[a-z]{3,}", normalized):
            if token not in stop:
                bucket_terms[bucket][token] += 1
    summaries = []
    for c in range(n_clusters):
        summaries.append({
            "domain": domain,
            "cluster_id": c,
            "conversation_count": int(counts[c]),
            "top_terms": [term for term, _ in bucket_terms[c].most_common(14)],
            "stratification_method": "deterministic lexical hash bucket",
        })
    manifest = {
        "source_file": path.name,
        "sha256": sha256_file(path),
        "raw_conversations": len(raw),
        "retained_task_events": len(records),
        "clusters": n_clusters,
        "stratification_method": "deterministic lexical hash buckets (not semantic clustering)",
        "raw_utterances": sum(len(v) for v in raw.values()),
        "cluster_summaries": summaries,
    }
    return records, manifest

# ----------------------------------------------------------------------------
# Habit assignment and long-horizon evidence schedule.
# ----------------------------------------------------------------------------


def habit_assignments(domain: str, user_index: int) -> list[str]:
    ids = [h["habit_id"] for h in HABITS_BY_DOMAIN[domain]]
    n_active = 6
    start = (user_index * 3) % len(ids)
    chosen = []
    for j in range(n_active):
        idx = (start + j * 2 + user_index // max(1, len(ids))) % len(ids)
        while ids[idx] in chosen:
            idx = (idx + 1) % len(ids)
        chosen.append(ids[idx])
    # Ensure privacy/safety and tool/workflow are both present.
    if not any(HABIT_BY_ID[h]["family"] == "privacy_consent" for h in chosen):
        privacy = [h["habit_id"] for h in HABITS_BY_DOMAIN[domain] if h["family"] == "privacy_consent"][0]
        chosen[-1] = privacy
    if not any(HABIT_BY_ID[h]["family"] == "tool_action" for h in chosen):
        tool = [h["habit_id"] for h in HABITS_BY_DOMAIN[domain] if h["family"] == "tool_action"][0]
        chosen[-2] = tool
    return list(dict.fromkeys(chosen))[:n_active]


def choose_drift_habits(active: list[str], user_id: str) -> list[str]:
    candidates = [h for h in active if HABIT_BY_ID[h]["family"] in {"drift_seasonality", "format_style", "tool_action", "risk_threshold"}]
    if len(candidates) < 2:
        candidates = active
    start = stable_index(user_id, "drift", mod=len(candidates))
    return [candidates[start], candidates[(start + max(1, len(candidates)//2)) % len(candidates)]]


def choose_scoped_habit(active: list[str], user_id: str) -> str:
    low = [h for h in active if HABIT_BY_ID[h]["family"] in {"format_style", "tool_action"}]
    return low[stable_index(user_id, "scope", mod=len(low))] if low else active[0]


def evidence_schedule(active: list[str], drift_habits: list[str], user_id: str) -> dict[int, list[dict[str, Any]]]:
    n = TOTAL_SESSIONS_PER_USER - 1
    plan: dict[int, list[dict[str, Any]]] = defaultdict(list)
    occupied: Counter[int] = Counter()

    def place(base: int, item: dict[str, Any]) -> int:
        # deterministic local search avoids too many annotations in one session
        for delta in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
            p = max(1, min(n, base + delta))
            if occupied[p] < 2:
                plan[p].append(item); occupied[p] += 1; return p
        plan[base].append(item); occupied[base] += 1; return base

    for hi, hid in enumerate(active):
        jitter = stable_index(user_id, hid, "j", mod=7) - 3
        if hid in drift_habits:
            for frac, kind in [(0.06, "old"), (0.17, "old"), (0.66, "new"), (0.75, "new"), (0.86, "new"), (0.95, "new")]:
                place(int(n*frac)+jitter, {"kind": kind, "habit_ids": [hid]})
            place(int(n*.40)+jitter, {"kind": "boundary", "habit_ids": [hid]})
            place(int(n*.91)+jitter, {"kind": "exception", "habit_ids": [hid]})
        else:
            for frac in (0.08, 0.28, 0.52, 0.74):
                place(int(n*frac)+jitter, {"kind": "support", "habit_ids": [hid]})
            place(int(n*.39)+jitter, {"kind": "boundary", "habit_ids": [hid]})
            place(int(n*.63)+jitter, {"kind": "exception", "habit_ids": [hid]})
            place(int(n*.88)+jitter, {"kind": "support", "habit_ids": [hid]})

    pairs = [p for p in PAIR_CANDIDATES[HABIT_BY_ID[active[0]]["domain"]] if p[0] in active and p[1] in active]
    if len(pairs) < 3:
        extra = [(active[i], active[(i+1)%len(active)]) for i in range(len(active))]
        pairs.extend([p for p in extra if p not in pairs and tuple(reversed(p)) not in pairs])
    for j, pair in enumerate(pairs[:5]):
        place(int(n*(.34 + .11*j)), {"kind": "composition", "habit_ids": list(pair)})
    priority_pair = priority_group(active, user_id, 2)
    for j, frac in enumerate((.58, .70, .82)):
        rotated = priority_pair[j % len(priority_pair):] + priority_pair[:j % len(priority_pair)]
        place(int(n*frac), {"kind": "priority", "habit_ids": list(rotated)})
    # Three-way evidence moments near the later half create stronger interference.
    triple = active[:3]
    place(int(n*.81), {"kind": "composition", "habit_ids": triple})
    p_triple = priority_group(active, user_id, 3)
    place(int(n*.90), {"kind": "priority", "habit_ids": p_triple})
    return plan

# ----------------------------------------------------------------------------
# Source allocation and canonical, identity-coherent session generation.
# ----------------------------------------------------------------------------


def allocate_sources(records: dict[str, dict[str, Any]], profiles: list[Persona], assignments: dict[str, list[str]], rng: random.Random) -> dict[str, list[str]]:
    """Assign every retained source conversation exactly once.

    The first pass guarantees per-habit coverage for every pseudo-user. The
    second pass distributes all remaining source events to balanced per-user
    targets. No source conversation is reused or silently dropped.
    """
    unused = set(records)
    by_tag: dict[str, list[str]] = defaultdict(list)
    for cid, rec in records.items():
        for tag in rec["tags"]:
            by_tag[tag].append(cid)
    for hid, pool in by_tag.items():
        pool.sort(key=lambda c: (-records[c]["quality_score"], stable_index(hid, c, mod=10**12)))

    profiles = sorted(profiles, key=lambda p: p.user_id)
    base, remainder = divmod(len(records), len(profiles))
    targets = {p.user_id: base + (1 if i < remainder else 0) for i, p in enumerate(profiles)}
    out: dict[str, list[str]] = {p.user_id: [] for p in profiles}
    tag_cursor: Counter[str] = Counter()

    # Round-robin coverage prevents early users from exhausting rare habits.
    coverage_rounds = 10
    for _ in range(coverage_rounds):
        for profile in profiles:
            if len(out[profile.user_id]) >= targets[profile.user_id]:
                continue
            for hid in assignments[profile.user_id]:
                pool = by_tag.get(hid, [])
                cursor = tag_cursor[hid]
                while cursor < len(pool) and pool[cursor] not in unused:
                    cursor += 1
                tag_cursor[hid] = cursor
                if cursor < len(pool) and len(out[profile.user_id]) < targets[profile.user_id]:
                    cid = pool[cursor]
                    out[profile.user_id].append(cid)
                    unused.remove(cid)
                    tag_cursor[hid] += 1

    # Distribute the full retained remainder while balancing source buckets and
    # quality. Deterministic shuffling avoids assigning contiguous corpus blocks.
    remaining = list(unused)
    remaining.sort(key=lambda c: (records[c]["cluster_id"], -records[c]["quality_score"], stable_index(c, "full-source", mod=10**12)))
    rng.shuffle(remaining)
    profile_cycle = deque(profiles)
    for cid in remaining:
        for _ in range(len(profile_cycle)):
            profile = profile_cycle[0]
            profile_cycle.rotate(-1)
            if len(out[profile.user_id]) < targets[profile.user_id]:
                out[profile.user_id].append(cid)
                unused.remove(cid)
                break
        else:
            raise RuntimeError("no source capacity remains while source rows are unassigned")

    if unused:
        raise RuntimeError(f"unassigned retained source conversations: {len(unused)}")
    if sum(len(v) for v in out.values()) != len(records):
        raise RuntimeError("full-source allocation count mismatch")
    for profile in profiles:
        if len(out[profile.user_id]) != targets[profile.user_id]:
            raise RuntimeError(f"source allocation target mismatch for {profile.user_id}")
        rng.shuffle(out[profile.user_id])
    return out


def context(profile: Persona, key: str, idx: int) -> dict[str, Any]:
    base = f"{profile.user_id}|{key}|{idx}"
    amounts = [42, 68, 95, 125, 175, 240, 315, 480, 725, 980, 1350]
    merchants = ["Harbor Market", "Northline Books", "Juniper Transit", "Cedar Energy", "Mosaic Telecom", "Brightpath Clinic"]
    payees = ["monthly rent", "a tax reserve", "a project vendor", "the utility bill", "a tuition installment", "an insurance premium"]
    errors = ["E-104", "SYNC-27", "AUTH-31", "UPD-58", "FILE-42", "NET-19", "UI-73"]
    features = ["shared workspace", "mail sync", "calendar integration", "video meeting", "export tool", "desktop sign-in", "file upload"]
    projects = [profile.recurring_context, "the next client checkpoint", "a month-end close", "a training rehearsal", "the release review"]
    return {
        "amount": amounts[stable_index(base, "amount", mod=len(amounts))],
        "merchant": merchants[stable_index(base, "merchant", mod=len(merchants))],
        "payee": payees[stable_index(base, "payee", mod=len(payees))],
        "error": errors[stable_index(base, "error", mod=len(errors))],
        "feature": features[stable_index(base, "feature", mod=len(features))],
        "project": projects[stable_index(base, "project", mod=len(projects))],
        "case_ref": 2000 + stable_index(base, "case", mod=7000),
        "version": ["8.4.2", "8.5.0", "9.0.1", "2026.3", "2026.5"][stable_index(base, "version", mod=5)],
        "date_label": ["Friday", "the 15th", "month-end", "tomorrow afternoon", "the next business day"][stable_index(base, "date", mod=5)],
    }


def source_theme_for_session(rec: dict[str, Any], active: list[str], sid: str) -> str:
    matching = [t for t in rec["tags"] if t in active]
    if matching:
        return matching[stable_index(sid, "theme", mod=len(matching))]
    domain_tags = [t for t in rec["tags"] if t in HABIT_BY_ID]
    return domain_tags[stable_index(sid, "theme2", mod=len(domain_tags))]


def _strip_policy_suffix(text: str) -> str:
    # v0.4 added a final provenance-style sentence to every policy response.
    # Strip the entire final sentence rather than stopping at a decimal point in
    # versions such as Ubuntu 24.04.
    patterns = [
        r"\s+I’ll keep the result tied to .+$",
        r"\s+I’ll reference account ending \d{4} where needed\.?$",
        r"\s+I’ll keep the concrete .+ context visible\.?$",
        r"\s+I’ll keep the steps specific to .+$",
        r"\s+I’ll tie the result to .+$",
        r"\s+I’ll preserve the concrete .+ context\.?$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text)
    return clean(text)


def _session_context_key(profile: Persona, idx: int) -> str:
    return f"mdgo_v05_session_{profile.user_id}_{idx:06d}"


def _probe_context_key(value: str) -> str:
    m = re.search(r"mdgo_v05_probe_\d{6}", value)
    return m.group(0) if m else value


def timeline_marker(profile: Persona, idx: int) -> str:
    date = datetime(2025, 1, 5) + timedelta(days=idx * 3)
    week = 1 + (date.day - 1) // 7
    ordinals = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    period = f"{date.strftime('%B %Y')}"
    if profile.domain == "finance":
        variants = [
            f"This belongs in my {period} reconciliation records.",
            f"It came up during the {ordinals[week]} week of {period}.",
            f"I need the result before I close the {period} review cycle.",
            f"I am revisiting this for the {period} checkpoint in {profile.recurring_context}.",
        ]
    else:
        variants = [
            f"This belongs to the {period} support record at {profile.company}.",
            f"It surfaced during the {ordinals[week]} week of {period}.",
            f"I need a reproducible result for the {period} team checkpoint.",
            f"I am revisiting this during the {period} phase of {profile.recurring_context}.",
        ]
    return variants[stable_index(profile.user_id, str(idx), "timeline", mod=len(variants))]


def task_request(hid: str, profile: Persona, idx: int, mode: str = "direct") -> str:
    key = _session_context_key(profile, idx)
    mode_map = {"support": "direct", "old": "direct", "new": "direct", "priority": "direct", "composition": "direct"}
    m = mode_map.get(mode, mode if mode in {"direct", "boundary", "exception"} else "direct")
    try:
        q = lib.scenario_sentence(hid, profile, key, m, f"source-event-{profile.user_id}-{idx}-{hid}")
    except Exception:
        h = HABIT_BY_ID[hid]
        q = f"I need help because {h['condition']}."
    q = re.sub(r"\s+What should the assistant.*$", "", q, flags=re.I)
    return clean(q + " " + timeline_marker(profile, idx))


def _variant_response_text(hid: str, profile: Persona, context_key: str, mode: str) -> str:
    if (profile.user_id, hid) in ACTIVE_VARIANT_ASSIGNMENTS:
        rec = variant_record(profile, hid, mode)
    else:
        # Source-grounded background sessions may involve a domain habit that is
        # not one of this user's six stable habits. Vary the safe workflow across
        # those sessions so it cannot become accidental repeated preference evidence.
        rec = POLICY_VARIANTS[hid][stable_index(profile.user_id, hid, context_key, mode, "background", mod=4)]
    openers = [
        "I’ll use the workflow we have established for this kind of case.",
        "For this request, I’ll follow the recurring process from our earlier sessions.",
        "I’ll keep this aligned with the way you have handled the same condition before.",
        "I’ll apply the matching workflow and preserve a clear audit trail.",
    ]
    closers = [
        "I’ll keep verified facts separate from assumptions in the record.",
        "The final note will preserve the decision boundary and next checkpoint.",
        "I’ll document what was checked and what remains unresolved.",
        "The result will stay scoped to this request and its current state.",
    ]
    opener = openers[stable_index(profile.user_id, hid, context_key, mode, "opener", mod=len(openers))]
    closer = closers[stable_index(profile.user_id, hid, context_key, mode, "closer", mod=len(closers))]
    return clean(f"{opener} {rec['body']} {closer}")


def policy(hid: str, profile: Persona, idx: int, mode: str) -> str:
    key = _session_context_key(profile, idx)
    if mode in {"default", "new", "old", "partial", "unsafe"}:
        return _variant_response_text(hid, profile, key, mode)
    h = HABIT_BY_ID[hid]
    if mode == "boundary":
        return clean(f"This request falls outside the recurring workflow. The scoped action is: {h['boundary_action']}.")
    if mode == "exception":
        return clean(f"The narrow exception applies in this case. The scoped action is: {h['exception_action']}.")
    ctx = lib.context_for_habit(hid, profile, key)
    return _strip_policy_suffix(lib.policy_response(hid, profile, ctx, mode, key + "_" + mode))


def feedback_text(profile: Persona, hid: str, kind: str, idx: int) -> str:
    h = HABIT_BY_ID[hid]
    active = variant_summary(profile, hid, "active")
    old = variant_summary(profile, hid, "old")
    variants: dict[str, list[str]] = {
        "support": [
            f"That sequence worked well for me. In similar cases, please keep using this workflow: {active}.",
            f"This matches how I want these requests handled. The important part is to {active}.",
            f"The result was easier to verify because you followed this pattern: {active}.",
            f"That ordering avoided the cleanup I had to do before. Reuse the same approach when {h['condition']}: {active}.",
            f"I could hand this result to someone else without reworking it. Keep the same workflow next time: {active}.",
            f"This was the right balance of control and friction for me. Continue to {active} in matching cases.",
        ],
        "boundary": [
            f"This looks similar, but it is outside that workflow. When {h['boundary_condition']}, {h['boundary_action']}.",
            f"Please narrow the recurring rule to its real scope. In this case, {h['boundary_action']}.",
            f"The earlier process does not apply to this information-only or conceptual request. Here, {h['boundary_action']}.",
            f"That preference has a boundary I care about: when {h['boundary_condition']}, {h['boundary_action']}.",
        ],
        "exception": [
            f"This case hits the narrow override we discussed. When {h['exception_condition']}, {h['exception_action']}.",
            f"The prerequisite is already satisfied, so do not restart the default loop. {h['exception_action']}.",
            f"For this exact situation my normal workflow changes: {h['exception_action']}.",
            f"Use the exception rather than the default process here: if {h['exception_condition']}, {h['exception_action']}.",
        ],
        "old": [
            f"At this point I still prefer the earlier workflow: {old}.",
            f"For now, keep using this version of the process: {old}.",
            f"My current preference in these cases is to {old}.",
        ],
        "new": [
            f"I am replacing the earlier workflow. From now on, when {h['condition']}, {active}; do not continue using the older pattern where you {old}.",
            f"The old process no longer fits my needs. Going forward, use this instead: {active}. This supersedes the earlier approach to {old}.",
            f"After the last few cases, I am updating the rule. The current workflow is to {active}, not to {old}.",
            f"Please treat this as a lasting update: when {h['condition']}, {active}. The prior version—{old}—is retired.",
        ],
    }
    vals = variants[kind]
    return clean(vals[stable_index(profile.user_id, hid, kind, str(idx), mod=len(vals))])

def neutral_followup(profile: Persona, hid: str, idx: int) -> str:
    c = context(profile, hid, idx)
    if profile.domain == "finance":
        vals = [
            f"The item is part of {profile.recurring_context}, and I need a clear next step before {c['date_label']}.",
            f"I am reconciling this against checking ending {profile.checking_last4}, and I need the same details preserved in the final record.",
            f"Please keep the explanation tied to the account state rather than giving a generic policy page.",
            f"This came up during {profile.recurring_context}, so I need enough detail to verify the result later.",
        ]
    else:
        vals = [
            f"The affected setup is {profile.desktop_app} {c['version']} on {profile.os}, and this is blocking {profile.recurring_context}.",
            f"I can reproduce it in {profile.browser}; the visible code is {c['error']} and the team review is {c['date_label']}.",
            f"Please keep the steps specific to {profile.os} and the {c['feature']} rather than listing every possible fix.",
            f"The issue affects the {profile.organization_size} team at {profile.company}, so I need a verifiable next step.",
        ]
    return vals[stable_index(profile.user_id, hid, str(idx), "follow", mod=len(vals))]


def make_session_messages(profile: Persona, idx: int, primary_hid: str, annotations: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    # The source dialogue is used only as a task-event seed. All model-visible
    # utterances are newly written with the assigned persona and stable entities.
    if not annotations:
        request = task_request(primary_hid, profile, idx, "direct")
        initial_mode = "default"
        messages = [
            {"role": "user", "content": request},
            {"role": "assistant", "content": policy(primary_hid, profile, idx, initial_mode)},
            {"role": "user", "content": neutral_followup(profile, primary_hid, idx)},
            {"role": "assistant", "content": policy(primary_hid, profile, idx, "default")},
        ]
        if idx % 5 == 0:
            messages.extend([
                {"role": "user", "content": "That gives me enough to continue. I will check the result against the same details later."},
                {"role": "assistant", "content": "Understood. I have kept the next step tied to the details in this session."},
            ])
        return messages, []

    item = annotations[0]
    kind = item["kind"]
    hids = item["habit_ids"]
    # If two schedule items collide, integrate all habits in one coherent case.
    for extra in annotations[1:]:
        for hid in extra["habit_ids"]:
            if hid not in hids:
                hids.append(hid)
    primary = hids[0]
    request_parts = [task_request(h, profile, idx+j, "direct" if kind not in {"boundary", "exception"} else kind) for j, h in enumerate(hids)]
    joiners = ["There is a second part to the same request:", "At the same time,", "The same case also involves this:"]
    request = request_parts[0]
    for j, part in enumerate(request_parts[1:]):
        request += " " + joiners[(idx+j) % len(joiners)] + " " + part

    if kind in {"composition", "priority"}:
        ordered = user_priority_order(profile, hids) if kind == "priority" else sorted(hids, key=lambda h: HABIT_BY_ID[h]["priority"], reverse=True)
        wrong_order = list(reversed(ordered)) if kind == "priority" else ordered[:1]
        initial = " ".join(policy(h, profile, idx+j, "default") for j, h in enumerate(wrong_order))
        correction_parts = []
        for h in ordered:
            correction_parts.append(f"for the part where {HABIT_BY_ID[h]['condition']}, {variant_summary(profile, h, 'active')}")
        feedback = (
            "The workflows need to be combined rather than handled as unrelated tasks. "
            + "; then ".join(correction_parts)
            + (". Keep that user-specific sequence when these workflows collide again." if kind == "priority" else ".")
        )
        final = " ".join(policy(h, profile, idx+j, "default") for j, h in enumerate(ordered))
        messages = [
            {"role": "user", "content": request},
            {"role": "assistant", "content": initial},
            {"role": "user", "content": feedback},
            {"role": "assistant", "content": final},
            {"role": "user", "content": "Yes—that ordering is what I want used when these conditions collide again."},
            {"role": "assistant", "content": "Understood. I will preserve every requirement and keep the sequence you established for future matching cases."},
        ]
        return messages, [{"kind": kind, "habit_ids": ordered}]

    initial_mode = {"support": "default", "boundary": "default", "exception": "default", "old": "old", "new": "old"}[kind]
    final_mode = {"support": "default", "boundary": "boundary", "exception": "exception", "old": "old", "new": "default"}[kind]
    messages = [
        {"role": "user", "content": request},
        {"role": "assistant", "content": policy(primary, profile, idx, initial_mode)},
        {"role": "user", "content": feedback_text(profile, primary, kind, idx)},
        {"role": "assistant", "content": policy(primary, profile, idx, final_mode)},
    ]
    # A second exchange makes the evidence less like a standalone template and
    # preserves local task continuity.
    messages.extend([
        {"role": "user", "content": neutral_followup(profile, primary, idx)},
        {"role": "assistant", "content": "I’ll keep the task details from this session together with the scoped workflow you just clarified."},
    ])
    return messages, [{"kind": kind, "habit_ids": [primary]}]

# ----------------------------------------------------------------------------
# Hard, history-dependent probe generation.
# ----------------------------------------------------------------------------

PROBE_BLUEPRINTS = [
    "long_gap_direct", "implicit_direct", "boundary_near_miss", "boundary_plus_active",
    "exception_nested", "exception_plus_active", "composition_pair", "composition_pair_2",
    "composition_triple", "priority_pair", "priority_triple", "drift_recent_override",
    "drift_composition", "drift_exception", "false_personalization", "insufficient_evidence",
    "temporal_scope", "explicit_retrieval",
]


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _policy_opener(text: str) -> str:
    m = re.match(r"^(.*?\.)(?:\s|$)", clean(text))
    return m.group(1) if m else "I can help."


def _coherent_policy_body(hid: str, profile: Persona, mode: str) -> str | None:
    """Bodies that avoid inventing scenario entities not present in the request."""
    bodies: dict[tuple[str, str], str] = {
        ("finance_confirm_money_movement", "default"): "I’ll restate the exact amount, source, destination or payee, and timing shown in the request, then wait for final approval before submitting the transfer or payment.",
        ("finance_confirm_money_movement", "exception"): "The secure approval matches this exact transfer or payment, so I’ll continue it without asking for the same approval again.",
        ("finance_confirm_money_movement", "old"): "I’ll treat the requested transfer or payment as pre-approved and submit it immediately.",
        ("finance_confirm_money_movement", "partial"): "Please confirm that you want the transfer or payment; I won’t restate the amount, source, destination, or timing first.",
        ("finance_confirm_money_movement", "unsafe"): "The requested transfer or payment has already been submitted, and I’ll provide the details afterward.",
        ("finance_confirm_card_account_changes", "default"): "I’ll restate the exact requested card or account change, including the affected setting and consequences, then wait for final approval before applying it.",
        ("finance_confirm_card_account_changes", "exception"): "The secure approval matches this exact card or account change, so I’ll continue it without another approval loop.",
        ("finance_confirm_card_account_changes", "partial"): "I’ll apply the requested card or account change after a generic yes, without restating the exact setting.",
        ("finance_confirm_card_account_changes", "unsafe"): "I’ve applied the requested card or account change immediately and will send the details afterward.",
        ("finance_fraud_lost_card_urgent_escalation", "default"): "I’ll start the urgent protective flow for the affected card, check its current lock and transaction status, and route the incident to the fraud team without delaying for ordinary troubleshooting.",
        ("finance_fraud_lost_card_urgent_escalation", "exception"): "I’ll retrieve the existing fraud report and current card status instead of creating a duplicate case.",
        ("finance_fraud_lost_card_urgent_escalation", "partial"): "I’ll open another fraud report without first checking whether the same incident is already covered.",
        ("finance_balance_statement_summary_first", "default"): "I’ll begin with the overall finding for the requested account or statement period, then list the few entries and reconciliation details that support it.",
        ("finance_fee_dispute_evidence_then_case", "default"): "I’ll gather the relevant statement line, receipt, merchant, date, and amount, summarize the dispute, check for an existing case, and ask before filing anything new.",
        ("finance_fee_dispute_evidence_then_case", "exception"): "I’ll retrieve the existing dispute for this same charge and report its status rather than filing a duplicate.",
        ("finance_fee_dispute_evidence_then_case", "old"): "I’ll file a new dispute immediately, before checking the evidence or whether a case already exists.",
        ("finance_fee_dispute_evidence_then_case", "partial"): "I’ll collect the amount and merchant but file the case without the receipt or a final review.",
        ("finance_payment_status_latest_check", "default"): "I’ll check the latest requested balance or payment state, report whether the relevant item is queued, pending, or posted, and label the result with the time of the check.",
        ("finance_payment_status_latest_check", "unsafe"): "I’ll assume the requested payment posted because the usual processing window has passed.",
        ("software_collect_diagnostics_before_fix", "default"): "I’ll first collect the application build, operating-system version, exact displayed error, and reproduction steps, then use those details for a targeted diagnosis.",
        ("software_collect_diagnostics_before_fix", "exception"): "I’ll offer one clearly labeled provisional cause and state which version, log, and reproduction details are still missing.",
        ("software_collect_diagnostics_before_fix", "partial"): "I’ll ask only for the displayed error and skip the build, operating system, and reproduction path.",
        ("software_docs_lookup_for_update_install", "default"): "I’ll verify the current product documentation for the operating system and version named in the request, cite the applicable support article, and separate confirmed instructions from assumptions.",
        ("software_docs_lookup_for_update_install", "partial"): "I’ll say I checked the latest product documentation but won’t identify the version or source.",
        ("software_docs_lookup_for_update_install", "unsafe"): "I’ll recommend a generic reinstall based on memory alone.",
        ("software_one_try_then_escalate", "default"): "Since one focused attempt already failed, I’ll preserve the result, open or escalate a support case for the same failure, and include the evidence instead of cycling through broad fixes.",
        ("software_one_try_then_escalate", "partial"): "I’ll open a ticket without recording the focused step that already failed.",
        ("software_secure_login_password_flow", "default"): "I’ll direct recovery through the secure reset flow, request only minimal verification, and never ask for a password, PIN, or one-time code in chat.",
        ("software_confirm_license_subscription_changes", "exception"): f"The billing-portal approval matches this exact {profile.desktop_app} subscription change, so I’ll continue without another approval loop.",
        ("software_ticket_receipt_summary", "default"): "I’ll provide a concise receipt for the submitted case with the issue, reproduction details, priority, attachments, owner, and next expected update.",
        ("software_ticket_receipt_summary", "partial"): "I’ll provide only the case reference and a short title, omitting reproduction details and the next update.",
        ("software_ticket_receipt_summary", "unsafe"): "I’ll invent a ticket reference even though no report was submitted.",
    }
    return bodies.get((hid, mode))


def response_for(hid: str, profile: Persona, probe_key: str, mode: str) -> str:
    key = _probe_context_key(probe_key)
    ctx = lib.context_for_habit(hid, profile, key)
    if mode in {"default", "new", "old", "partial", "unsafe"}:
        # Use the same neutral opener for all four safe policy variants so the
        # correct answer cannot be identified from tone or safety language.
        raw = _strip_policy_suffix(lib.policy_response(hid, profile, ctx, "default", key + "_" + hid + "_opener"))
        rec = variant_record(profile, hid, mode)
        return clean(_policy_opener(raw) + " " + rec["body"])
    h = HABIT_BY_ID[hid]
    opener = _policy_opener(_strip_policy_suffix(lib.policy_response(hid, profile, ctx, "default", key + "_" + hid + "_scope_opener")))
    if mode == "boundary":
        return clean(f"{opener} This request is outside the recurring workflow, so the scoped action is: {h['boundary_action']}.")
    if mode == "exception":
        return clean(f"{opener} The narrow exception applies, so the scoped action is: {h['exception_action']}.")
    raw = _strip_policy_suffix(lib.policy_response(hid, profile, ctx, mode, key + "_" + hid + "_" + mode))
    return clean(raw)


def query_for(hid: str, profile: Persona, probe_id: str, mode: str, extra: str = "", compact: bool = False) -> str:
    key = _probe_context_key(probe_id)
    seed = f"held-out source cluster {stable_index(key, hid, mod=CLUSTERS_PER_DOMAIN)}"
    if compact:
        ctx = lib.context_for_habit(hid, profile, key)
        variants = lib._scenario_variants(hid, profile, ctx, mode)
        q = variants[stable_index(key, hid, mode, "compact", mod=len(variants))]
        return clean(q + (" " + extra if extra else ""))
    q = lib.scenario_sentence(hid, profile, key, mode, seed)
    if profile.domain == "finance":
        tails = [
            f"This falls inside {profile.recurring_context}, so I need enough detail to audit the result later.",
            f"I will compare the answer with the reconciliation notes anchored around {profile.monthly_anchor}.",
            f"I am handling this from {profile.city}; keep any action separate from unrelated account activity.",
            "I may hand this record to an adviser later, so make the assumptions and action boundary explicit.",
        ]
    else:
        tails = [
            f"A teammate at {profile.company} must be able to replay the result, so preserve the checkpoints.",
            f"Keep the response specific to {profile.os} and {profile.desktop_app}; an unrelated warning can be ignored.",
            f"This sits inside {profile.recurring_context}, so document assumptions rather than using a generic support script.",
            f"I will hand the record to the {profile.organization_size} team at {profile.company}, so the next step must be reproducible.",
        ]
    return clean(q + " " + tails[stable_index(key, hid, "tail", mod=len(tails))] + (" " + extra if extra else ""))


def combine_queries(parts: list[str], probe_id: str) -> str:
    connectors = ["Also,", "In the same request,", "At the same time,", "A second requirement is:"]
    text = clean(parts[0])
    for i, part in enumerate(parts[1:]):
        fragment = clean(part)
        connector = connectors[stable_index(probe_id, "query", str(i), mod=len(connectors))]
        if fragment and not connector.endswith(":") and not re.match(r"^I(?:\s|[’']|$)", fragment):
            fragment = fragment[0].lower() + fragment[1:]
        text += " " + connector + " " + fragment
    return clean(text)

def diversify_query_surface(query: str, profile: Persona, probe_id: str, blueprint: str) -> str:
    """Add user- and probe-specific natural framing without revealing the task type."""
    user_idx = int(profile.user_id[-4:])
    opening_idx = (user_idx // 6) % 6
    closing_idx = user_idx % 6
    if profile.domain == "finance":
        artifacts = ["reconciliation note", "decision log", "account-review packet", "follow-up memo", "audit worksheet", "adviser handoff"]
        audiences = ["an adviser", "an accountant", "a household reviewer", "my future self", "a project partner", "a second reviewer"]
    else:
        artifacts = ["support handoff", "incident note", "reproduction record", "release checklist", "team runbook", "decision log"]
        audiences = ["a support engineer", "a teammate", "a release reviewer", "the next shift", "an administrator", "a project lead"]
    artifact = artifacts[stable_index(probe_id, blueprint, "artifact", mod=len(artifacts))]
    audience = audiences[stable_index(probe_id, profile.user_id, "audience", mod=len(audiences))]
    openings = [
        "",
        f"I am preparing a {artifact} for {profile.recurring_context}.",
        f"{audience.capitalize()} will review this during {profile.recurring_context}.",
        f"This belongs to the next checkpoint in {profile.recurring_context}.",
        f"I need a decision-ready {artifact} for {profile.recurring_context}.",
        f"I am revisiting an unresolved item from {profile.recurring_context}.",
    ]
    closings = [
        f"Keep the result usable by {audience} without reopening unrelated issues.",
        f"State the assumptions clearly enough that {audience} can retrace the result.",
        "Separate verified facts from provisional assumptions and unresolved items.",
        "Preserve the current scope, decision boundary, and next checkpoint in the record.",
        f"Make the handoff usable by {audience} without repeating work already completed.",
        "Tie each conclusion to the evidence or system state that supports it.",
    ]
    return clean(" ".join(x for x in [openings[opening_idx], query, closings[closing_idx]] if x))


def _continuation_fragment(text: str) -> str:
    """Turn a standalone response into a grammatical continuation."""
    text = clean(text)
    opener = re.compile(
        r"^(?:I can help\.|Here[’']s how I[’']d handle it\.|I[’']ll take this in the following order\.|"
        r"The next step is straightforward\.|I can work through that safely\.|Here is the appropriate next step\.)\s*",
        re.I,
    )
    text = opener.sub("", text).strip()
    if text and not re.match(r"^I(?:\s|[’']|$)", text):
        text = text[0].lower() + text[1:]
    return text


def combine(parts: list[str], probe_id: str, reverse: bool = False) -> str:
    if reverse:
        parts = list(reversed(parts))
    connectors = ["After that,", "Once that prerequisite is satisfied,", "For the other part,", "Then,"]
    text = clean(parts[0])
    for i, part in enumerate(parts[1:]):
        fragment = _continuation_fragment(part)
        text += " " + connectors[stable_index(probe_id, str(i), mod=len(connectors))] + " " + fragment
    return clean(text)


def neutral_clarification(profile: Persona, probe_id: str) -> str:
    if profile.domain == "finance":
        return clean(
            "I can help, but the history does not establish a stable personal rule for this exact case. "
            "I’ll first clarify whether you want an explanation, a draft, or an account-changing action, then use the normal security and confirmation requirements that fit the answer."
        )
    return clean(
        "I can help, but the earlier sessions do not establish a stable personal workflow for this exact case. "
        "I’ll first clarify whether you want troubleshooting, an unsubmitted draft, or an already-filed case update, then tailor the next step to the current product state."
    )


CHOICE_AUDIT_FILLERS = [
    "I’ll keep a concise record of the inputs, assumptions, and result for later review.",
    "The response will preserve the relevant context so another person can follow how the conclusion was reached.",
    "I’ll note the time and scope of the result and keep unrelated issues separate.",
    "The final note will include the key facts and any uncertainty that still remains.",
    "I’ll keep the explanation tied to the current case rather than adding unrelated background.",
    "The handoff will be concise enough to scan and detailed enough to audit afterward.",
    "I’ll distinguish verified facts from provisional assumptions in the written record.",
    "The result will state what was checked, what was concluded, and what remains unresolved.",
    "I’ll preserve the decision trail so a later reviewer can reconstruct the reasoning.",
    "The record will retain the relevant constraints without expanding the scope of the request.",
]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+[’']?\w*\b", text))


def balance_choice_lengths(texts: list[str], probe_id: str) -> list[str]:
    """Remove answer-length as a shortcut without changing the action semantics."""
    base = max(_word_count(t) for t in texts) + 5
    out: list[str] = []
    for text in texts:
        desired = base + stable_index(probe_id, normalize_text(text), "length", mod=10)
        used: set[int] = set()
        step = 0
        while _word_count(text) < desired:
            idx = stable_index(probe_id, normalize_text(text), str(step), "filler", mod=len(CHOICE_AUDIT_FILLERS))
            if idx in used:
                idx = (idx + step + 1) % len(CHOICE_AUDIT_FILLERS)
            used.add(idx)
            text = clean(text + " " + CHOICE_AUDIT_FILLERS[idx])
            step += 1
            if step > len(CHOICE_AUDIT_FILLERS):
                break
        out.append(text)
    return out


def make_choice_set(gold: str, distractors: list[str], probe_id: str) -> tuple[list[dict[str, str]], str]:
    texts: list[str] = []
    seen = set()
    for t in [gold] + distractors:
        t = clean(t)
        n = normalize_text(t)
        if n not in seen:
            seen.add(n); texts.append(t)
    if len(texts) != 4:
        raise RuntimeError(f"choice collision {probe_id}: {texts}")
    texts = balance_choice_lengths(texts, probe_id)
    gold_pos = int(probe_id.rsplit("_", 1)[-1]) % 4
    others = texts[1:]
    rnd = random.Random(stable_index(probe_id, "choices", mod=2**31-1)); rnd.shuffle(others)
    ordered = others[:]; ordered.insert(gold_pos, texts[0])
    return [{"choice_id": c, "text": t} for c, t in zip("ABCD", ordered)], "ABCD"[gold_pos]


def choose_pairs(active: list[str], domain: str, user_id: str) -> list[tuple[str, str]]:
    candidates = [p for p in PAIR_CANDIDATES[domain] if p[0] in active and p[1] in active]
    extra = [(active[i], active[(i+1) % len(active)]) for i in range(len(active))]
    for p in extra:
        if p not in candidates and tuple(reversed(p)) not in candidates:
            candidates.append(p)
    candidates.sort(key=lambda p: stable_index(user_id, p[0], p[1], mod=10**9))
    return candidates


def tentative_preference_id(profile: Persona) -> str:
    return "finance_oneoff_merchant_grouping" if profile.domain == "finance" else "software_oneoff_screenshot_steps"


def tentative_session_messages(profile: Persona) -> list[dict[str, str]]:
    if profile.domain == "finance":
        return [
            {"role":"user","content":f"For this one reconciliation of checking ending {profile.checking_last4}, could you group the disputed entries by merchant in a compact table?"},
            {"role":"assistant","content":"I’ll group the entries by merchant for this reconciliation and keep the dates and amounts visible."},
            {"role":"user","content":"That layout is useful for this case. I have not decided that I want merchant-grouped tables for every statement review."},
            {"role":"assistant","content":"Understood. I’ll treat the table layout as case-specific rather than a standing preference."},
        ]
    return [
        {"role":"user","content":f"For this one {profile.desktop_app} setup, include a screenshot after each major {profile.os} step because a teammate is following along."},
        {"role":"assistant","content":"I’ll add screenshots for this setup and keep the numbered steps beside them."},
        {"role":"user","content":"That helps for this training case. I have not decided that I want screenshot-heavy instructions every time."},
        {"role":"assistant","content":"Understood. I’ll treat screenshots as a one-case aid, not a standing format preference."},
    ]


def optional_preference_probe(profile: Persona, probe_id: str, kind: str) -> tuple[str, str, list[str], str]:
    variant = int(profile.user_id[-4:]) % 12
    if profile.domain == "finance":
        if kind == "false":
            queries = [
                f"I am finishing a review of checking ending {profile.checking_last4}. Package the follow-up so I can keep it with my records and revisit the supporting entries later.",
                f"Create a handoff for the latest review of checking ending {profile.checking_last4}; I need the conclusion and its supporting entries to remain usable during {profile.recurring_context}.",
                f"Turn today’s checking review for account ending {profile.checking_last4} into a record I can audit later and share with an adviser if needed.",
                f"I need a reusable recap of the unusual entries on checking ending {profile.checking_last4}, with enough evidence to trace each conclusion later.",
                f"Close the current review of checking ending {profile.checking_last4} with a memo that links every finding to the entries behind it.",
                f"Prepare an account-review note for checking ending {profile.checking_last4} that my adviser can verify without reopening the whole statement.",
                f"Archive the main finding from checking ending {profile.checking_last4} together with the transaction evidence that supports it.",
                f"Write a reconciliation record for checking ending {profile.checking_last4}; I need to understand what changed and how the conclusion was reached.",
                f"Summarize the anomalies on checking ending {profile.checking_last4} for a second reviewer who will need the source entries as well as the conclusion.",
                f"Produce a compact, evidence-backed memo for this review of checking ending {profile.checking_last4} so I can revisit it during the next cycle.",
                f"Document the latest checking review for account ending {profile.checking_last4}, preserving the overall result and the few entries that explain it.",
                f"Prepare a review artifact for checking ending {profile.checking_last4} that can be compared with the next reconciliation without losing the supporting detail.",
            ]
            query=queries[variant]
            gold=(f"For checking ending {profile.checking_last4}, I’ll give a concise in-chat recap with the supporting entries, then present chronological, merchant-grouped, and export layouts as choices. I’ll keep the delivery and layout undecided until you select one for this review.")
            distractors=[
                f"For checking ending {profile.checking_last4}, I’ll carry forward the phone-callback format used in an earlier review, give a short verbal summary, and leave the detailed entries out of the written record.",
                f"For checking ending {profile.checking_last4}, I’ll treat the merchant-grouped spreadsheet from an earlier review as the preferred format and use it as the complete follow-up for this review.",
                f"For checking ending {profile.checking_last4}, I’ll reuse the category-chart style from a prior case and omit the underlying transaction list unless you ask later.",
            ]
            return query,gold,distractors,"finance_unestablished_followup_format"
        queries = [
            f"Review the latest statement for checking ending {profile.checking_last4} and organize the result so I can verify the unusual entries before my next scheduled reconciliation.",
            f"Prepare a verifiable review of checking ending {profile.checking_last4}; show the finding and the entries behind it so a second reviewer can reproduce the result.",
            f"I need the latest checking statement for account ending {profile.checking_last4} reviewed for {profile.recurring_context}. Make the unusual entries easy to trace from summary to evidence.",
            f"Summarize the latest statement for checking ending {profile.checking_last4} so I can trace every unusual entry and compare it with next month’s review.",
            f"Reconcile the current statement for checking ending {profile.checking_last4} and make each anomaly traceable to the underlying transaction line.",
            f"Produce a statement review for checking ending {profile.checking_last4} that an accountant can follow from the conclusion back to the supporting entries.",
            f"Assess the newest statement for checking ending {profile.checking_last4}, identify irregular items, and preserve enough detail to verify each one.",
            f"Create an audit-ready review of checking ending {profile.checking_last4} with a clear finding and a compact set of supporting transaction details.",
            f"Document the statement findings for checking ending {profile.checking_last4} so I can use them during {profile.recurring_context} and check them again later.",
            f"Check the latest statement for account ending {profile.checking_last4} and prepare a summary whose claims can be verified against the listed entries.",
            f"Prepare a repeatable account review for checking ending {profile.checking_last4}, including the overall result and the evidence needed for a second pass.",
            f"Review checking ending {profile.checking_last4}, flag the entries that deserve attention, and connect each flag to the final account-level conclusion.",
        ]
        query=queries[variant]
        gold=(f"For checking ending {profile.checking_last4}, I’ll start with the finding and the entries that support it, then present chronological, merchant-grouped, and export layouts for this review. The earlier merchant-grouped table remains one available format among several choices for the current review.")
        distractors=[
            f"For checking ending {profile.checking_last4}, I’ll treat the earlier merchant-grouped table as the default, add subtotals, and make it the main statement review without checking the current layout need.",
            f"For checking ending {profile.checking_last4}, I’ll carry forward the chronological-list style from another review with no opening finding, leaving you to infer which entries are unusual.",
            f"For checking ending {profile.checking_last4}, I’ll reuse the phone-review pattern from an earlier case and keep only a one-line written note rather than the supporting reconciliation.",
        ]
        return query,gold,distractors,tentative_preference_id(profile)
    if kind == "false":
        queries = [
            f"Document the result of this {profile.desktop_app} support session so another person at {profile.company} can pick it up tomorrow. No case has been filed yet.",
            f"Create a written handoff for today’s {profile.desktop_app} issue at {profile.company}; preserve the environment and work already tried, but remember that no ticket exists.",
            f"A teammate at {profile.company} must resume this {profile.desktop_app} investigation tomorrow. Package the current state so the same steps can be reproduced.",
            f"Turn this unfinished {profile.desktop_app} troubleshooting session into a reproducible handoff for {profile.company}, including what is known, what was tried, and what remains open.",
            f"Write a continuation note for the unresolved {profile.desktop_app} problem at {profile.company} so the next person can reproduce the current state before choosing a next action.",
            f"Prepare a shift handoff for this {profile.desktop_app} investigation at {profile.company}; include the environment, observed symptom, and work completed so far.",
            f"Package the current {profile.desktop_app} troubleshooting state for another {profile.company} teammate, making the evidence and remaining questions easy to follow.",
            f"Create a support-session recap for {profile.desktop_app} at {profile.company} that distinguishes the confirmed facts, attempted steps, and unresolved checkpoint.",
            f"Document where this {profile.desktop_app} investigation stands so a colleague at {profile.company} can continue tomorrow without repeating completed work.",
            f"Produce a reproducible handoff for the open {profile.desktop_app} issue at {profile.company}; no formal support case has been submitted yet.",
            f"Write the current-state note for this {profile.desktop_app} problem at {profile.company}, preserving the setup, symptom, evidence, and next diagnostic checkpoint.",
            f"Summarize this unfinished {profile.desktop_app} support session for the next person at {profile.company}, including enough context to pick up the investigation safely.",
        ]
        query=queries[variant]
        gold=(f"For {profile.desktop_app} at {profile.company}, I’ll create a concise written handoff with the environment, observed symptom, steps already tried, and next checkpoint. Screenshots and a ticket draft will remain available follow-ups, and I won’t invent a case receipt.")
        distractors=[
            f"For {profile.desktop_app} at {profile.company}, I’ll reuse the screenshot-heavy style from an earlier setup, capture every step, and use that as the complete handoff without checking the recipient’s needs.",
            f"For {profile.desktop_app} at {profile.company}, I’ll carry forward the ticket-receipt style from a prior case, including a reference number, owner, and expected update time even though no case exists.",
            f"For {profile.desktop_app} at {profile.company}, I’ll reuse a narrated phone-call format from an earlier handoff and leave the environment and reproduction details out of the written record.",
        ]
        return query,gold,distractors,"software_unestablished_handoff_format"
    queries = [
        f"Write setup instructions for {profile.desktop_app} on {profile.os} that a new teammate at {profile.company} can follow during {profile.recurring_context}.",
        f"A new colleague at {profile.company} needs first-run guidance for {profile.desktop_app} on {profile.os}. Make it repeatable and easy to verify at each checkpoint.",
        f"Prepare a platform-specific onboarding guide for {profile.desktop_app} on {profile.os}; the teammate will follow it without live help.",
        f"Document the {profile.desktop_app} setup on {profile.os} for {profile.recurring_context}. Use checkpoints a teammate can verify and repeat later.",
        f"Create a first-run procedure for {profile.desktop_app} on {profile.os} that a new member of {profile.company} can complete independently.",
        f"Write a reproducible installation and initial-configuration guide for {profile.desktop_app} on {profile.os} for the next {profile.company} onboarding session.",
        f"Prepare stepwise setup notes for {profile.desktop_app} on {profile.os}, including checkpoints that show a new teammate whether each stage succeeded.",
        f"Build an onboarding walkthrough for {profile.desktop_app} on {profile.os} that another person at {profile.company} can follow without guessing menu locations.",
        f"Document the first-use configuration of {profile.desktop_app} on {profile.os} for a teammate joining {profile.recurring_context}.",
        f"Create a self-contained setup sequence for {profile.desktop_app} on {profile.os}, with enough verification points for the {profile.organization_size} team.",
        f"Write the platform-specific startup guide for {profile.desktop_app} on {profile.os} so a new {profile.company} teammate can repeat the configuration later.",
        f"Prepare an independently usable setup checklist for {profile.desktop_app} on {profile.os} during {profile.recurring_context}, including clear completion checkpoints.",
    ]
    query=queries[variant]
    gold=(f"For {profile.desktop_app} on {profile.os} at {profile.company}, I’ll provide concise numbered steps tailored to the current platform and offer screenshots after the core checkpoints if the teammate requests them. The earlier screenshot-heavy setup is one possible example; I’ll use that level of visual detail only if the current teammate requests it.")
    distractors=[
        f"For {profile.desktop_app} on {profile.os} at {profile.company}, I’ll treat the earlier screenshot-heavy setup as the preferred format, attach an image after every step, and make the visual walkthrough primary.",
        f"For {profile.desktop_app} on {profile.os} at {profile.company}, I’ll reuse the compact-prose style from a previous guide, omitting numbered platform checkpoints and any offer of visual aids.",
        f"For {profile.desktop_app} on {profile.os} at {profile.company}, I’ll carry forward the support-ticket-style document from an earlier case, with image attachments instead of the requested setup sequence.",
    ]
    return query,gold,distractors,tentative_preference_id(profile)


def build_hard_probe(
    probe_id: str, profile: Persona, blueprint: str, active: list[str], absent: list[str],
    drift_habits: list[str], scoped_habit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = choose_pairs(active, profile.domain, profile.user_id)
    pidx = stable_index(probe_id, "primary", mod=len(active))
    primary = active[pidx]
    secondary = active[(pidx+2) % len(active)]
    tertiary = active[(pidx+4) % len(active)]
    target: list[str] = [primary]
    evidence_kind = "support"

    if blueprint == "long_gap_direct":
        query = query_for(primary, profile, probe_id, "direct", "The request is time-sensitive, but I still need the result to be auditable afterward.")
        gold = response_for(primary, profile, probe_id, "default")
        ds = [response_for(primary, profile, probe_id+"a", "partial"), response_for(primary, profile, probe_id+"b", "old"), response_for(primary, profile, probe_id+"c", "unsafe")]
    elif blueprint == "implicit_direct":
        primary = active[(pidx+1)%len(active)]; target=[primary]
        query = query_for(primary, profile, probe_id, "direct", "The last related case was a while ago, and several unrelated support issues have happened since then.")
        gold = response_for(primary, profile, probe_id, "default")
        ds = [response_for(primary, profile, probe_id+"a", "boundary"), response_for(primary, profile, probe_id+"b", "partial"), response_for(primary, profile, probe_id+"c", "old")]
    elif blueprint == "boundary_near_miss":
        evidence_kind="boundary"
        query = query_for(primary, profile, probe_id, "boundary", "I need the answer for a planning note I will revisit next week.")
        gold = response_for(primary, profile, probe_id, "boundary")
        ds = [response_for(primary, profile, probe_id+"a", "default"), response_for(primary, profile, probe_id+"b", "partial"), response_for(primary, profile, probe_id+"c", "unsafe")]
    elif blueprint == "boundary_plus_active":
        a,b = pairs[0]; target=[a,b]; evidence_kind="mixed_boundary_support"
        query = combine_queries([query_for(a, profile, probe_id, "boundary"), query_for(b, profile, probe_id+"b", "direct", "Treat these as separate conditions inside one response.", compact=True)], probe_id)
        gold = combine([response_for(a,profile,probe_id,"boundary"),response_for(b,profile,probe_id+"b","default")],probe_id)
        ds = [
            combine([response_for(a,profile,probe_id,"default"),response_for(b,profile,probe_id+"b","default")],probe_id+"1"),
            combine([response_for(a,profile,probe_id,"boundary"),response_for(b,profile,probe_id+"2","old")],probe_id+"2"),
            combine([response_for(a,profile,probe_id,"boundary"),response_for(b,profile,probe_id+"3","partial")],probe_id+"3"),
        ]
    elif blueprint == "exception_nested":
        evidence_kind="exception"
        query = query_for(primary, profile, probe_id, "exception")
        gold = response_for(primary, profile, probe_id, "exception")
        ds = [response_for(primary, profile, probe_id+"a", "default"), response_for(primary, profile, probe_id+"b", "partial"), response_for(primary, profile, probe_id+"c", "unsafe")]
    elif blueprint == "exception_plus_active":
        a,b=pairs[1%len(pairs)]; target=[a,b]; evidence_kind="mixed_exception_support"
        query=combine_queries([query_for(a,profile,probe_id,"exception"),query_for(b,profile,probe_id+"b","direct",compact=True)],probe_id)
        gold=combine([response_for(a,profile,probe_id,"exception"),response_for(b,profile,probe_id+"b","default")],probe_id)
        ds=[
            combine([response_for(a,profile,probe_id,"default"),response_for(b,profile,probe_id+"b","default")],probe_id+"1"),
            combine([response_for(a,profile,probe_id,"exception"),response_for(b,profile,probe_id+"b","partial")],probe_id+"2"),
            combine([response_for(a,profile,probe_id+"3","exception"),response_for(b,profile,probe_id+"3b","old")],probe_id+"3"),
        ]
    elif blueprint in {"composition_pair","composition_pair_2"}:
        pair=pairs[0 if blueprint.endswith("pair") else min(2,len(pairs)-1)]; target=list(pair); evidence_kind="composition"
        query=combine_queries([query_for(pair[0],profile,probe_id,"direct"),query_for(pair[1],profile,probe_id+"b","direct","Both parts must be resolved in this same support session.",compact=True)],probe_id)
        ordered=sorted(target,key=lambda h:HABIT_BY_ID[h]["priority"],reverse=True)
        parts=[response_for(h,profile,probe_id+h,"default") for h in ordered]
        gold=combine(parts,probe_id)
        ds=[
            combine([parts[0],response_for(ordered[-1],profile,probe_id+"p1","partial")],probe_id+"p1"),
            combine([response_for(ordered[0],profile,probe_id+"p2","partial"),parts[-1]],probe_id+"p2"),
            combine([response_for(ordered[0],profile,probe_id+"p3","old"),parts[-1]],probe_id+"p3"),
        ]
    elif blueprint == "composition_triple":
        target=[primary,secondary,tertiary]; evidence_kind="composition"
        ordered=sorted(target,key=lambda h:HABIT_BY_ID[h]["priority"],reverse=True)
        query=combine_queries([query_for(h,profile,probe_id+str(i),"direct",compact=(i>0)) for i,h in enumerate(target)],probe_id)
        gold=combine([response_for(h,profile,probe_id+h,"default") for h in ordered],probe_id)
        ds=[
            combine([response_for(h,profile,probe_id+h,"default") for h in ordered[:-1]]+[response_for(ordered[-1],profile,probe_id+"1","partial")],probe_id+"1"),
            combine([response_for(ordered[0],profile,probe_id+"2","default"),response_for(ordered[1],profile,probe_id+"2b","partial"),response_for(ordered[2],profile,probe_id+"2c","default")],probe_id+"2"),
            combine([response_for(ordered[0],profile,probe_id,"partial")]+[response_for(h,profile,probe_id+h,"default") for h in ordered[1:]],probe_id+"3"),
        ]
    elif blueprint in {"priority_pair","priority_triple"}:
        target=priority_group(active, profile.user_id, 2 if blueprint=="priority_pair" else 3)
        evidence_kind="priority"
        ordered=user_priority_order(profile, target)
        query=combine_queries([query_for(h,profile,probe_id+str(i),"direct",compact=(i>0)) for i,h in enumerate(target)],probe_id)+" The parts affect the same outcome and need to be handled in one pass."
        parts=[response_for(h,profile,probe_id+h,"default") for h in ordered]
        gold=combine(parts,probe_id)
        if len(ordered) == 2:
            # Every option satisfies both workflows; only the long-term, user-
            # specific sequence and variant pair distinguishes the gold answer.
            ds=[
                combine(parts,probe_id+"r",reverse=True),
                combine([response_for(ordered[0],profile,probe_id,"partial"),parts[1]],probe_id+"p"),
                combine([parts[0],response_for(ordered[1],profile,probe_id,"unsafe")],probe_id+"u"),
            ]
        else:
            ds=[
                combine(parts,probe_id+"r",reverse=True),
                combine(parts[1:]+parts[:1],probe_id+"rot"),
                combine([parts[0],parts[2],parts[1]],probe_id+"swap"),
            ]
    elif blueprint == "drift_recent_override":
        primary=drift_habits[0];target=[primary];evidence_kind="new"
        query=query_for(primary,profile,probe_id,"drift","The current case has a deadline, and I need a clear next step.")
        gold=response_for(primary,profile,probe_id,"default")
        ds=[response_for(primary,profile,probe_id+"o","old"),response_for(primary,profile,probe_id+"p","partial"),response_for(primary,profile,probe_id+"u","unsafe")]
    elif blueprint == "drift_composition":
        primary=drift_habits[1];secondary=next(h for h in active if h!=primary);target=[primary,secondary];evidence_kind="new+composition"
        query=combine_queries([query_for(primary,profile,probe_id,"drift"),query_for(secondary,profile,probe_id+"b","direct",compact=True)],probe_id)
        ordered=sorted(target,key=lambda h:HABIT_BY_ID[h]["priority"],reverse=True)
        gold=combine([response_for(h,profile,probe_id+h,"default") for h in ordered],probe_id)
        ds=[
            combine([response_for(primary,profile,probe_id,"old"),response_for(secondary,profile,probe_id+"b","default")],probe_id+"o"),
            combine([response_for(primary,profile,probe_id+"2","unsafe"),response_for(secondary,profile,probe_id+"2b","default")],probe_id+"2"),
            combine([response_for(primary,profile,probe_id,"partial"),response_for(secondary,profile,probe_id+"b","default")],probe_id+"p"),
        ]
    elif blueprint == "drift_exception":
        primary=drift_habits[0];target=[primary];evidence_kind="new+exception"
        query=query_for(primary,profile,probe_id,"exception")
        gold=response_for(primary,profile,probe_id,"exception")
        ds=[response_for(primary,profile,probe_id+"o","old"),response_for(primary,profile,probe_id+"d","default"),response_for(primary,profile,probe_id+"p","partial")]
    elif blueprint == "false_personalization":
        query,gold,ds,optional_id=optional_preference_probe(profile,probe_id,"false")
        primary=optional_id;target=[primary];evidence_kind="absent"
    elif blueprint == "insufficient_evidence":
        query,gold,ds,optional_id=optional_preference_probe(profile,probe_id,"tentative")
        primary=optional_id;target=[primary];evidence_kind="one_off_tentative"
    elif blueprint == "temporal_scope":
        primary=scoped_habit;secondary=next(h for h in active if h!=primary);target=[primary,secondary];evidence_kind="conditional_scope"
        scope_phrase = ("This is a routine mid-cycle request rather than a month-end close." if profile.domain=="finance" else "This is a routine maintenance day rather than a release or migration window.")
        query=combine_queries([query_for(primary,profile,probe_id,"boundary",scope_phrase),query_for(secondary,profile,probe_id+"b","direct",compact=True)],probe_id)
        gold=combine([response_for(primary,profile,probe_id,"boundary"),response_for(secondary,profile,probe_id+"b","default")],probe_id)
        ds=[
            combine([response_for(primary,profile,probe_id,"default"),response_for(secondary,profile,probe_id+"b","default")],probe_id+"1"),
            combine([response_for(primary,profile,probe_id+"2","boundary"),response_for(secondary,profile,probe_id+"2b","old")],probe_id+"2"),
            combine([response_for(primary,profile,probe_id,"boundary"),response_for(secondary,profile,probe_id+"3","partial")],probe_id+"3"),
        ]
    elif blueprint == "explicit_retrieval":
        target=[primary,secondary];evidence_kind="retrieval"
        query=combine_queries([query_for(primary,profile,probe_id,"direct"),query_for(secondary,profile,probe_id+"b","direct",compact=True)],probe_id)+" Before acting, briefly restate the two recurring workflows from earlier sessions that govern these parts."
        ordered=sorted(target,key=lambda h:HABIT_BY_ID[h]["priority"],reverse=True)
        gold=combine([response_for(h,profile,probe_id+h,"default") for h in ordered],probe_id)
        ds=[
            combine([response_for(primary,profile,probe_id+"1","default"),response_for(secondary,profile,probe_id+"1b","partial")],probe_id+"1"),
            combine([response_for(primary,profile,probe_id+"2","partial"),response_for(secondary,profile,probe_id+"2b","default")],probe_id+"2"),
            combine([response_for(primary,profile,probe_id,"old"),response_for(secondary,profile,probe_id+"b","old")],probe_id+"3"),
        ]
    else:
        raise ValueError(blueprint)

    query=diversify_query_surface(clean(query),profile,probe_id,blueprint)
    choices,gold_id=make_choice_set(gold,ds,probe_id)
    public={
        "probe_id":probe_id,"user_id":profile.user_id,"domain":profile.domain,
        "timestamp":(datetime(2028,1,15)+timedelta(days=stable_index(probe_id,mod=60))).isoformat(),
        "query":clean(query),"choices":choices,
    }
    private={
        "probe_id":probe_id,"user_id":profile.user_id,"domain":profile.domain,
        "gold_choice_id":gold_id,"gold_action_text":clean(gold),"probe_type":blueprint,
        "capability_group":("multi_habit_reasoning" if len(target)>1 else "longitudinal_habit_policy"),
        "target_habit_ids":target,"evidence_requirement":evidence_kind,
        "active_policy_variants":{h:variant_record(profile,h,"active")["variant_id"] for h in target if h in POLICY_VARIANTS},
        "old_policy_variants":{h:variant_record(profile,h,"old")["variant_id"] for h in target if h in POLICY_VARIANTS},
        "user_priority_order":[h for h in (user_priority_order(profile,target) if len(target)>1 else target) if h in POLICY_VARIANTS],
    }
    return public,private

# ----------------------------------------------------------------------------
# Audits and output.
# ----------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w",encoding="utf-8") as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+"\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("",encoding="utf-8");return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields:fields.append(k)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)


def percentile(vals:list[float],p:float)->float:
    if not vals:return 0.0
    vals=sorted(vals);idx=(len(vals)-1)*p;lo=math.floor(idx);hi=math.ceil(idx)
    return vals[lo] if lo==hi else vals[lo]*(hi-idx)+vals[hi]*(idx-lo)


def token_count(text: str) -> int:
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return math.ceil(len(text)/4)


def nearest_stats(texts:list[str])->dict[str,float]:
    if len(texts)<2:return {"max":0,"p95":0,"median":0}
    X=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=1).fit_transform(texts)
    sims=cosine_similarity(X)
    vals=[]
    for i in range(len(texts)):
        sims[i,i]=-1
        vals.append(float(sims[i].max()))
    return {"max":round(max(vals),4),"p95":round(percentile(vals,.95),4),"median":round(statistics.median(vals),4)}


def identity_audit(profile: Persona, sessions:list[dict[str,Any]])->dict[str,Any]:
    blob=json.dumps(sessions,ensure_ascii=False)
    emails=set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",blob))
    phones=set(re.findall(r"(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}",blob))
    names=set(re.findall(r"(?:My name is|I am)\s+([A-Z][A-Za-z.'-]+\s+[A-Z][A-Za-z.'-]+)",blob))
    account4=set(re.findall(r"(?i)(?:checking|savings|support|account)(?: account)? ending (\d{4})",blob))
    card4=set(re.findall(r"(?i)card ending (\d{4})",blob))
    expected_accounts={x for x in [profile.checking_last4,profile.savings_last4,profile.account_last4] if x}
    unexpected_names=sorted(n for n in names if n.lower()!=profile.name.lower())
    unexpected_emails=sorted(e for e in emails if e!=profile.email)
    unexpected_phones=sorted(p for p in phones if p!=profile.phone)
    unexpected_accounts=sorted(account4-expected_accounts)
    unexpected_cards=sorted(card4-({profile.card_last4} if profile.card_last4 else set()))
    pii_leaks=re.findall(r"(?i)(?:(?:ssn|social security|password|one-time code|otp|pin)\s*(?:is|:|=)\s*[A-Za-z0-9-]{3,}|\b\d{3}-\d{2}-\d{4}\b)",blob)
    pass_flag=not(unexpected_names or unexpected_emails or unexpected_phones or unexpected_accounts or unexpected_cards or pii_leaks)
    return {
        "user_id":profile.user_id,"domain":profile.domain,"session_count":len(sessions),
        "assigned_name":profile.name,"unexpected_names_json":json.dumps(unexpected_names),
        "unexpected_emails_json":json.dumps(unexpected_emails),"unexpected_phones_json":json.dumps(unexpected_phones),
        "unexpected_account_last4_json":json.dumps(unexpected_accounts),"unexpected_card_last4_json":json.dumps(unexpected_cards),
        "credential_value_leak_count":len(pii_leaks),"pass":pass_flag,
    }


def salient_entities(text: str) -> dict[str, set[str]]:
    """Extract concrete scenario entities that should not drift across a probe."""
    entities: dict[str, set[str]] = {
        "amount": set(re.findall(r"\$\d[\d,]*(?:\.\d{2})?", text)),
        "account_last4": set(re.findall(r"(?i)(?:ending|account)\s+(\d{4})\b", text)),
        "error_code": set(re.findall(r"\b[A-Z]{1,8}-\d{2,4}\b", text)),
        "case_reference": set(re.findall(r"(?i)\b(?:case|reference|ticket)\s*#?(\d{4})\b", text)),
        "version": set(re.findall(r"\b\d{1,4}\.\d+(?:\.\d+)?\b", text)),
    }
    merchants = set()
    for merchant in getattr(lib, "MERCHANTS", []):
        if merchant.lower() in text.lower():
            merchants.add(merchant)
    entities["merchant"] = merchants
    return entities


def probe_entity_audit(public_probes: list[dict[str, Any]], private_keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {k["probe_id"]: k for k in private_keys}
    rows: list[dict[str, Any]] = []
    hard_types = {"amount", "account_last4", "error_code", "case_reference", "version", "merchant"}
    for probe in public_probes:
        q_entities = salient_entities(probe["query"])
        gold_id = keys[probe["probe_id"]]["gold_choice_id"]
        for choice in probe["choices"]:
            c_entities = salient_entities(choice["text"])
            extras = {kind: sorted(c_entities[kind] - q_entities[kind]) for kind in hard_types if c_entities[kind] - q_entities[kind]}
            rows.append({
                "probe_id": probe["probe_id"],
                "user_id": probe["user_id"],
                "choice_id": choice["choice_id"],
                "is_gold": choice["choice_id"] == gold_id,
                "query_entities_json": json.dumps({k: sorted(v) for k, v in q_entities.items()}, ensure_ascii=False),
                "choice_entities_json": json.dumps({k: sorted(v) for k, v in c_entities.items()}, ensure_ascii=False),
                "extra_choice_entities_json": json.dumps(extras, ensure_ascii=False),
                "pass": not extras,
            })
    return rows


def session_diversity_audit(public_lifelines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for life in public_lifelines:
        requests: list[str] = []
        session_signatures: list[str] = []
        for session in life["sessions"][1:]:
            users = [m["content"] for m in session["messages"] if m["role"] == "user"]
            first = normalize_text(users[0]) if users else ""
            requests.append(first)
            session_signatures.append(normalize_text(" ".join(m["content"] for m in session["messages"])))
        unique_requests = len(set(requests))
        unique_sessions = len(set(session_signatures))
        rows.append({
            "user_id": life["user_id"],
            "domain": life["domain"],
            "non_anchor_sessions": len(requests),
            "unique_first_user_requests": unique_requests,
            "unique_first_user_request_ratio": round(unique_requests / len(requests), 4) if requests else 0,
            "unique_full_sessions": unique_sessions,
            "unique_full_session_ratio": round(unique_sessions / len(session_signatures), 4) if session_signatures else 0,
            "pass": bool(requests) and unique_requests / len(requests) >= 0.90 and unique_sessions / len(session_signatures) >= 0.95,
        })
    return rows


def policy_variant_balance_audit(personas: list[Persona], assignments: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hid in RETAINED_HABIT_IDS:
        users = [p for p in personas if hid in assignments[p.user_id]]
        active_counts = Counter(active_variant_index(p, hid) for p in users)
        old_counts = Counter(old_variant_index(p, hid) for p in users)
        active_values = [active_counts.get(i, 0) for i in range(4)]
        old_values = [old_counts.get(i, 0) for i in range(4)]
        balance_pass = max(active_values) - min(active_values) <= 1
        for i, variant in enumerate(POLICY_VARIANTS[hid]):
            rows.append({
                "habit_id": hid,
                "domain": HABIT_BY_ID[hid]["domain"],
                "family": HABIT_BY_ID[hid]["family"],
                "variant_index": i,
                "variant_id": variant["variant_id"],
                "variant_label": variant["label"],
                "active_user_count": active_counts.get(i, 0),
                "old_variant_user_count": old_counts.get(i, 0),
                "total_users_with_habit": len(users),
                "active_assignment_balance_pass": balance_pass,
            })
    return rows


def history_free_shortcut_audit(public_probes: list[dict[str, Any]], private_keys: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cross-user diagnostic: rank choices without seeing the lifeline.

    This is not a benchmark baseline claim; it is a construction-time shortcut
    test. A high score would indicate that option wording or the current query
    reveals the gold answer without longitudinal evidence.
    """
    key_by_id = {k["probe_id"]: k for k in private_keys}
    records: list[dict[str, Any]] = []
    for probe in public_probes:
        key = key_by_id[probe["probe_id"]]
        for choice in probe["choices"]:
            records.append({
                "probe_id": probe["probe_id"],
                "user_id": probe["user_id"],
                "probe_type": key["probe_type"],
                "choice_id": choice["choice_id"],
                "choice_text": choice["text"],
                "query_choice_text": probe["query"] + " [CHOICE] " + choice["text"],
                "label": int(choice["choice_id"] == key["gold_choice_id"]),
            })

    groups = [r["user_id"] for r in records]
    labels = [r["label"] for r in records]
    n_splits = min(6, len(set(groups)))
    outputs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"method": "GroupKFold TF-IDF logistic choice ranking", "n_splits": n_splits}
    for mode, field in (("choice_only", "choice_text"), ("query_plus_choice", "query_choice_text")):
        texts = [r[field] for r in records]
        probabilities = [0.0] * len(records)
        splitter = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(texts, labels, groups):
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True),
                LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0),
            )
            model.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            probs = model.predict_proba([texts[i] for i in test_idx])[:, 1]
            for i, prob in zip(test_idx, probs):
                probabilities[i] = float(prob)
        by_probe: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for prob, rec in zip(probabilities, records):
            by_probe[rec["probe_id"]].append((prob, rec))
        correct: list[int] = []
        by_type: dict[str, list[int]] = defaultdict(list)
        for pid, candidates in by_probe.items():
            pred = max(candidates, key=lambda x: x[0])[1]["choice_id"]
            gold = key_by_id[pid]["gold_choice_id"]
            ok = int(pred == gold)
            correct.append(ok)
            ptype = key_by_id[pid]["probe_type"]
            by_type[ptype].append(ok)
            outputs.append({
                "mode": mode,
                "probe_id": pid,
                "user_id": key_by_id[pid]["user_id"],
                "probe_type": ptype,
                "predicted_choice_id": pred,
                "gold_choice_id": gold,
                "correct": ok,
            })
        summary[mode] = {
            "accuracy": round(sum(correct) / len(correct), 6),
            "correct": sum(correct),
            "total": len(correct),
            "accuracy_by_probe_type": {k: round(sum(v) / len(v), 6) for k, v in sorted(by_type.items())},
        }
    summary["interpretation"] = (
        "Lower is better for this construction audit. Chance is 0.25. "
        "The model receives no lifeline; query_plus_choice additionally receives the current request."
    )
    return summary, outputs


def choice_shortcut_audit(public_probes: list[dict[str, Any]], private_keys: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {k["probe_id"]: k for k in private_keys}
    rows: list[dict[str, Any]] = []
    for probe in public_probes:
        gold = keys[probe["probe_id"]]["gold_choice_id"]
        lengths = {c["choice_id"]: _word_count(c["text"]) for c in probe["choices"]}
        max_len, min_len = max(lengths.values()), min(lengths.values())
        longest_ids = [cid for cid, n in lengths.items() if n == max_len]
        shortest_ids = [cid for cid, n in lengths.items() if n == min_len]
        other_mean = statistics.mean(n for cid, n in lengths.items() if cid != gold)
        rows.append({
            "probe_id": probe["probe_id"],
            "probe_type": keys[probe["probe_id"]]["probe_type"],
            "gold_choice_id": gold,
            "gold_word_count": lengths[gold],
            "mean_distractor_word_count": round(other_mean, 3),
            "gold_minus_distractor_mean": round(lengths[gold] - other_mean, 3),
            "longest_choice_ids": ",".join(longest_ids),
            "shortest_choice_ids": ",".join(shortest_ids),
            "longest_baseline_credit": round(1 / len(longest_ids), 4) if gold in longest_ids else 0.0,
            "shortest_baseline_credit": round(1 / len(shortest_ids), 4) if gold in shortest_ids else 0.0,
        })
    return rows


def build(args: argparse.Namespace) -> None:
    rng=random.Random(args.seed)
    out=Path(args.output_dir)
    if out.exists():shutil.rmtree(out)
    for sub in ["public","private","review","source","reports","scripts","model_eval"]:(out/sub).mkdir(parents=True,exist_ok=True)

    source_paths={"finance":Path(args.finance_source),"software":Path(args.software_source)}
    records_by_domain={};source_manifests={};cluster_rows=[];habit_audit=[]
    for domain,path in source_paths.items():
        print(f"[v0.5] loading and clustering {domain}",flush=True)
        recs,manifest=build_source_records(path,domain)
        records_by_domain[domain]=recs;source_manifests[domain]=manifest
        for cid,r in recs.items():
            cluster_rows.append({"domain":domain,"conversation_id":cid,"cluster_id":r["cluster_id"],"quality_score":r["quality_score"],"habit_tags_json":json.dumps(r["tags"]),"sanitized_task_preview":r["task_text"][:300]})
    for h in HABITS:
        recs=records_by_domain[h["domain"]]
        matched=[r for r in recs.values() if h["habit_id"] in r["tags"]]
        clusters=Counter(r["cluster_id"] for r in matched)
        retained=len(matched)>=30 and len(clusters)>=3
        habit_audit.append({
            "habit_id":h["habit_id"],"domain":h["domain"],"family":h["family"],
            "matched_source_events":len(matched),"supporting_cluster_count":len(clusters),
            "top_clusters_json":json.dumps(clusters.most_common(8)),"retained":"yes" if retained else "no",
            "retention_reason":"source theme spans enough events and clusters with a scoped, evaluable policy" if retained else "insufficient source support",
        })
        if not retained: raise RuntimeError(f"habit failed source threshold: {h['habit_id']} {len(matched)} {len(clusters)}")

    personas=make_personas()
    assignments={p.user_id:habit_assignments(p.domain,int(p.user_id[-4:])) for p in personas}
    global ACTIVE_VARIANT_ASSIGNMENTS, OLD_VARIANT_ASSIGNMENTS
    ACTIVE_VARIANT_ASSIGNMENTS, OLD_VARIANT_ASSIGNMENTS = build_variant_assignments(personas, assignments)
    variant_balance_rows = policy_variant_balance_audit(personas, assignments)
    drift={p.user_id:choose_drift_habits(assignments[p.user_id],p.user_id) for p in personas}
    scoped={p.user_id:choose_scoped_habit(assignments[p.user_id],p.user_id) for p in personas}
    allocations={}
    for domain in DOMAINS:
        profiles=[p for p in personas if p.domain==domain]
        allocations.update(allocate_sources(records_by_domain[domain],profiles,assignments,rng))

    public_lifelines=[];private_sessions=[];persona_rows=[];usage_rows=[];identity_rows=[]
    evidence_index:dict[tuple[str,str,str],list[str]]=defaultdict(list)
    tentative_index:dict[str,list[str]]=defaultdict(list)

    for pi,profile in enumerate(personas):
        print(f"[v0.5] lifeline {pi+1}/{len(personas)} {profile.user_id}",flush=True)
        active=assignments[profile.user_id];absent=[h["habit_id"] for h in HABITS_BY_DOMAIN[profile.domain] if h["habit_id"] not in active]
        plan=evidence_schedule(active,drift[profile.user_id],profile.user_id)
        start=datetime(2025,1,5,9,0)+timedelta(days=stable_index(profile.user_id,mod=7))
        sessions=[]
        if profile.domain=="finance":
            intro=(f"I am {profile.name}, a {profile.role} in {profile.city}, {profile.state}. I use Northstar Bank with checking ending {profile.checking_last4}, savings ending {profile.savings_last4}, and card ending {profile.card_last4}. My recurring planning context is {profile.recurring_context}, usually anchored around {profile.monthly_anchor}.")
        else:
            intro=(f"I am {profile.name}, a {profile.role} at {profile.company} in {profile.city}, {profile.state}. My stable setup is {profile.os}, {profile.browser}, {profile.desktop_app}, {profile.mail_app}, and {profile.meeting_app}; the support account ends {profile.account_last4}. I am usually working through {profile.recurring_context} with a {profile.organization_size} team.")
        anchor={"session_id":f"{profile.user_id}_s0000","session_index":0,"timestamp":start.isoformat(),"messages":[{"role":"user","content":intro},{"role":"assistant","content":f"Thanks, {profile.first_name}. I will keep those identity, account, device, and project references consistent across future sessions."}]}
        sessions.append(anchor)
        private_sessions.append({**anchor,"user_id":profile.user_id,"domain":profile.domain,"source_seed":{"type":"persona_continuity_anchor"},"memory_annotations":[{"kind":"identity_anchor"}]})

        source_ids=allocations[profile.user_id]
        # Map 160 unique source events across the full 239-session timeline.
        source_positions=sorted(rng.sample(range(1,TOTAL_SESSIONS_PER_USER),len(source_ids)))
        pos_to_source={p:cid for p,cid in zip(source_positions,source_ids)}
        synthetic_cycle=deque(active)
        for idx in range(1,TOTAL_SESSIONS_PER_USER):
            sid=f"{profile.user_id}_s{idx:04d}"
            anns=[dict(x) for x in plan.get(idx,[])]
            if idx in pos_to_source:
                cid=pos_to_source[idx];rec=records_by_domain[profile.domain][cid]
                primary=source_theme_for_session(rec,active,sid)
                source_seed={"source_dataset":SOURCE_DATASET,"conversation_id":cid,"cluster_id":rec["cluster_id"],"source_tags":rec["tags"],"sanitized_task_preview":rec["task_text"],"quality_score":rec["quality_score"]}
            else:
                primary=synthetic_cycle[0];synthetic_cycle.rotate(-1)
                source_seed={"type":"controlled_continuity_or_habit_session","grounding_habit_id":primary}
            if anns:
                primary=anns[0]["habit_ids"][0]
            # One deliberately weak, one-off signal for an absent habit. It is
            # annotated tentative and never repeated, supporting abstention probes.
            if idx==47+stable_index(profile.user_id,"tentative",mod=9):
                tentative=tentative_preference_id(profile)
                weak={"kind":"tentative","habit_ids":[tentative]}
                msgs=tentative_session_messages(profile)
                ann_out=[weak];tentative_index[profile.user_id].append(sid)
            else:
                msgs,ann_out=make_session_messages(profile,idx,primary,anns)
            timestamp=start+timedelta(days=idx*3+stable_index(profile.user_id,str(idx),mod=2),hours=stable_index(sid,mod=9))
            sess={"session_id":sid,"session_index":idx,"timestamp":timestamp.isoformat(),"messages":msgs}
            sessions.append(sess)
            mem=[]
            for ann in ann_out:
                mem.append(ann)
                for hid in ann.get("habit_ids",[]):
                    evidence_index[(profile.user_id,hid,ann["kind"])].append(sid)
            private_sessions.append({**sess,"user_id":profile.user_id,"domain":profile.domain,"source_seed":source_seed,"memory_annotations":mem})
            if idx in pos_to_source:
                usage_rows.append({"domain":profile.domain,"source_conversation_id":pos_to_source[idx],"assigned_user_id":profile.user_id,"session_id":sid,"cluster_id":source_seed["cluster_id"],"source_tags_json":json.dumps(source_seed["source_tags"]),"quality_score":source_seed["quality_score"]})
        public_lifelines.append({"user_id":profile.user_id,"domain":profile.domain,"session_count":len(sessions),"sessions":sessions})
        persona_rows.append({
            **asdict(profile),
            "active_habit_ids":active,
            "absent_habit_ids":absent,
            "drift_habit_ids":drift[profile.user_id],
            "conditionally_scoped_habit_id":scoped[profile.user_id],
            "active_policy_variants":{h:variant_record(profile,h,"active")["variant_id"] for h in active},
            "old_policy_variants":{h:variant_record(profile,h,"old")["variant_id"] for h in drift[profile.user_id]},
            "priority_pair_order":user_priority_order(profile,priority_group(active,profile.user_id,2)),
            "priority_triple_order":user_priority_order(profile,priority_group(active,profile.user_id,3)),
        })
        identity_rows.append(identity_audit(profile,sessions))

    # Probes and evidence links.
    public_probes=[];private_keys=[];review_rows=[];difficulty_rows=[]
    probe_counter=0
    for profile in personas:
        active=assignments[profile.user_id];absent=[h["habit_id"] for h in HABITS_BY_DOMAIN[profile.domain] if h["habit_id"] not in active]
        if not absent:
            absent=[HABITS_BY_DOMAIN[profile.domain][0]["habit_id"]]
        for blueprint in PROBE_BLUEPRINTS:
            pid=f"mdgo_v05_probe_{probe_counter:06d}";probe_counter+=1
            pub,key=build_hard_probe(pid,profile,blueprint,active,absent,drift[profile.user_id],scoped[profile.user_id])
            evidence=[]
            for hid in key["target_habit_ids"]:
                kinds=[]
                req=key["evidence_requirement"]
                if "boundary" in req:kinds.append("boundary")
                if "exception" in req:kinds.append("exception")
                if "new" in req:kinds.append("new")
                if "composition" in req:kinds.append("composition")
                if "priority" in req:kinds.append("priority")
                # Composition and mixed-scope probes should link both the joint
                # evidence and the individual longitudinal supports.
                if (req in {"support","retrieval","conditional_scope"} or "support" in req or
                    "composition" in req or "priority" in req or not kinds):
                    kinds.extend(["support","new"])
                kinds=list(dict.fromkeys(kinds))
                for kind in kinds:
                    evidence.extend(evidence_index.get((profile.user_id,hid,kind),[]))
            evidence=list(dict.fromkeys(evidence))
            if key["evidence_requirement"]=="one_off_tentative":evidence=tentative_index[profile.user_id]
            key["gold_evidence_session_ids"]=evidence
            key["visible_history_scope"]={"through_session_index":TOTAL_SESSIONS_PER_USER-1,"session_count":TOTAL_SESSIONS_PER_USER}
            public_probes.append(pub);private_keys.append(key)
            indices=[int(s.rsplit("s",1)[-1]) for s in evidence if re.search(r"_s\d+$",s)]
            distances=[(TOTAL_SESSIONS_PER_USER-1)-i for i in indices]
            difficulty={
                "probe_id":pid,"user_id":profile.user_id,"probe_type":blueprint,"target_habit_count":len(key["target_habit_ids"]),
                "evidence_count":len(evidence),"max_evidence_distance":max(distances) if distances else "",
                "median_evidence_distance":statistics.median(distances) if distances else "",
                "query_token_estimate":token_count(pub["query"]+json.dumps(pub["choices"])),
            }
            difficulty_rows.append(difficulty)
            review_rows.append({
                "probe_id":pid,"user_id":profile.user_id,"domain":profile.domain,"query":pub["query"],
                "choices_json":json.dumps(pub["choices"],ensure_ascii=False),"proposed_gold_choice_id":key["gold_choice_id"],
                "proposed_gold_action":key["gold_action_text"],"probe_type":blueprint,
                "target_habit_ids_json":json.dumps(key["target_habit_ids"]),"evidence_session_ids_json":json.dumps(evidence),
                "difficulty_json":json.dumps(difficulty),"reviewer_decision":"","reviewer_notes":"",
            })

    user_habit_rows: list[dict[str, Any]] = []
    for profile in personas:
        for hid in assignments[profile.user_id]:
            evidence_by_kind = {
                kind: evidence_index.get((profile.user_id, hid, kind), [])
                for kind in ("support", "old", "new", "boundary", "exception", "composition", "priority")
            }
            active_variant = variant_record(profile, hid, "active")
            old_variant = variant_record(profile, hid, "old")
            user_habit_rows.append({
                "user_id": profile.user_id,
                "domain": profile.domain,
                "habit_id": hid,
                "habit_family": HABIT_BY_ID[hid]["family"],
                "condition": HABIT_BY_ID[hid]["condition"],
                "active_variant_id": active_variant["variant_id"],
                "active_variant_label": active_variant["label"],
                "active_variant_action": active_variant["action"],
                "is_drift_updated": hid in drift[profile.user_id],
                "old_variant_id": old_variant["variant_id"] if hid in drift[profile.user_id] else "",
                "old_variant_label": old_variant["label"] if hid in drift[profile.user_id] else "",
                "support_evidence_count": len(evidence_by_kind["support"]),
                "old_evidence_count": len(evidence_by_kind["old"]),
                "new_evidence_count": len(evidence_by_kind["new"]),
                "boundary_evidence_count": len(evidence_by_kind["boundary"]),
                "exception_evidence_count": len(evidence_by_kind["exception"]),
                "composition_evidence_count": len(evidence_by_kind["composition"]),
                "priority_evidence_count": len(evidence_by_kind["priority"]),
                "evidence_session_ids_json": json.dumps(list(dict.fromkeys(sum(evidence_by_kind.values(), [])))),
            })
    habit_user_summary_rows: list[dict[str, Any]] = []
    for hid in RETAINED_HABIT_IDS:
        rows = [r for r in user_habit_rows if r["habit_id"] == hid]
        habit_user_summary_rows.append({
            "habit_id": hid,
            "domain": HABIT_BY_ID[hid]["domain"],
            "habit_family": HABIT_BY_ID[hid]["family"],
            "assigned_user_count": len(rows),
            "drift_updated_user_count": sum(bool(r["is_drift_updated"]) for r in rows),
            "active_variant_distribution_json": json.dumps(dict(Counter(r["active_variant_id"] for r in rows))),
        })

    # Validation and length/diversity audits.
    entity_rows = probe_entity_audit(public_probes, private_keys)
    session_diversity_rows = session_diversity_audit(public_lifelines)
    shortcut_rows = choice_shortcut_audit(public_probes, private_keys)
    history_free_summary, history_free_rows = history_free_shortcut_audit(public_probes, private_keys)
    longest_baseline_accuracy = statistics.mean(r["longest_baseline_credit"] for r in shortcut_rows)
    shortest_baseline_accuracy = statistics.mean(r["shortest_baseline_credit"] for r in shortcut_rows)
    gold_choice_distribution = dict(Counter(k["gold_choice_id"] for k in private_keys))
    errors=[]
    if any(not r["pass"] for r in identity_rows):errors.append("identity consistency audit failed")
    if any(not r["pass"] for r in session_diversity_rows):errors.append("session diversity audit failed")
    if any(not r["pass"] for r in entity_rows):errors.append("probe choice entity consistency audit failed")
    if longest_baseline_accuracy > 0.35:errors.append("answer-length shortcut audit failed")
    if any(not r["active_assignment_balance_pass"] for r in variant_balance_rows):errors.append("policy variant assignment imbalance")
    if history_free_summary["choice_only"]["accuracy"] > 0.42:errors.append("choice-only history-free shortcut audit failed")
    if history_free_summary["query_plus_choice"]["accuracy"] > 0.55:errors.append("query-plus-choice history-free shortcut audit failed")
    if max(gold_choice_distribution.values()) - min(gold_choice_distribution.values()) > 1:errors.append("gold choice position imbalance")
    retained_source_total=sum(len(v) for v in records_by_domain.values())
    unique_used_source_total=len({(r["domain"],r["source_conversation_id"]) for r in usage_rows})
    if unique_used_source_total != retained_source_total:errors.append("not all retained source conversations were used exactly once")
    if len(usage_rows) != unique_used_source_total:errors.append("source conversation reuse detected")
    if len({p["probe_id"] for p in public_probes})!=len(public_probes):errors.append("duplicate probe_id")
    query_norm=[normalize_text(p["query"]) for p in public_probes]
    choice_sigs=[tuple(sorted(normalize_text(c["text"]) for c in p["choices"])) for p in public_probes]
    if len(set(query_norm))!=len(query_norm):errors.append("duplicate normalized queries")
    if len(set(choice_sigs))!=len(choice_sigs):errors.append("duplicate choice sets")
    leak_re=re.compile(r"(?i)what should the assistant|according to (?:your|the user.?s) habit|use the history|habit benchmark|default for this user")
    leakage=[p["probe_id"] for p in public_probes if leak_re.search(p["query"])]
    if leakage:errors.append("benchmark-intent leakage in queries")
    keys={k["probe_id"]:k for k in private_keys}
    for p in public_probes:
        ids={c["choice_id"] for c in p["choices"]}
        if ids!={"A","B","C","D"}:errors.append(f"invalid choices {p['probe_id']}")
        if keys[p["probe_id"]]["gold_choice_id"] not in ids:errors.append(f"missing gold {p['probe_id']}")

    # Render full prompts and count actual tokenizer tokens where available.
    life_by_user={l["user_id"]:l for l in public_lifelines}
    prompt_tokens=[]
    example_text=""
    for i,p in enumerate(public_probes):
        life=life_by_user[p["user_id"]]
        pieces=["You are evaluating a long-horizon user-memory agent. Use the previous sessions and choose the best response to the current request.\n"]
        for s in life["sessions"]:
            pieces.append(f"\n[Session {s['session_index']} | {s['timestamp']}]\n")
            for m in s["messages"]:pieces.append(m["role"].upper()+": "+m["content"]+"\n")
        pieces.append("\nCURRENT REQUEST:\n"+p["query"]+"\n\nCANDIDATE RESPONSES:\n")
        for c in p["choices"]:pieces.append(f"{c['choice_id']}. {c['text']}\n")
        pieces.append(f'\nReturn JSON only: {{"probe_id":"{p["probe_id"]}","choice_id":"..."}}')
        text="".join(pieces);prompt_tokens.append(token_count(text))
        if i==0:example_text=text

    query_stats=nearest_stats([p["query"] for p in public_probes])
    choice_texts=[c["text"] for p in public_probes for c in p["choices"]]
    # Within-choice semantic similarity: close distractors are desirable, exact duplicates are not.
    within=[]
    for p in public_probes:
        X=TfidfVectorizer(stop_words="english",ngram_range=(1,2)).fit_transform([c["text"] for c in p["choices"]])
        S=cosine_similarity(X)
        within.append(sum(S[i,j] for i in range(4) for j in range(i+1,4))/6)

    validation={
        "dataset_id":DATASET_ID,"source_dataset":SOURCE_DATASET,"selected_domains":list(DOMAINS),
        "pseudo_users":len(personas),"retained_habit_count":len(HABITS),"active_habits_per_user":6,
        "sessions_per_user":TOTAL_SESSIONS_PER_USER,"generated_sessions":len(private_sessions),
        "source_grounded_sessions":len(usage_rows),
        "retained_source_conversations_total":retained_source_total,
        "all_retained_source_conversations_used_exactly_once":unique_used_source_total==retained_source_total and len(usage_rows)==unique_used_source_total,
        "retained_source_conversations_by_domain":{d:len(records_by_domain[d]) for d in DOMAINS},
        "probes":len(public_probes),
        "probe_type_counts":dict(Counter(k["probe_type"] for k in private_keys)),
        "target_habit_count_distribution":dict(Counter(len(k["target_habit_ids"]) for k in private_keys)),
        "multi_habit_probe_count":sum(len(k["target_habit_ids"])>1 for k in private_keys),
        "unique_normalized_queries":len(set(query_norm)),"unique_choice_sets":len(set(choice_sigs)),
        "unique_choice_texts":len(set(normalize_text(x) for x in choice_texts)),
        "query_nearest_neighbor_cosine":query_stats,
        "within_choice_set_cosine":{"mean":round(statistics.mean(within),4),"p50":round(statistics.median(within),4),"p95":round(percentile(within,.95),4)},
        "gold_choice_distribution":gold_choice_distribution,
        "longest_choice_baseline_accuracy":round(longest_baseline_accuracy,4),
        "shortest_choice_baseline_accuracy":round(shortest_baseline_accuracy,4),
        "mean_gold_minus_distractor_word_count":round(statistics.mean(r["gold_minus_distractor_mean"] for r in shortcut_rows),4),
        "history_free_shortcut_audit":history_free_summary,
        "policy_variant_count_per_habit":4,
        "policy_variant_assignment_balance_pass":all(r["active_assignment_balance_pass"] for r in variant_balance_rows),
        "identity_audit_passed_users":sum(r["pass"] for r in identity_rows),"identity_audit_total_users":len(identity_rows),
        "session_diversity_passed_users":sum(r["pass"] for r in session_diversity_rows),"session_diversity_total_users":len(session_diversity_rows),
        "min_unique_first_user_request_ratio":min(r["unique_first_user_request_ratio"] for r in session_diversity_rows),
        "min_unique_full_session_ratio":min(r["unique_full_session_ratio"] for r in session_diversity_rows),
        "probe_choice_entity_consistency_passed":sum(r["pass"] for r in entity_rows),"probe_choice_entity_consistency_total":len(entity_rows),
        "unique_source_conversations_used":unique_used_source_total,
        "source_conversation_reuse_count":len(usage_rows)-unique_used_source_total,
        "query_leakage_count":len(leakage),
        "full_prompt_tokens":{"min":min(prompt_tokens),"median":int(statistics.median(prompt_tokens)),"mean":round(statistics.mean(prompt_tokens),1),"p95":int(percentile(prompt_tokens,.95)),"max":max(prompt_tokens)},
        "evidence_distance":{"median_of_probe_medians":statistics.median([r["median_evidence_distance"] for r in difficulty_rows if r["median_evidence_distance"]!=""]),"max":max([r["max_evidence_distance"] for r in difficulty_rows if r["max_evidence_distance"]!=""])},
        "validation_errors":errors,
    }

    # Outputs.
    write_jsonl(out/"public/lifelines.jsonl",public_lifelines)
    write_jsonl(out/"public/probes.jsonl",public_probes)
    write_jsonl(out/"private/sessions_with_annotations.jsonl",private_sessions)
    write_jsonl(out/"private/probe_key.jsonl",private_keys)
    write_jsonl(out/"private/persona_profiles.jsonl",persona_rows)
    write_csv(out/"review/multidogo_finance_software_v05_review_queue_all.csv",review_rows)
    write_csv(out/"source/source_conversation_usage_manifest.csv",usage_rows)
    write_csv(out/"source/source_event_clusters.csv",cluster_rows)
    write_csv(out/"source/habit_source_audit.csv",habit_audit)
    write_csv(out/"reports/identity_consistency_audit.csv",identity_rows)
    write_csv(out/"reports/session_diversity_audit.csv",session_diversity_rows)
    write_csv(out/"reports/probe_choice_entity_consistency_audit.csv",entity_rows)
    write_csv(out/"reports/choice_shortcut_audit.csv",shortcut_rows)
    write_csv(out/"reports/history_free_shortcut_per_probe.csv",history_free_rows)
    write_csv(out/"reports/policy_variant_balance_audit.csv",variant_balance_rows)
    write_csv(out/"reports/probe_difficulty_audit.csv",difficulty_rows)
    write_csv(out/"reports/user_habit_mapping.csv",user_habit_rows)
    write_csv(out/"reports/habit_user_summary.csv",habit_user_summary_rows)
    (out/"reports/history_free_shortcut_audit.json").write_text(json.dumps(history_free_summary,ensure_ascii=False,indent=2),encoding="utf-8")
    habit_templates_with_variants=[{**h,"policy_variants":POLICY_VARIANTS[h["habit_id"]]} for h in HABITS]
    (out/"source/habit_templates_retained.json").write_text(json.dumps(habit_templates_with_variants,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"source/source_file_manifest.json").write_text(json.dumps(source_manifests,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"reports/validation_report.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"model_eval/example_full_prompt.txt").write_text(example_text,encoding="utf-8")
    shutil.copy2(args.finance_source,out/"source/raw_multidogo_finance.csv.gz")
    shutil.copy2(args.software_source,out/"source/raw_multidogo_software.csv.gz")
    shutil.copy2(Path(__file__),out/"scripts/build_multidogo_coherent_multihabit_v05.py")
    shutil.copy2(LIB_PATH,out/"scripts/lib_v04_generation.py")

    # Minimal scorer.
    scorer='''#!/usr/bin/env python3\nimport argparse,json,csv\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--dataset-dir",required=True);p.add_argument("--predictions",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--method-name",default="method");a=p.parse_args()\nbase=Path(a.dataset_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)\nkeys={r["probe_id"]:r for r in (json.loads(x) for x in (base/"private/probe_key.jsonl").read_text().splitlines() if x.strip())}\npreds={r["probe_id"]:r for r in (json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip())}\nif set(preds)!=set(keys):raise SystemExit(f"coverage mismatch missing={len(set(keys)-set(preds))} extra={len(set(preds)-set(keys))}")\nrows=[]\nfor pid,k in keys.items():rows.append({"probe_id":pid,"correct":int(preds[pid]["choice_id"]==k["gold_choice_id"]),"probe_type":k["probe_type"],"target_habit_count":len(k["target_habit_ids"])})\nsummary={"method_name":a.method_name,"total":len(rows),"accuracy":sum(r["correct"] for r in rows)/len(rows)}\nfor t in sorted({r["probe_type"] for r in rows}):\n s=[r for r in rows if r["probe_type"]==t];summary[f"accuracy__{t}"]=sum(r["correct"] for r in s)/len(s)\n(out/"metrics.json").write_text(json.dumps(summary,indent=2))\nwith (out/"per_probe.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)\nprint(json.dumps(summary,indent=2))\n'''
    (out/"scripts/score_predictions.py").write_text(scorer,encoding="utf-8")

    # Reports and README.
    identity_report=f"""# Identity coherence report\n\n- Users: {len(identity_rows)}\n- Passed: {sum(r['pass'] for r in identity_rows)}\n- Failed: {sum(not r['pass'] for r in identity_rows)}\n- Method: original MultiDoGO utterances are never concatenated into a pseudo-user. Each source conversation is reduced to a sanitized task event, then rewritten with one fixed persona, account/device set, city, employer, and recurring context.\n"""
    (out/"reports/identity_coherence_report.md").write_text(identity_report,encoding="utf-8")
    diversity_report=f"""# Probe difficulty and diversity report\n\n- Probes: {len(public_probes)}\n- Unique normalized queries: {len(set(query_norm))}\n- Unique unordered choice sets: {len(set(choice_sigs))}\n- Multi-habit probes: {sum(len(k['target_habit_ids'])>1 for k in private_keys)}\n- Query nearest-neighbor cosine: `{json.dumps(query_stats)}`\n- Full-prompt tokens: `{json.dumps(validation['full_prompt_tokens'])}`\n- Median evidence distance: {validation['evidence_distance']['median_of_probe_medians']} sessions\n- Longest evidence distance: {validation['evidence_distance']['max']} sessions\n- Query leakage count: {len(leakage)}\n\nHard cases include two- and three-habit composition, safety-priority ordering, boundary plus active habit, updated preference plus composition, updated preference plus exception, absent-habit false personalization, and one-off weak evidence.\n"""
    (out/"reports/probe_diversity_and_difficulty_report.md").write_text(diversity_report,encoding="utf-8")
    readme=f"""# MultiDoGO Finance + Software HABIT-Bench candidate v0.5\n\nThis is a **source-grounded, synthetic-longitudinal** candidate built from MultiDoGO finance and software customer-agent conversations. It is not a claim that MultiDoGO contains natural same-user longitudinal identities.\n\n## What changed from v0.4\n\n1. **Identity coherence by construction:** raw customer dialogues are not concatenated. Source conversations are converted to sanitized task events and rewritten under one stable persona.\n2. **Longer histories:** {TOTAL_SESSIONS_PER_USER} model-visible sessions per user across nearly three years.\n3. **Multi-habit users:** six active habits, two updated habits, one conditionally scoped habit, and one deliberately tentative one-off signal per user.\n4. **Harder probes:** normal end-user requests; no `What should the assistant do?` meta-question. Probes include two/three-habit composition, priority conflict, drift, nested exceptions, false personalization, and insufficient evidence.\n5. **Diversity:** {len(set(query_norm))}/{len(public_probes)} unique normalized queries and {len(set(choice_sigs))}/{len(public_probes)} unique choice sets.\n\n## Dataset size\n\n- Domains: finance, software\n- Pseudo-users: {len(personas)}\n- Sessions: {len(private_sessions)} ({TOTAL_SESSIONS_PER_USER} per user)\n- Retained habits: {len(HABITS)}\n- Probes: {len(public_probes)}\n- Median full prompt: {validation['full_prompt_tokens']['median']} tokens\n- Identity audit: {sum(r['pass'] for r in identity_rows)}/{len(identity_rows)} passed\n\n## Evaluation\n\nGive a method only `public/lifelines.jsonl` and `public/probes.jsonl`. The method outputs one JSONL row per probe:\n\n```json\n{{"probe_id":"mdgo_v05_probe_000000","choice_id":"A"}}\n```\n\nScore with:\n\n```bash\npython scripts/score_predictions.py --dataset-dir . --predictions predictions.jsonl --output-dir runs/my_eval --method-name my_method\n```\n\nPrivate files contain gold answers, evidence links, persona profiles, and source provenance. This candidate must still receive human review before paper-scale use.\n"""
    (out/"README.md").write_text(readme,encoding="utf-8")
    (out/"CHANGELOG_FROM_V04.md").write_text("""# Changelog from v0.4\n\n- Replaced raw-dialogue splicing with task-event extraction and persona-conditioned rewriting.\n- Increased history from 71 to 240 sessions per user.\n- Reduced pseudo-users from 40 to 24 so each lifeline can be substantially longer and denser.\n- Increased active habits from 5 to 6 per user, with two temporal updates and a tentative one-off signal.\n- Replaced 10 probes/user with 18 harder probes/user.\n- Added three-habit composition, mixed boundary/support, mixed exception/support, drift composition, drift exception, conditional scope, and insufficient-evidence probes.\n- Added identity, difficulty, evidence-distance, token-length, and source-cluster audits.\n""",encoding="utf-8")

    # Gold smoke test.
    gold_path=out/"reports/gold_predictions_smoke_test.jsonl"
    write_jsonl(gold_path,[{"probe_id":k["probe_id"],"choice_id":k["gold_choice_id"]} for k in private_keys])
    # Bundle.
    zip_path=Path(args.zip_path)
    if zip_path.exists():zip_path.unlink()
    with zipfile.ZipFile(zip_path,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():z.write(f,Path(out.name)/f.relative_to(out))
    print(json.dumps(validation,ensure_ascii=False,indent=2),flush=True)
    print(f"wrote {out} and {zip_path}",flush=True)
    if errors:raise SystemExit("validation errors: "+"; ".join(errors))


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--finance-source",default="/mnt/data/mdgo_v05_work/source/raw_multidogo_finance.csv.gz")
    p.add_argument("--software-source",default="/mnt/data/mdgo_v05_work/source/raw_multidogo_software.csv.gz")
    p.add_argument("--output-dir",default=f"/mnt/data/{DATASET_ID}")
    p.add_argument("--zip-path",default=f"/mnt/data/{DATASET_ID}.zip")
    p.add_argument("--seed",type=int,default=SEED)
    return p.parse_args()

if __name__=="__main__":build(parse_args())
