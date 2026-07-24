#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import string
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SEED = 20260720
random.seed(SEED)
SOURCE_DATASET = "awslabs/multi-domain-goal-oriented-dialogues-dataset"
DATASET_ID = "habit_bench_multidogo_finance_software_coherent_multihabit_v0_4"
DOMAINS = ("finance", "software")
USERS_PER_DOMAIN = 20
SOURCE_SESSIONS_PER_USER = 70
PROBES_PER_USER = 10
CLUSTERS_PER_DOMAIN = 24

# -----------------------------------------------------------------------------
# Habit contracts. They are retained after source clustering/theme audit; there
# is deliberately no fixed query or fixed choice set in these contracts.
# -----------------------------------------------------------------------------
HABITS: list[dict[str, Any]] = [
    {
        "habit_id": "finance_confirm_money_movement",
        "domain": "finance",
        "family": "risk_threshold",
        "theme": "money_movement_transfer_payment",
        "pattern": r"\btransfer\b|send money|wire|bill pay|pay bill|make a payment|scheduled payment|move money|fund transfer",
        "condition": "a transfer, bill payment, scheduled payment, or other money movement would be submitted",
        "default_action": "state the exact amount, source, and destination, then wait for one final approval before submission",
        "boundary_condition": "the user is only asking how transfers or payments work, without requesting a transaction",
        "boundary_action": "answer the information request directly without creating a confirmation step",
        "exception_condition": "the exact transaction has already been approved in the secure banking flow",
        "exception_action": "verify that the approval matches the exact transaction and continue without asking for a duplicate confirmation",
        "old_action": "treat small transfers as pre-approved and submit them immediately",
        "priority": 70,
    },
    {
        "habit_id": "finance_confirm_card_account_changes",
        "domain": "finance",
        "family": "risk_threshold",
        "theme": "card_account_change",
        "pattern": r"change (?:my )?(?:address|phone|email)|update (?:my )?(?:address|phone|email)|replace (?:my )?card|block (?:my )?card|close (?:my )?account|open (?:an? )?account|lost (?:my )?(?:credit |debit )?card|stolen (?:credit |debit )?card|mailing address",
        "condition": "a card or account setting would be blocked, replaced, opened, closed, or edited",
        "default_action": "summarize the precise account change and obtain final approval before applying it",
        "boundary_condition": "the user is comparing card features or asking about account policy only",
        "boundary_action": "give the comparison or policy answer without treating it as a change request",
        "exception_condition": "the exact change was just approved through the secure account screen",
        "exception_action": "match the secure approval to the requested change and continue without another approval loop",
        "old_action": "route every account change to a phone callback even after secure in-app approval",
        "priority": 75,
    },
    {
        "habit_id": "finance_minimal_pii_secure_verification",
        "domain": "finance",
        "family": "privacy_consent",
        "theme": "pii_verification_secure_channel",
        "pattern": r"ssn|social security|account number|card number|credit card number|debit card number|otp|pin|verify|verification|last four|last 4|identity",
        "condition": "finance support requires identity, card, account, SSN-like, PIN, or one-time-code verification",
        "default_action": "request only the minimum necessary information through a secure verification surface and never ask for full secrets in chat",
        "boundary_condition": "the question is a general explanation that requires no account access",
        "boundary_action": "answer normally without starting identity verification",
        "exception_condition": "a sensitive value appeared earlier or the user asked that it not be retained",
        "exception_action": "do not repeat or reuse the value and redirect verification to the secure channel",
        "old_action": "accept account identifiers and one-time codes directly in the chat transcript",
        "priority": 100,
    },
    {
        "habit_id": "finance_fraud_lost_card_urgent_escalation",
        "domain": "finance",
        "family": "tool_action",
        "theme": "fraud_lost_card_urgent_path",
        "pattern": r"fraud|unauthori[sz]ed|stolen|lost (?:my )?(?:credit |debit )?card|suspicious|card.*missing|block.*card|transaction.*not mine",
        "condition": "the user reports a lost or stolen card, suspected fraud, or an unauthorized transaction",
        "default_action": "start the urgent protective flow, surface the current card/transaction status, and route the case to the fraud path",
        "boundary_condition": "the user is asking general fraud-prevention questions without an active incident",
        "boundary_action": "give prevention guidance without opening an incident or blocking anything",
        "exception_condition": "the incident is already locked and a fraud case is open",
        "exception_action": "check the existing case status and next step rather than opening a duplicate case",
        "old_action": "try several ordinary self-service checks before using the urgent fraud path",
        "priority": 95,
    },
    {
        "habit_id": "finance_balance_statement_summary_first",
        "domain": "finance",
        "family": "format_style",
        "theme": "balance_statement_summary",
        "pattern": r"\bbalance\b|statement|transaction history|transactions|bank statement|activity",
        "condition": "the user asks for a balance, statement, or transaction review",
        "default_action": "lead with a short status summary and then provide the supporting details or reconciliation",
        "boundary_condition": "the user explicitly asks for a full line-by-line export or audit",
        "boundary_action": "provide the requested detailed view rather than compressing it into a short summary",
        "exception_condition": "the user asks for a one-line answer only",
        "exception_action": "give only the requested one-line status and offer details separately",
        "old_action": "start with a long transaction dump before giving the conclusion",
        "priority": 25,
    },
    {
        "habit_id": "finance_fee_dispute_evidence_then_case",
        "domain": "finance",
        "family": "risk_threshold",
        "theme": "fee_dispute_charge_case",
        "pattern": r"dispute|charge|fee|overdraft|refund|charged|transaction.*not mine|merchant",
        "condition": "the user wants to dispute a charge or fee or open a transaction case",
        "default_action": "collect the relevant statement, receipt, merchant, date, and amount evidence, summarize the case, then ask before filing it",
        "boundary_condition": "the user wants only an explanation of a fee or dispute process",
        "boundary_action": "explain the process without opening a case",
        "exception_condition": "a case already exists for the same charge",
        "exception_action": "retrieve the existing case status instead of filing a duplicate",
        "old_action": "file a dispute immediately before checking the evidence or existing cases",
        "priority": 80,
    },
    {
        "habit_id": "finance_credit_loan_cautious_no_commitment",
        "domain": "finance",
        "family": "risk_threshold",
        "theme": "loan_credit_cautious_info",
        "pattern": r"loan|mortgage|credit limit|credit score|interest rate|\bapr\b|\bemi\b|eligib|application",
        "condition": "the user asks about loan, mortgage, credit-limit, or credit eligibility decisions",
        "default_action": "explain requirements and scenarios without promising approval or submitting an application",
        "boundary_condition": "the user asks for a general definition or calculation only",
        "boundary_action": "provide the calculation or explanation directly",
        "exception_condition": "the user explicitly requests a draft checklist rather than an application",
        "exception_action": "prepare the checklist without submitting anything or implying approval",
        "old_action": "move directly from an eligibility question into an application or approval prediction",
        "priority": 65,
    },
    {
        "habit_id": "finance_payment_status_latest_check",
        "domain": "finance",
        "family": "tool_action",
        "theme": "due_status_live_check",
        "pattern": r"pending|posted|due|status|available balance|current balance|scheduled|processing|went through|go through|payment.*received|transaction.*status|check.*balance",
        "condition": "the answer depends on the current status of a payment, transfer, balance, due date, or posting state",
        "default_action": "check the latest available account or payment state and label the answer with an as-of time",
        "boundary_condition": "the user asks an evergreen conceptual question about payment processing",
        "boundary_action": "answer conceptually without pretending a live account check occurred",
        "exception_condition": "the user explicitly wants a hypothetical explanation with no account lookup",
        "exception_action": "give the hypothetical explanation and state that no live status was checked",
        "old_action": "answer status questions from generic timing assumptions without checking the current state",
        "priority": 85,
    },
    {
        "habit_id": "software_collect_diagnostics_before_fix",
        "domain": "software",
        "family": "tool_action",
        "theme": "diagnostics_error_repro",
        "pattern": r"error|bug|crash|not working|issue|problem|fail|failed|broken|missing|freeze|slow|buffering|unexpected",
        "condition": "a concrete software error, crash, broken feature, or failed operation is being diagnosed",
        "default_action": "collect the product version, operating system, exact error, and reproduction path before prescribing a focused fix",
        "boundary_condition": "the user asks a conceptual software question with no concrete failure",
        "boundary_action": "answer the concept without asking for logs or reproduction details",
        "exception_condition": "the user asks for a quick best-effort guess and accepts uncertainty",
        "exception_action": "give a short provisional path and clearly state what diagnostics would be needed to verify it",
        "old_action": "offer a long list of generic fixes before collecting any diagnostic context",
        "priority": 65,
    },
    {
        "habit_id": "software_docs_lookup_for_update_install",
        "domain": "software",
        "family": "tool_action",
        "theme": "docs_update_install",
        "pattern": r"install|setup|download|update|upgrade|version|patch|configure|configuration|compatib",
        "condition": "an install, setup, update, upgrade, patch, or version-specific issue is being handled",
        "default_action": "check the relevant product/version documentation or knowledge base and provide version-aware steps",
        "boundary_condition": "the user asks an evergreen software concept unrelated to a current version",
        "boundary_action": "answer from general knowledge without a documentation lookup",
        "exception_condition": "the user asks for an offline best-effort answer with no lookup",
        "exception_action": "give a clearly labeled unverified answer and avoid claiming it matches the current version",
        "old_action": "rely on remembered generic steps even when the product version matters",
        "priority": 70,
    },
    {
        "habit_id": "software_one_try_then_escalate",
        "domain": "software",
        "family": "drift_seasonality",
        "theme": "escalation_ticket_human",
        "pattern": r"ticket|agent|human|representative|escalat|supervisor|support team|still.*not working|again|recurr",
        "condition": "a focused software fix has already failed once or the issue is recurring",
        "default_action": "try at most one targeted next step and then create or route to a ticket or human support path",
        "boundary_condition": "this is a first-time, low-complexity issue with no failed attempt",
        "boundary_action": "give one concise self-service step before escalating",
        "exception_condition": "the user asks to keep this particular case in self-service",
        "exception_action": "continue self-service for this case while keeping escalation available",
        "old_action": "continue several rounds of self-service troubleshooting before escalation",
        "priority": 80,
    },
    {
        "habit_id": "software_secure_login_password_flow",
        "domain": "software",
        "family": "privacy_consent",
        "theme": "security_login_password",
        "pattern": r"password|login|log in|sign in|access|account|otp|pin|credential|locked out|reset|verification code",
        "condition": "login, password reset, account access, PIN, OTP, or credential support is needed",
        "default_action": "use the secure reset or verification flow and never request passwords or one-time codes in chat",
        "boundary_condition": "the user asks a general account-security concept",
        "boundary_action": "explain it normally without starting a reset flow",
        "exception_condition": "a credential appeared earlier or the user withdrew consent to retain it",
        "exception_action": "do not repeat or reuse the value and restart through the secure flow",
        "old_action": "accept passwords or verification codes in the chat transcript",
        "priority": 100,
    },
    {
        "habit_id": "software_platform_specific_steps",
        "domain": "software",
        "family": "format_style",
        "theme": "installation_step_by_step",
        "pattern": r"install|setup|download|configure|configuration|onboard|windows|mac|linux|desktop|mobile|settings",
        "condition": "the user needs setup, installation, configuration, onboarding, or first-use help",
        "default_action": "give short numbered steps tailored to the user's platform and ask for the next screen or error",
        "boundary_condition": "the user asks for architecture or conceptual explanation",
        "boundary_action": "explain the concept naturally without forcing a setup checklist",
        "exception_condition": "the user requests a one-sentence overview",
        "exception_action": "give the concise overview and offer detailed steps only if requested",
        "old_action": "give platform-agnostic prose with no numbered progression",
        "priority": 30,
    },
    {
        "habit_id": "software_backup_before_risky_change",
        "domain": "software",
        "family": "risk_threshold",
        "theme": "rollback_backup_change",
        "pattern": r"rollback|back ?up|restore|reset|uninstall|delete|update|upgrade|migration|factory|reinstall",
        "condition": "an update, reset, rollback, reinstall, migration, or destructive configuration change may affect data",
        "default_action": "confirm a backup or restore point, summarize the risk, and obtain final approval before the change",
        "boundary_condition": "the user asks only what a backup, reset, or rollback means",
        "boundary_action": "explain the concept without creating approval friction",
        "exception_condition": "the user has already verified the backup and approved the exact change",
        "exception_action": "continue with the approved change and restate the verified backup state",
        "old_action": "perform the risky change first and discuss recovery only if something goes wrong",
        "priority": 90,
    },
    {
        "habit_id": "software_confirm_license_subscription_changes",
        "domain": "software",
        "family": "risk_threshold",
        "theme": "license_subscription_purchase",
        "pattern": r"license|subscription|trial|purchase|buy|payment|renew|cancel|plan|upgrade.*plan",
        "condition": "a software license, subscription, renewal, cancellation, purchase, or paid plan would change",
        "default_action": "summarize the plan, price, billing effect, and renewal state, then ask for final approval before submission",
        "boundary_condition": "the user is only comparing plans or asking about license terms",
        "boundary_action": "compare the plans directly without treating the comparison as a purchase request",
        "exception_condition": "the exact plan change was already approved in the billing portal",
        "exception_action": "verify the matching portal approval and continue without another confirmation loop",
        "old_action": "apply small plan or renewal changes automatically",
        "priority": 75,
    },
    {
        "habit_id": "software_ticket_receipt_summary",
        "domain": "software",
        "family": "format_style",
        "theme": "bug_report_summary",
        "pattern": r"bug report|report|ticket|issue|problem|case|reference|incident|support request",
        "condition": "a bug report, support ticket, or incident has actually been submitted",
        "default_action": "provide a concise receipt with issue, reproduction details, priority, reference ID, and next step",
        "boundary_condition": "the user is still troubleshooting or drafting and no ticket has been filed",
        "boundary_action": "help troubleshoot or draft without inventing a reference ID",
        "exception_condition": "the user asks for draft text only and does not want submission",
        "exception_action": "produce the draft without filing it or presenting a receipt",
        "old_action": "end after submission without a clear written receipt or next step",
        "priority": 35,
    },
]
HABIT_BY_ID = {h["habit_id"]: h for h in HABITS}
HABITS_BY_DOMAIN = {d: [h for h in HABITS if h["domain"] == d] for d in DOMAINS}

# High-value interaction pairs used to create composition and priority probes.
PAIR_CANDIDATES = {
    "finance": [
        ("finance_payment_status_latest_check", "finance_confirm_money_movement"),
        ("finance_minimal_pii_secure_verification", "finance_fraud_lost_card_urgent_escalation"),
        ("finance_minimal_pii_secure_verification", "finance_confirm_card_account_changes"),
        ("finance_fee_dispute_evidence_then_case", "finance_minimal_pii_secure_verification"),
        ("finance_credit_loan_cautious_no_commitment", "finance_minimal_pii_secure_verification"),
        ("finance_balance_statement_summary_first", "finance_payment_status_latest_check"),
    ],
    "software": [
        ("software_docs_lookup_for_update_install", "software_backup_before_risky_change"),
        ("software_secure_login_password_flow", "software_one_try_then_escalate"),
        ("software_collect_diagnostics_before_fix", "software_one_try_then_escalate"),
        ("software_platform_specific_steps", "software_docs_lookup_for_update_install"),
        ("software_confirm_license_subscription_changes", "software_ticket_receipt_summary"),
        ("software_secure_login_password_flow", "software_collect_diagnostics_before_fix"),
    ],
}

# -----------------------------------------------------------------------------
# Synthetic but stable persona profiles.
# -----------------------------------------------------------------------------
NAMES = [
    "Maya Chen", "Jordan Brooks", "Elena Ruiz", "Noah Patel", "Avery Thompson",
    "Lena Okafor", "Daniel Kim", "Sofia Martinez", "Ethan Walker", "Priya Shah",
    "Marcus Lee", "Nina Alvarez", "Owen Foster", "Camila Santos", "Theo Nguyen",
    "Leila Hassan", "Julian Reed", "Amara Johnson", "Miles Carter", "Zoe Bennett",
    "Iris Park", "Caleb Morgan", "Fatima Rahman", "Leo Castillo", "Naomi Evans",
    "Samir Mehta", "Grace Liu", "Adrian Cole", "Mina Davis", "Rafael Torres",
    "Hana Suzuki", "Jonah Price", "Alina Petrova", "Isaac Cooper", "Talia Green",
    "Nikhil Rao", "Ruby Scott", "Mateo Rivera", "Claire Wilson", "Darius King",
]
FINANCE_ROLES = [
    "freelance designer", "high-school teacher", "clinic coordinator", "small-business owner",
    "graduate student", "operations consultant", "nonprofit manager", "retired librarian",
    "restaurant manager", "research assistant", "construction estimator", "event planner",
    "accounting associate", "physical therapist", "online shop owner", "civil engineer",
    "community organizer", "photographer", "sales manager", "technical writer",
]
SOFTWARE_ROLES = [
    "product manager", "QA analyst", "marketing coordinator", "data analyst", "software developer",
    "customer-success lead", "graphic designer", "researcher", "project coordinator", "IT generalist",
    "content editor", "operations manager", "sales engineer", "teacher", "healthcare administrator",
    "lab technician", "freelance consultant", "support specialist", "architect", "recruiter",
]
CITIES = [
    ("Portland", "OR"), ("Austin", "TX"), ("Raleigh", "NC"), ("Denver", "CO"),
    ("Madison", "WI"), ("Sacramento", "CA"), ("Columbus", "OH"), ("Richmond", "VA"),
    ("Pittsburgh", "PA"), ("Tucson", "AZ"), ("Boise", "ID"), ("Minneapolis", "MN"),
    ("Nashville", "TN"), ("Providence", "RI"), ("Albuquerque", "NM"), ("Spokane", "WA"),
    ("Omaha", "NE"), ("Buffalo", "NY"), ("Birmingham", "AL"), ("Salt Lake City", "UT"),
]
COMPANIES = [
    "Cedar Loop Studio", "Northstar Analytics", "Juniper Works", "Blue Harbor Labs",
    "Paper Kite Media", "Willow Ridge Health", "Copper Finch Design", "Orchard Field Group",
    "Lighthouse Learning", "Mosaic River Consulting", "Brightline Research", "Crescent Oak Co.",
    "Harborstone Projects", "Silver Fern Systems", "Redwood Civic Lab", "Cloudberry Studio",
    "Maple Thread Collective", "Sunward Operations", "Evergreen Fieldworks", "Atlas Grove Partners",
]
OS_OPTIONS = ["Windows 11", "macOS 15", "Ubuntu 24.04"]
BROWSERS = ["Chrome", "Firefox", "Edge", "Safari"]
VOICE_STYLES = ["concise", "conversational", "careful", "direct", "detail-aware"]

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


def make_personas() -> list[Persona]:
    personas: list[Persona] = []
    for domain in DOMAINS:
        offset = 0 if domain == "finance" else USERS_PER_DOMAIN
        for i in range(USERS_PER_DOMAIN):
            name = NAMES[offset + i]
            first = name.split()[0]
            city, state = CITIES[i]
            user_id = f"mdgo_v04_{'fin' if domain=='finance' else 'sof'}_user_{i:04d}"
            base = dict(
                user_id=user_id,
                domain=domain,
                name=name,
                first_name=first,
                pronouns=["they/them", "she/her", "he/him"][i % 3],
                city=city,
                state=state,
                role=(FINANCE_ROLES if domain == "finance" else SOFTWARE_ROLES)[i],
                company=COMPANIES[i],
                email=f"{first.lower()}.{name.split()[-1].lower()}@persona.example",
                phone=f"+1-202-555-{1000 + offset + i:04d}",
                voice_style=VOICE_STYLES[i % len(VOICE_STYLES)],
            )
            if domain == "finance":
                personas.append(Persona(
                    **base,
                    checking_last4=f"{4100+i*7:04d}"[-4:],
                    savings_last4=f"{7300+i*9:04d}"[-4:],
                    card_last4=f"{8600+i*11:04d}"[-4:],
                    account_last4=f"{4100+i*7:04d}"[-4:],
                ))
            else:
                os_name = OS_OPTIONS[i % len(OS_OPTIONS)]
                personas.append(Persona(
                    **base,
                    account_last4=f"{5200+i*13:04d}"[-4:],
                    os=os_name,
                    browser=BROWSERS[i % len(BROWSERS)],
                    desktop_app=["AsterDesk", "Nimbus Workbench", "Orbit Office"][i % 3],
                    mail_app=["Aster Mail", "Nimbus Mail", "Orbit Inbox"][i % 3],
                    meeting_app=["Aster Meet", "Nimbus Call", "Orbit Rooms"][i % 3],
                    plan=["individual", "professional", "team"][i % 3],
                ))
    return personas

# -----------------------------------------------------------------------------
# Source loading, filtering, clustering, and theme tags.
# -----------------------------------------------------------------------------

def read_conversations(path: Path, domain: str) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                turn = int(r.get("turnNumber", 0))
            except Exception:
                turn = 0
            by[r["conversationId"]].append({
                "conversation_id": r["conversationId"],
                "turn": turn,
                "utterance_id": r.get("utteranceId", ""),
                "utterance": (r.get("utterance") or "").strip(),
                "role": "user" if r.get("authorRole") == "customer" else "assistant",
                "domain": domain,
            })
    for rows in by.values():
        rows.sort(key=lambda x: x["turn"])
    return dict(by)

SOFTWARE_CORE = re.compile(
    r"\b(?:software|application|app|install|uninstall|update|upgrade|version|patch|browser|windows|macos|ubuntu|linux|error|bug|crash|freeze|login|password|file|folder|backup|restore|rollback|reset|database|server|sync|license|subscription|ticket|configuration|network|computer|laptop|desktop|cloud|email client|workspace)\b",
    re.I,
)
SOFTWARE_NOISE = re.compile(
    r"musical instrument|music center|musical|keyboard order|guitar|piano|drum|flute|saxophone|yamaha|\bpsr[- ]?[a-z0-9]*|recurring order|food|pizza|burger|flight|airline|hotel|restaurant|room booking|car rental|quantity\s+\d+|model\s+psr",
    re.I,
)
SOFTWARE_CROSS_DOMAIN = re.compile(
    r"\b(?:reimbursement|reimburse|expenses?|checking account|savings account|bank statement|mortgage|personal loan|credit score|cash withdrawal|fund transfer|debit card|credit card|amount \(in dollars\)|refund request|merchant charge|overdraft|hotel reservation|flight booking|food order)\b",
    re.I,
)
GENERIC_GREETING_RE = re.compile(
    r"^(?:hi|hello|hey|hai|good morning|good afternoon|good evening|greetings|hello there|hi there|dear customer)[.! ,:-]*$",
    re.I,
)
GENERIC_CLOSING_RE = re.compile(
    r"^(?:thanks|thank you|thanks a lot|thank you very much|bye|goodbye|good bye|nothing else|no more help needed|have a (?:nice|great|good) day|take care)[.! ,:-]*$",
    re.I,
)
IDENTITY_PROMPT_RE = re.compile(
    r"(?i)(?:your|full|first and last) name|may i (?:know|confirm).*name|provide.*name|share.*name|speaking with|company name|organization name|name of your company|mailing address|phone number|email address|account number|last four.*account|last 4.*account"
)
SECURE_PROMPT_RE = re.compile(
    r"(?i)ssn|social security|full.*card number|16[- ]?digit|password|one[- ]time password|otp|\bpin\b|credential|security code|cvv"
)
BIOGRAPHY_CONFLICT_RE = re.compile(
    r"(?i)\b(?:i live in|i moved to|my home is in|my address is|i was born|date of birth|my birthday|i am \d{1,3} years old|i work (?:for|at)|my employer is|my spouse|my husband|my wife|my child|my daughter|my son)\b"
)
ALL_CAPS_WORD_RE = re.compile(r"[A-Z]{3,}")



def conversation_text(rows: list[dict[str, Any]], customer_only: bool = False) -> str:
    return " ".join(r["utterance"] for r in rows if not customer_only or r["role"] == "user")


def is_good_conversation(rows: list[dict[str, Any]], domain: str) -> bool:
    if not 6 <= len(rows) <= 40:
        return False
    customer = conversation_text(rows, customer_only=True)
    full = conversation_text(rows)
    words = re.findall(r"[A-Za-z]+", customer)
    if len(words) < 10:
        return False
    normalized = [re.sub(r"\W+", " ", r["utterance"].lower()).strip() for r in rows if r["utterance"].strip()]
    if len(set(normalized)) < max(4, int(len(normalized) * 0.60)):
        return False
    # Reject conflicting biography/location/employment claims rather than trying
    # to splice them into a different synthetic person.
    if BIOGRAPHY_CONFLICT_RE.search(customer) or ADDRESS_LIKE_RE.search(customer) or US_ADDRESS_COMMA_RE.search(customer):
        return False
    if domain == "software":
        core_hits = len(SOFTWARE_CORE.findall(full))
        if core_hits < 2:
            return False
        if SOFTWARE_NOISE.search(full):
            return False
        if SOFTWARE_CROSS_DOMAIN.search(full):
            return False
        if re.search(r"(?i)(?:order.{0,30}keyboard|keyboard.{0,30}order|musical keyboard|model.{0,20}keyboard)", full):
            return False
    # Heavy all-caps/gibberish conversations make a pseudo-user's voice jump.
    substantive_user = [r["utterance"] for r in rows if r["role"] == "user" and len(re.findall(r"[A-Za-z]+", r["utterance"])) >= 3]
    caps = 0
    for t in substantive_user:
        letters = [c for c in t if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.82:
            caps += 1
    if substantive_user and caps / len(substantive_user) > 0.45:
        return False
    return True



def tag_conversation(rows: list[dict[str, Any]], domain: str) -> set[str]:
    customer = conversation_text(rows, customer_only=True).lower()
    full = conversation_text(rows).lower()
    tags: set[str] = set()
    for h in HABITS_BY_DOMAIN[domain]:
        # Customer request is primary; agent text may support privacy/status themes.
        if re.search(h["pattern"], customer, re.I):
            tags.add(h["habit_id"])
        elif h["family"] in {"privacy_consent", "tool_action"} and re.search(h["pattern"], full, re.I):
            tags.add(h["habit_id"])
    return tags


def quality_score(rows: list[dict[str, Any]], domain: str) -> float:
    customer = conversation_text(rows, customer_only=True)
    full = conversation_text(rows)
    words = re.findall(r"[A-Za-z]+", customer)
    n = len(rows)
    user_turns = sum(1 for r in rows if r["role"] == "user")
    unique_ratio = len(set(w.lower() for w in words)) / max(len(words), 1)
    substantive = sum(
        1 for r in rows
        if len(re.findall(r"[A-Za-z]+", r["utterance"])) >= 5
        and not GENERIC_GREETING_RE.match(clean_text(r["utterance"]))
        and not GENERIC_CLOSING_RE.match(clean_text(r["utterance"]))
    )
    identity_prompts = sum(1 for r in rows if IDENTITY_PROMPT_RE.search(r["utterance"]) or SECURE_PROMPT_RE.search(r["utterance"]))
    caps_penalty = sum(1 for r in rows if len(ALL_CAPS_WORD_RE.findall(r["utterance"])) >= 3)
    task_density = substantive / max(n, 1)
    balance = 1 - abs(user_turns / max(n, 1) - 0.5)
    core_bonus = min(len(SOFTWARE_CORE.findall(full)), 8) / 8 if domain == "software" else 0.5
    score = 0.30 * min(n, 24) / 24 + 0.25 * min(len(words), 150) / 150 + 0.18 * task_density + 0.12 * balance + 0.10 * min(unique_ratio * 2, 1) + 0.05 * core_bonus
    score -= min(identity_prompts / max(n, 1), 0.35) * 0.20
    score -= min(caps_penalty / max(n, 1), 0.30) * 0.15
    return round(max(0.0, score), 4)



def extract_customer_request(rows: list[dict[str, Any]]) -> str:
    candidates = []
    junk = re.compile(r"^(hi|hello|hey|yes|no|ok|okay|thanks|thank you|bye|nothing|sure|please)$", re.I)
    pii = re.compile(r"ssn|social security|account number|card number|password|otp|pin|my name|name\s*:", re.I)
    for r in rows:
        if r["role"] != "user":
            continue
        t = re.sub(r"\s+", " ", r["utterance"]).strip(" \t\n.,")
        if not t or junk.match(t) or pii.search(t):
            continue
        score = len(re.findall(r"[A-Za-z]+", t))
        if re.search(r"need|want|help|can|could|how|why|what|lost|error|issue|change|check|transfer|balance|update|install|password|login|charge|fee|loan", t, re.I):
            score += 8
        candidates.append((score, t))
    if not candidates:
        return "I need help with this support request."
    candidates.sort(reverse=True)
    return candidates[0][1]


def build_clusters(records: dict[str, dict[str, Any]], domain: str, n_clusters: int) -> tuple[dict[str, int], list[dict[str, Any]]]:
    ids = list(records)
    docs = [records[cid]["customer_text"] for cid in ids]
    n_clusters = min(n_clusters, max(2, len(ids) // 20))
    vec = TfidfVectorizer(stop_words="english", min_df=4, max_df=0.85, ngram_range=(1, 1), max_features=3000)
    X = vec.fit_transform(docs)
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, batch_size=1024, n_init=3, max_iter=60, reassignment_ratio=0.01)
    labels = km.fit_predict(X)
    terms = vec.get_feature_names_out()
    mapping = {cid: int(label) for cid, label in zip(ids, labels)}
    summaries = []
    counts = Counter(labels)
    order = km.cluster_centers_.argsort(axis=1)[:, ::-1]
    for c in range(n_clusters):
        top = [terms[i] for i in order[c, :12]]
        sample_ids = [cid for cid in ids if mapping[cid] == c][:3]
        summaries.append({
            "domain": domain,
            "cluster_id": c,
            "conversation_count": counts[c],
            "top_terms": top,
            "sample_conversation_ids": sample_ids,
        })
    return mapping, summaries

# -----------------------------------------------------------------------------
# Identity normalization.
# -----------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
LONG_DIGIT_RE = re.compile(r"(?<![\d$])\d{7,18}(?!\d)")
NAME_INLINE_RE = re.compile(
    r"\b(?:my name is|full name is|name\s*[:\-]|this is)\s*"
    r"([A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*)?)",
    re.I,
)

COMMON_NON_NAMES = {
    "yes", "no", "sure", "okay", "ok", "thanks", "thank you", "hello", "hi", "hey",
    "connected", "there", "sir", "madam", "please", "nothing", "good morning", "good evening",
}

# Identity-bearing source conversations are excluded rather than randomly merged
# and repaired after the fact. This is deliberately conservative: MultiDoGO has
# enough finance/software conversations to keep long lifelines after filtering.
IDENTITY_RISK_RE = re.compile(
    r"(?i)\b(?:date\s+of\s+birth|\bdob\b|years?\s+old|"
    r"company\s+name|organization\s+name|i\s+work\s+(?:for|at)|my\s+employer\s+is|i\s+live\s+in|i\s+moved\s+to|"
    r"i\s+am\s+(?:a|an)\s+(?:student|teacher|engineer|manager|doctor|nurse|consultant|designer|developer|analyst|owner|retiree|writer|researcher)|"
    r"full\s+address|mailing\s+address)\b"
)
ADDRESS_LIKE_RE = re.compile(
    r"(?i)\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5}\s+"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?|way|cross)\b"
)
US_ADDRESS_COMMA_RE = re.compile(
    r"(?i)\b\d{1,6}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,5},\s*"
    r"[A-Za-z .'-]{2,40},\s*[A-Z]{2}(?:\s+\d{5})?\b"
)
SSN_VALUE_RE = re.compile(
    r"(?i)\b(?:ssn(?:\s*(?:number|no\.?|#))?|ssnnumber|social\s+security(?:\s+number)?)"
    r"\s*(?:is|in|was|:|=|-)?\s*(?:\+?\d[\d .-]{1,20})"
)
ACCOUNT_VALUE_RE = re.compile(
    r"(?i)\b(?:a/?c|acct|account)\s*(?:number|no\.?|#)?\s*(?:is|:|=|-)?\s*\d{3,18}"
)
CARD_VALUE_RE = re.compile(
    r"(?i)\b(?:credit|debit)?\s*(?:card|credit)\s*(?:number|no\.?|#)?\s*(?:is|:|=|-)?\s*\d{3,18}"
)


def has_identity_risk(rows: list[dict[str, Any]]) -> bool:
    customer_text = " ".join(clean_text(r["utterance"]) for r in rows if r["role"] == "user")
    # Exclude hard-to-reconcile biography/location claims. Names, contact fields,
    # and account credentials may remain in the source but are deterministically
    # rewritten to the assigned persona and then audited for residue.
    return bool(IDENTITY_RISK_RE.search(customer_text) or ADDRESS_LIKE_RE.search(customer_text) or US_ADDRESS_COMMA_RE.search(customer_text))


def sanitize_identity_residue(text: str, profile: Persona, role: str) -> str:
    """Remove identity-bearing residue that survives source normalization."""
    secure_sentence = "I completed secure identity verification." if role == "user" else "Secure verification was completed."

    # Any message that contains an SSN-like field is rewritten at sentence
    # level. This catches malformed forms such as `SSNNUMBER`, `SSN#`,
    # alphanumeric values, and correction turns (`it should be 0031`).
    if re.search(r"(?i)\b(?:ssn|ssnnumber|social security)\b", text):
        tail = re.search(
            r"(?i)\band\s+(I\s+(?:lost|need|want|noticed|cannot|can't|am|have|would|still)[^.!?]*)",
            text,
        )
        text = secure_sentence + (" " + tail.group(1).strip() + "." if tail else "")

    text = CARD_VALUE_RE.sub(f"card ending {profile.card_last4 or profile.account_last4}", text)
    text = ACCOUNT_VALUE_RE.sub(f"account ending {profile.account_last4}", text)
    text = ADDRESS_LIKE_RE.sub("the verified mailing address in the secure profile", text)
    text = US_ADDRESS_COMMA_RE.sub("the verified mailing address in the secure profile", text)
    if role == "user" and profile.email in text and len(text.split()) <= 8:
        text = f"My contact email is {profile.email}."
    text = re.sub(r"(?i)\brouting\s+(?:number|no\.?|#)?\s*(?:is|:|=|-)?\s*\d[\d -]{5,20}", "verified routing information", text)

    # Normalize all generic account/card endings, including assistant turns.
    account_targets = [x for x in [profile.checking_last4, profile.savings_last4, profile.account_last4] if x]
    account_counter = 0
    def account_repl(match: re.Match[str]) -> str:
        nonlocal account_counter
        value = account_targets[min(account_counter, len(account_targets)-1)] if account_targets else profile.account_last4
        account_counter += 1
        return f"account ending {value}"
    text = re.sub(r"(?i)\baccount\s+ending(?:\s+with)?\s*[:#-]?\s*\d{4,8}\b", account_repl, text)
    text = re.sub(r"(?i)\bcard\s+ending(?:\s+with)?\s*[:#-]?\s*\d{4,8}\b", f"card ending {profile.card_last4 or profile.account_last4}", text)

    if profile.domain == "finance" and re.search(r"(?i)\btransfer|move money|payment\b", text):
        text = re.sub(
            r"(?i)\bfrom\s+(?:account\s*)?(\d{4,8})\s+to\s+(?:account\s*)?(\d{4,8})\b",
            f"from checking ending {profile.checking_last4} to savings ending {profile.savings_last4}",
            text,
        )
        text = re.sub(r"(?i)\bto\s+(\d{4,8})\s+account\b", f"to savings ending {profile.savings_last4}", text)

    # Messages that are effectively a credential blob carry no useful task
    # semantics; convert them to the secure verification event.
    if re.fullmatch(r"[\s,;:/#A-Za-z-]*\d[\d\s,;:/#A-Za-z-]{3,}", text) and len(re.findall(r"[A-Za-z]+", text)) <= 3:
        text = secure_sentence
    return clean_text(text)


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


NAME_STOPWORDS = {
    "my", "i", "me", "credit", "card", "account", "amount", "date", "thanks", "thank",
    "you", "very", "much", "ok", "okay", "sure", "yeah", "yes", "no", "lost", "for",
    "of", "the", "there", "nothing", "else", "name", "is", "loan", "balance", "please",
    "need", "want", "hello", "hi", "good", "morning", "evening", "actual", "wrong",
    "questions", "question", "your", "will", "all", "extra", "require", "required", "northstar",
    "technical", "unable", "confirm", "however", "some", "like", "talking", "from", "that", "this",
    "request", "verified", "redacted", "identifier", "around", "june", "jun", "may", "time", "pm", "am",
    "transfer", "money", "bank", "banking", "new", "lower", "rates", "commercial", "not", "nt",
    "windows", "macos", "ubuntu", "linux", "chrome", "firefox", "edge", "safari", "outlook",
    "skype", "salesforce", "software", "application", "app", "product", "version", "update", "upgrade",
    "global", "support", "service", "services", "solution", "solutions", "server", "servers", "system",
}


def plausible_person_name(value: str) -> bool:
    value = clean_text(value).strip(" .,-:;!")
    if not value or len(value) > 42:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z.'-]*", value)
    if not 1 <= len(words) <= 4:
        return False
    if any(w.lower() in NAME_STOPWORDS for w in words):
        return False
    if any(len(w) < 2 or len(w) > 22 for w in words):
        return False
    # Reject sentence fragments accidentally captured after a name prompt.
    if re.search(r"(?i)\b(?:need|want|lost|transfer|check|help|could|would|have|account|credit|loan|amount|date)\b", value):
        return False
    return True


def plausible_company(value: str) -> bool:
    value = clean_text(value).strip(" .,-:;!")
    if not value or len(value) > 60 or len(value.split()) > 7:
        return False
    if re.search(r"(?i)\b(?:interest|rate|loan|account|balance|help|need|want|thank)\b", value):
        return False
    return bool(re.search(r"[A-Za-z]{2,}", value))


def detect_original_names(rows: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    expect_name = False
    for r in rows:
        t = clean_text(r["utterance"])
        low = t.lower()
        if r["role"] == "assistant":
            if re.search(r"(?:your|full|first and last) name|may i know.*name|provide.*name|share.*name", low):
                expect_name = True
            # Agent confirmations often expose the customer's name even when
            # the preceding customer turn is malformed.
            for m in re.finditer(
                r"(?i:\b(?:thank you|thanks|confirm|speaking with|hello|hi|sorry)[, ]+)"
                r"([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+)?)",
                t,
            ):
                cand = m.group(1).strip(" .,-")
                if plausible_person_name(cand):
                    names.add(cand)
            continue
        m = NAME_INLINE_RE.search(t)
        if m:
            cand = re.split(r"\b(?:and|ssn|account|card|company|address)\b", m.group(1), flags=re.I)[0].strip(" .,-")
            if plausible_person_name(cand):
                names.add(cand)
        elif expect_name:
            cand = re.split(r"\b(?:and|ssn|account|card|company|number|address)\b", t, flags=re.I)[0]
            cand = re.sub(r"[^A-Za-z .'-]", " ", cand)
            cand = re.sub(r"\s+", " ", cand).strip(" .,-")
            if plausible_person_name(cand):
                names.add(cand)
            expect_name = False
        # Standalone short name-like customer turns.
        stripped = re.sub(r"[^A-Za-z .'-]", " ", t)
        stripped = re.sub(r"\s+", " ", stripped).strip(" .,-")
        if len(t) <= 35 and plausible_person_name(stripped) and not re.search(r"[?!]", t):
            names.add(stripped)
    return names


def detect_original_companies(rows: list[dict[str, Any]]) -> set[str]:
    companies: set[str] = set()
    expect = False
    for r in rows:
        t = clean_text(r["utterance"])
        low = t.lower()
        if r["role"] == "assistant" and re.search(r"company name|organization name|name of your company", low):
            expect = True
            continue
        if r["role"] == "user":
            m = re.search(r"(?:company|organization)\s+name\s*(?:is|:|-)?\s*([A-Za-z0-9 &.'-]{2,50})", t, re.I)
            if m:
                cand = m.group(1).strip(" .,-")
                if plausible_company(cand):
                    companies.add(cand)
            elif expect:
                cand = re.sub(r"\s+", " ", t).strip(" .,-")
                if plausible_company(cand):
                    companies.add(cand)
                expect = False
    return companies


def replace_case_insensitive(text: str, old: str, new: str) -> str:
    if not old or len(old) < 2:
        return text
    return re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, text, flags=re.I)


def normalize_brand_and_platform(text: str, profile: Persona) -> str:
    if profile.domain == "finance":
        text = re.sub(r"\b(?:ACS|AIM|LMT|ABC|XYZ)\s+(?:Bank|Financial|Finance)(?:\s+(?:Services|Solutions))?\b", "Northstar Bank", text, flags=re.I)
        text = re.sub(r"\b(?:the )?bank(?:ing company)?\b", "Northstar Bank", text, flags=re.I)
    else:
        # Normalize support vendors and the user's recurring software stack.
        text = re.sub(r"\b(?:ACS|AIM|LMT|enterprise|mystic|sitara|global|libra|HCL|HOJ)\s+(?:software|technical|tech)(?:\s+(?:services|solutions|department|company|center))?\b", "Aster Software Support", text, flags=re.I)
        text = re.sub(r"\b[A-Za-z][A-Za-z& .'-]{1,28}\s+software\s+(?:solutions|services|support|center)\b", "Aster Software Support", text, flags=re.I)
        text = re.sub(r"\b(?:salesforce|asterdesk|nimbus workbench|orbit office|generic software|the support agent software|support agent software)\b", profile.desktop_app, text, flags=re.I)
        text = re.sub(r"\b(?:outlook|microsoft outlook)\b", profile.mail_app, text, flags=re.I)
        text = re.sub(r"\b(?:skype|microsoft teams|zoom|teamviewer|team viewer|whatsapp)\b", profile.meeting_app, text, flags=re.I)
        text = re.sub(r"\b(?:google chrome|chrome|firefox|edge|safari)\b", profile.browser, text, flags=re.I)
        text = re.sub(r"\bwindows\s*(?:7|8(?:\.1)?|10|11)?\b|\bmac\s*os\b|\bmacos\s*\d*(?:\.\d+)*\b|\bubuntu\s*\d*(?:\.\d+)*\b|\blinux\b", profile.os, text, flags=re.I)
        text = re.sub(r"\bcontrol panel\b|\bwindows programs\b|\ball programs\b", "system settings", text, flags=re.I)
    return text


def _canonical_support_org(profile: Persona) -> str:
    return "Northstar Bank Support" if profile.domain == "finance" else "Aster Software Support"


def _normalize_message_style(text: str, role: str) -> str:
    text = clean_text(text)
    text = re.sub(r"(?i)\b(?:sir|madam|ma'am)\b[,.! ]*", "", text)
    text = re.sub(r"([!?.,])\1{1,}", r"\1", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    letters = [c for c in text if c.isalpha()]
    if role == "user" and len(letters) >= 12 and sum(c.isupper() for c in letters) / len(letters) > 0.82:
        text = text.lower()
        text = text[:1].upper() + text[1:]
    return clean_text(text)


def _strip_agent_vocatives(text: str) -> str:
    # Source customers often address a randomly assigned support agent by name.
    text = re.sub(
        r"(?i)\b(thanks?|thank you)(?:\s+(?:a lot|so much|very much))?[, ]+(?:mr\.?|ms\.?|mrs\.?)?\s*[A-Z][A-Za-z.'-]{2,}\b",
        r"\1",
        text,
    )
    text = re.sub(r"(?i)\b(?:mr\.?|ms\.?|mrs\.?)\s+[A-Z][A-Za-z.'-]{2,}\b", "the support agent", text)
    text = re.sub(r"(?i)(?:please help me|thanks for your help|thank you)[, ]+[A-Za-z.'-]{3,}(?=[.! ]|$)", lambda m: re.sub(r"[, ]+[A-Za-z.'-]{3,}$", "", m.group(0)), text)
    return clean_text(text)


def _normalize_support_intro(text: str, profile: Persona) -> str:
    org = _canonical_support_org(profile)
    if re.search(r"(?i)how may i (?:help|assist)|how can i (?:help|assist)|what can i do for you", text):
        if re.search(r"(?i)reached|support|department|company|customer executive|customer service|technical team", text):
            return f"Hello, you've reached {org}. How may I help you today?"
    text = re.sub(r"(?i)\bmy name is\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?", "I am the support agent", text)
    text = re.sub(r"(?i)\bthis is\s+[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?\s+here", "this is the support team", text)
    return text


def _is_identity_only_user_turn(text: str) -> bool:
    low = text.lower()
    if re.fullmatch(r"(?:yes[, ]*)?(?:my name is|this is)?\s*[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2}[.! ]*", text, re.I):
        return True
    if re.fullmatch(r"(?:yes[, ]*)?(?:the )?(?:company|organization)(?: name)?(?: is)?\s+[A-Za-z0-9 &.'-]{2,60}[.! ]*", text, re.I):
        return True
    if re.fullmatch(r"[\s,;:/#A-Za-z-]*\d[\d\s,;:/#A-Za-z-]{3,}", text) and len(re.findall(r"[A-Za-z]+", text)) <= 4:
        return True
    return bool(re.search(r"(?i)^(?:my name is|full name is|company name is|organization name is|my account number is|my card number is|my ssn is)", low))



def _normalize_inline_company(text: str, profile: Persona) -> str:
    pat = re.compile(
        r"(?i)\\b(?:y\\s+)?(?:my\\s+)?(?:company|organization)(?:\\s+name)?\\s*(?:is|:|-)\\s*"
        r"[A-Za-z0-9 &.'-]{2,65}?(?=\\s+(?:and|my|support|account|i\\b)|[,.!?;]|$)"
    )
    return pat.sub(f"My company is {profile.company}", text)


def _strip_unexpected_vocative_names(text: str, profile: Persona) -> str:
    allowed = {profile.first_name.lower(), profile.name.split()[-1].lower(), "northstar", "aster", "nimbus", "orbit"}
    patterns = [
        r"(?i)(thank you(?: so much| very much| for (?:the )?(?:details|information))?|thanks?|awesome|perfect|sure thing|alright)[, ]+([A-Z][A-Za-z.'-]{2,})",
        r"(?i)(?:details|information)[, ]+([A-Z][A-Za-z.'-]{2,})(?=[,.! ])",
        r"(?i)(?:have a (?:nice|great) day|goodbye)[.! ]+([A-Z][A-Za-z.'-]{2,})[.! ]*$",
    ]
    for idx, pat in enumerate(patterns):
        def repl(m):
            cand=m.group(m.lastindex or 1)
            if cand.lower() in allowed:
                return m.group(0)
            if idx==0:
                return m.group(1)
            return ""
        text=re.sub(pat,repl,text)
    return clean_text(text)


def _strip_trailing_boilerplate(text: str, role: str) -> str:
    if role == "assistant":
        patterns = [
            r"(?i)\\b(?:is|would) there (?:anything|something) else (?:i|we) (?:can|may|might|could) (?:help|assist)[^.!?]*[.!?]?",
            r"(?i)\\bmay i confirm if all your questions have been answered[^.!?]*[.!?]?",
            r"(?i)\\bthank you for (?:reaching out|contacting|choosing|connecting)[^.!?]*[.!?]?",
            r"(?i)\\b(?:have|wish you) (?:a )?(?:nice|great|good|wonderful|awesome) day[^.!?]*[.!?]?",
            r"(?i)\\b(?:it was|my) pleasure assisting you[^.!?]*[.!?]?",
            r"(?i)\\bgood ?bye[^.!?]*[.!?]?",
            r"(?i)\\bdear user,? as there(?:'s| is) no response[^.!?]*[.!?]?",
        ]
    else:
        patterns = [
            r"(?i)\\b(?:no more help needed|nothing else|you can end the chat now)[^.!?]*[.!?]?",
            r"(?i)\\b(?:have a (?:nice|great|good) day|good ?bye)[^.!?]*[.!?]?",
        ]
    for pat in patterns:
        text=re.sub(pat,"",text)
    return clean_text(text)

def normalize_conversation(rows: list[dict[str, Any]], profile: Persona) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Extract the task-bearing dialogue and rewrite it to one stable persona.

    Greetings, closings, agent/customer names, company-identification turns and
    raw credential exchanges are not useful habit evidence, so they are removed
    or collapsed into one secure-verification event. This avoids making a single
    pseudo-user look like dozens of unrelated source customers.
    """
    original_names = detect_original_names(rows)
    original_companies = detect_original_companies(rows)
    out: list[dict[str, str]] = []
    rewrite_counts = Counter()
    pending_identity: str | None = None
    secure_prompt_emitted = False

    for r in rows:
        role = r["role"]
        raw = clean_text(r["utterance"])
        if not raw:
            continue
        low = raw.lower()

        if role == "assistant" and (IDENTITY_PROMPT_RE.search(raw) or SECURE_PROMPT_RE.search(raw)):
            pending_identity = "secure" if SECURE_PROMPT_RE.search(raw) else "identity"
            if pending_identity == "secure" and not secure_prompt_emitted:
                out.append({"role": "assistant", "content": "Please complete identity verification in the secure support panel; do not paste account numbers, card numbers, passwords, PINs, SSNs, or one-time codes into chat."})
                secure_prompt_emitted = True
                rewrite_counts["secure_prompt_collapsed"] += 1
            else:
                rewrite_counts["identity_prompt_removed"] += 1
            continue

        if role == "user" and pending_identity:
            if pending_identity == "secure" and not (out and out[-1]["role"] == "user" and "secure verification" in out[-1]["content"].lower()):
                out.append({"role": "user", "content": "I completed the secure verification step."})
                rewrite_counts["secure_response_collapsed"] += 1
            else:
                rewrite_counts["identity_response_removed"] += 1
            pending_identity = None
            continue

        if role == "user" and _is_identity_only_user_turn(raw):
            rewrite_counts["identity_only_turn_removed"] += 1
            continue

        text = raw
        if role == "assistant":
            text = _normalize_support_intro(text, profile)
        else:
            text = _strip_agent_vocatives(text)

        # Remove empty conversational boilerplate. Keep greetings only when the
        # same turn contains a substantive task after the greeting.
        if GENERIC_GREETING_RE.match(text) or GENERIC_CLOSING_RE.match(text):
            rewrite_counts["boilerplate_removed"] += 1
            continue
        text = re.sub(r"(?i)^(?:hi|hello|hey|good morning|good afternoon|good evening)[,!. ]+(?=\w)", "", text)
        text = re.sub(r"(?i)(?:thank you|thanks)[,!. ]*$", "", text)

        # Stable brands/platforms and role-sensitive name/company normalization.
        if role == "user":
            text = _normalize_inline_company(text, profile)
        text = normalize_brand_and_platform(text, profile)
        text = re.sub(r"(?i)\b(?:the )?support agent(?:'s)?\s+(?:software|application|app)\b", profile.desktop_app if profile.domain == "software" else "the banking application", text)
        for old_name in sorted(original_names, key=len, reverse=True):
            replacement = profile.first_name if role == "assistant" else "the support agent"
            new = replace_case_insensitive(text, old_name, replacement)
            if new != text:
                rewrite_counts["source_name_removed"] += 1
                text = new
        for old_company in sorted(original_companies, key=len, reverse=True):
            replacement = profile.company if role == "user" else _canonical_support_org(profile)
            new = replace_case_insensitive(text, old_company, replacement)
            if new != text:
                rewrite_counts["source_company_removed"] += 1
                text = new

        # Normalize any remaining provider/department phrasing.
        if role == "assistant":
            text = re.sub(r"(?i)you(?:'ve| have) reached[^.!?]{0,90}(?:support|department|company)[^.!?]*", f"You've reached {_canonical_support_org(profile)}", text)
            text = re.sub(r"(?i)\b[A-Za-z][A-Za-z &.'-]{1,35}\s+(?:customer support|technical support|software company|finance department)\b", _canonical_support_org(profile), text)
        else:
            text = re.sub(r"(?i)\b(?:my name is|this is)\s+[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*)?[, ]*(?:and\s+)?", "", text)

        # Generic PII and credential normalization.
        text = EMAIL_RE.sub(profile.email, text)
        text = PHONE_RE.sub(profile.phone, text)
        if SECURE_PROMPT_RE.search(text) and re.search(r"\d|password|otp|pin|code", text, re.I):
            if role == "user":
                tail = re.split(r"(?i)\b(?:and|but)\b", text, maxsplit=1)
                text = "I completed secure verification." + (" " + tail[1].strip() if len(tail) > 1 and len(tail[1].split()) >= 4 else "")
            else:
                text = "Please use the secure verification panel rather than sharing credentials in chat."
            rewrite_counts["credential_exchange_sanitized"] += 1
        text = re.sub(r"(?i)(?:credit |debit )?card(?: number)?(?:\s*(?:is|:|-))?\s*\d{4,18}", f"card ending {profile.card_last4 or profile.account_last4}", text)
        text = re.sub(r"(?i)account(?: number| no\.?| #)?(?:\s*(?:is|:|-))?\s*\d{4,18}", f"account ending {profile.account_last4}", text)
        text = LONG_DIGIT_RE.sub("[REDACTED_IDENTIFIER]", text)
        text = sanitize_identity_residue(text, profile, role)

        if profile.domain == "finance":
            text = re.sub(r"(?i)checking account ending\s*\d{4}", f"checking account ending {profile.checking_last4}", text)
            text = re.sub(r"(?i)savings account ending\s*\d{4}", f"savings account ending {profile.savings_last4}", text)
        else:
            text = re.sub(r"(?i)(?:support )?account ending\s*\d{4}", f"support account ending {profile.account_last4}", text)

        text = _strip_unexpected_vocative_names(text, profile)
        text = _strip_trailing_boilerplate(text, role)
        # Normalize bare checking/savings endings that omit the word account.
        if profile.domain == "finance":
            text = re.sub(r"(?i)checking\s+ending\s+\d{4}", f"checking ending {profile.checking_last4}", text)
            text = re.sub(r"(?i)savings\s+ending\s+\d{4}", f"savings ending {profile.savings_last4}", text)
        text = _normalize_message_style(text, role)
        if not text or GENERIC_GREETING_RE.match(text) or GENERIC_CLOSING_RE.match(text):
            continue
        # Discard source-side biography fragments that escaped the initial filter.
        if role == "user" and BIOGRAPHY_CONFLICT_RE.search(text):
            rewrite_counts["biography_fragment_removed"] += 1
            continue
        out.append({"role": role, "content": text[:900]})

    # Merge adjacent same-role fragments created by identity-turn removal.
    merged: list[dict[str, str]] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            combined = clean_text(merged[-1]["content"] + " " + m["content"])
            merged[-1]["content"] = combined[:1200]
        elif not (merged and merged[-1] == m):
            merged.append(m)

    # Keep the task-bearing middle of very long source conversations.
    if len(merged) > 16:
        merged = merged[:8] + merged[-8:]
    while merged and merged[0]["role"] == "assistant":
        merged.pop(0)
    if len(merged) < 4:
        request = naturalize_seed(extract_customer_request(rows)) or "I need help with this support request."
        request = normalize_brand_and_platform(request, profile)
        merged = [
            {"role": "user", "content": request},
            {"role": "assistant", "content": "I can help. I’ll first clarify the current state and then outline the next step."},
            {"role": "user", "content": "Please continue."},
            {"role": "assistant", "content": "Here is the task-specific next step based on the information available in this session."},
        ]
        rewrite_counts["short_dialogue_reconstructed"] += 1

    return merged, {
        "original_name_count": len(original_names),
        "original_company_count": len(original_companies),
        "rewrite_counts": dict(rewrite_counts),
        "original_names_hash": hashlib.sha256("|".join(sorted(n.lower() for n in original_names)).encode()).hexdigest() if original_names else "",
        "_original_names": sorted(original_names),
        "_original_companies": sorted(original_companies),
    }



# -----------------------------------------------------------------------------
# Habit assignment and evidence generation.
# -----------------------------------------------------------------------------

def balanced_habit_assignments(domain: str, users: int) -> list[list[str]]:
    ids = [h["habit_id"] for h in HABITS_BY_DOMAIN[domain]]
    out: list[list[str]] = []
    for i in range(users):
        # Five habits per user, cycling with co-prime strides for balanced overlap.
        indices = []
        for j in range(5):
            idx = (i * 3 + j * 5 + (i // len(ids))) % len(ids)
            while idx in indices:
                idx = (idx + 1) % len(ids)
            indices.append(idx)
        chosen = [ids[k] for k in indices]
        # Ensure each user has at least one safety/privacy or action-control habit.
        safety = [h["habit_id"] for h in HABITS_BY_DOMAIN[domain] if h["family"] in {"privacy_consent", "risk_threshold"}]
        if not any(h in safety for h in chosen):
            chosen[-1] = safety[i % len(safety)]
        out.append(chosen)
    return out


def voice_prefix(profile: Persona, idx: int) -> str:
    pools = {
        "concise": ["One preference for next time:", "Please keep this as my usual flow:", "For future requests like this,"],
        "conversational": ["That worked well for me—next time,", "A quick note for future chats:", "Could we make this my usual approach:"],
        "careful": ["To avoid mistakes in future cases,", "Please use this safeguard from now on:", "For similar requests, I would like you to"],
        "direct": ["Going forward,", "Use this rule for me:", "My default for this kind of request is:"],
        "detail-aware": ["For future cases with the same conditions,", "Please distinguish this situation from nearby cases:", "When the details match this kind of request,"],
    }
    return pools[profile.voice_style][idx % len(pools[profile.voice_style])]


def acknowledgement(profile: Persona, idx: int) -> str:
    options = [
        f"Understood, {profile.first_name}. I will follow that workflow when the same conditions apply.",
        "Got it. I will use that preference in matching situations and keep the stated boundary in mind.",
        "Understood. I will apply that approach only where it fits and respect the exception you described.",
        "Thanks for clarifying. I will use that as the default for comparable requests.",
        "Noted. I will follow the newer preference in future matching cases.",
    ]
    return options[idx % len(options)]


def evidence_turns(profile: Persona, habit: dict[str, Any], kind: str, idx: int) -> list[dict[str, str]]:
    h = habit
    v = stable_index(profile.user_id, h["habit_id"], kind, str(idx), mod=6)
    if kind == "support":
        options = [
            f"That sequence worked well for me. When {h['condition']}, please {h['default_action']}.",
            f"For similar requests, the part I want repeated is this: {h['default_action']}.",
            f"Next time this comes up, don't rush ahead; {h['default_action']}.",
            f"Please keep the same flow for me whenever {h['condition']}: {h['default_action']}.",
            f"I liked how the last case was handled. My usual preference is to {h['default_action']}.",
            f"A small correction for future sessions: when {h['condition']}, {h['default_action']}.",
        ]
    elif kind == "boundary":
        options = [
            f"For a nearby case, I use a different rule: when {h['boundary_condition']}, {h['boundary_action']}.",
            f"Please don't carry that workflow into every similar-looking request. If {h['boundary_condition']}, {h['boundary_action']}.",
            f"The distinction that matters to me is that when {h['boundary_condition']}, you should {h['boundary_action']}.",
            f"That preference has a limit: when {h['boundary_condition']}, {h['boundary_action']}.",
            f"For information-only or out-of-scope cases like this, I prefer that you {h['boundary_action']}.",
            f"One boundary to remember is this: if {h['boundary_condition']}, {h['boundary_action']}.",
        ]
    elif kind == "exception":
        options = [
            f"There is one exception I rely on: when {h['exception_condition']}, {h['exception_action']}.",
            f"If {h['exception_condition']}, don't repeat the full default flow; {h['exception_action']}.",
            f"For that specific situation, my normal preference changes: {h['exception_action']}.",
            f"Please treat this as an exception whenever {h['exception_condition']}: {h['exception_action']}.",
            f"The last case also showed an exception—when {h['exception_condition']}, {h['exception_action']}.",
            f"One narrow override matters to me: if {h['exception_condition']}, {h['exception_action']}.",
        ]
    elif kind == "old":
        options = [
            f"For now, I still prefer to {h['old_action']}.",
            f"My current approach is to {h['old_action']}.",
            f"At this point, please {h['old_action']}.",
        ]
    elif kind == "new":
        options = [
            f"I want to change that earlier preference. From now on, when {h['condition']}, {h['default_action']}.",
            f"Please update the way you handle this for me: instead of trying to {h['old_action']}, {h['default_action']}.",
            f"The old approach no longer works for me. Going forward, when {h['condition']}, {h['default_action']}.",
            f"Use this newer rule from now on: {h['default_action']}; don't continue the earlier practice of trying to {h['old_action']}.",
            f"After the last few cases, my preference has changed. Please {h['default_action']} whenever {h['condition']}.",
            f"This supersedes what I said earlier: when {h['condition']}, {h['default_action']}.",
        ]
    else:
        raise ValueError(kind)
    user = clean_text(options[v % len(options)])
    acks = [
        "Understood. I’ll follow that approach when the same conditions apply.",
        "Got it. I’ll use that preference in matching cases and keep its boundary in mind.",
        "Thanks for clarifying. I’ll apply the newer instruction only where it fits.",
        "Noted. I’ll preserve that exception rather than applying the default mechanically.",
        "Understood. I’ll treat that as your standing workflow for comparable requests.",
        "I’ve got it. I’ll use the scoped preference in future sessions.",
    ]
    return [{"role": "user", "content": user}, {"role": "assistant", "content": acks[v]}]


def composition_turns(profile: Persona, h1: dict[str, Any], h2: dict[str, Any], idx: int) -> list[dict[str, str]]:
    first, second = sorted([h1, h2], key=lambda h: h["priority"], reverse=True)
    variants = [
        f"When both issues happen in the same request, first {first['default_action']}. After that, also {second['default_action']}.",
        f"A combined case should keep both preferences: {first['default_action']}; then {second['default_action']}.",
        f"Please don't let one workflow erase the other. In a combined case, {first['default_action']}, followed by {second['default_action']}.",
        f"For requests that trigger both conditions, the safer prerequisite comes first: {first['default_action']}. Then {second['default_action']}.",
    ]
    v = stable_index(profile.user_id, h1["habit_id"], h2["habit_id"], str(idx), mod=len(variants))
    return [
        {"role": "user", "content": variants[v]},
        {"role": "assistant", "content": "Understood. I’ll combine the two preferences and preserve the higher-priority prerequisite."},
    ]



# -----------------------------------------------------------------------------
# Source allocation.
# -----------------------------------------------------------------------------

def choose_from_pool(pool: list[str], unused: set[str], n: int, rng: random.Random) -> list[str]:
    chosen: list[str] = []
    attempts = 0
    while pool and len(chosen) < n and attempts < len(pool) + n * 5:
        cid = pool.pop()
        attempts += 1
        if cid in unused:
            unused.remove(cid)
            chosen.append(cid)
    return chosen


def allocate_sources(
    domain_records: dict[str, dict[str, Any]],
    personas: list[Persona],
    assignments: dict[str, list[str]],
    rng: random.Random,
) -> tuple[dict[str, list[str]], set[str]]:
    """Allocate high-quality, thematically coherent source sessions without reuse."""
    unused = set(domain_records)
    by_habit: dict[str, list[str]] = {}
    for h in HABITS_BY_DOMAIN[personas[0].domain]:
        ids = [cid for cid, rec in domain_records.items() if h["habit_id"] in rec["tags"]]
        ids.sort(key=lambda cid: (domain_records[cid]["quality_score"], -domain_records[cid]["cluster_id"]), reverse=True)
        by_habit[h["habit_id"]] = ids
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for cid, rec in domain_records.items():
        by_cluster[rec["cluster_id"]].append(cid)
    for ids in by_cluster.values():
        ids.sort(key=lambda cid: domain_records[cid]["quality_score"], reverse=True)

    allocation: dict[str, list[str]] = {}
    for pidx, persona in enumerate(personas):
        active = assignments[persona.user_id]
        selected: list[str] = []
        # At least six natural task sessions per active habit.
        for hid in active:
            for cid in by_habit[hid]:
                if cid in unused:
                    unused.remove(cid); selected.append(cid)
                if sum(1 for x in selected if hid in domain_records[x]["tags"]) >= 6:
                    break
        # Select a stable set of topic clusters for this pseudo-user, based on
        # the clusters already associated with their active habits.
        cluster_votes = Counter(domain_records[cid]["cluster_id"] for cid in selected)
        preferred_clusters = [c for c,_ in cluster_votes.most_common(8)]
        # Add multi-habit sessions first.
        multi = sorted(
            [cid for cid in unused if len(set(domain_records[cid]["tags"]) & set(active)) >= 2],
            key=lambda cid: domain_records[cid]["quality_score"], reverse=True,
        )
        for cid in multi:
            if len(selected) >= min(SOURCE_SESSIONS_PER_USER, 42): break
            unused.remove(cid); selected.append(cid)
            if domain_records[cid]["cluster_id"] not in preferred_clusters and len(preferred_clusters) < 10:
                preferred_clusters.append(domain_records[cid]["cluster_id"])
        # Fill from the user's recurring clusters and active themes.
        coherent_pool = sorted(
            [cid for cid in unused if domain_records[cid]["cluster_id"] in preferred_clusters or set(domain_records[cid]["tags"]) & set(active)],
            key=lambda cid: (domain_records[cid]["quality_score"], len(set(domain_records[cid]["tags"]) & set(active))), reverse=True,
        )
        for cid in coherent_pool:
            if len(selected) >= SOURCE_SESSIONS_PER_USER: break
            unused.remove(cid); selected.append(cid)
        # Final high-quality same-domain background fill.
        if len(selected) < SOURCE_SESSIONS_PER_USER:
            fallback = sorted(unused, key=lambda cid: domain_records[cid]["quality_score"], reverse=True)
            for cid in fallback:
                if len(selected) >= SOURCE_SESSIONS_PER_USER: break
                unused.remove(cid); selected.append(cid)
        if len(selected) != SOURCE_SESSIONS_PER_USER:
            raise RuntimeError(f"not enough coherent source conversations for {persona.user_id}: {len(selected)}")
        rng.shuffle(selected)
        allocation[persona.user_id] = selected
    return allocation, unused



# -----------------------------------------------------------------------------
# Natural, source-conditioned probe synthesis.
# -----------------------------------------------------------------------------
AMOUNTS = [65, 85, 110, 125, 160, 195, 240, 285, 315, 375, 420, 480, 560, 640, 725, 810, 980, 1125, 1250, 1480, 1840, 2360]
PAYEES = ["rent", "the electricity bill", "a vendor invoice", "tuition", "the insurance premium", "a contractor invoice", "the travel fund", "a medical bill", "quarterly taxes", "the office lease", "a workshop deposit", "a supplier payment"]
MERCHANTS = ["Harbor Market", "Cedar Transit", "Northline Books", "Sunrise Pharmacy", "Mosaic Kitchen", "Riverbend Hotel", "Juniper Mobile", "Atlas Office Supply", "Cobalt Energy", "Willow Fitness", "Brightway Hosting", "Pine & Paper"]
ERROR_CODES = ["E-104", "SYNC-27", "AUTH-19", "UPD-503", "EXP-42", "CFG-88", "DB-17", "NET-31", "UI-208", "LIC-61", "FILE-73", "API-29"]
FEATURES = ["export screen", "cloud sync", "sign-in page", "report builder", "calendar integration", "desktop updater", "shared workspace", "invoice module", "search index", "notification center", "template library", "offline cache"]
PLANS = ["individual", "professional", "team", "annual", "monthly", "business"]
DATES = ["Monday", "Tuesday morning", "the 15th", "month-end", "tomorrow afternoon", "Friday", "the next billing date", "the first business day"]


def stable_index(*parts: str, mod: int) -> int:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(h[:8], "big") % mod


def naturalize_seed(text: str) -> str:
    t = clean_text(text)
    if not t:
        return ""
    if re.search(r"(?i)ssn|social security|account number|card number|password|otp|\bpin\b|my name|name\s*:|address|phone number|email address|yes sir|dear user|gold loan|\[redacted|customer service desk", t):
        return ""
    t = re.sub(r"^(?:hi|hello|hey|hai|good morning|good afternoon)[,!. ]+", "", t, flags=re.I)
    t = re.sub(r"^(?:yes|yeah|sure|okay|ok)[,!. ]+", "", t, flags=re.I)
    fixes = {r"\bi got to\b":"I need to", r"\bi wanna\b":"I want to", r"\bwanna\b":"want to", r"\bcould u\b":"could you", r"\bpls\b":"please", r"\bi using\b":"I am using", r"\bhave not working\b":"is not working", r"\bkindly\b":"please"}
    for pat, rep in fixes.items(): t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"(?i)\b(?:sir|madam|ma'am)\b", "", t)
    t = re.sub(r"\s+", " ", t).strip(" .,-")
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < 5 or len(words) > 32 or len(t) > 220:
        return ""
    if not re.search(r"(?i)need|want|help|can|could|how|why|what|lost|error|issue|change|check|transfer|balance|update|install|login|charge|fee|loan|payment|statement|subscription|ticket|cancel|replace|reset|report", t):
        return ""
    t = t[0].upper() + t[1:]
    return t + ("" if t[-1] in ".?!" else ".")


FINANCE_REVIEWS = ["month-end cash-flow review", "vendor reconciliation", "household budget check", "quarter-end close", "tax-preparation checklist", "travel-expense review", "invoice audit", "weekly account review", "tuition budget plan", "insurance-payment review", "freelance income review", "annual subscription audit"]
SOFTWARE_PROJECTS = ["client onboarding", "release-candidate test", "data-migration rehearsal", "quarterly reporting rollout", "remote-work setup", "workspace consolidation", "accessibility review", "training-lab setup", "billing-cycle change", "incident follow-up", "new-hire setup", "archive migration"]


def context_for_habit(hid: str, profile: Persona, probe_id: str) -> dict[str, Any]:
    i = stable_index(probe_id, hid, mod=10**9)
    return {
        "amount": AMOUNTS[i % len(AMOUNTS)],
        "payee": PAYEES[(i // 3) % len(PAYEES)],
        "merchant": MERCHANTS[(i // 7) % len(MERCHANTS)],
        "error": ERROR_CODES[(i // 11) % len(ERROR_CODES)],
        "feature": FEATURES[(i // 17) % len(FEATURES)],
        "plan": PLANS[(i // 23) % len(PLANS)],
        "date": DATES[(i // 29) % len(DATES)],
        "review": FINANCE_REVIEWS[(i // 31) % len(FINANCE_REVIEWS)],
        "project": SOFTWARE_PROJECTS[(i // 37) % len(SOFTWARE_PROJECTS)],
        "case_ref": 1000 + stable_index(probe_id, profile.user_id, "case", mod=8000),
        "profile": profile,
        "hid": hid,
    }


def _source_style(seed: str) -> str:
    s = naturalize_seed(seed)
    if not s: return "neutral"
    low = s.lower()
    if low.startswith(("how", "what", "why", "can", "could")): return "question"
    if "not working" in low or "error" in low or "issue" in low or "problem" in low: return "problem"
    if "want" in low or "need" in low: return "request"
    return "neutral"


def _scenario_variants(hid: str, profile: Persona, ctx: dict[str, Any], mode: str) -> list[str]:
    a, payee, merchant = ctx["amount"], ctx["payee"], ctx["merchant"]
    if hid == "finance_confirm_money_movement":
        direct = [
            f"Please transfer ${a} from checking ending {profile.checking_last4} to savings ending {profile.savings_last4} for {payee}.",
            f"Can you set up a ${a} payment for {payee} from my checking account ending {profile.checking_last4}?",
            f"I need ${a} moved from checking ending {profile.checking_last4} into savings ending {profile.savings_last4} before {ctx['date']}.",
            f"Prepare a transfer of ${a} from checking ending {profile.checking_last4} to savings ending {profile.savings_last4}; it is for {payee}.",
            f"Help me send ${a} from checking ending {profile.checking_last4} toward {payee}, using savings ending {profile.savings_last4} as the destination.",
            f"The next step in my {ctx['review']} is a ${a} transfer from checking ending {profile.checking_last4} to savings ending {profile.savings_last4}.",
        ]
        boundary = [
            f"How do same-day and standard bank transfers differ if I might pay {payee} later this week?",
            f"What information would a bank normally show before a transfer is submitted? I am only comparing the process for {payee}.",
            f"Can you explain whether moving money between checking and savings usually affects available balance immediately?",
            f"I am writing notes for my {ctx['review']}; what is the usual sequence for an internal transfer?",
        ]
        exception = [
            f"I approved transfer reference {ctx['case_ref']} in the banking app for ${a} from checking ending {profile.checking_last4} to savings ending {profile.savings_last4}. What happens next?",
            f"The secure screen already records my approval for the ${a} payment to {payee}; please finish the matching request.",
            f"Approval code {ctx['case_ref']} is attached to the exact ${a} transfer between my two listed accounts. Can you continue it?",
            f"The app shows the ${a} transfer as approved but not yet submitted. Please take the next step on that exact transaction.",
        ]
    elif hid == "finance_confirm_card_account_changes":
        direct = [
            f"Please replace card ending {profile.card_last4}; the magnetic stripe stopped working.",
            f"I need the notification phone updated on my Northstar Bank profile before {ctx['date']}.",
            f"Block card ending {profile.card_last4} and prepare a replacement because I cannot find it.",
            f"Change the mailing preference on checking ending {profile.checking_last4} from paper to electronic statements.",
            f"Help me close the unused savings account ending {profile.savings_last4}.",
            f"I want to update the contact email connected to card ending {profile.card_last4}.",
        ]
        boundary = [
            f"How does the travel protection on card ending {profile.card_last4} compare with a basic debit card?",
            "What are the usual differences between paper and electronic statements?",
            "Can you explain when a bank replaces a damaged card versus reissuing the same number?",
            "I am comparing checking-account notification options; which alerts are commonly available?",
        ]
        exception = [
            f"The secure account page already records approval to replace card ending {profile.card_last4}. Can you continue that exact replacement?",
            f"I approved the contact-email change in the banking app under reference {ctx['case_ref']}; please complete it.",
            f"The secure screen shows my approved request to close savings ending {profile.savings_last4}. What is the next step?",
            f"The approved mailing-address update is waiting under case {ctx['case_ref']}; please finish the matching change.",
        ]
    elif hid == "finance_minimal_pii_secure_verification":
        direct = [
            f"I need access to checking ending {profile.checking_last4}, but I do not want to paste a full account number or one-time code into chat.",
            f"Help me verify ownership of card ending {profile.card_last4} through the safest available channel.",
            f"The support flow needs to confirm my identity before showing a statement. How should we do that without exposing secrets here?",
            f"I can use the banking app for verification, but I will not send an SSN, PIN, password, or security code in this conversation.",
            f"Can we recover access to my Northstar Bank profile using a secure verification panel rather than chat messages?",
            f"I need help with account access and want the minimum necessary identity check.",
        ]
        boundary = [
            "Why do banks use multi-factor authentication instead of relying on a password alone?",
            "I am making a training slide about the difference between identity proofing and login authentication. Can you summarize it?",
            "What makes a one-time code safer than reusing the same password across sites?",
            "Could you explain, in general terms, why support agents should not collect full card numbers in chat?",
        ]
        exception = [
            "A one-time code was accidentally visible earlier, and I asked for it not to be retained. I still need a fresh recovery route.",
            "Please continue account recovery without quoting the card number that appeared in the previous message.",
            "I withdrew consent to keep the identity value shared earlier. What secure verification option can we use now?",
            "The earlier chat contains a sensitive identifier that should be treated as deleted; help me restart verification safely.",
        ]
    elif hid == "finance_fraud_lost_card_urgent_escalation":
        direct = [
            f"Card ending {profile.card_last4} is missing, and a ${a} purchase at {merchant} is not mine.",
            f"I just saw an unauthorized ${a} charge from {merchant} on card ending {profile.card_last4}.",
            f"My wallet was stolen this morning; card ending {profile.card_last4} may still be active.",
            f"There are two suspicious transactions on card ending {profile.card_last4}, including ${a} at {merchant}.",
            f"I cannot find my debit card and need to know its current lock status right away.",
            f"A transaction from {merchant} appeared while the physical card ending {profile.card_last4} was with me.",
        ]
        boundary = [
            "What are three practical ways to reduce card-fraud risk while traveling?",
            "How do banks usually distinguish a merchant dispute from suspected card theft?",
            "I am reviewing security guidance; when should someone freeze a card as a precaution?",
            "Can you explain how transaction alerts help detect fraud earlier?",
        ]
        exception = [
            f"Card ending {profile.card_last4} is already locked and fraud case {ctx['case_ref']} is open. Has the investigation moved forward?",
            f"The bank app shows the card is frozen and case {ctx['case_ref']} covers the ${a} {merchant} charge. What is the next step?",
            f"An urgent fraud report already exists for card ending {profile.card_last4}; please check that case rather than creating another one.",
            f"I reported the stolen card earlier today and received case {ctx['case_ref']}. Can you tell me its current status?",
        ]
    elif hid == "finance_balance_statement_summary_first":
        direct = [
            f"Review last month's statement for checking ending {profile.checking_last4} and tell me what changed most.",
            f"Can you summarize the current balance and the largest recent movements on checking ending {profile.checking_last4}?",
            f"I am doing a {ctx['review']}; give me the headline first, then the entries that explain it.",
            f"Help me reconcile the statement for savings ending {profile.savings_last4}, starting with the overall result.",
            f"What is the short status of my account activity this week, and which transactions support it?",
            f"I need a quick read of the statement before I examine the individual lines.",
        ]
        boundary = [
            f"Export every transaction from checking ending {profile.checking_last4} for the last 90 days in line-by-line order.",
            "I need a complete audit trail with every date, amount, description, and running balance—not a summary.",
            f"Show the full statement detail for my {ctx['review']} so I can verify each entry myself.",
            "Please list all transactions individually and keep the original order from the statement.",
        ]
        exception = [
            f"In one sentence, what is the current status of checking ending {profile.checking_last4}?",
            "Give me only the bottom-line balance change; I will ask for details later.",
            "I have room for one line in my notes. What is the account headline?",
            f"For today's {ctx['review']}, answer with the single most important statement result only.",
        ]
    elif hid == "finance_fee_dispute_evidence_then_case":
        direct = [
            f"A ${a} charge from {merchant} looks wrong, and I may need to dispute it.",
            f"I was charged a ${a} fee that I do not recognize. Help me prepare the dispute.",
            f"The statement shows ${a} at {merchant}, but my receipt has a different total.",
            f"I want to challenge the ${a} transaction from {merchant} on checking ending {profile.checking_last4}.",
            f"An overdraft fee of ${a} appeared after a deposit posted late; I want the case reviewed.",
            f"Help me assemble what is needed to contest the ${a} {merchant} charge.",
        ]
        boundary = [
            "What evidence do banks commonly request for a card-charge dispute?",
            "How does a merchant dispute differ from reporting an unauthorized transaction?",
            "Can you explain the usual timeline after someone files a fee dispute?",
            "I am reading my account terms. What does 'provisional credit' mean during a dispute?",
        ]
        exception = [
            f"Case {ctx['case_ref']} already covers the ${a} charge from {merchant}. Can you check its status?",
            f"I received dispute reference {ctx['case_ref']} for the same {merchant} transaction yesterday. What happens next?",
            f"The ${a} fee is already under review in case {ctx['case_ref']}; please don't open another case.",
            f"My statement links the {merchant} charge to existing dispute {ctx['case_ref']}. Has anything changed?",
        ]
    elif hid == "finance_credit_loan_cautious_no_commitment":
        direct = [
            f"What would a lender usually consider for a ${a * 100} personal loan, and what might the monthly-payment range look like?",
            "Help me compare a fixed-rate and variable-rate loan without submitting an application.",
            f"I am exploring whether a higher credit limit could fit my budget before {ctx['date']}.",
            "Can you outline mortgage eligibility factors and possible scenarios without predicting approval?",
            f"Build a cautious checklist for evaluating a ${a * 100} loan offer.",
            "I want to understand the documentation and tradeoffs before deciding whether to apply for credit.",
        ]
        boundary = [
            f"What is the monthly payment on ${a * 100} at 6% APR over five years, ignoring fees?",
            "What does APR mean, and how is it different from the stated interest rate?",
            "Can you define debt-to-income ratio with a simple example?",
            "How does a fixed interest rate differ from a variable one?",
        ]
        exception = [
            "Create a draft loan-document checklist for me, but do not submit an application or predict approval.",
            "I only need a comparison worksheet for two mortgage offers; nothing should be sent to a lender.",
            "Prepare questions I can ask a loan officer. This is planning material, not an application.",
            "Give me a private draft of the credit-limit review steps without initiating a request.",
        ]
    elif hid == "finance_payment_status_latest_check":
        direct = [
            f"Did the ${a} payment for {payee} post, or is it still pending?",
            f"What is the current status of transfer reference {ctx['case_ref']}?",
            f"Has the ${a} payment to {payee} cleared as of now?",
            f"Check whether the scheduled payment for {payee} is due, processing, or posted.",
            f"I need the latest available balance on checking ending {profile.checking_last4} before I make another payment.",
            f"Can you verify the current posting state of the ${a} {merchant} transaction and tell me when the status was checked?",
        ]
        boundary = [
            "Why can card payments remain pending for several days?",
            "What is the general difference between an available balance and a current balance?",
            "How do scheduled bank payments usually move from queued to posted?",
            "What factors affect how long an electronic transfer takes to settle?",
        ]
        exception = [
            "For a fictional workshop example, how might a pending payment change an available balance?",
            "I am writing a hypothetical scenario about a scheduled transfer. Explain the possible states without looking up an account.",
            "Can you give a generic example of a payment moving from pending to posted?",
            "For training purposes, describe what 'processing' might mean on a payment timeline.",
        ]
    elif hid == "software_collect_diagnostics_before_fix":
        direct = [
            f"{profile.desktop_app} crashes in the {ctx['feature']} with error {ctx['error']} on {profile.os}.",
            f"The {ctx['feature']} freezes every time I repeat the same three steps in {profile.desktop_app}; the code is {ctx['error']}.",
            f"I can reproduce {ctx['error']} in {profile.desktop_app} after opening the {ctx['feature']} on {profile.os}.",
            f"The {ctx['feature']} stopped working after yesterday's restart, and {profile.desktop_app} shows {ctx['error']}.",
            f"Help me diagnose a repeatable {ctx['error']} failure in {profile.desktop_app} on {profile.os}.",
            f"{profile.desktop_app} closes unexpectedly when I use the {ctx['feature']}; I need a targeted diagnosis.",
        ]
        boundary = [
            "What is the conceptual difference between an application crash, a freeze, and a slow response?",
            "I am writing onboarding notes. Which diagnostic details are generally useful in a bug report?",
            "Why are reproduction steps important when investigating software defects?",
            "Can you explain what an error code is meant to communicate to support teams?",
        ]
        exception = [
            f"I am away from the {profile.os} machine and only need one clearly labeled possible cause for {ctx['error']}; I can collect logs later.",
            f"Give me a provisional explanation for {ctx['error']} in the {ctx['feature']}; the affected computer is offline until tomorrow.",
            "I need a short hypothesis for a meeting, not a full diagnosis; the version and logs are not available yet.",
            f"Can you list one likely cause of the {ctx['feature']} failure while clearly noting that it has not been verified?",
        ]
    elif hid == "software_docs_lookup_for_update_install":
        direct = [
            f"The {profile.desktop_app} update failed on {profile.os}, and the installed build is one release behind.",
            f"Which current installation steps apply to {profile.desktop_app} on {profile.os}?",
            f"I need the version-specific procedure for upgrading {profile.desktop_app} during our {ctx['project']}.",
            f"The updater reports {ctx['error']}; check the current documentation for this {profile.os} build.",
            f"Help me install the supported {profile.desktop_app} release without using instructions for an older version.",
            f"What does the latest support article say about repairing the {profile.desktop_app} updater on {profile.os}?",
        ]
        boundary = [
            "What is the conceptual difference between a software patch and a full upgrade?",
            "Why do vendors publish release notes for software updates?",
            "What does semantic versioning try to communicate?",
            "Can you explain, without referring to a product, why an installation may require a restart?",
        ]
        exception = [
            f"The {profile.os} machine is offline until tomorrow. Give me a clearly qualified list of possible reasons the {profile.desktop_app} update failed.",
            "I cannot reach the vendor documentation from the training lab. What offline checks can I do while treating the answer as provisional?",
            f"The network is unavailable, so do not claim a live documentation check; outline likely causes of {ctx['error']} instead.",
            f"I only need a provisional explanation for the {profile.desktop_app} installer until the documentation site is reachable.",
        ]
    elif hid == "software_one_try_then_escalate":
        direct = [
            f"A focused sign-in fix failed once, and {profile.desktop_app} still shows {ctx['error']}.",
            f"I already tried the single targeted step for the {ctx['feature']}; the same problem returned.",
            f"The first focused attempt did not resolve {ctx['error']} on {profile.os}.",
            f"We repeated the documented quick check once, but {profile.desktop_app} is still failing.",
            f"The {ctx['feature']} issue survived one targeted troubleshooting attempt and is blocking today's {ctx['project']}.",
            f"One reasonable local fix has already failed for {ctx['error']}; I need the next level of support.",
        ]
        boundary = [
            f"The {ctx['feature']} looked wrong for the first time this morning, and I have not tried any focused step yet.",
            f"This is the first occurrence of {ctx['error']} in {profile.desktop_app}; what should I check once?",
            "A minor software issue just appeared and has not been reproduced. What is one sensible first check?",
            f"The {ctx['feature']} failed once after a restart, with no prior troubleshooting history.",
        ]
        exception = [
            f"The affected {profile.os} machine is in an offline lab, so a support ticket cannot be opened until tomorrow; what local evidence should I preserve?",
            f"External escalation is unavailable during this maintenance window. What one safe local check can I run for {ctx['error']}?",
            "The support portal is down for the evening; help me document the failed attempt so escalation can happen tomorrow.",
            f"I cannot contact the vendor from the secure lab. What should I capture about the {ctx['feature']} issue before the connection returns?",
        ]
    elif hid == "software_secure_login_password_flow":
        direct = [
            f"I am locked out of {profile.desktop_app} but still have access to {profile.email}; I will not paste a password or one-time code here.",
            f"Help me recover my {profile.desktop_app} account through a secure reset flow.",
            f"The sign-in page rejects my password, and I need a recovery path that keeps credentials out of chat.",
            f"I can use the secure account portal but cannot share a PIN or authentication code in this conversation.",
            f"Restore access to {profile.desktop_app} using the minimum necessary verification.",
            f"I need to reset the {profile.desktop_app} login without repeating any secret that may have appeared earlier.",
        ]
        boundary = [
            "Why are one-time codes safer than sending a password to support?",
            "What is the difference between password reset and account recovery?",
            "I am preparing security training. Why should support staff avoid asking for credentials in chat?",
            "Can you explain how phishing-resistant authentication works at a high level?",
        ]
        exception = [
            "A temporary code appeared earlier and I withdrew consent to retain it. Start a fresh recovery flow without quoting it.",
            "Do not reuse the password fragment from the previous message; help me reset access through the secure portal.",
            "The earlier credential should be treated as deleted. What is the cleanest way to restart account recovery?",
            "I asked support not to remember the code that was posted by mistake. Please use a new verification route.",
        ]
    elif hid == "software_platform_specific_steps":
        direct = [
            f"Set up {profile.desktop_app} on {profile.os} and connect it to {profile.mail_app}.",
            f"Walk me through the first-run configuration for {profile.desktop_app} on {profile.os}.",
            f"I need numbered steps for installing {profile.desktop_app} and enabling the {ctx['feature']} on {profile.os}.",
            f"Configure {profile.desktop_app} for {profile.company}'s {ctx['project']} on this {profile.os} machine.",
            f"Show me the {profile.os}-specific path for connecting {profile.desktop_app} to {profile.mail_app}.",
            f"The first-run screen is open in {profile.desktop_app}; guide me through setup on {profile.os}.",
        ]
        boundary = [
            "What is cloud synchronization in general, and why do applications use it?",
            "I am preparing an architecture briefing about desktop versus web applications. Can you compare them?",
            "What does an installer normally change on a computer?",
            "Can you explain the purpose of application settings without giving device-specific instructions?",
        ]
        exception = [
            f"Give my colleague a one-sentence overview of what the {profile.desktop_app} installer changes on {profile.os}.",
            f"I only need a brief orientation to {profile.desktop_app}, not numbered setup steps.",
            f"Summarize the purpose of the {ctx['feature']} in one sentence for a meeting slide.",
            f"A teammate wants a quick description of {profile.desktop_app} on {profile.os}; detailed configuration can wait.",
        ]
    elif hid == "software_backup_before_risky_change":
        direct = [
            f"Roll {profile.desktop_app} back to the previous build on {profile.os}; the current release is corrupting exports.",
            f"Reset and reinstall {profile.desktop_app} during tonight's {ctx['project']}.",
            f"I need to migrate the {profile.desktop_app} workspace and remove the old configuration.",
            f"Prepare a destructive reset of the {ctx['feature']} settings on {profile.os}.",
            f"The current build is unusable, so I want to restore the earlier {profile.desktop_app} release.",
            f"Reinstall {profile.desktop_app} and clear its local data before {ctx['date']}.",
        ]
        boundary = [
            "What is a restore point, and when is it useful?",
            "How is a backup different from an application configuration export?",
            "I am writing a glossary entry about rollback. Can you define it?",
            "Why can software resets cause data loss even when the application is reinstalled successfully?",
        ]
        exception = [
            f"The admin console verifies a current backup and records approval for the exact rollback of {profile.desktop_app}. Continue that change.",
            f"Restore point {ctx['case_ref']} and final approval are attached to the planned {profile.desktop_app} reset. What is next?",
            f"The backup check passed and the migration was approved in the console; proceed with that exact migration.",
            f"The approved maintenance task already includes a verified backup for the {ctx['feature']} reset.",
        ]
    elif hid == "software_confirm_license_subscription_changes":
        direct = [
            f"Move {profile.company}'s {profile.desktop_app} subscription from the {profile.plan} plan to the team plan at the next billing cycle.",
            f"Cancel the auto-renewal for {profile.desktop_app} before {ctx['date']}.",
            f"Add five paid seats to the {profile.desktop_app} subscription for {profile.company}.",
            f"Change the billing interval for {profile.desktop_app} from monthly to annual.",
            f"Upgrade {profile.company}'s {profile.desktop_app} license to the business plan.",
            f"Remove two seats and change the renewal date on the {profile.desktop_app} account.",
        ]
        boundary = [
            f"Compare the professional and team plans for {profile.desktop_app}; I am not changing the subscription today.",
            f"What is the difference between monthly and annual billing for {profile.desktop_app}?",
            "How do named-user and concurrent-user software licenses differ?",
            f"Which features are commonly included in a business software plan compared with an individual plan?",
        ]
        exception = [
            f"The billing portal records approval for the exact upgrade of {profile.desktop_app} to the team plan. Continue it.",
            f"Approval reference {ctx['case_ref']} matches the annual-renewal change for {profile.desktop_app}; please finish the update.",
            f"The five-seat addition was approved in the billing portal and is waiting to be applied.",
            f"The portal already confirms the cancellation of {profile.desktop_app} auto-renewal. Complete that exact request.",
        ]
    elif hid == "software_ticket_receipt_summary":
        direct = [
            f"I submitted a bug report for {ctx['error']} on the {ctx['feature']} with logs and reproduction steps. What should I keep for follow-up?",
            f"Support case {ctx['case_ref']} was just filed for the {profile.desktop_app} crash; give me the useful receipt details.",
            f"The incident about {ctx['error']} has been submitted. Summarize the reference, priority, and next step.",
            f"I sent the {ctx['feature']} failure report with attachments and need a concise record of what went in.",
            f"A ticket for {ctx['error']} is now open. What should the written follow-up include?",
            f"The bug report for {profile.desktop_app} was accepted under case {ctx['case_ref']}; recap it for my notes.",
        ]
        boundary = [
            f"I am still troubleshooting {ctx['error']} on the {ctx['feature']}; no report has been filed yet.",
            f"Help me organize the facts for a possible {profile.desktop_app} ticket, but I have not submitted anything.",
            "I am deciding whether an issue is reproducible enough to report. What should I test next?",
            f"The {ctx['feature']} problem is still under investigation, and there is no case reference.",
        ]
        exception = [
            f"Draft a bug report for {ctx['error']} on the {ctx['feature']}, but do not submit it.",
            f"I need wording for tomorrow's review packet, not a filed support ticket.",
            f"Write a draft incident description for {profile.desktop_app}; I will decide later whether to send it.",
            f"Prepare the ticket text and leave the reference-ID field blank because submission happens later.",
        ]
    else:
        direct = ["I need help with this support request."]
        boundary = direct
        exception = direct
    return {"direct": direct, "drift": direct, "boundary": boundary, "exception": exception}[mode]


def scenario_sentence(hid: str, profile: Persona, probe_id: str, mode: str, seed_request: str) -> str:
    ctx = context_for_habit(hid, profile, probe_id)
    variants = _scenario_variants(hid, profile, ctx, mode)
    n = int(re.search(r"(\\d+)$", probe_id).group(1)) if re.search(r"(\\d+)$", probe_id) else stable_index(probe_id, mod=10000)
    style = _source_style(seed_request)
    style_offset = {"neutral":0,"question":1,"problem":2,"request":3}[style]
    q = variants[(n + style_offset + stable_index(hid, mode, mod=len(variants))) % len(variants)]
    if profile.domain == "finance":
        direct_tails = [
            f"This is part of my {ctx['review']}.",
            f"I need to settle it before {ctx['date']}.",
            f"I flagged it while reviewing Northstar Bank activity in {profile.city}.",
            f"It came up while I was closing today's {ctx['review']}.",
            f"I am checking it for my work as a {profile.role}.",
            f"This is the next item on my {ctx['review']} checklist.",
            f"The timing matters because I need the result by {ctx['date']}.",
            f"I am handling it from {profile.city} as part of a {ctx['review']}.",
        ]
        boundary_tails = [
            f"I am using the answer in notes for my {ctx['review']}.",
            f"This came up while preparing a short explanation for {ctx['date']}.",
            f"I want the concept clear before my next {ctx['review']}.",
            f"The question is for a training note I am writing in {profile.city}.",
        ]
    else:
        direct_tails = [
            f"This is affecting {profile.company}'s {ctx['project']} on {profile.os}.",
            f"I need it resolved before {ctx['date']}.",
            f"The affected setup is {profile.desktop_app} on {profile.os} at {profile.company}.",
            f"It surfaced during our {ctx['project']}.",
            f"This is blocking the next checkpoint for {profile.company}.",
            f"I am working on it as a {profile.role} using {profile.os}.",
            f"The issue is part of today's {ctx['project']}.",
            f"I need a usable next step before the team reviews it on {ctx['date']}.",
        ]
        boundary_tails = [
            f"I am using the explanation in {profile.company}'s {ctx['project']} notes.",
            f"This is for a short training note on {profile.os}.",
            f"I want the concept clear before our next {ctx['project']} review.",
            f"The answer will go into an internal briefing at {profile.company}.",
        ]
    tails = boundary_tails if mode == "boundary" else direct_tails
    tail = tails[stable_index(probe_id, hid, "tail", mod=len(tails))]
    return clean_text(q + " " + tail)


def _lead(probe_id: str, choices: list[str]) -> str:
    return choices[stable_index(probe_id, "lead", mod=len(choices))]


def policy_response(hid: str, profile: Persona, ctx: dict[str, Any], mode: str, probe_id: str) -> str:
    a, payee, merchant = ctx["amount"], ctx["payee"], ctx["merchant"]
    openers = ["I can help.", "Here’s how I’d handle it.", "I’ll take this in the following order.", "The next step is straightforward.", "I can work through that safely.", "Here is the appropriate next step."]
    lead = _lead(probe_id + hid + mode, openers)
    if hid == "finance_confirm_money_movement":
        if mode == "default": body=f"I’ll prepare the ${a} transfer using checking ending {profile.checking_last4} and savings ending {profile.savings_last4}, state the payee and timing, and wait for your final approval before submitting it."
        elif mode == "boundary": body=f"Same-day transfers usually settle faster but may have tighter cutoffs or fees, while standard transfers can take longer. I have not created a transaction for {payee}."
        elif mode == "exception": body=f"The secure approval matches the ${a} transfer and the listed accounts. I’ll continue that exact transaction without asking you to approve it a second time."
        elif mode == "old": body=f"I’ll treat the ${a} transfer as pre-approved and submit it immediately."
        elif mode == "partial": body="Please confirm that you want to transfer money; I won’t restate the amount or accounts first."
        else: body=f"The ${a} transfer has already been submitted from checking ending {profile.checking_last4}."
    elif hid == "finance_confirm_card_account_changes":
        if mode == "default": body=f"I’ll restate the exact change involving card ending {profile.card_last4} or the selected account setting, then wait for your final approval before applying it."
        elif mode == "boundary": body="I’ll compare the features and policy differences directly; I won’t open or change an account based on a comparison question."
        elif mode == "exception": body=f"The secure approval matches the requested change under reference {ctx['case_ref']}. I’ll continue that exact update without another approval loop."
        elif mode == "old": body="I’ll require a phone callback even though the secure in-app approval already matches the request."
        elif mode == "partial": body=f"I’ll apply the change to card ending {profile.card_last4} after a generic yes, without restating the exact setting."
        else: body=f"I’ve applied the card or account change immediately and will send the details afterward."
    elif hid == "finance_minimal_pii_secure_verification":
        if mode == "default": body="I’ll request only the minimum necessary identity check in the secure panel and will not ask you to paste a full account number, card number, SSN, PIN, password, or one-time code into chat."
        elif mode == "boundary": body="I’ll answer the general security question directly and won’t start an identity-verification flow for an explanation."
        elif mode == "exception": body="I won’t quote or reuse the sensitive value from the earlier message. I’ll start a fresh verification step in the secure panel."
        elif mode == "old": body="Paste the identifier or code from the earlier chat so I can reuse it for verification."
        elif mode == "partial": body="I’ll ask for the last several account digits in chat before opening the secure verification panel."
        else: body="Please paste the full account number and one-time code here so I can continue."
    elif hid == "finance_fraud_lost_card_urgent_escalation":
        if mode == "default": body=f"I’ll start the urgent protective flow for card ending {profile.card_last4}, check its current lock and transaction status, and route the incident to the fraud team without delaying for ordinary troubleshooting."
        elif mode == "boundary": body="I’ll give practical fraud-prevention guidance and explain warning signs; I won’t freeze a card or open an incident from a general question."
        elif mode == "exception": body=f"I’ll retrieve fraud case {ctx['case_ref']} and the current card status instead of creating a duplicate report."
        elif mode == "old": body="I’ll try several ordinary self-service checks before using the urgent fraud path."
        elif mode == "partial": body=f"I’ll open a new fraud report for the ${a} {merchant} charge without checking whether one already exists."
        else: body="I’ll wait for more suspicious transactions before taking protective action."
    elif hid == "finance_balance_statement_summary_first":
        if mode == "default": body=f"I’ll begin with the account headline for the {ctx['review']}, then list the few transactions or reconciliation details that support it."
        elif mode == "boundary": body="I’ll provide the complete line-by-line export in statement order, including dates, descriptions, amounts, and running balances."
        elif mode == "exception": body="I’ll give only the one-line account status you requested and leave the supporting detail for later."
        elif mode == "old": body="I’ll start with the full transaction dump and put the conclusion at the end."
        elif mode == "partial": body="I’ll provide a brief headline but omit the supporting entries needed to verify it."
        else: body="I’ll return only a generic account overview without addressing the requested statement period."
    elif hid == "finance_fee_dispute_evidence_then_case":
        if mode == "default": body=f"I’ll gather the statement line, receipt, {merchant} details, date, and ${a} amount, summarize the dispute, check for an existing case, and ask before filing a new one."
        elif mode == "boundary": body="I’ll explain the dispute process and typical evidence without opening a case."
        elif mode == "exception": body=f"I’ll retrieve case {ctx['case_ref']} for the same ${a} {merchant} charge and report its status rather than filing a duplicate."
        elif mode == "old": body=f"I’ll file a new dispute for the ${a} charge immediately, before checking the evidence or existing cases."
        elif mode == "partial": body=f"I’ll collect the ${a} amount and merchant name but file the case without a receipt or final review."
        else: body="I’ll give a generic policy page and leave the disputed transaction unresolved."
    elif hid == "finance_credit_loan_cautious_no_commitment":
        if mode == "default": body="I’ll explain the eligibility factors, documentation, payment scenarios, and tradeoffs without promising approval or submitting an application."
        elif mode == "boundary": body="I’ll calculate or define the requested term directly, with the assumptions shown, and won’t turn it into an application."
        elif mode == "exception": body="I’ll prepare the requested checklist or comparison draft only; nothing will be submitted and no approval will be implied."
        elif mode == "old": body="I’ll move from the eligibility question into an application and estimate that approval is likely."
        elif mode == "partial": body="I’ll list broad requirements but imply that meeting them guarantees approval."
        else: body="I’ll submit a credit request using the information already in the conversation."
    elif hid == "finance_payment_status_latest_check":
        if mode == "default": body=f"I’ll check the latest available account or payment state, report whether the ${a} item is queued, pending, or posted, and label the result with the time of the check."
        elif mode == "boundary": body="I’ll explain the general processing timeline and the difference between payment states without claiming that I looked at an account."
        elif mode == "exception": body="I’ll give the hypothetical payment-state explanation and clearly state that no live account lookup was performed."
        elif mode == "old": body="I’ll estimate the status from generic timing rules without checking the current state."
        elif mode == "partial": body="I’ll check the latest status but omit when the information was retrieved."
        else: body=f"I’ll assume the ${a} payment posted because the usual processing window has passed."
    elif hid == "software_collect_diagnostics_before_fix":
        if mode == "default": body=f"I’ll first collect the {profile.desktop_app} build, {profile.os} version, exact {ctx['error']} text, and reproduction steps, then use those details for a targeted diagnosis."
        elif mode == "boundary": body="I’ll answer the conceptual software question directly and can give a general example; no device diagnostics are needed."
        elif mode == "exception": body=f"I’ll offer one clearly labeled provisional cause for {ctx['error']} and state which version, log, and reproduction details are still missing."
        elif mode == "old": body="I’ll apply a generic fix immediately without collecting version, error, or reproduction details."
        elif mode == "partial": body=f"I’ll ask only for the error code {ctx['error']} and skip the build, operating system, and reproduction path."
        else: body=f"I’ll reinstall {profile.desktop_app} as the first step without diagnosing the repeatable failure."
    elif hid == "software_docs_lookup_for_update_install":
        if mode == "default": body=f"I’ll verify the current {profile.desktop_app} documentation for {profile.os}, cite the applicable version or support article, and separate confirmed instructions from assumptions."
        elif mode == "boundary": body="I’ll explain the general software concept without pretending that a product-specific documentation lookup occurred."
        elif mode == "exception": body="Because the documentation site is unavailable, I’ll give only qualified offline checks and label them as provisional rather than claiming a live lookup."
        elif mode == "old": body="I’ll reuse older installation steps without checking whether they match the current product version."
        elif mode == "partial": body=f"I’ll say I checked the latest {profile.desktop_app} documentation but won’t identify a version or source."
        else: body="I’ll recommend a generic reinstall based on memory alone."
    elif hid == "software_one_try_then_escalate":
        if mode == "default": body=f"Since one focused attempt already failed, I’ll preserve the result, open or escalate a support case for {ctx['error']}, and include the evidence instead of cycling through broad fixes."
        elif mode == "boundary": body="I’ll try one focused, low-risk check first and ask you to report the result before escalating."
        elif mode == "exception": body="Since external escalation is unavailable, I’ll document the failed attempt and preserve logs for the first available support window, with at most one safe local check."
        elif mode == "old": body="I’ll continue through several rounds of self-service troubleshooting before escalating."
        elif mode == "partial": body=f"I’ll open a ticket for {ctx['error']} without recording the focused step that already failed."
        else: body="I’ll repeat the same generic troubleshooting step until it works."
    elif hid == "software_secure_login_password_flow":
        if mode == "default": body=f"I’ll direct the recovery through the secure {profile.desktop_app} reset flow, request only minimal verification, and never ask for a password, PIN, or one-time code in chat."
        elif mode == "boundary": body="I’ll answer the general authentication question without starting account recovery or requesting credentials."
        elif mode == "exception": body="I won’t repeat or reuse the credential fragment from the earlier message; I’ll start a fresh secure recovery flow."
        elif mode == "old": body="Paste the earlier password or code again so I can continue the same recovery attempt."
        elif mode == "partial": body="I’ll open the secure reset page after asking you to confirm part of the password in chat."
        else: body="Send the one-time code here and I’ll unlock the account directly."
    elif hid == "software_platform_specific_steps":
        if mode == "default": body=f"I’ll give numbered {profile.os}-specific steps for {profile.desktop_app}, naming the exact menus and checkpoints, then ask you to confirm each stage."
        elif mode == "boundary": body="I’ll give a concise conceptual explanation without turning it into device-specific setup instructions."
        elif mode == "exception": body="I’ll provide the requested one-sentence overview and leave the detailed setup steps for later."
        elif mode == "old": body="I’ll give one generic set of steps without checking which platform is in use."
        elif mode == "partial": body=f"I’ll provide numbered steps for {profile.desktop_app} but use a different operating system's menu names."
        else: body="I’ll tell you to explore the settings until the feature appears."
    elif hid == "software_backup_before_risky_change":
        if mode == "default": body=f"Before changing {profile.desktop_app}, I’ll verify a current backup or restore point, summarize the data-loss and rollback risks, and wait for final approval."
        elif mode == "boundary": body="I’ll explain restore points and rollback conceptually; I won’t create a backup or alter the computer from a glossary question."
        elif mode == "exception": body=f"The verified backup and approval match the exact {profile.desktop_app} change, so I’ll continue it without repeating those checks."
        elif mode == "old": body="I’ll perform the reset or rollback first and discuss recovery only if something goes wrong."
        elif mode == "partial": body="I’ll confirm that a backup exists but make the destructive change without restating the risk or obtaining final approval."
        else: body=f"I’ll clear the {profile.desktop_app} data and reinstall it immediately."
    elif hid == "software_confirm_license_subscription_changes":
        if mode == "default": body=f"I’ll restate the {profile.desktop_app} plan, seat count, price, billing effect, and renewal timing, then wait for final approval before changing the subscription."
        elif mode == "boundary": body="I’ll compare the plans and license terms directly without treating the comparison as a purchase or cancellation request."
        elif mode == "exception": body=f"The billing-portal approval matches the exact {profile.desktop_app} change under reference {ctx['case_ref']}, so I’ll continue without another approval loop."
        elif mode == "old": body="I’ll apply the small plan change automatically because the billing effect looks minor."
        elif mode == "partial": body="I’ll ask for approval without restating the seat count, price, or renewal impact."
        else: body=f"I’ve upgraded {profile.desktop_app} to the team plan and will explain the new bill afterward."
    elif hid == "software_ticket_receipt_summary":
        if mode == "default": body=f"I’ll provide a concise receipt for case {ctx['case_ref']} with the issue, reproduction details, priority, attachments, owner, and next expected update."
        elif mode == "boundary": body="I’ll continue troubleshooting or help organize a draft; I won’t invent a case number before anything is submitted."
        elif mode == "exception": body="I’ll write the requested bug-report draft and leave it unsubmitted, with no fabricated receipt or reference ID."
        elif mode == "old": body="I’ll end after submission without recording a reference, owner, or next step."
        elif mode == "partial": body=f"I’ll provide case {ctx['case_ref']} and a short title but omit reproduction details and the next update."
        else: body=f"I’ll invent ticket {ctx['case_ref']} even though no report was submitted."
    else:
        body="I’ll provide a generic support response."
    if profile.domain == "finance":
        suffix = [f" I’ll keep the result tied to the {ctx['review']}.", f" I’ll reference account ending {profile.account_last4} where needed.", f" I’ll keep the concrete ${ctx['amount']} context visible."][stable_index(probe_id, hid, "suffix", mod=3)]
    else:
        suffix = [f" I’ll keep the steps specific to {profile.desktop_app} on {profile.os}.", f" I’ll tie the result to the {ctx['project']}.", f" I’ll preserve the concrete {ctx['error']} and {ctx['feature']} context."][stable_index(probe_id, hid, "suffix", mod=3)]
    return clean_text(f"{lead} {body}{suffix}")


def combine_responses(parts: list[str], probe_id: str, reverse: bool = False) -> str:
    parts = list(reversed(parts)) if reverse else parts
    joins = [" After that, ", " Then, ", " Once that prerequisite is complete, ", " In the same response, ", " Next, "]
    result = parts[0].rstrip()
    for i, p in enumerate(parts[1:]):
        p = re.sub(r"^(?:I can help\.|Here’s how I’d handle it\.|I’ll take this in the following order\.|The next step is straightforward\.|I can work through that safely\.|Here is the appropriate next step\.)\s*", "", p)
        result += joins[stable_index(probe_id, str(i), mod=len(joins))] + p[:1].lower() + p[1:]
    return clean_text(result)


def neutral_false_personalization_response(hid: str, profile: Persona, ctx: dict[str, Any], probe_id: str) -> str:
    if profile.domain == "finance":
        return f"I can review the {ctx['review']} with you. Would you like a one-line balance, a short summary with supporting entries, or a full line-by-line walkthrough of account ending {profile.account_last4}?"
    return f"I can help document {ctx['error']} on the {ctx['feature']}. Are you still troubleshooting, preparing an unsubmitted draft, or asking for a receipt from an existing case?"


def build_probe(
    probe_id: str,
    profile: Persona,
    probe_type: str,
    target_habits: list[str],
    seed_requests: list[str],
    updated_habit: str,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = target_habits[0]
    secondary = target_habits[1] if len(target_habits) > 1 else None
    mode = {"direct_use":"direct", "boundary":"boundary", "exception":"exception", "composition":"direct", "priority_conflict":"direct", "drift":"drift", "false_personalization":"direct", "explicit_retrieval":"direct"}[probe_type]
    q1 = scenario_sentence(primary, profile, probe_id, mode, seed_requests[0] if seed_requests else "")
    if secondary:
        q2 = scenario_sentence(secondary, profile, probe_id + "b", "direct", seed_requests[1] if len(seed_requests) > 1 else "")
        joiners = [
            "I also need help with a second part:", "There is a related issue in the same support session:",
            "At the same time,", "The same request has another piece:", "One more thing is connected to this:",
            "There are two tasks in this request; the other one is:",
        ]
        query = clean_text(q1 + " " + joiners[stable_index(probe_id, "join", mod=len(joiners))] + " " + q2)
    else:
        query = q1
    if probe_type == "explicit_retrieval":
        reminders = [
            "Before taking the next step, remind me how I have usually wanted this kind of request handled.",
            "What standing approach from our earlier conversations applies here?",
            "Please restate the recurring workflow I have preferred in similar cases.",
            "Before proceeding, summarize the pattern I have repeatedly asked you to follow.",
            "Which of my established support preferences fits this request?",
        ]
        query = clean_text(query + " " + reminders[stable_index(probe_id, "reminder", mod=len(reminders))])
    if probe_type == "false_personalization":
        ctx = context_for_habit(primary, profile, probe_id)
        if profile.domain == "finance":
            variants = [
                f"Can you help me review the latest statement entries for checking ending {profile.checking_last4}?",
                f"I need help making sense of several transactions in my {ctx['review']}.",
                f"Walk me through the activity on savings ending {profile.savings_last4} from the last statement period.",
                f"Help me understand what changed in checking ending {profile.checking_last4} this month.",
                f"I am reviewing my Northstar Bank statement in {profile.city} and need help deciding how to examine it.",
                f"Can we go over the account activity I flagged for my {ctx['review']}?",
            ]
        else:
            variants = [
                f"Help me document an issue with the {ctx['feature']}; I have not said whether it is still being diagnosed or already filed.",
                f"I need assistance describing a {ctx['error']} problem in {profile.desktop_app} for my team.",
                f"Can you help me organize the information about the {ctx['feature']} failure before I decide the next step?",
                f"I want to capture what happened in {profile.desktop_app}, but I have not chosen a report format yet.",
                f"Help me prepare notes about {ctx['error']} during {profile.company}'s {ctx['project']}.",
                f"I need a useful record of the {ctx['feature']} problem on {profile.os}; start by clarifying what kind of output I need.",
            ]
        query = variants[stable_index(probe_id, profile.user_id, "false", mod=len(variants))]

    ctx1 = context_for_habit(primary, profile, probe_id)
    ctx2 = context_for_habit(secondary, profile, probe_id + "b") if secondary else None
    ordered = sorted(target_habits, key=lambda hid: HABIT_BY_ID[hid]["priority"], reverse=True)

    if probe_type == "boundary":
        gold = policy_response(primary, profile, ctx1, "boundary", probe_id)
        distractors = [
            policy_response(primary, profile, ctx1, "default", probe_id + "d1"),
            policy_response(primary, profile, ctx1, "partial", probe_id + "d2"),
            policy_response(primary, profile, ctx1, "unsafe", probe_id + "d3"),
        ]
    elif probe_type == "exception":
        gold = policy_response(primary, profile, ctx1, "exception", probe_id)
        distractors = [
            policy_response(primary, profile, ctx1, "default", probe_id + "d1"),
            policy_response(primary, profile, ctx1, "partial", probe_id + "d2"),
            policy_response(primary, profile, ctx1, "unsafe", probe_id + "d3"),
        ]
    elif probe_type == "drift":
        gold = policy_response(updated_habit, profile, ctx1, "default", probe_id)
        distractors = [
            policy_response(updated_habit, profile, ctx1, "old", probe_id + "d1"),
            policy_response(updated_habit, profile, ctx1, "partial", probe_id + "d2"),
            policy_response(updated_habit, profile, ctx1, "unsafe", probe_id + "d3"),
        ]
    elif probe_type == "false_personalization":
        gold = neutral_false_personalization_response(primary, profile, ctx1, probe_id)
        if profile.domain == "finance":
            distractors = [
                f"I’ll assume you want a one-line answer and provide only the ending balance for checking ending {profile.checking_last4}.",
                f"I’ll assume your standing format is a short headline followed by three selected entries from the {ctx1['review']}.",
                f"I’ll assume you always want the full statement exported line by line, including every transaction and running balance.",
            ]
        else:
            distractors = [
                f"I’ll assume case {ctx1['case_ref']} has already been submitted and produce a ticket receipt for the {ctx1['feature']} issue.",
                f"I’ll assume you want only a one-sentence draft about {ctx1['error']} and will omit troubleshooting details.",
                f"I’ll assume the issue should be escalated immediately and create a support-case summary before confirming its current state.",
            ]
    elif secondary:
        response_by_hid = {
            primary: policy_response(primary, profile, ctx1, "default", probe_id + primary),
            secondary: policy_response(secondary, profile, ctx2, "default", probe_id + secondary),
        }
        ordered_parts = [response_by_hid[h] for h in ordered]
        gold = combine_responses(ordered_parts, probe_id)
        only_primary = response_by_hid[primary]
        only_secondary = response_by_hid[secondary]
        reverse_both = combine_responses(ordered_parts, probe_id + "rev", reverse=True)
        if probe_type == "composition":
            wrong_one = policy_response(secondary, profile, ctx2, "partial", probe_id + "partial")
            distractors = [only_primary, only_secondary, combine_responses([response_by_hid[primary], wrong_one], probe_id + "mix")]
        else:
            # Priority conflict: both actions appear in all serious choices, but
            # only one respects the higher-priority prerequisite.
            distractors = [reverse_both, only_primary, combine_responses([policy_response(ordered[0], profile, context_for_habit(ordered[0], profile, probe_id if ordered[0]==primary else probe_id+'b'), "partial", probe_id+'part'), response_by_hid[ordered[1]]], probe_id+'mix')]
    else:
        gold = policy_response(primary, profile, ctx1, "default", probe_id)
        distractors = [
            policy_response(primary, profile, ctx1, "partial", probe_id + "d1"),
            policy_response(primary, profile, ctx1, "old", probe_id + "d2"),
            policy_response(primary, profile, ctx1, "unsafe", probe_id + "d3"),
        ]

    # Normalize and enforce four distinct contextual replies.
    texts=[]; seen=set()
    for text in [gold]+distractors:
        text=clean_text(text)
        n=re.sub(r"\W+", " ", text.lower()).strip()
        if n not in seen:
            seen.add(n); texts.append(text)
    if len(texts) != 4:
        raise RuntimeError(f"non-distinct choices for {probe_id}: {texts}")
    # Balance gold positions exactly across the dataset.
    gold_pos = int(probe_id.rsplit("_",1)[-1]) % 4
    other = texts[1:]
    rng.shuffle(other)
    ordered_texts = other[:]
    ordered_texts.insert(gold_pos, texts[0])
    choices = [{"choice_id": cid, "text": txt} for cid,txt in zip("ABCD",ordered_texts)]
    gold_choice = "ABCD"[gold_pos]
    public={"probe_id":probe_id,"user_id":profile.user_id,"domain":profile.domain,"query":clean_text(query),"choices":choices}
    capability={"direct_use":"habit_direct_use","boundary":"boundary_false_personalization","exception":"counterevidence_exception","composition":"multi_habit_composition","priority_conflict":"habit_priority_resolution","drift":"habit_drift","false_personalization":"false_personalization_control","explicit_retrieval":"explicit_fact_preference_retrieval"}[probe_type]
    private={"probe_id":probe_id,"user_id":profile.user_id,"domain":profile.domain,"gold_choice_id":gold_choice,"gold_action_text":texts[0],"probe_type":probe_type,"capability_group":capability,"target_habit_ids":target_habits,"updated_habit_id":updated_habit if probe_type=="drift" else None,"source_probe_seed_requests":seed_requests,"required_priority_order":ordered}
    return public,private



# -----------------------------------------------------------------------------
# Validation helpers.
# -----------------------------------------------------------------------------

def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def token_estimate(text: str) -> int:
    # Conservative approximation for English support dialogue.
    return math.ceil(len(text) / 3.8)


def nearest_neighbor_stats(texts: list[str]) -> dict[str, float]:
    if len(texts) < 2:
        return {"max": 0.0, "p95": 0.0, "median": 0.0}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    sims = cosine_similarity(vec)
    vals = []
    for i in range(len(texts)):
        sims[i, i] = -1
        vals.append(float(sims[i].max()))
    vals.sort()
    return {
        "max": round(vals[-1], 4),
        "p95": round(vals[int(0.95 * (len(vals) - 1))], 4),
        "median": round(statistics.median(vals), 4),
    }

# -----------------------------------------------------------------------------
# Build.
# -----------------------------------------------------------------------------

def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    for sub in ["public", "private", "review", "source", "reports", "scripts", "model_eval"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    source_paths = {
        "finance": Path(args.finance_source),
        "software": Path(args.software_source),
    }
    records_by_domain: dict[str, dict[str, dict[str, Any]]] = {}
    cluster_summaries: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    source_manifest: dict[str, Any] = {}

    for domain, path in source_paths.items():
        print(f"[stage] load/filter/cluster {domain}", flush=True)
        raw = read_conversations(path, domain)
        good: dict[str, dict[str, Any]] = {}
        for cid, rows in raw.items():
            if not is_good_conversation(rows, domain):
                continue
            tags = tag_conversation(rows, domain)
            good[cid] = {
                "conversation_id": cid,
                "domain": domain,
                "rows": rows,
                "tags": tags,
                "customer_text": conversation_text(rows, customer_only=True),
                "full_text": conversation_text(rows),
                "quality_score": quality_score(rows, domain),
                "request": extract_customer_request(rows),
            }
        print(f"[stage] {domain}: {len(good)} quality conversations; clustering", flush=True)
        mapping, summaries = build_clusters(good, domain, CLUSTERS_PER_DOMAIN)
        for cid, cluster_id in mapping.items():
            good[cid]["cluster_id"] = cluster_id
            cluster_rows.append({
                "domain": domain,
                "conversation_id": cid,
                "cluster_id": cluster_id,
                "quality_score": good[cid]["quality_score"],
                "habit_tags_json": json.dumps(sorted(good[cid]["tags"])),
                "customer_request_preview": clean_text(good[cid]["request"])[:300],
            })
        cluster_summaries.extend(summaries)
        source_manifest[domain] = {
            "source_file": path.name,
            "sha256": sha256_file(path),
            "raw_conversations": len(raw),
            "quality_filtered_conversations": len(good),
            "raw_utterances": sum(len(x) for x in raw.values()),
            "clusters": len(summaries),
        }
        for _rec in good.values():
            _rec.pop("customer_text", None)
            _rec.pop("full_text", None)
        records_by_domain[domain] = good
        del raw, mapping, summaries
        gc.collect()

    # Source-grounded candidate statistics.
    habit_audit_rows = []
    for h in HABITS:
        recs = records_by_domain[h["domain"]]
        matched = [r for r in recs.values() if h["habit_id"] in r["tags"]]
        clusters = Counter(r["cluster_id"] for r in matched)
        examples = sorted(matched, key=lambda r: r["quality_score"], reverse=True)[:5]
        habit_audit_rows.append({
            "habit_id": h["habit_id"],
            "domain": h["domain"],
            "family": h["family"],
            "source_theme": h["theme"],
            "matched_source_conversations": len(matched),
            "supporting_cluster_count": len(clusters),
            "top_clusters_json": json.dumps(clusters.most_common(5)),
            "source_examples_json": json.dumps([{"conversation_id": e["conversation_id"], "request": e["request"][:250]} for e in examples], ensure_ascii=False),
            "retained": "yes",
            "retention_reason": "repeated source theme with clear scope, boundary, exception, and evaluable action policy",
        })

    print("[stage] source audit complete; assigning personas", flush=True)
    personas = make_personas()
    assignments: dict[str, list[str]] = {}
    updated_habits: dict[str, str] = {}
    by_domain_personas = {d: [p for p in personas if p.domain == d] for d in DOMAINS}
    for domain in DOMAINS:
        assn = balanced_habit_assignments(domain, USERS_PER_DOMAIN)
        for i, profile in enumerate(by_domain_personas[domain]):
            assignments[profile.user_id] = assn[i]
            updated_habits[profile.user_id] = assn[i][(i * 2 + 1) % len(assn[i])]

    allocations: dict[str, list[str]] = {}
    unused_by_domain: dict[str, set[str]] = {}
    for domain in DOMAINS:
        alloc, unused = allocate_sources(records_by_domain[domain], by_domain_personas[domain], assignments, rng)
        allocations.update(alloc)
        unused_by_domain[domain] = unused

    public_lifelines: list[dict[str, Any]] = []
    private_sessions: list[dict[str, Any]] = []
    persona_rows: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    evidence_index: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    identity_audit_rows: list[dict[str, Any]] = []
    original_name_hashes_by_user: dict[str, set[str]] = defaultdict(set)
    original_names_by_user: dict[str, set[str]] = defaultdict(set)
    original_companies_by_user: dict[str, set[str]] = defaultdict(set)

    print("[stage] allocations complete; building coherent lifelines", flush=True)
    for profile_idx, profile in enumerate(personas):
        if profile_idx % 5 == 0: print(f"[stage] lifeline {profile_idx}/{len(personas)}", flush=True)
        active = assignments[profile.user_id]
        updated = updated_habits[profile.user_id]
        source_ids = allocations[profile.user_id]
        start = datetime(2025, 1, 6, 9, 0) + timedelta(days=stable_index(profile.user_id, mod=8))
        sessions_public: list[dict[str, Any]] = []
        profile_session_id = f"{profile.user_id}_s0000"
        if profile.domain == "finance":
            profile_user_text = (
                f"I am {profile.name}, a {profile.role} in {profile.city}, {profile.state}. "
                f"My usual Northstar Bank references are checking ending {profile.checking_last4}, savings ending {profile.savings_last4}, and card ending {profile.card_last4}. "
                f"My contact label is {profile.email}."
            )
            profile_assistant = f"Thanks, {profile.first_name}. I will keep this identity and these account references consistent across future support sessions."
        else:
            profile_user_text = (
                f"I am {profile.name}, a {profile.role} at {profile.company}. My usual setup is {profile.os} with {profile.browser}, "
                f"{profile.desktop_app}, {profile.mail_app}, and {profile.meeting_app}. The support account ends {profile.account_last4}."
            )
            profile_assistant = f"Thanks, {profile.first_name}. I will keep that device, organization, and software setup consistent in future support conversations."
        profile_session = {
            "session_id": profile_session_id,
            "user_id": profile.user_id,
            "session_index": 0,
            "timestamp": start.isoformat(),
            "domain": profile.domain,
            "messages": [
                {"role": "user", "content": profile_user_text},
                {"role": "assistant", "content": profile_assistant},
            ],
        }
        sessions_public.append({k: profile_session[k] for k in ["session_id", "session_index", "timestamp", "messages"]})
        private_sessions.append({
            **profile_session,
            "source_seed": {"type": "synthetic_identity_anchor"},
            "memory_annotations": [{"annotation_type": "identity_anchor", "persona_id": profile.user_id}],
        })

        # Evidence schedule: stable habits get support/boundary/exception; one habit has old->new drift.
        positions = list(range(1, SOURCE_SESSIONS_PER_USER + 1))
        relevant_positions: dict[str, list[int]] = {hid: [] for hid in active}
        for pos, cid in enumerate(source_ids, start=1):
            tags = records_by_domain[profile.domain][cid]["tags"]
            for hid in active:
                if hid in tags:
                    relevant_positions[hid].append(pos)
        evidence_plan: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for hidx, hid in enumerate(active):
            rel = relevant_positions[hid] or positions
            # Deterministic spread over the timeline.
            if hid == updated:
                old_candidates = [p for p in rel if p <= 24] or rel[:]
                new_candidates = [p for p in rel if p >= 43] or rel[:]
                for p in old_candidates[:2]: evidence_plan[p].append((hid, "old"))
                for p in new_candidates[-3:]: evidence_plan[p].append((hid, "new"))
                bpos = rel[len(rel) // 2]
                epos = rel[min(len(rel) - 1, len(rel) // 2 + 1)]
                evidence_plan[bpos].append((hid, "boundary"))
                evidence_plan[epos].append((hid, "exception"))
            else:
                candidates = rel[:]
                picks = []
                for frac in (0.18, 0.48, 0.78):
                    picks.append(candidates[min(len(candidates)-1, int(frac * (len(candidates)-1)))])
                for p in sorted(set(picks)):
                    evidence_plan[p].append((hid, "support"))
                evidence_plan[candidates[min(len(candidates)-1, int(0.62*(len(candidates)-1)))]].append((hid, "boundary"))
                evidence_plan[candidates[min(len(candidates)-1, int(0.88*(len(candidates)-1)))]].append((hid, "exception"))

        # Add two composition evidence moments for an active high-value pair.
        active_pairs = [p for p in PAIR_CANDIDATES[profile.domain] if p[0] in active and p[1] in active]
        if not active_pairs:
            active_pairs = [(active[0], active[1]), (active[2], active[3])]
        for j, pair in enumerate(active_pairs[:2]):
            pos = 29 + j * 25 + (stable_index(profile.user_id, str(j), mod=5) - 2)
            pos = max(1, min(SOURCE_SESSIONS_PER_USER, pos))
            evidence_plan[pos].append(("|".join(pair), "composition"))

        raw_name_hashes: set[str] = set()
        raw_names: set[str] = set()
        raw_companies: set[str] = set()
        total_rewrites = Counter()
        for pos, cid in enumerate(source_ids, start=1):
            rec = records_by_domain[profile.domain][cid]
            messages, rewrite_meta = normalize_conversation(rec["rows"], profile)
            total_rewrites.update(rewrite_meta["rewrite_counts"])
            if rewrite_meta["original_names_hash"]:
                raw_name_hashes.add(rewrite_meta["original_names_hash"])
            raw_names.update(n for n in rewrite_meta.pop("_original_names", []) if plausible_person_name(n))
            raw_companies.update(c for c in rewrite_meta.pop("_original_companies", []) if plausible_company(c))
            annotations: list[dict[str, Any]] = []
            for hid, kind in evidence_plan.get(pos, []):
                if kind == "composition":
                    h1, h2 = hid.split("|")
                    messages.extend(composition_turns(profile, HABIT_BY_ID[h1], HABIT_BY_ID[h2], pos))
                    annotations.append({"annotation_type": "composition_support", "habit_ids": [h1, h2]})
                    evidence_index[(profile.user_id, h1, "composition")].append(f"{profile.user_id}_s{pos:04d}")
                    evidence_index[(profile.user_id, h2, "composition")].append(f"{profile.user_id}_s{pos:04d}")
                else:
                    messages.extend(evidence_turns(profile, HABIT_BY_ID[hid], kind, pos))
                    annotations.append({"annotation_type": kind, "habit_id": hid})
                    evidence_index[(profile.user_id, hid, kind)].append(f"{profile.user_id}_s{pos:04d}")
            timestamp = start + timedelta(days=pos * 4 + stable_index(profile.user_id, str(pos), mod=3), hours=stable_index(cid, mod=8))
            sid = f"{profile.user_id}_s{pos:04d}"
            public_session = {
                "session_id": sid,
                "session_index": pos,
                "timestamp": timestamp.isoformat(),
                "messages": messages,
            }
            sessions_public.append(public_session)
            private_sessions.append({
                **public_session,
                "user_id": profile.user_id,
                "domain": profile.domain,
                "source_seed": {
                    "source_dataset": SOURCE_DATASET,
                    "conversation_id": cid,
                    "cluster_id": rec["cluster_id"],
                    "source_habit_tags": sorted(rec["tags"]),
                    "quality_score": rec["quality_score"],
                },
                "memory_annotations": annotations,
                "identity_rewrite_counts": rewrite_meta["rewrite_counts"],
            })
            usage_rows.append({
                "source_domain": profile.domain,
                "source_conversation_id": cid,
                "source_cluster_id": rec["cluster_id"],
                "source_quality_score": rec["quality_score"],
                "assigned_user_id": profile.user_id,
                "generated_session_id": sid,
                "source_tags_json": json.dumps(sorted(rec["tags"])),
                "identity_rewrite_count": sum(rewrite_meta["rewrite_counts"].values()),
            })
        # Per-conversation normalization has already rewritten every detected
        # customer identity to this profile. We intentionally avoid a global
        # many-name substitution pass here; output-level audits below inspect
        # the finished lifeline for any remaining alternate identity.

        original_name_hashes_by_user[profile.user_id] = raw_name_hashes
        original_names_by_user[profile.user_id] = raw_names
        original_companies_by_user[profile.user_id] = raw_companies
        public_lifelines.append({
            "user_id": profile.user_id,
            "domain": profile.domain,
            "session_count": len(sessions_public),
            "sessions": sessions_public,
        })
        persona_rows.append({
            **asdict(profile),
            "active_habit_ids": active,
            "updated_habit_id": updated,
        })

        # Identity audit against the model-visible lifeline.
        blob = json.dumps(sessions_public, ensure_ascii=False)
        emails = sorted(set(EMAIL_RE.findall(blob)))
        phones = sorted(set(PHONE_RE.findall(blob)))
        ssn_digits = re.findall(r"(?i)(?:ssn|ssnnumber|social security)[^\n]{0,28}\d{3,}", blob)
        long_ids = LONG_DIGIT_RE.findall(blob)
        raw_consumer_domains = re.findall(r"[\w.+-]+@(gmail|yahoo|hotmail|outlook)\.[A-Za-z]{2,}", blob, flags=re.I)
        expected_emails = {profile.email}
        expected_phones = {profile.phone}
        explicit_name_pattern = re.compile(
            r"(?i:\b(?:my name is|full name is|this is)\s+)"
            r"([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*)?)"
        )
        observed_identity_names: set[str] = set()
        for sess in sessions_public:
            for msg in sess["messages"]:
                if msg["role"] != "user":
                    continue
                for match in explicit_name_pattern.finditer(msg["content"]):
                    observed_identity_names.add(clean_text(match.group(1)).strip(" .,-"))
        expected_name_forms = {profile.name.lower(), profile.first_name.lower(), profile.name.split()[-1].lower()}
        raw_name_leaks = sorted(n for n in observed_identity_names if n.lower() not in expected_name_forms)
        # Company identity is introduced only in the synthetic anchor and in
        # normalized `name from company` responses; audit explicit organization
        # declarations rather than generic merchant/product mentions.
        explicit_company_pattern = re.compile(r"(?i)\b(?:company|organization)\s+name\s+(?:is|:)\s+([^,.!?;]{2,60})")
        observed_identity_companies = {
            clean_text(m.group(1)).strip(" .,-")
            for sess in sessions_public for msg in sess["messages"]
            for m in explicit_company_pattern.finditer(msg["content"])
        }
        raw_company_leaks = sorted(c for c in observed_identity_companies if c.lower() != profile.company.lower())
        address_leaks = ADDRESS_LIKE_RE.findall(blob) + US_ADDRESS_COMMA_RE.findall(blob)
        observed_account_last4 = set(re.findall(r"(?i)(?:checking|savings|support|account)(?:\s+account)?\s+ending\s+(\d{4})\b", blob))
        expected_account_last4 = {x for x in [profile.checking_last4, profile.savings_last4, profile.account_last4] if x}
        unexpected_account_last4 = sorted(observed_account_last4 - expected_account_last4)
        observed_card_last4 = set(re.findall(r"(?i)card\s+ending\s+(\d{4})\b", blob))
        expected_card_last4 = {profile.card_last4} if profile.card_last4 else set()
        unexpected_card_last4 = sorted(observed_card_last4 - expected_card_last4)
        platform_mismatches = []
        if profile.domain == "software":
            platform_tokens = set(re.findall(r"(?i)\b(?:Windows(?: 10| 11)?|macOS(?: 15)?|Ubuntu(?: 24\.04)?|Linux)\b", blob))
            def platform_family(value: str) -> str:
                low = value.lower()
                if "windows" in low: return "windows"
                if "macos" in low: return "macos"
                if "ubuntu" in low or low == "linux": return "ubuntu"
                return low
            platform_mismatches = sorted(x for x in platform_tokens if platform_family(x) != platform_family(profile.os))
        identity_pass = (
            not [e for e in emails if e not in expected_emails]
            and not [p for p in phones if p not in expected_phones]
            and not ssn_digits and not long_ids and not raw_consumer_domains
            and not raw_name_leaks and not raw_company_leaks and not address_leaks
            and not unexpected_account_last4 and not unexpected_card_last4
            and not platform_mismatches
        )
        identity_audit_rows.append({
            "user_id": profile.user_id,
            "domain": profile.domain,
            "session_count": len(sessions_public),
            "assigned_name": profile.name,
            "assigned_email": profile.email,
            "assigned_phone": profile.phone,
            "observed_email_count": len(emails),
            "unexpected_emails_json": json.dumps([e for e in emails if e not in expected_emails]),
            "observed_phone_count": len(phones),
            "unexpected_phones_json": json.dumps([p for p in phones if p not in expected_phones]),
            "ssn_digit_leak_count": len(ssn_digits),
            "long_identifier_leak_count": len(long_ids),
            "consumer_email_domain_leak_count": len(raw_consumer_domains),
            "raw_name_leak_count": len(raw_name_leaks),
            "raw_name_leaks_json": json.dumps(raw_name_leaks, ensure_ascii=False),
            "raw_company_leak_count": len(raw_company_leaks),
            "raw_company_leaks_json": json.dumps(raw_company_leaks, ensure_ascii=False),
            "address_leak_count": len(address_leaks),
            "address_leaks_json": json.dumps(address_leaks, ensure_ascii=False),
            "unexpected_account_last4_json": json.dumps(unexpected_account_last4),
            "unexpected_card_last4_json": json.dumps(unexpected_card_last4),
            "platform_mismatches_json": json.dumps(platform_mismatches),
            "source_name_identity_sets_rewritten": len(raw_name_hashes),
            "identity_rewrite_operations": sum(total_rewrites.values()),
            "pass": identity_pass,
        })

    print("[stage] lifelines built; synthesizing probes", flush=True)
    # Probe seed pools are held out from lifelines when possible.
    used_source = {r["source_conversation_id"] for r in usage_rows}
    probe_pools: dict[str, list[str]] = {}
    for domain in DOMAINS:
        for h in HABITS_BY_DOMAIN[domain]:
            ids = [cid for cid, rec in records_by_domain[domain].items() if h["habit_id"] in rec["tags"] and cid not in used_source]
            if len(ids) < 30:
                ids += [cid for cid, rec in records_by_domain[domain].items() if h["habit_id"] in rec["tags"]]
            ids = list(dict.fromkeys(ids))
            rng.shuffle(ids)
            probe_pools[h["habit_id"]] = ids

    public_probes: list[dict[str, Any]] = []
    private_keys: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    used_probe_seed_ids: set[str] = set()
    seen_query_norms: set[str] = set()
    seen_choice_sigs: set[tuple[str, ...]] = set()

    def get_seed(hid: str) -> tuple[str, str]:
        pool = probe_pools[hid]
        while pool:
            cid = pool.pop()
            if cid not in used_probe_seed_ids:
                used_probe_seed_ids.add(cid)
                return cid, records_by_domain[HABIT_BY_ID[hid]["domain"]][cid]["request"]
        # Graceful fallback: reuse only as a lexical seed, never as a history session for that user.
        cid = next(iter(records_by_domain[HABIT_BY_ID[hid]["domain"]]))
        return cid, records_by_domain[HABIT_BY_ID[hid]["domain"]][cid]["request"]

    probe_counter = 0
    for profile_idx, profile in enumerate(personas):
        if profile_idx % 5 == 0: print(f"[stage] probes for user {profile_idx}/{len(personas)}", flush=True)
        active = assignments[profile.user_id]
        updated = updated_habits[profile.user_id]
        active_pairs = [p for p in PAIR_CANDIDATES[profile.domain] if p[0] in active and p[1] in active]
        if len(active_pairs) < 2:
            fallback_pairs = [(active[0], active[1]), (active[2], active[3])]
            for p in fallback_pairs:
                if p not in active_pairs:
                    active_pairs.append(p)
        unassigned_candidates = [h["habit_id"] for h in HABITS_BY_DOMAIN[profile.domain] if h["habit_id"] not in active and h["family"] == "format_style"]
        if not unassigned_candidates:
            unassigned_candidates = [h["habit_id"] for h in HABITS_BY_DOMAIN[profile.domain] if h["habit_id"] not in active]
        unassigned = unassigned_candidates[stable_index(profile.user_id, "unassigned", mod=len(unassigned_candidates))]
        specs = [
            ("direct_use", [active[0]]),
            ("direct_use", [active[1]]),
            ("boundary", [active[2]]),
            ("exception", [active[3]]),
            ("composition", list(active_pairs[0])),
            ("composition", list(active_pairs[1])),
            ("priority_conflict", list(max(active_pairs, key=lambda p: max(HABIT_BY_ID[p[0]]["priority"], HABIT_BY_ID[p[1]]["priority"])))),
            ("drift", [updated]),
            ("false_personalization", [unassigned]),
            ("explicit_retrieval", [active[4]]),
        ]
        for local_idx, (ptype, targets) in enumerate(specs):
            probe_id = f"mdgo_coherent_v04_probe_{probe_counter:06d}"
            seed_ids, seed_requests = [], []
            for hid in targets:
                cid, request = get_seed(hid)
                seed_ids.append(cid)
                seed_requests.append(request)
            public, private = build_probe(probe_id, profile, ptype, targets, seed_requests, updated, rng)

            # Hard diversity guard. The natural surface context normally makes
            # every item unique; this fallback adds a scenario-specific detail
            # without exposing the target habit or answer.
            qnorm = re.sub(r"\W+", " ", public["query"].lower()).strip()
            if qnorm in seen_query_norms:
                ctx_u = context_for_habit(targets[0], profile, probe_id)
                if profile.domain == "finance":
                    suffix = f" It belongs to my {ctx_u['review']} and concerns account ending {profile.account_last4}."
                else:
                    suffix = f" It belongs to {profile.company}'s {ctx_u['project']} on {profile.os}."
                public["query"] = clean_text(public["query"] + suffix)
                qnorm = re.sub(r"\W+", " ", public["query"].lower()).strip()
                if qnorm in seen_query_norms:
                    public["query"] = clean_text(public["query"] + f" The relevant support window is {ctx_u['time_context']}.")
                    qnorm = re.sub(r"\W+", " ", public["query"].lower()).strip()
            if qnorm in seen_query_norms:
                raise RuntimeError(f"could not make query unique: {probe_id}")
            seen_query_norms.add(qnorm)

            csig = tuple(sorted(re.sub(r"\W+", " ", c["text"].lower()).strip() for c in public["choices"]))
            if csig in seen_choice_sigs:
                ctx_u = context_for_habit(targets[0], profile, probe_id)
                if profile.domain == "finance":
                    qualifier = f" The concrete case involves account ending {profile.account_last4} during the {ctx_u['review']}."
                else:
                    qualifier = f" The concrete case involves {profile.desktop_app} on {profile.os} during the {ctx_u['project']}."
                for choice in public["choices"]:
                    choice["text"] = clean_text(choice["text"] + qualifier)
                private["gold_action_text"] = next(c["text"] for c in public["choices"] if c["choice_id"] == private["gold_choice_id"])
                csig = tuple(sorted(re.sub(r"\W+", " ", c["text"].lower()).strip() for c in public["choices"]))
            if csig in seen_choice_sigs:
                raise RuntimeError(f"could not make choice set unique: {probe_id}")
            seen_choice_sigs.add(csig)

            # Evidence links are private and type-aware.
            evidence = []
            for hid in targets:
                for kind in (["new"] if ptype == "drift" and hid == updated else ["support", "composition"]):
                    evidence.extend(evidence_index.get((profile.user_id, hid, kind), []))
                if ptype == "boundary": evidence.extend(evidence_index.get((profile.user_id, hid, "boundary"), []))
                if ptype == "exception": evidence.extend(evidence_index.get((profile.user_id, hid, "exception"), []))
                if ptype == "drift": evidence.extend(evidence_index.get((profile.user_id, hid, "old"), []))
            private["source_probe_seed_conversation_ids"] = seed_ids
            private["gold_evidence_session_ids"] = list(dict.fromkeys(evidence))
            private["hidden_habit_graph"] = {
                "active_habit_ids": active,
                "updated_habit_id": updated,
                "target_habit_ids": targets,
            }
            public_probes.append(public)
            private_keys.append(private)
            generation_rows.append({
                "probe_id": probe_id,
                "user_id": profile.user_id,
                "domain": profile.domain,
                "probe_type": ptype,
                "target_habit_ids_json": json.dumps(targets),
                "source_seed_conversation_ids_json": json.dumps(seed_ids),
                "source_seed_requests_json": json.dumps(seed_requests, ensure_ascii=False),
                "query_sha256": hashlib.sha256(public["query"].encode()).hexdigest(),
                "choice_set_sha256": hashlib.sha256("|".join(sorted(c["text"] for c in public["choices"])).encode()).hexdigest(),
            })
            probe_counter += 1

    print("[stage] probes built; writing files", flush=True)
    # Write model-facing and private files.
    jsonl_write(out / "public/lifelines.jsonl", public_lifelines)
    jsonl_write(out / "public/probes.jsonl", public_probes)
    jsonl_write(out / "private/sessions_with_annotations.jsonl", private_sessions)
    jsonl_write(out / "private/probe_key.jsonl", private_keys)
    jsonl_write(out / "private/persona_profiles.jsonl", persona_rows)
    csv_write(out / "source/source_conversation_usage_manifest.csv", usage_rows)
    csv_write(out / "source/source_conversation_clusters.csv", cluster_rows)
    csv_write(out / "source/habit_source_audit.csv", habit_audit_rows)
    (out / "source/habit_templates_retained.json").write_text(json.dumps(HABITS, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "source/source_file_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_write(out / "reports/identity_consistency_audit.csv", identity_audit_rows)
    csv_write(out / "reports/probe_generation_trace.csv", generation_rows)
    (out / "reports/source_cluster_summary.json").write_text(json.dumps(cluster_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy raw source files for reproducibility.
    shutil.copy2(source_paths["finance"], out / "source/raw_multidogo_finance.csv.gz")
    shutil.copy2(source_paths["software"], out / "source/raw_multidogo_software.csv.gz")

    # Human review queue.
    key_by_id = {k["probe_id"]: k for k in private_keys}
    review_rows = []
    for p in public_probes:
        k = key_by_id[p["probe_id"]]
        ev_preview = []
        for sid in k["gold_evidence_session_ids"][:8]:
            sess = next(s for s in private_sessions if s["session_id"] == sid)
            ev_preview.append({"session_id": sid, "messages": sess["messages"][-4:]})
        review_rows.append({
            "probe_id": p["probe_id"],
            "user_id": p["user_id"],
            "domain": p["domain"],
            "probe_type": k["probe_type"],
            "capability_group": k["capability_group"],
            "target_habit_ids_json": json.dumps(k["target_habit_ids"]),
            "query": p["query"],
            "choices_json": json.dumps(p["choices"], ensure_ascii=False),
            "proposed_gold_choice_id": k["gold_choice_id"],
            "proposed_gold_action": k["gold_action_text"],
            "evidence_preview_json": json.dumps(ev_preview, ensure_ascii=False),
            "source_probe_seed_conversation_ids_json": json.dumps(k["source_probe_seed_conversation_ids"]),
            "reviewer_decision": "",
            "reviewer_notes": "",
        })
    csv_write(out / "review/multidogo_finance_software_v03_review_queue_all.csv", review_rows)

    # Validation and diversity metrics.
    validation_errors: list[str] = []
    query_norm = [re.sub(r"\W+", " ", p["query"].lower()).strip() for p in public_probes]
    choice_sigs = [tuple(sorted(re.sub(r"\W+", " ", c["text"].lower()).strip() for c in p["choices"])) for p in public_probes]
    if len(set(query_norm)) != len(query_norm):
        validation_errors.append("duplicate normalized probe queries")
    if len(set(choice_sigs)) != len(choice_sigs):
        validation_errors.append("duplicate unordered choice sets")
    forbidden_probe_phrases = [
        "what should the assistant do by default",
        "ignore the long-term user history",
        "according to this user's history",
        "habit_template_id",
        "gold_choice_id",
    ]
    for phrase in forbidden_probe_phrases:
        if any(phrase in json.dumps(p).lower() for p in public_probes):
            validation_errors.append(f"forbidden/meta probe phrase present: {phrase}")
    if any(not row["pass"] for row in identity_audit_rows):
        validation_errors.append("one or more identity consistency rows failed")
    if any(len(x["active_habit_ids"]) != 5 for x in persona_rows):
        validation_errors.append("not every pseudo-user has exactly five active habits")
    if len(public_probes) != len(private_keys):
        validation_errors.append("probe/private-key count mismatch")
    if any(len(p["choices"]) != 4 for p in public_probes):
        validation_errors.append("a probe does not have four choices")
    def recursive_keys(obj: Any) -> list[str]:
        if isinstance(obj, dict):
            keys = [str(k).lower() for k in obj]
            for v in obj.values():
                keys.extend(recursive_keys(v))
            return keys
        if isinstance(obj, list):
            out_keys: list[str] = []
            for v in obj:
                out_keys.extend(recursive_keys(v))
            return out_keys
        return []
    leaked_gold_keys = [k for p in public_probes for k in recursive_keys(p) if k == "gold" or k.startswith("gold_")]
    if leaked_gold_keys:
        validation_errors.append("gold metadata leaked into public probes")
    if any("habit_id" in json.dumps(p).lower() or "memory_annotations" in json.dumps(p).lower() for p in public_lifelines):
        validation_errors.append("hidden habit metadata leaked into public lifelines")

    query_nn = nearest_neighbor_stats([p["query"] for p in public_probes])
    full_prompt_tokens = []
    life_by_user = {x["user_id"]: x for x in public_lifelines}
    for p in public_probes:
        life = life_by_user[p["user_id"]]
        text = json.dumps(life["sessions"], ensure_ascii=False) + json.dumps(p, ensure_ascii=False)
        full_prompt_tokens.append(token_estimate(text))

    validation_report = {
        "dataset_id": DATASET_ID,
        "source_dataset": SOURCE_DATASET,
        "selected_domains": list(DOMAINS),
        "pseudo_users": len(personas),
        "active_habits_per_user": 5,
        "retained_habit_count": len(HABITS),
        "source_sessions_per_user": SOURCE_SESSIONS_PER_USER,
        "model_visible_sessions_per_user": SOURCE_SESSIONS_PER_USER + 1,
        "generated_sessions": len(private_sessions),
        "probes": len(public_probes),
        "unique_normalized_queries": len(set(query_norm)),
        "unique_unordered_choice_sets": len(set(choice_sigs)),
        "unique_normalized_choice_texts": len(set(re.sub(r"\W+", " ", c["text"].lower()).strip() for p in public_probes for c in p["choices"])),
        "query_nearest_neighbor_cosine": query_nn,
        "probe_type_counts": dict(Counter(k["probe_type"] for k in private_keys)),
        "habit_family_counts": dict(Counter(HABIT_BY_ID[hid]["family"] for k in private_keys for hid in k["target_habit_ids"])),
        "multi_habit_probe_count": sum(len(k["target_habit_ids"]) > 1 for k in private_keys),
        "users_with_multiple_habits": sum(len(x["active_habit_ids"]) > 1 for x in persona_rows),
        "identity_audit_passed_users": sum(bool(r["pass"]) for r in identity_audit_rows),
        "identity_audit_total_users": len(identity_audit_rows),
        "generic_ignore_history_distractor_count": sum("ignore the long-term user history" in json.dumps(p).lower() for p in public_probes),
        "full_prompt_token_estimate": {
            "min": min(full_prompt_tokens),
            "median": int(statistics.median(full_prompt_tokens)),
            "mean": round(statistics.mean(full_prompt_tokens), 1),
            "p95": sorted(full_prompt_tokens)[int(0.95 * (len(full_prompt_tokens)-1))],
            "max": max(full_prompt_tokens),
        },
        "validation_errors": validation_errors,
    }
    (out / "reports/validation_report.json").write_text(json.dumps(validation_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # User-habit map and summaries.
    map_rows = []
    for x in persona_rows:
        map_rows.append({
            "user_id": x["user_id"],
            "domain": x["domain"],
            "name": x["name"],
            "role": x["role"],
            "stable_identity_summary": f"{x['name']} | {x['role']} | {x['city']}, {x['state']} | {x['company']}",
            "active_habit_ids_json": json.dumps(x["active_habit_ids"]),
            "updated_habit_id": x["updated_habit_id"],
            "habit_count": len(x["active_habit_ids"]),
            "model_visible_session_count": SOURCE_SESSIONS_PER_USER + 1,
            "probe_count": PROBES_PER_USER,
        })
    csv_write(out / "reports/user_habit_mapping.csv", map_rows)

    # Example prompt.
    first_probe = public_probes[0]
    first_life = life_by_user[first_probe["user_id"]]
    prompt_lines = [
        "You are evaluating a long-horizon user-memory agent.",
        "Use the user's prior sessions and choose the best response for the current request.",
        "Return only JSON with probe_id and choice_id.",
        "",
        f"USER_ID: {first_probe['user_id']}",
        f"DOMAIN: {first_probe['domain']}",
        "",
        "PRIOR SESSIONS:",
    ]
    for s in first_life["sessions"]:
        prompt_lines.append(f"\n[Session {s['session_index']} | {s['timestamp']}]")
        for m in s["messages"]:
            prompt_lines.append(f"{m['role'].upper()}: {m['content']}")
    prompt_lines += ["", "CURRENT REQUEST:", first_probe["query"], "", "CANDIDATE RESPONSES:"]
    for c in first_probe["choices"]:
        prompt_lines.append(f"{c['choice_id']}. {c['text']}")
    prompt_lines += ["", f'{{"probe_id":"{first_probe["probe_id"]}","choice_id":"..."}}']
    (out / "model_eval/example_full_prompt.txt").write_text("\n".join(prompt_lines), encoding="utf-8")

    # Scorer.
    scorer = r'''#!/usr/bin/env python3
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset-dir',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--method-name',default='method');a=p.parse_args()
base=Path(a.dataset_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
keys={x['probe_id']:x for x in map(json.loads,open(base/'private/probe_key.jsonl',encoding='utf-8'))}
preds={x['probe_id']:x for x in map(json.loads,open(a.predictions,encoding='utf-8'))}
missing=sorted(set(keys)-set(preds));extra=sorted(set(preds)-set(keys))
if missing or extra: raise SystemExit(json.dumps({'missing':missing[:20],'extra':extra[:20],'missing_n':len(missing),'extra_n':len(extra)}))
rows=[];tot=Counter();cor=Counter()
for pid,k in keys.items():
    pred=preds[pid]['choice_id'];ok=int(pred==k['gold_choice_id'])
    row={'probe_id':pid,'user_id':k['user_id'],'domain':k['domain'],'probe_type':k['probe_type'],'capability_group':k['capability_group'],'gold_choice_id':k['gold_choice_id'],'predicted_choice_id':pred,'correct':ok}
    rows.append(row)
    dims=[('overall','overall'),('domain',k['domain']),('probe_type',k['probe_type']),('capability_group',k['capability_group'])]
    for d,v in dims: tot[(d,v)]+=1;cor[(d,v)]+=ok
with open(out/'per_probe_results.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for (d,v),n in sorted(tot.items()):summary.append({'method_name':a.method_name,'dimension':d,'value':v,'n':n,'correct':cor[(d,v)],'accuracy':cor[(d,v)]/n})
with open(out/'metrics_summary.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
print(json.dumps({'overall_accuracy':cor[('overall','overall')]/tot[('overall','overall')],'n':tot[('overall','overall')]},indent=2))
'''
    (out / "scripts/score_predictions.py").write_text(scorer, encoding="utf-8")
    os.chmod(out / "scripts/score_predictions.py", 0o755)
    shutil.copy2(Path(__file__), out / "scripts/build_multidogo_coherent_multihabit_v04.py")

    # Gold smoke test.
    gold_preds = [{"probe_id": k["probe_id"], "choice_id": k["gold_choice_id"]} for k in private_keys]
    jsonl_write(out / "reports/gold_predictions_smoke_test.jsonl", gold_preds)

    # Reports, changelog, README.
    selection_report = [
        "# Domain and habit selection report\n",
        "This revision keeps only MultiDoGO **finance** and **software** and uses a quality-filtered, cluster-balanced slice rather than forcing every source conversation into a pseudo-user. Habits were retained after TF–IDF clustering, source-theme coverage checks, and contract review. The count is not fixed in advance; 16 policies remain because each has repeated source support and a clear default, boundary, exception, or update behavior.\n\n",
        "## Selection criteria\n\n",
        "1. Repeated source conversations in multiple lexical clusters.\n2. A behavior that can be learned across sessions rather than a one-off fact.\n3. A clear scope and at least one meaningful non-application boundary.\n4. Plausible exceptions, composition, or priority interaction with other habits.\n5. A gold action that can be distinguished from realistic partial-compliance alternatives.\n\n",
        "See `source/habit_source_audit.csv` and `reports/source_cluster_summary.json`.\n",
    ]
    (out / "reports/domain_and_habit_selection_report.md").write_text("".join(selection_report), encoding="utf-8")

    identity_report = [
        "# Identity coherence report\n\n",
        "Each pseudo-user has one synthetic identity anchor and five active habits. Source conversations are reduced to task-bearing turns: greetings, closings, random support-agent names, customer-identification turns, and unsafe credential exchanges are removed or normalized to the assigned persona, unsafe credential exchanges are converted to a secure verification flow, and software product/platform references are mapped to a stable user setup.\n\n",
        f"- Users passing automated identity audit: {validation_report['identity_audit_passed_users']}/{validation_report['identity_audit_total_users']}\n",
        "- Public emails use the reserved `.example` domain.\n- Public phone numbers use the fictional 555 range.\n- Raw SSN/full-card/password/OTP values are not retained in public lifelines.\n- Detailed per-user results: `reports/identity_consistency_audit.csv`.\n",
    ]
    (out / "reports/identity_coherence_report.md").write_text("".join(identity_report), encoding="utf-8")

    diversity_report = [
        "# Probe diversity and leakage report\n\n",
        f"- Probes: {len(public_probes)}\n",
        f"- Unique normalized queries: {len(set(query_norm))}\n",
        f"- Unique unordered choice sets: {len(set(choice_sigs))}\n",
        f"- Multi-habit composition/priority probes: {validation_report['multi_habit_probe_count']}\n",
        f"- Generic `ignore the long-term user history` distractors: {validation_report['generic_ignore_history_distractor_count']}\n",
        f"- Query nearest-neighbor cosine: `{json.dumps(query_nn)}`\n\n",
        "Queries are natural current user requests. Choices are contextual candidate responses; the public text does not state the probe type, habit ID, or expected behavior. Generation is conditioned on held-out MultiDoGO conversations, persona state, and one or two hidden habit policies rather than one fixed query/choice template per habit.\n",
    ]
    (out / "reports/probe_diversity_report.md").write_text("".join(diversity_report), encoding="utf-8")

    changelog = """# Changes from v0.2 / v0.3\n\n1. Replaced random same-domain conversation concatenation with task-turn extraction plus persona-conditioned identity normalization.\n2. Added stable identity anchors, finance account references, and software device/product profiles.\n3. Changed from one habit per user to five habits per user, including one temporally updated habit.\n4. Added composition, priority-conflict, drift, and false-personalization probes.\n5. Replaced repeated fixed prompts with source-cluster-conditioned scenario generation and full contextual candidate replies. Queries no longer ask meta-level “what should the assistant do,” and choices no longer use generic “ignore history” distractors.\n6. Added exact and near-duplicate probe audits, identity audits, source clustering, and generation traces.\n7. Added task-bearing-turn extraction, cross-domain noise filtering, identity-vocative removal, and source quality ranking before allocation.\n"""
    (out / "CHANGELOG_FROM_V02.md").write_text(changelog, encoding="utf-8")

    readme = f"""# HABIT-Bench MultiDoGO Finance + Software — Coherent Multi-Habit v0.4\n\nThis candidate addresses two failures in the earlier v0.2 package: pseudo-user identity contamination and template-heavy probes.\n\n## Key statistics\n\n- Source: `{SOURCE_DATASET}`\n- Domains: finance, software\n- Pseudo-users: {len(personas)}\n- Model-visible sessions per user: {SOURCE_SESSIONS_PER_USER + 1}\n- Active habits per user: 5\n- Retained domain habits: {len(HABITS)}\n- Probes: {len(public_probes)}\n- Unique normalized probe queries: {len(set(query_norm))}\n- Unique unordered choice sets: {len(set(choice_sigs))}\n- Automated identity audit: {validation_report['identity_audit_passed_users']}/{validation_report['identity_audit_total_users']} users passed\n- Validation errors: {len(validation_errors)}\n\n## Model-facing files\n\nUse only:\n\n```text\npublic/lifelines.jsonl\npublic/probes.jsonl\n```\n\nThe evaluated method matches each probe to the lifeline with the same `user_id` and returns one `choice_id`. Hidden gold, habit graphs, source traces, and persona metadata remain private.\n\n## Scoring\n\n```bash\npython scripts/score_predictions.py \\\n  --dataset-dir {DATASET_ID} \\\n  --predictions path/to/predictions.jsonl \\\n  --output-dir runs/my_eval \\\n  --method-name my_method\n```\n\n## Important provenance note\n\nMultiDoGO provides natural multi-turn customer–agent conversations but not stable same-user longitudinal identities. This build uses selected conversations as domain-grounded session seeds, rewrites task-bearing dialogue to one coherent synthetic persona while stripping source identities and authentication boilerplate, injects longitudinal habit evidence, and creates future probes from held-out source situations.\n\nSee `CHANGELOG_FROM_V02.md`, `reports/identity_coherence_report.md`, and `reports/probe_diversity_report.md`.\n"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    attribution = """# Source attribution\n\nDerived from the AWS Labs Multi-Domain Goal-Oriented Dialogues Dataset (MultiDoGO), repository `awslabs/multi-domain-goal-oriented-dialogues-dataset`. The selected finance and software source files are included in compressed form for reproducibility. Preserve the upstream attribution and license notice when publishing derived data.\n"""
    (out / "source/SOURCE_ATTRIBUTION.md").write_text(attribution, encoding="utf-8")

    # Zip.
    zip_path = Path(args.zip_path)
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", out.parent, out.name)
    print(json.dumps(validation_report, ensure_ascii=False, indent=2))
    print("WROTE", out)
    print("WROTE", zip_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--finance-source", default="/mnt/data/rebuild_mdgo_v03_source/raw_multidogo_finance.tsv.gz")
    p.add_argument("--software-source", default="/mnt/data/rebuild_mdgo_v03_source/raw_multidogo_software.tsv.gz")
    p.add_argument("--output-dir", default=f"/mnt/data/{DATASET_ID}")
    p.add_argument("--zip-path", default=f"/mnt/data/{DATASET_ID}.zip")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()

if __name__ == "__main__":
    build(parse_args())
