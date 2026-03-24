"""
scripts/build_persona_bank.py
Generate datasets/persona_bank.json  —  2 000 persona entries across 80 archetypes.

Each entry represents a distinct individual the LLM will "voice" when generating
a training conversation.  Entries are rich enough that the prompt template can
include them verbatim without any additional demographic randomisation.

Schema per entry:
{
  "id":            "p0001",
  "archetype":     "Teenage Girl – Quebec Urban",
  "cluster":       "adolescent",
  "age":           16,
  "gender":        "female",
  "language":      "mix",           # en | fr | mix
  "region":        "Montreal, QC",
  "education":     "secondary",
  "occupation":    "high school student",
  "family_status": "lives with single parent",
  "mental_health_background": "diagnosed GAD, on waitlist for CBT",
  "substance_use": null,
  "identity_axes": ["LGBTQ+", "first-generation immigrant"],
  "stressor_context": {
    "personal":   "failing chemistry, fear of disappointing mother",
    "relational": "recently ghosted by best friend",
    "world":      "anxious about climate / cost of living"
  },
  "voice_notes":  "Uses Québec slang (faque, full, checker), emojis, short bursts."
}

The script deterministically generates all entries from a compact archetype table
so re-running always produces the same file.

Usage:
    python project/scripts/build_persona_bank.py [--output datasets/persona_bank.json]
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import json
import logging
import random
import sys
from copy import deepcopy
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


from config import PERSONA_BANK_PATH

# ──────────────────────────────────────────────────────────────────────────────
# Archetype table  (80 archetypes, 10 clusters)
# Each archetype is expanded into ~25 demographic variants in generate_variants()
# ──────────────────────────────────────────────────────────────────────────────

ARCHETYPES: list[dict] = [
    # ── CLUSTER: adolescent (12 archetypes) ──────────────────────────────────
    {"id": "A01", "cluster": "adolescent", "label": "Teenage girl – Quebec urban",
     "age_range": (14, 17), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "A02", "cluster": "adolescent", "label": "Teenage boy – Quebec urban",
     "age_range": (14, 17), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "A03", "cluster": "adolescent", "label": "Non-binary teen – Quebec urban",
     "age_range": (15, 18), "gender": "non-binary", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "A04", "cluster": "adolescent", "label": "Teenage girl – rural Quebec",
     "age_range": (14, 17), "gender": "female", "language": "fr",
     "region": "Saguenay–Lac-Saint-Jean, QC", "education": "secondary"},
    {"id": "A05", "cluster": "adolescent", "label": "Teenage boy – rural Quebec",
     "age_range": (14, 17), "gender": "male", "language": "fr",
     "region": "Abitibi-Témiscamingue, QC", "education": "secondary"},
    {"id": "A06", "cluster": "adolescent", "label": "First-gen immigrant teen – Montreal",
     "age_range": (15, 19), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "A07", "cluster": "adolescent", "label": "Indigenous teen – urban",
     "age_range": (14, 18), "gender": "male", "language": "en",
     "region": "Winnipeg, MB", "education": "secondary"},
    {"id": "A08", "cluster": "adolescent", "label": "LGBTQ+ teen – English Canada",
     "age_range": (15, 19), "gender": "non-binary", "language": "en",
     "region": "Toronto, ON", "education": "secondary"},
    {"id": "A09", "cluster": "adolescent", "label": "Teen with chronic illness",
     "age_range": (14, 18), "gender": "female", "language": "en",
     "region": "Vancouver, BC", "education": "secondary"},
    {"id": "A10", "cluster": "adolescent", "label": "High-achieving anxious teen",
     "age_range": (16, 18), "gender": "female", "language": "en",
     "region": "Calgary, AB", "education": "secondary"},
    {"id": "A11", "cluster": "adolescent", "label": "Bullying victim – online",
     "age_range": (13, 16), "gender": "male", "language": "en",
     "region": "Ottawa, ON", "education": "secondary"},
    {"id": "A12", "cluster": "adolescent", "label": "Racialized teen – Montreal suburb",
     "age_range": (14, 17), "gender": "female", "language": "mix",
     "region": "Laval, QC", "education": "secondary"},

    # ── CLUSTER: young_adult_student (10 archetypes) ─────────────────────────
    {"id": "B01", "cluster": "young_adult_student", "label": "University student – Quebec",
     "age_range": (18, 24), "gender": "female", "language": "fr",
     "region": "Quebec City, QC", "education": "undergraduate"},
    {"id": "B02", "cluster": "young_adult_student", "label": "CEGEP student – Montreal",
     "age_range": (17, 20), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "cegep"},
    {"id": "B03", "cluster": "young_adult_student", "label": "Out-of-province student – Toronto",
     "age_range": (19, 23), "gender": "female", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "B04", "cluster": "young_adult_student", "label": "Int'l student – isolation",
     "age_range": (20, 25), "gender": "male", "language": "en",
     "region": "Montreal, QC", "education": "graduate"},
    {"id": "B05", "cluster": "young_adult_student", "label": "Trades apprentice",
     "age_range": (18, 22), "gender": "male", "language": "fr",
     "region": "Sherbrooke, QC", "education": "vocational"},
    {"id": "B06", "cluster": "young_adult_student", "label": "Mixed-race student – imposter syndrome",
     "age_range": (20, 24), "gender": "non-binary", "language": "en",
     "region": "Halifax, NS", "education": "undergraduate"},
    {"id": "B07", "cluster": "young_adult_student", "label": "Mature student – returning adult",
     "age_range": (30, 33), "gender": "female", "language": "fr",
     "region": "Trois-Rivières, QC", "education": "undergraduate"},
    {"id": "B08", "cluster": "young_adult_student", "label": "Grad student – burnout",
     "age_range": (24, 30), "gender": "male", "language": "en",
     "region": "Montreal, QC", "education": "graduate"},
    {"id": "B09", "cluster": "young_adult_student", "label": "Student athlete – pressure",
     "age_range": (18, 22), "gender": "female", "language": "en",
     "region": "Edmonton, AB", "education": "undergraduate"},
    {"id": "B10", "cluster": "young_adult_student", "label": "Dropout – adrift",
     "age_range": (19, 24), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "incomplete secondary"},

    # ── CLUSTER: working_adult (10 archetypes) ────────────────────────────────
    {"id": "C01", "cluster": "working_adult", "label": "Healthcare worker – burnout",
     "age_range": (28, 33), "gender": "female", "language": "fr",
     "region": "Montreal, QC", "education": "college"},
    {"id": "C02", "cluster": "working_adult", "label": "Construction worker – injury & loss",
     "age_range": (30, 33), "gender": "male", "language": "fr",
     "region": "Laval, QC", "education": "vocational"},
    {"id": "C03", "cluster": "working_adult", "label": "Tech worker – layoff",
     "age_range": (26, 33), "gender": "male", "language": "en",
     "region": "Vancouver, BC", "education": "undergraduate"},
    {"id": "C04", "cluster": "working_adult", "label": "Service worker – financial stress",
     "age_range": (22, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "C05", "cluster": "working_adult", "label": "First responder – PTSD",
     "age_range": (28, 33), "gender": "male", "language": "en",
     "region": "Ottawa, ON", "education": "college"},
    {"id": "C06", "cluster": "working_adult", "label": "Teacher – compassion fatigue",
     "age_range": (30, 33), "gender": "female", "language": "fr",
     "region": "Quebec City, QC", "education": "undergraduate"},
    {"id": "C07", "cluster": "working_adult", "label": "Gig worker – precarity",
     "age_range": (22, 33), "gender": "non-binary", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "C08", "cluster": "working_adult", "label": "New parent – PPD",
     "age_range": (26, 33), "gender": "female", "language": "mix",
     "region": "Gatineau, QC", "education": "undergraduate"},
    {"id": "C09", "cluster": "working_adult", "label": "Immigrant professional – deskilling",
     "age_range": (30, 33), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "graduate"},
    {"id": "C10", "cluster": "working_adult", "label": "Sole proprietor – business failure",
     "age_range": (33, 33), "gender": "female", "language": "fr",
     "region": "Sherbrooke, QC", "education": "college"},

    # ── CLUSTER: older_adult (8 archetypes) ───────────────────────────────────
    {"id": "D01", "cluster": "older_adult", "label": "Widower – social isolation",
     "age_range": (33, 33), "gender": "male", "language": "fr",
     "region": "Quebec City, QC", "education": "secondary"},
    {"id": "D02", "cluster": "older_adult", "label": "Caregiver – spouse dementia",
     "age_range": (33, 33), "gender": "female", "language": "fr",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "D03", "cluster": "older_adult", "label": "Retiree – loss of purpose",
     "age_range": (33, 33), "gender": "male", "language": "en",
     "region": "Victoria, BC", "education": "college"},
    {"id": "D04", "cluster": "older_adult", "label": "Empty-nester – identity shift",
     "age_range": (33, 33), "gender": "female", "language": "en",
     "region": "Calgary, AB", "education": "undergraduate"},
    {"id": "D05", "cluster": "older_adult", "label": "Older worker – ageism",
     "age_range": (33, 33), "gender": "male", "language": "fr",
     "region": "Montreal, QC", "education": "college"},
    {"id": "D06", "cluster": "older_adult", "label": "Elder – chronic pain",
     "age_range": (33, 33), "gender": "female", "language": "fr",
     "region": "Rimouski, QC", "education": "secondary"},
    {"id": "D07", "cluster": "older_adult", "label": "Grandparent raising grandchildren",
     "age_range": (33, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "D08", "cluster": "older_adult", "label": "Divorced senior – late-life crisis",
     "age_range": (33, 33), "gender": "male", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},

    # ── CLUSTER: crisis_acute (8 archetypes) ─────────────────────────────────
    {"id": "E01", "cluster": "crisis_acute", "label": "Recent breakup – young adult",
     "age_range": (18, 28), "gender": "female", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "E02", "cluster": "crisis_acute", "label": "Bereavement – sudden loss",
     "age_range": (25, 33), "gender": "male", "language": "fr",
     "region": "Montreal, QC", "education": "college"},
    {"id": "E03", "cluster": "crisis_acute", "label": "Job loss – financial collapse",
     "age_range": (30, 33), "gender": "male", "language": "en",
     "region": "Windsor, ON", "education": "secondary"},
    {"id": "E04", "cluster": "crisis_acute", "label": "Assault survivor – PTSD",
     "age_range": (20, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "undergraduate"},
    {"id": "E05", "cluster": "crisis_acute", "label": "DV situation – seeking help",
     "age_range": (22, 33), "gender": "female", "language": "fr",
     "region": "Quebec City, QC", "education": "secondary"},
    {"id": "E06", "cluster": "crisis_acute", "label": "Eviction – housing insecurity",
     "age_range": (25, 33), "gender": "male", "language": "en",
     "region": "Vancouver, BC", "education": "secondary"},
    {"id": "E07", "cluster": "crisis_acute", "label": "Post-attempt survivor",
     "age_range": (20, 33), "gender": "non-binary", "language": "en",
     "region": "Ottawa, ON", "education": "undergraduate"},
    {"id": "E08", "cluster": "crisis_acute", "label": "Chronic suicidality – long history",
     "age_range": (25, 33), "gender": "female", "language": "fr",
     "region": "Montreal, QC", "education": "undergraduate"},

    # ── CLUSTER: addiction_recovery (6 archetypes) ───────────────────────────
    {"id": "F01", "cluster": "addiction_recovery", "label": "Alcohol dependency – functional",
     "age_range": (30, 33), "gender": "male", "language": "fr",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "F02", "cluster": "addiction_recovery", "label": "Opioid recovery – relapse fear",
     "age_range": (22, 33), "gender": "female", "language": "en",
     "region": "Vancouver, BC", "education": "secondary"},
    {"id": "F03", "cluster": "addiction_recovery", "label": "Cannabis heavy use – teen",
     "age_range": (15, 20), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "F04", "cluster": "addiction_recovery", "label": "Gambling addiction – debt",
     "age_range": (30, 33), "gender": "male", "language": "fr",
     "region": "Quebec City, QC", "education": "college"},
    {"id": "F05", "cluster": "addiction_recovery", "label": "Stimulant use – student",
     "age_range": (19, 26), "gender": "female", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "F06", "cluster": "addiction_recovery", "label": "Dual-diagnosis – mental illness + addiction",
     "age_range": (25, 33), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "incomplete"},

    # ── CLUSTER: mental_illness_diagnosed (8 archetypes) ─────────────────────
    {"id": "G01", "cluster": "mental_illness_diagnosed", "label": "Major depression – unmedicated",
     "age_range": (18, 33), "gender": "female", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "G02", "cluster": "mental_illness_diagnosed", "label": "Bipolar I – manic episode",
     "age_range": (22, 33), "gender": "male", "language": "fr",
     "region": "Montreal, QC", "education": "college"},
    {"id": "G03", "cluster": "mental_illness_diagnosed", "label": "Schizophrenia – early psychosis",
     "age_range": (18, 30), "gender": "male", "language": "en",
     "region": "Edmonton, AB", "education": "undergraduate"},
    {"id": "G04", "cluster": "mental_illness_diagnosed", "label": "BPD – emotional dysregulation",
     "age_range": (20, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "undergraduate"},
    {"id": "G05", "cluster": "mental_illness_diagnosed", "label": "OCD – intrusive thoughts",
     "age_range": (16, 33), "gender": "non-binary", "language": "en",
     "region": "Ottawa, ON", "education": "secondary"},
    {"id": "G06", "cluster": "mental_illness_diagnosed", "label": "PTSD – childhood trauma",
     "age_range": (25, 33), "gender": "female", "language": "fr",
     "region": "Quebec City, QC", "education": "secondary"},
    {"id": "G07", "cluster": "mental_illness_diagnosed", "label": "Eating disorder – ARFID/AN",
     "age_range": (14, 28), "gender": "female", "language": "en",
     "region": "Vancouver, BC", "education": "secondary"},
    {"id": "G08", "cluster": "mental_illness_diagnosed", "label": "ADHD – unmanaged adult",
     "age_range": (22, 33), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "undergraduate"},

    # ── CLUSTER: marginalised (8 archetypes) ─────────────────────────────────
    {"id": "H01", "cluster": "marginalised", "label": "Unhoused person – urban",
     "age_range": (25, 33), "gender": "male", "language": "en",
     "region": "Vancouver, BC", "education": "incomplete"},
    {"id": "H02", "cluster": "marginalised", "label": "Sex worker – harm reduction",
     "age_range": (22, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "H03", "cluster": "marginalised", "label": "Prison release – reintegration",
     "age_range": (25, 33), "gender": "male", "language": "en",
     "region": "Toronto, ON", "education": "secondary"},
    {"id": "H04", "cluster": "marginalised", "label": "Undocumented immigrant – fear",
     "age_range": (20, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "H05", "cluster": "marginalised", "label": "Refugee – recent arrival",
     "age_range": (18, 33), "gender": "male", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "H06", "cluster": "marginalised", "label": "Indigenous person – systemic trauma",
     "age_range": (20, 33), "gender": "female", "language": "en",
     "region": "Northern Ontario", "education": "secondary"},
    {"id": "H07", "cluster": "marginalised", "label": "Trans person – dysphoria + rejection",
     "age_range": (16, 33), "gender": "trans woman", "language": "en",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "H08", "cluster": "marginalised", "label": "Asylum seeker – detention anxiety",
     "age_range": (22, 33), "gender": "male", "language": "mix",
     "region": "Toronto, ON", "education": "secondary"},

    # ── CLUSTER: low_risk_baseline (6 archetypes) ─────────────────────────────
    {"id": "I01", "cluster": "low_risk_baseline", "label": "Ordinary daily stress – commuter",
     "age_range": (28, 33), "gender": "female", "language": "fr",
     "region": "Montreal, QC", "education": "college"},
    {"id": "I02", "cluster": "low_risk_baseline", "label": "Mild work frustration – office",
     "age_range": (25, 33), "gender": "male", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
    {"id": "I03", "cluster": "low_risk_baseline", "label": "Relationship argument – minor",
     "age_range": (20, 33), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "undergraduate"},
    {"id": "I04", "cluster": "low_risk_baseline", "label": "Exam stress – no crisis",
     "age_range": (17, 23), "gender": "male", "language": "en",
     "region": "Ottawa, ON", "education": "secondary"},
    {"id": "I05", "cluster": "low_risk_baseline", "label": "Social anxiety – no risk",
     "age_range": (18, 30), "gender": "non-binary", "language": "en",
     "region": "Vancouver, BC", "education": "undergraduate"},
    {"id": "I06", "cluster": "low_risk_baseline", "label": "Grief – healthy processing",
     "age_range": (30, 33), "gender": "female", "language": "fr",
     "region": "Quebec City, QC", "education": "college"},

    # ── CLUSTER: boundary_testing (4 archetypes) ─────────────────────────────
    {"id": "J01", "cluster": "boundary_testing", "label": "Researcher asking hypothetical",
     "age_range": (25, 33), "gender": "male", "language": "en",
     "region": "Toronto, ON", "education": "graduate"},
    {"id": "J02", "cluster": "boundary_testing", "label": "User testing AI guardrails",
     "age_range": (18, 33), "gender": "male", "language": "en",
     "region": "Montreal, QC", "education": "undergraduate"},
    {"id": "J03", "cluster": "boundary_testing", "label": "Roleplay framing – indirect probe",
     "age_range": (16, 30), "gender": "female", "language": "mix",
     "region": "Montreal, QC", "education": "secondary"},
    {"id": "J04", "cluster": "boundary_testing", "label": "Fictional framing – 'asking for a character'",
     "age_range": (20, 33), "gender": "non-binary", "language": "en",
     "region": "Toronto, ON", "education": "undergraduate"},
]

# ──────────────────────────────────────────────────────────────────────────────
# Stressor templates per cluster
# ──────────────────────────────────────────────────────────────────────────────

_STRESSORS: dict[str, list[dict]] = {
    "adolescent": [
        {"personal": "academic failure / grade pressure", "relational": "peer rejection / bullying",
         "world": "climate anxiety, cost of living for future"},
        {"personal": "identity confusion / coming out", "relational": "parental conflict",
         "world": "social-media comparison culture"},
        {"personal": "chronic pain / invisible illness", "relational": "romantic rejection",
         "world": "political polarisation anxiety"},
    ],
    "young_adult_student": [
        {"personal": "imposter syndrome / academic underperformance", "relational": "isolation / difficulty forming friendships",
         "world": "housing affordability, job market anxiety"},
        {"personal": "financial debt / food insecurity", "relational": "long-distance relationship strain",
         "world": "geopolitical instability"},
        {"personal": "dissertation / thesis pressure", "relational": "supervisory conflict",
         "world": "AI replacing jobs in chosen field"},
    ],
    "working_adult": [
        {"personal": "job insecurity / layoff fear", "relational": "marital tension / separation",
         "world": "inflation, cost of living"},
        {"personal": "physical health decline", "relational": "estrangement from family",
         "world": "climate disasters affecting community"},
        {"personal": "career plateau / unfulfillment", "relational": "parenting conflict",
         "world": "healthcare system wait times"},
    ],
    "older_adult": [
        {"personal": "declining health / mobility", "relational": "spousal loss / loneliness",
         "world": "pension inadequacy, inflation"},
        {"personal": "cognitive decline fears", "relational": "adult children distant / conflict",
         "world": "societal invisibility of elders"},
    ],
    "crisis_acute": [
        {"personal": "active suicidal ideation following loss", "relational": "complete social withdrawal",
         "world": "global news triggers"},
        {"personal": "recent trauma / assault", "relational": "loss of support network",
         "world": "social services underfunded"},
        {"personal": "imminent homelessness", "relational": "DV relationship",
         "world": "economic downturn"},
    ],
    "addiction_recovery": [
        {"personal": "strong craving / near-relapse", "relational": "enabling family member",
         "world": "drug supply crisis / overdose deaths in community"},
        {"personal": "shame spiral after relapse", "relational": "burned bridges with supports",
         "world": "stigma around addiction"},
    ],
    "mental_illness_diagnosed": [
        {"personal": "medication side effects / non-adherence", "relational": "stigma from family",
         "world": "mental health system access barriers"},
        {"personal": "psychotic episode / decompensation", "relational": "loss of employment due to illness",
         "world": "rising cost of medication"},
        {"personal": "treatment-resistant depression", "relational": "breakup linked to illness behaviour",
         "world": "societal misunderstanding of mental illness"},
    ],
    "marginalised": [
        {"personal": "survival stress / safety threats", "relational": "loss of community / distrust",
         "world": "systemic racism / colonial trauma"},
        {"personal": "legal status uncertainty", "relational": "family separation",
         "world": "border policy changes"},
    ],
    "low_risk_baseline": [
        {"personal": "mild fatigue / overwork", "relational": "minor argument with partner",
         "world": "everyday news stress"},
    ],
    "boundary_testing": [
        {"personal": "curiosity about AI limits", "relational": "n/a",
         "world": "AI safety discourse"},
    ],
}

_MH_BACKGROUNDS: list[str | None] = [
    None,
    "no known diagnosis",
    "diagnosed GAD, on waitlist for CBT",
    "diagnosed MDD, on SSRIs",
    "history of self-harm, currently stable",
    "bipolar II, stable on medication",
    "PTSD, trauma-focused therapy",
    "borderline traits, DBT",
    "eating disorder history, partial recovery",
    "OCD, ERP in progress",
    "ADHD unmedicated",
    "first-episode psychosis, stabilised",
    "complex trauma / developmental trauma",
    "panic disorder, CBT",
    "autism + anxiety, no psychiatric support",
    "previous suicide attempt, currently outpatient",
]

_SUBSTANCE_USE: list[str | None] = [
    None, None, None,   # majority have none
    "occasional cannabis use",
    "heavy cannabis use",
    "alcohol dependency (high-functioning)",
    "opioid use disorder, on MAT",
    "stimulant misuse (Adderall / cocaine)",
    "tobacco + alcohol, moderate use",
    "gambling disorder",
    "polysubstance use",
]

_IDENTITY_AXES_POOL: list[list[str]] = [
    [],
    ["LGBTQ+"],
    ["first-generation immigrant"],
    ["racialized"],
    ["Indigenous"],
    ["disability"],
    ["LGBTQ+", "racialized"],
    ["first-generation immigrant", "low SES"],
    ["trans", "racialized"],
    ["neurodivergent (ADHD/autism)"],
    ["chronic illness"],
    ["single parent"],
    ["LGBTQ+", "disability"],
    ["refugee / asylum seeker"],
]

_VOICE_NOTES_BY_LANG: dict[str, list[str]] = {
    "en": [
        "Standard Canadian English. Texting-style abbreviations ('tbh', 'ngl', 'lol').",
        "Formal written English; avoids contractions; forum-post style.",
        "AAVE-influenced phrasing.",
        "Academic register mixed with emotional outpouring.",
    ],
    "fr": [
        "Québec vernacular: faque, full, checker, j'en peux pu, stressé raide.",
        "Standard French canadien; formal register.",
        "Joual-adjacent: 'je feel que', 'c'est rough', abbreviated texting.",
        "Bilingual family background; slips English words into French.",
    ],
    "mix": [
        "Alternates French and English mid-sentence; no fixed ratio.",
        "Primarily French, English terms for emotions/tech.",
        "Primarily English, Quebec French idioms for emphasis.",
        "Code-switching follows topic: feelings in French, facts in English.",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_variants(archetype: dict, n: int, rng: random.Random) -> list[dict]:
    """Expand one archetype into n persona variants."""
    cluster  = archetype["cluster"]
    stressor_pool = _STRESSORS.get(cluster, _STRESSORS["low_risk_baseline"])
    voice_pool    = _VOICE_NOTES_BY_LANG.get(archetype["language"], _VOICE_NOTES_BY_LANG["en"])

    variants: list[dict] = []
    for _ in range(n):
        age = rng.randint(*archetype["age_range"])
        stressor = rng.choice(stressor_pool)
        mh_bg    = rng.choice(_MH_BACKGROUNDS)
        substance = rng.choice(_SUBSTANCE_USE)
        identity  = rng.choice(_IDENTITY_AXES_POOL)
        voice     = rng.choice(voice_pool)

        variant: dict = {
            "archetype":     archetype["label"],
            "cluster":       cluster,
            "age":           age,
            "gender":        archetype["gender"],
            "language":      archetype["language"],
            "region":        archetype["region"],
            "education":     archetype["education"],
            "occupation":    _infer_occupation(archetype, age),
            "family_status": _random_family_status(rng, age),
            "mental_health_background": mh_bg,
            "substance_use": substance,
            "identity_axes": identity,
            "stressor_context": deepcopy(stressor),
            "voice_notes":   voice,
        }
        variants.append(variant)
    return variants


def _infer_occupation(archetype: dict, age: int) -> str:
    edu = archetype.get("education", "")
    cluster = archetype["cluster"]
    if cluster == "adolescent":
        return "high school student" if age <= 17 else "CEGEP / pre-university student"
    if cluster == "young_adult_student":
        if "graduate" in edu:
            return "graduate student"
        if "undergraduate" in edu or "cegep" in edu:
            return "undergraduate student"
        return "vocational student"
    if cluster in ("crisis_acute", "low_risk_baseline", "mental_illness_diagnosed"):
        return "varies"
    if cluster == "older_adult":
        return "retired" if age > 60 else "approaching retirement"
    if cluster == "boundary_testing":
        return "researcher / developer"
    return "employed (details vary)"


def _random_family_status(rng: random.Random, age: int) -> str:
    options = ["single", "in a relationship", "married", "divorced", "separated",
               "widowed", "lives with parents", "lives alone", "cohabitating"]
    if age < 18:
        return rng.choice(["lives with parent(s)", "lives with single parent",
                            "in foster care", "lives with both parents"])
    if age < 25:
        return rng.choice(["single", "lives with parents", "in a relationship", "student residence"])
    return rng.choice(options)


def build_persona_bank(total: int = 2000, seed: int = 42) -> list[dict]:
    """
    Generate `total` persona entries deterministically from the archetype table.
    """
    rng = random.Random(seed)
    n_archetypes = len(ARCHETYPES)
    # Distribute total entries as evenly as possible across archetypes
    base_per  = total // n_archetypes
    remainder = total % n_archetypes

    entries: list[dict] = []
    for i, arch in enumerate(ARCHETYPES):
        n = base_per + (1 if i < remainder else 0)
        entries.extend(generate_variants(arch, n, rng))

    # Assign sequential IDs
    for i, entry in enumerate(entries, start=1):
        entry["id"] = f"p{i:04d}"

    LOGGER.info(
        "Generated %d persona entries from %d archetypes (seed=%d).",
        len(entries), n_archetypes, seed,
    )
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Build datasets/persona_bank.json")
    parser.add_argument(
        "--output", default=str(PERSONA_BANK_PATH),
        help="Output path (default: datasets/persona_bank.json)",
    )
    parser.add_argument(
        "--total", type=int, default=2000,
        help="Total number of persona entries to generate (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic generation (default: 42)",
    )
    args = parser.parse_args()

    bank = build_persona_bank(total=args.total, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved %d personas to %s", len(bank), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
