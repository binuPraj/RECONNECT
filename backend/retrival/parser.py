"""Categorise a patient question. Invalid JSON becomes intent=other."""
from __future__ import annotations

import json
import re

from retrival.config import INTENTS, MEMORY_CATEGORIES, PARSER_FIELDS

PARSER_PROMPT = """You classify one question from a memory-aid wearer with dementia.
Return ONLY JSON with these keys:
intent: who_is | what_happening | recall | schedule | other
person_ref: the person's words or null (e.g. "my sister", "Sam", "my daughter")
time_ref: the time words or null (e.g. "next week", "last Monday")
occasion_ref: occasion words or null (e.g. "my granddaughter's birthday")
location_ref: place words or null
topic_ref: short topic words or null
tense: past | present | future | null
category_hint: array of relevant memory categories from: ["identity", "relationship", "commitment", "significant_event", "routine", "preference"] or []
is_followup: true only if this utterance continues the previous question using the same matter

INTENT GUIDELINES:
- "who_is": STRICTLY for visual questions looking at someone in person with no name (e.g. "who is this?", "who is that?", "who am I looking at?").
- "recall": For verbal questions asking about a named person, relationship, identity, or past events (e.g. "Who is Sam?", "Who is my daughter?", "What happened when I met my daughter?", "What did Sam say?").
- "schedule": For future plans, commitments, appointments (e.g. "When am I meeting my daughter?", "What do I have tomorrow?").
- "what_happening": For current situation or what just took place (e.g. "What is happening right now?").
- "other": General / broad questions.

EXAMPLES:
- "who is this?" -> {{"intent": "who_is", "person_ref": null, "category_hint": ["identity", "relationship"], "is_followup": false}}
- "who is Sam?" -> {{"intent": "recall", "person_ref": "Sam", "category_hint": ["identity", "relationship"], "tense": "present", "is_followup": false}}
- "who is my daughter?" -> {{"intent": "recall", "person_ref": "my daughter", "category_hint": ["identity", "relationship"], "tense": "present", "is_followup": false}}
- "what happened when I met my daughter?" -> {{"intent": "recall", "person_ref": "my daughter", "category_hint": ["significant_event", "relationship"], "tense": "past", "is_followup": false}}
- "what am I supposed to do next week?" -> {{"intent": "schedule", "time_ref": "next week", "category_hint": ["commitment"], "tense": "future", "is_followup": false}}
- "what is happening?" -> {{"intent": "what_happening", "tense": "present", "is_followup": false}}


Previous resolved question JSON: {last_query}
Recent Q&A: {turns}

Question: {question}
"""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    blob = fenced.group(1) if fenced else stripped
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _clean_ref(value) -> str | None:
    if value is None or value is False:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a"}:
        return None
    return text


def normalize_parsed(raw: dict | None) -> dict:
    data = raw or {}
    intent = str(data.get("intent") or "other").strip().casefold().replace(" ", "_")
    aliases = {"whois": "who_is", "who_is_this": "who_is", "whats_happening": "what_happening",
               "what_is_happening": "what_happening"}
    intent = aliases.get(intent, intent)
    if intent not in INTENTS:
        intent = "other"
    tense = _clean_ref(data.get("tense"))
    if tense:
        tense = tense.casefold()
        if tense not in {"past", "present", "future"}:
            tense = None
    follow = data.get("is_followup")
    if isinstance(follow, str):
        follow = follow.strip().casefold() in {"true", "1", "yes"}

    # Clean category_hint
    raw_cats = data.get("category_hint")
    category_hint = []
    if isinstance(raw_cats, list):
        for c in raw_cats:
            c_str = str(c).strip().casefold()
            if c_str in MEMORY_CATEGORIES and c_str not in category_hint:
                category_hint.append(c_str)
    elif isinstance(raw_cats, str):
        c_str = raw_cats.strip().casefold()
        if c_str in MEMORY_CATEGORIES:
            category_hint.append(c_str)

    parsed = {
        "intent": intent,
        "person_ref": _clean_ref(data.get("person_ref")),
        "time_ref": _clean_ref(data.get("time_ref")),
        "occasion_ref": _clean_ref(data.get("occasion_ref")),
        "location_ref": _clean_ref(data.get("location_ref")),
        "topic_ref": _clean_ref(data.get("topic_ref")),
        "tense": tense,
        "category_hint": category_hint,
        "is_followup": bool(follow),
    }
    for key in PARSER_FIELDS:
        parsed.setdefault(key, None)
    return parsed


def heuristic_parse(question: str) -> dict:
    text = (question or "").casefold()
    intent = "other"
    tense = None
    category_hint = []
    if re.search(r"\bwho is (this|that)\b", text) or "who am i looking" in text:
        intent = "who_is"
        category_hint = ["identity", "relationship"]
    elif re.search(r"\bwho is\b", text):
        intent = "recall"
        category_hint = ["identity", "relationship"]
    elif re.search(r"what('?s| is) happening|what is going on|what are they saying", text):
        intent = "what_happening"
        tense = "present"
    elif re.search(r"supposed to|schedule|appointment|what do i have|next week|tomorrow", text):
        intent = "schedule"
        category_hint = ["commitment"]
        tense = "future"
    elif re.search(r"what happened|last (monday|week|night)|yesterday|remember when|did .+ say", text):
        intent = "recall"
        category_hint = ["significant_event", "relationship"]
        tense = "past"
    time_ref = None
    for phrase in ("day after tomorrow", "day before yesterday", "next week", "last week",
                   "this week", "next month", "last month", "yesterday", "tomorrow", "today",
                   "last monday", "last tuesday", "last wednesday", "last thursday",
                   "last friday", "last saturday", "last sunday", "next monday", "next tuesday",
                   "next wednesday", "next thursday", "next friday", "next saturday", "next sunday"):
        if phrase in text:
            time_ref = phrase
            break
    person_ref = None
    match = re.search(r"\b(my|your)\s+(granddaughter|grandson|daughter|son|sister|brother|mother|father|wife|husband|friend)\b", text)
    if match:
        person_ref = match.group(0)
    else:
        named = re.search(r"\bwho is ([A-Z][a-z]+)\b", question or "", re.IGNORECASE) or re.search(r"\bwith ([A-Z][a-z]+)\b", question or "")
        if named:
            person_ref = named.group(1)
    location_ref = None
    loc = re.search(r"\b(?:in|at) (?:the )?([a-z][a-z ]{2,20})\b", text)
    if loc and loc.group(1) not in {"the", "a", "my"}:
        location_ref = loc.group(1)
    return normalize_parsed({
        "intent": intent,
        "person_ref": person_ref,
        "time_ref": time_ref,
        "occasion_ref": None,
        "location_ref": location_ref,
        "topic_ref": None,
        "tense": tense,
        "is_followup": False,
    })


def parse_question(question: str, llm, *, session: dict, patient_id: int) -> tuple[dict, str | None]:
    prompt = PARSER_PROMPT.format(
        last_query=json.dumps(session.get("last_query")),
        turns=json.dumps(session.get("turns") or []),
        question=question,
    )
    raw = llm.complete(
        prompt, kind="parser", patient_id=patient_id,
        session_id=session.get("session_id"), question=question,
        extra={"last_query": session.get("last_query")},
    )
    extracted = _extract_json(raw or "")
    if extracted is None:
        # Invalid JSON becomes other. Heuristic is only used when the LLM is down.
        parsed = heuristic_parse(question) if raw is None else normalize_parsed(None)
        return parsed, raw
    return normalize_parsed(extracted), raw


def inherit_slots(parsed: dict, last_query: dict | None) -> dict:
    if not last_query:
        return dict(parsed)
    merged = dict(parsed)
    for key in ("person_ref", "time_ref", "occasion_ref", "location_ref", "topic_ref", "tense"):
        if not merged.get(key) and last_query.get(key):
            merged[key] = last_query[key]
    return merged


def is_vague(parsed: dict) -> bool:
    slots = [parsed.get(key) for key in ("person_ref", "time_ref", "occasion_ref", "location_ref", "topic_ref")]
    if parsed.get("intent") in {"who_is", "what_happening"}:
        return False
    if parsed.get("intent") == "other" and not any(slots):
        return True
    if parsed.get("intent") in {"schedule", "recall"} and not any(slots):
        return True
    return False
