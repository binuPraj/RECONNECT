"""Consolidate closed events into durable LLM-synthesized memories."""
from __future__ import annotations
import json
import re
import uuid
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any
from app.events.db import get_connection, get_event, init_db, list_events
from app.memory.store import MemoryStore
from app.memory.time_resolution import TZ, direction_ok, resolve_time

BATCH_INTERVAL_SECONDS = 5 * 60
MEMORY_CATEGORIES = {"identity", "relationship", "commitment", "significant_event", "preference", "routine_update"}
PROMPT_TEMPLATE = """You are consolidating raw conversational events into durable long-term
memories for a dementia-care assistive system.

TODAY: {today}

PATIENT CONTEXT:
{patient_context}

EXISTING MEMORIES (merge candidates; M1.T1 labels are existing TimeRefs):
{existing_block}

NEW EVENTS THIS BATCH:
{events_block}

IDENTIFIER RULES (must be followed exactly):
- Use only the short labels shown in square brackets: E1 for source_events,
  M1 for target_memory_id, and M1.T1 for supersedes_time_refs. Never invent a
  label. The backend converts valid labels to its internal IDs.

MEMORY BOUNDARY RULES (separate by default):
- Create a NEW memory unless the new event is a genuine continuation, update,
  correction, or confirmation of the same specific fact, relationship, plan,
  or ongoing matter as an existing memory.
- Sharing a participant, person, place, date, or broad theme is NOT enough to
  merge. For example, an event involving Kris and Sarah must not merge into
  Kris's identity memory merely because Kris is present.
- A merge is allowed only when the existing memory and result have the SAME
  category and the merged summary remains one coherent, narrowly focused fact.
- Identity memories contain stable facts about one person's identity, health,
  role, or enduring preferences. Never merge a relationship, disagreement,
  appointment, or one-time event into an identity memory.
- A conflict, reconciliation, or change in how two people relate belongs in a
  separate relationship memory (or significant_event when it is a one-off
  incident), even if either person has an existing identity memory.
- When uncertain whether two matters are genuinely the same, choose action
  "new". Prefer several precise memories over one broad, mixed summary.
- Never put speaker IDs in the summary.
- Summaries must focus on the underlying fact, event, or ongoing
  situation rather than who reported it.
- Preserve the information source in about.reported_by.
- Do not add uncertainty markers solely because the speaker is
  unenrolled. Use cautious wording only when the content itself
  is uncertain, speculative, or explicitly unconfirmed.

INSTRUCTIONS:
Read all new events above as a whole. Identify only the small number of
distinct memories genuinely worth keeping long-term — merging multiple
events into a single memory wherever they describe the same person,
relationship, or ongoing matter. Do not produce one output per input
event; produce only the final set of memories that should exist after
this batch. Silently omit routine or trivial events entirely — do not
list what you are excluding. Meaningful life updates about friends/relatives (e.g. jobs, projects, news) are NOT trivial and must be preserved as separate memories if in the same event or batch has different topics.

Keep only memories matching one of these categories:
  - identity: establishes/clarifies who a person is or their relation to the patient
  - relationship: reinforces or updates a known relationship
  - commitment: a plan, promise, or appointment tied to a date/time
  - significant_event: notable news, milestones, personal achievements, health changes, or life updates about family and friends
  - preference: a stated like/dislike/habit worth remembering
  - routine_update: a recurring caregiving fact worth tracking

Merge into an existing memory listed above if it concerns the same
person or ongoing matter; otherwise create a new one. Entities may be PLACE,
ITEM, EVENT, PERSON, or RELATION. Use RELATION when the transcript and its
conversation context establish that it is a relation of the patient, such as
"my daughter" or "your daughter"; set text to the relation word (for example
"daughter"). Code resolves a confirmed caregiver relationship to its enrolled
person. For somebody else's relation such as "Sam's daughter", use PERSON with
the complete phrase as text. Never use DATE as an entity: dates use
time_refs.

Keep these roles distinct: participants are people present or speaking. Return
the participants field. It must include every labelled event speaker EXCEPT
the patient. Never list the patient as a participant and never imply that the
patient has a PRESENT_AT relationship: the system wearer is always present.
about contains ONLY genuinely absent third parties, each as
{{"person":"<name>", "type":"PERSON|RELATION", "reported_by":"<participant>",
"what_was_said":"<concise supported claim>"}}. Do not duplicate a participant
in about. If someone reports a fact about an absent person, describe the
absent person's situation or the underlying event in the summary
(for example, "Sarah is planning a visit"). Preserve who reported
the information in about.reported_by. Do not present reported
information as directly observed fact. Use [] when no absent third party
is discussed. Do not assume a speaker is the patient. "I/me/my" refers to the
labelled speaker. "You/your" refers to the named patient when direct speech
clearly addresses the system wearer. Rewrite these pronouns as names in the summary: if
Sam says "I will pick you up tomorrow" to patient Rita, write "Sam will pick
up Rita tomorrow." For a specific time phrase, copy its exact words into time_refs
with tense (past, present, future) and the source event ID. Never calculate a
date; code resolves it from that event timestamp. Omit vague words such as
"soon" and "recently", or facts unsupported by the transcript.
Create a TimeRef only for a past/future date or period different from the
immediate present. Never create one for "now" or "right now". For every
time_refs item, add what_was_mentioned: a concise supported sentence
describing the plan or event that the phrase refers to; concerned_with: the
person scheduled for the plan or the person whose past incident/experience is
being discussed; topic: a short specific matter label; and commitment_by for
a commitment, naming the person who promised or is responsible for it. For
example, for "last Monday", use "Ram picked up Sita last Monday" if that is
what the transcript says.
Do not merge commitments merely because their dates overlap. They must concern
the same specific matter, concerned person, and commitment maker. A different
person may replace the maker only when the new statement explicitly says they
are taking over that same plan.

For each memory, summaries must be fact-centered and event-oriented, not
speaker-centered. Describe what happened, what is ongoing, or
what is known about the person rather than narrating who said it. Write the single most accurate conclusion
that synthesizes everything relevant across ALL contributing source
events, not a restatement of just one of them and not a concatenation
of each event in turn. If this memory is a merge into an existing one,
the new summary must supersede and incorporate the prior summary's
still-relevant content, not simply append to it — write it as if
composing the memory fresh, informed by everything now known.
When action is "merge", also return what_was_updated: a very short,
supported description of the new information or correction. Do not return a
time for it; code records the exact time the system received the update.

Label the overall emotional tone of each memory as one of: happy, sad,
angry, anxious, calm, distressed, neutral. Base this on the tone of the
conversation across the contributing events as a whole, not a single
word or moment taken out of context. If tone is genuinely mixed or
unclear, use neutral rather than guessing.

Do not compute or report any timestamp yourself. Only report which
source events contributed to each memory; timestamps are derived
separately from those events by the calling system.

Return ONLY a JSON array, no other text, this exact schema:

[{{
  "action": "merge" | "new",
  "category": "identity" | "relationship" | "commitment" | "significant_event" | "preference" | "routine_update",
  "target_memory_id": "<only if action=merge>",
  "what_was_updated": "<very short change description; required only for merge>",
  "summary": "<the single best synthesized conclusion across all source events>",
  "participants": ["<people present or speaking>"],
  "about": [{{"person": "<absent third party>", "type":"PERSON|RELATION", "reported_by": "<participant who said it>", "what_was_said": "<concise claim>"}}],
  "source_events": ["<event_id, ...>"],
  "importance": "high" | "medium" | "low",
  "entities": [{{"type": "PLACE|ITEM|EVENT|PERSON|RELATION", "text": "..."}}],
  "topic": "<short free-text label>",
  "emotion": "happy" | "sad" | "angry" | "anxious" | "calm" | "distressed" | "neutral",
  "time_refs": [{{"ref_key":"T1","phrase":"<exact words>","tense":"past|present|future","from_event":"<event id>","topic":"<specific matter>","concerned_with":"<person this time concerns>","commitment_by":"<person responsible; commitment only>","takes_over_from":"<only when explicitly taking over a prior commitment>","what_was_mentioned":"<concise supported statement this time refers to>"}}],
  "supersedes_time_refs": {{"M3.T1":"T1"}}
}}]"""

def utc_now() -> datetime:
    """Compatibility name; RECONNECT stores generated timestamps in Kathmandu time."""
    return datetime.now(TZ)
def _dt(value: datetime | str) -> datetime: return value if isinstance(value, datetime) else datetime.fromisoformat(value)
def _iso(value: datetime | str) -> str: return _dt(value).isoformat()


def _name_key(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

def flatten_transcript(event: dict) -> list[dict]:
    """Merge adjacent same-speaker raw segments, retaining the run's endpoints."""
    lines = []
    for segment in event.get("segments", []):
        if lines and lines[-1]["speaker"] == segment["speaker"]:
            lines[-1]["text"] = f"{lines[-1]['text']} {segment['text']}".strip()
            lines[-1]["end"] = segment["end"]
        else:
            lines.append({"speaker": segment["speaker"], "start": segment["start"],
                          "end": segment["end"], "text": segment["text"]})
    return lines

def render_transcript_text(event: dict) -> str:
    return "\n".join(f"[{_dt(line['start']):%Y-%m-%d %H:%M:%S%z} - {_dt(line['end']):%Y-%m-%d %H:%M:%S%z}] {line['speaker']}: {line['text']}" for line in flatten_transcript(event))

def should_run_batch(last_batch_time: datetime | str | None, unconsumed_event_count: int, now: datetime | None = None) -> bool:
    return bool(unconsumed_event_count and (last_batch_time is None or (now or utc_now()) - _dt(last_batch_time) >= timedelta(seconds=BATCH_INTERVAL_SECONDS)))

def _dedupe_memories(memories: Iterable[dict], cap: int = 14) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for memory in memories:
        memory_id = memory.get("memory_id")
        if memory_id and memory_id not in seen:
            seen.add(memory_id)
            out.append(memory)
        if len(out) >= cap:
            break
    return out


def _merge_candidates(events: list[dict], memory_store: MemoryStore, now: datetime) -> list[dict]:
    """Provide plans, person context, and semantic context to consolidation.

    An absent person may be named only in transcript text, so this deliberately
    does not restrict the upcoming-plan or vector buckets to participants.
    """
    patient_ids = {event.get("patient_id") for event in events if event.get("patient_id") is not None}
    patient_id = next(iter(patient_ids)) if len(patient_ids) == 1 else None
    upcoming = getattr(memory_store, "get_upcoming", lambda *_: [])(now.isoformat(), 10, patient_id)
    participants = {person for event in events for person in event.get("participants", [])}
    recent = [memory for person in sorted(participants) for memory in memory_store.get_recent(person, 2, patient_id)]
    text = " ".join(render_transcript_text(event) for event in events)
    semantic: list[dict] = []
    vector = getattr(memory_store, "vector", None)
    get_many = getattr(memory_store, "get_many", None)
    if vector is not None and get_many is not None:
        semantic = get_many(vector.search(text, k=5))
    return _dedupe_memories([*upcoming, *recent, *semantic])


def _patient_name(events: Iterable[dict]) -> str | None:
    patient_ids = {event.get("patient_id") for event in events if event.get("patient_id") is not None}
    if len(patient_ids) != 1:
        return None
    try:
        from database.db import get_patient_name
        return get_patient_name(next(iter(patient_ids)))
    except Exception:
        return None


def _patient_context(events: Iterable[dict]) -> str:
    name = _patient_name(events)
    if name:
        patient_ids = {event.get("patient_id") for event in events if event.get("patient_id") is not None}
        relations: list[str] = []
        if len(patient_ids) == 1:
            try:
                from database.db import get_patient_enrolled_people
                relations = [
                    f"{row['relation']}: {row['name']}"
                    for row in get_patient_enrolled_people(next(iter(patient_ids)))
                    if row.get("relation") and row.get("name")
                ]
            except Exception:
                pass
        relation_context = (
            " Known patient relations: " + "; ".join(relations) + "."
            if relations else " No confirmed patient relations are currently available."
        )
        return (
            f"The patient wearing the system is {name}. {name} is not automatically a speaker or participant. "
            "Use this name only when direct second-person speech clearly addresses the patient."
            + relation_context +
            " Infer a RELATION entity from the whole conversation and this context, including direct speech such as 'your daughter'."
        )
    return "No patient name is available. Do not assume any speaker is the patient."


def build_batch_context(
    closed_events: Iterable[dict], memory_store: MemoryStore, now: datetime | None = None
) -> tuple[str, str, dict[str, str]]:
    events = sorted(list(closed_events), key=lambda event: _dt(event["start"]))
    current = now or utc_now()
    aliases: dict[str, str] = {}
    lines: list[str] = []
    for index, memory in enumerate(_merge_candidates(events, memory_store, current), 1):
        memory_alias = f"M{index}"
        aliases[memory_alias] = memory["memory_id"]
        refs: list[str] = []
        for ref_index, ref in enumerate(memory.get("time_refs", []), 1):
            if ref.get("status") != "active":
                continue
            time_alias = f"{memory_alias}.T{ref_index}"
            aliases[time_alias] = ref["time_ref_id"]
            start = ref.get("start") or "unresolved"
            end = ref.get("end")
            flag = "UPCOMING" if end and _dt(end) > current else "PAST"
            refs.append(
                f'[{time_alias}] "{ref.get("phrase", "")}"; '
                f'topic={ref.get("topic") or "unspecified"}; '
                f'concerned_with={ref.get("concerned_with") or "unspecified"}; '
                f'commitment_by={ref.get("commitment_by") or "unspecified"}; '
                f'= {start} ({flag})'
            )
        lines.append(
            f"[{memory_alias}] participants={memory.get('participants', [])}; about={memory.get('about', [])} "
            f"— {memory['summary']} | times: {' ; '.join(refs) or 'none'}"
        )
    event_lines: list[str] = []
    for index, event in enumerate(events, 1):
        alias = f"E{index}"
        aliases[alias] = event["event_id"]
        start = _dt(event["start"])
        event_lines.append(
            f"[{alias}] event_start={start:%Y-%m-%d (%A) %H:%M %z} "
            f"participants={', '.join(event.get('participants', []))}\n{render_transcript_text(event)}"
        )
    return "\n".join(lines) or "(none)", "\n\n".join(event_lines), aliases


def build_batch_prompt(closed_events: Iterable[dict], memory_store: MemoryStore) -> tuple[str, str]:
    """Compatibility helper used by diagnostics/tests; the runner uses aliases too."""
    existing, new_events, _ = build_batch_context(closed_events, memory_store)
    return existing, new_events


def make_batch_prompt(events: Iterable[dict], memory_store: MemoryStore, now: datetime | None = None) -> str:
    existing, new_events, _ = build_batch_context(events, memory_store, now)
    return PROMPT_TEMPLATE.format(
        today=(now or utc_now()).astimezone(TZ).strftime("%Y-%m-%d (%A) %H:%M"),
        patient_context=_patient_context(events),
        existing_block=existing,
        events_block=new_events,
    )


def _normalize_aliases(
    results: list[dict], aliases: dict[str, str], allowed_participants: set[str],
    event_participants: dict[str, list[str]] | None = None, patient_name: str | None = None,
) -> list[dict]:
    """Convert model-facing labels to durable IDs and flag invented labels."""
    normalized: list[dict] = []
    for raw in results:
        item = dict(raw)
        flagged = False
        source_events = []
        for alias in item.get("source_events", []):
            value = aliases.get(str(alias))
            if value and str(alias).startswith("E"):
                source_events.append(value)
            else:
                flagged = True
        if not source_events:
            continue
        item["source_events"] = list(dict.fromkeys(source_events))
        participant_lookup = {name.casefold(): name for name in allowed_participants}
        participants: list[str] = []
        for name in item.get("participants", []):
            canonical = participant_lookup.get(str(name).strip().casefold())
            if canonical:
                participants.append(canonical)
            else:
                flagged = True
        # Presence is observed from diarization, not left to an LLM choice.
        # This preserves labels such as ``unenrolled_9`` until caregiver
        # confirmation can resolve them to an enrolled person.
        observed = [
            person
            for event_id in item["source_events"]
            for person in (event_participants or {}).get(event_id, [])
        ]
        patient_key = _name_key(patient_name)
        item["participants"] = list(dict.fromkeys(
            person for person in [*observed, *participants]
            if _name_key(person) != patient_key
        ))
        target = item.get("target_memory_id")
        if item.get("action") == "merge":
            value = aliases.get(str(target))
            if value and str(target).startswith("M") and ".T" not in str(target):
                item["target_memory_id"] = value
            else:
                item["action"] = "new"
                item.pop("target_memory_id", None)
                flagged = True
        supersedes: dict[str, str | None] = {}
        raw_supersedes = item.get("supersedes_time_refs", {})
        if not isinstance(raw_supersedes, dict):
            flagged = True
            raw_supersedes = {}
        for alias, replacement_key in raw_supersedes.items():
            value = aliases.get(str(alias))
            if value and ".T" in str(alias) and (replacement_key is None or isinstance(replacement_key, str)):
                supersedes[value] = replacement_key
            else:
                flagged = True
        item["supersedes_time_refs"] = supersedes
        normalized_refs: list[dict] = []
        for ref in item.get("time_refs", []):
            if not isinstance(ref, dict):
                flagged = True
                continue
            normalized_ref = dict(ref)
            event_alias = normalized_ref.get("from_event")
            event_id = aliases.get(str(event_alias))
            if event_id and str(event_alias).startswith("E"):
                normalized_ref["from_event"] = event_id
            else:
                # Keep the proposal, but let code choose an event that actually
                # contains the phrase and mark this output for review.
                normalized_ref.pop("from_event", None)
                flagged = True
            normalized_refs.append(normalized_ref)
        item["time_refs"] = normalized_refs
        if flagged:
            item["needs_review"] = True
        normalized.append(item)
    return normalized

def parse_llm_json_response(response: str | list[dict]) -> list[dict]:
    """Accept raw JSON or the common Markdown-fenced JSON response form."""
    if isinstance(response, list):
        return response
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    return json.loads(text)


def _normalize_concern_entities(result: dict) -> None:
    """A person in CONCERNS must not also become a separate mentioned entity."""
    valid_types = {"PLACE", "ITEM", "EVENT", "PERSON", "RELATION"}
    entities = [
        {**entity, "type": str(entity.get("type", "")).upper(), "text": str(entity.get("text", "")).strip()}
        for entity in result.get("entities", []) if isinstance(entity, dict)
        and str(entity.get("type", "")).upper() in valid_types and str(entity.get("text", "")).strip()
    ]
    entity_types = {_name_key(entity["text"]): entity["type"] for entity in entities}
    concerns: list[dict] = []
    for entry in result.get("about", []):
        if not isinstance(entry, dict):
            concerns.append(entry)
            continue
        concern = dict(entry)
        inferred_type = entity_types.get(_name_key(concern.get("person")), "PERSON")
        concern["type"] = str(concern.get("type") or inferred_type).upper()
        concerns.append(concern)
    concern_keys = {
        (_name_key(entry.get("person")), str(entry.get("type", "PERSON")).upper())
        for entry in concerns if isinstance(entry, dict)
    }
    result["about"] = concerns
    result["entities"] = [
        entity for entity in entities
        if (_name_key(entity["text"]), entity["type"]) not in concern_keys
    ]


def _validate_results(results: Any, eligible_ids: set[str]) -> list[dict]:
    if not isinstance(results, list): raise ValueError("LLM response must be a JSON array")
    for result in results:
        if result.get("action") not in {"new", "merge"} or not result.get("source_events"): raise ValueError("Each result needs action and source_events")
        if result.get("category") not in MEMORY_CATEGORIES: raise ValueError("Each result needs a valid memory category")
        result.setdefault("about", [])
        if not isinstance(result["about"], list):
            raise ValueError("Each result needs an about list")
        result.setdefault("participants", [])
        if not isinstance(result["participants"], list) or not all(
            isinstance(person, str) and person.strip() for person in result["participants"]
        ):
            raise ValueError("Each result needs a participants list of names")
        result.setdefault("entities", [])
        _normalize_concern_entities(result)
        result.setdefault("time_refs", [])
        result.setdefault("supersedes_time_refs", {})
        if not isinstance(result["supersedes_time_refs"], dict):
            raise ValueError("supersedes_time_refs must map an existing TimeRef to a new ref_key")
        for entry in result["about"]:
            if (not isinstance(entry, dict) or not all(isinstance(entry.get(key), str) and entry[key].strip()
                    for key in ("person", "reported_by", "what_was_said"))
                    or entry.get("type") not in {"PERSON", "RELATION"}):
                raise ValueError("Each about entry needs person, type, reported_by, and what_was_said")
        for ref in result["time_refs"]:
            if not isinstance(ref, dict) or not isinstance(ref.get("phrase"), str) or not ref["phrase"].strip():
                raise ValueError("Each time reference needs a phrase")
            mentioned = ref.get("what_was_mentioned")
            if mentioned is not None and (not isinstance(mentioned, str) or not mentioned.strip()):
                raise ValueError("what_was_mentioned must be a non-empty string when supplied")
        if set(result["source_events"]) - eligible_ids: raise ValueError("LLM response references event outside this batch")
        if result["action"] == "merge" and not result.get("target_memory_id"): raise ValueError("Merge result needs target_memory_id")
        if result["action"] == "merge" and (
            not isinstance(result.get("what_was_updated"), str)
            or not result["what_was_updated"].strip()
        ):
            raise ValueError("Merge result needs a short what_was_updated description")
    return results

def _get_valid_merge_target(result: dict, memory_store: MemoryStore) -> dict:
    """Prevent a model-selected target from mixing unrelated memory categories."""
    existing = memory_store.get(result["target_memory_id"])
    if not existing: raise KeyError(f"Merge target does not exist: {result['target_memory_id']}")
    if result["category"] != existing["category"]:
        raise ValueError(
            f"Cannot merge {result['category']} memory into {existing['category']} memory; create a new memory instead"
        )
    return existing

def _merge_event_times(memory: dict, events: list[dict]) -> None:
    """Keep durable event ranges without needing old event rows after cleanup."""
    starts = set(memory.get("event_starts", [])) | {_iso(event["start"]) for event in events}
    ends = [memory.get("latest_event_time")] + [_iso(event["end"]) for event in events]
    memory["event_starts"] = sorted(starts)
    memory["earliest_event_time"] = min(starts)
    memory["latest_event_time"] = max(value for value in ends if value)


def _event_for_phrase(proposal: dict, source_events: list[dict]) -> dict:
    """Use the cited event only when it actually contains the time phrase."""
    phrase = str(proposal.get("phrase", "")).casefold().strip()
    by_id = {event["event_id"]: event for event in source_events}
    cited = by_id.get(proposal.get("from_event"))
    if cited and phrase and phrase in render_transcript_text(cited).casefold():
        return cited
    for event in source_events:
        if phrase and phrase in render_transcript_text(event).casefold():
            return event
    return source_events[-1]


def _same_time_matter(left: dict, right: dict) -> bool:
    """A time update is safe only for the same narrowly-defined matter."""
    return bool(_name_key(left.get("topic")) and _name_key(left.get("concerned_with"))) and (
        _name_key(left.get("topic")) == _name_key(right.get("topic"))
        and _name_key(left.get("concerned_with")) == _name_key(right.get("concerned_with"))
    )


def _same_commitment_maker(left: dict, right: dict) -> bool:
    old_maker, new_maker = _name_key(left.get("commitment_by")), _name_key(right.get("commitment_by"))
    if old_maker == new_maker:
        return True
    # A changed responsible person is safe only when the model explicitly
    # identifies the person whose responsibility is being taken over.
    return bool(old_maker and _name_key(right.get("takes_over_from")) == old_maker)


def _explicit_reschedule(result: dict) -> bool:
    """Permit a differently-worded topic only for an explicit plan change."""
    text = " ".join(str(result.get(field) or "") for field in (
        "what_was_updated", "summary",
    )).casefold()
    return bool(re.search(r"\b(reschedul|postpon|delay|mov(?:e|ed|ing)|chang(?:e|ed|ing)|replac(?:e|ed|ing)|take over)\w*", text))


def _plan_overlap(left: dict, right: dict, result: dict) -> bool:
    """Require concrete shared plan words before accepting a renamed TimeRef."""
    ignored = {
        "will", "would", "could", "should", "take", "next", "this", "that",
        "with", "from", "into", "then", "than", "plan", "plans", "visit",
        "rescheduled", "reschedule", "meeting", "house", "instead",
    }
    def words(value: object) -> set[str]:
        return {
            word for word in re.findall(r"[a-z0-9_]+", str(value or "").casefold())
            if len(word) > 2 and word not in ignored
        }
    old_words = words(left.get("what_was_mentioned")) | words(left.get("topic"))
    new_words = (
        words(right.get("what_was_mentioned")) | words(right.get("topic"))
        | words(result.get("summary")) | words(result.get("topic"))
    )
    return len(old_words & new_words) >= 2


def _valid_supersession(
    old: dict, replacement: dict, result: dict, memory: dict, statement_anchor: str,
) -> bool:
    if not (old.get("status") == "active" and old.get("anchor", "") < statement_anchor < (old.get("end") or "")):
        return False
    same_maker = _same_commitment_maker(old, replacement)
    if _same_time_matter(old, replacement) and same_maker:
        return True
    # The model explicitly chose this old ref. A reschedule can word its
    # TimeRef topic differently and may name a replacement driver. Keep it in
    # the same chain only when the memory's own narrow topic and concerned
    # person still match; unrelated plans therefore cannot merge by date.
    return _explicit_reschedule(result) and _plan_overlap(old, replacement, result)


def _time_ref_plan(memory: dict, result: dict, source_events: list[dict], now: str) -> tuple[list[dict], list[dict], bool]:
    """Validate TimeRefs and only permit same-matter replacements."""
    refs = list(memory.get("time_refs", []))
    by_id = {ref["time_ref_id"]: ref for ref in refs}
    statement_anchor = max(_iso(event["start"]) for event in source_events)
    review = bool(result.get("needs_review"))
    created: list[dict] = []
    created_by_key: dict[str, dict] = {}
    for index, proposal in enumerate(result.get("time_refs", [])[:8], 1):
        if not isinstance(proposal, dict) or not proposal.get("phrase"):
            review = True
            continue
        event = _event_for_phrase(proposal, source_events)
        tense = proposal.get("tense", "present")
        if tense not in {"past", "present", "future"}:
            tense = "present"
            review = True
        window = resolve_time(proposal["phrase"], event["start"], tense)
        if window and not direction_ok(window, event["start"], tense):
            window = None
            review = True
        proposal_key = str(proposal.get("ref_key") or f"T{index}").strip()
        if not proposal_key or proposal_key in created_by_key:
            review = True
            continue
        proposal_matter = {
            "topic": str(proposal.get("topic") or result.get("topic") or "").strip(),
            "concerned_with": str(proposal.get("concerned_with") or "").strip(),
            "commitment_by": str(proposal.get("commitment_by") or "").strip(),
            "takes_over_from": str(proposal.get("takes_over_from") or "").strip(),
        }
        same = next((ref for ref in refs if ref.get("status") == "active" and window
                     and ref.get("start") == window["start"] and ref.get("end") == window["end"]
                     and _same_time_matter(ref, proposal_matter)
                     and _same_commitment_maker(ref, proposal_matter)), None)
        if same:
            same["last_confirmed_at"] = now
            continue
        created_ref = {"time_ref_id": str(uuid.uuid4()), "phrase": proposal["phrase"], "tense": tense,
                        "anchor": _iso(event["start"]), "start": window["start"] if window else None,
                        "end": window["end"] if window else None, "part": window["part"] if window else None,
                        "granularity": window["granularity"] if window else None,
                        "status": "active" if window else "unresolved",
                        **proposal_matter,
                        "what_was_mentioned": str(proposal.get("what_was_mentioned") or result.get("summary", "")).strip(),
                        "created_at": now,
                        "superseded_at": None, "last_confirmed_at": None}
        created.append(created_ref)
        created_by_key[proposal_key] = created_ref

    old_refs: list[dict] = []
    for ref_id, replacement_key in result.get("supersedes_time_refs", {}).items():
        old, replacement = by_id.get(ref_id), created_by_key.get(str(replacement_key))
        if not old or old.get("status") != "active" or not replacement:
            review = True
            continue
        if not _valid_supersession(old, replacement, result, memory, statement_anchor):
            review = True
            continue
        # Preserve the canonical old topic along the same plan's time chain.
        # This avoids later wording changes splitting the plan again.
        replacement["topic"] = old.get("topic") or replacement.get("topic", "")
        old_refs.append(old)
        old["_replacement_id"] = replacement["time_ref_id"]
    return old_refs, created, review


def _apply_time_refs(memory: dict, result: dict, source_events: list[dict], now: str) -> tuple[list[dict], list[dict], bool]:
    """Code, rather than the LLM, validates dates and update history."""
    refs = list(memory.get("time_refs", []))
    old_refs, created, review = _time_ref_plan(memory, result, source_events, now)
    for old in old_refs:
        old["status"] = "superseded"
        old["superseded_at"] = now
        old["superseded_by"] = old.pop("_replacement_id")
    memory["time_refs"] = refs + created
    memory["needs_review"] = bool(memory.get("needs_review")) or review
    return old_refs, created, review


def apply_batch_response(results: list[dict], events_by_id: dict[str, dict], memory_store: MemoryStore, batch_processing_time: datetime | None = None) -> None:
    """Store source-event times and validated TimeRefs without trusting LLM dates."""
    processed_at = (batch_processing_time or utc_now()).isoformat()
    for result in results:
        # Also protect direct/internal callers that did not pass through the
        # normal LLM validation path.
        _normalize_concern_entities(result)
        source_events = [events_by_id[event_id] for event_id in result["source_events"]]
        patient_name = _patient_name(source_events)
        patient_key = _name_key(patient_name)
        # The wearer can be addressed in speech, but is deliberately never
        # stored as a participant or a PRESENT_AT graph edge.
        participants = list(dict.fromkeys(
            person for person in (
                [person for event in source_events for person in event.get("participants", [])]
                + result.get("participants", [])
            ) if _name_key(person) != patient_key
        ))
        if result["action"] == "new":
            patient_ids = {event.get("patient_id") for event in source_events if event.get("patient_id") is not None}
            about = [entry for entry in result.get("about", []) if entry["person"].casefold() not in {person.casefold() for person in participants}]
            memory = {"memory_id": str(uuid.uuid4()), "patient_id": next(iter(patient_ids)) if len(patient_ids) == 1 else None, "category": result["category"], "summary": result["summary"], "participants": participants, "about": about, "source_events": result["source_events"], "importance": result["importance"], "entities": result["entities"], "topic": result.get("topic"), "emotion": result.get("emotion", "neutral"), "created_at": processed_at, "updated_at": processed_at, "event_starts": [], "time_refs": [], "update_history": [{"at": processed_at, "action": "new", "source_events": result["source_events"]}], "updated_events": [], "needs_review": bool(result.get("needs_review"))}
            _merge_event_times(memory, source_events)
            _apply_time_refs(memory, result, source_events, processed_at)
            memory_store.create(memory)
            continue
        existing = _get_valid_merge_target(result, memory_store)
        valid_old, planned, review = _time_ref_plan(existing, result, source_events, processed_at)
        active = [ref for ref in existing.get("time_refs", []) if ref.get("status") == "active"]
        windows = [ref for ref in planned if ref.get("start") and ref.get("end")]
        overlaps = lambda left, right: left["start"] < right["end"] and left["end"] > right["start"]
        compatible_active = [
            old for old in active for window in windows
            if overlaps(window, old) and _same_time_matter(old, window)
            and _same_commitment_maker(old, window)
        ]
        # Dates alone never make two plans the same. A merge requires an
        # explicit valid supersession or a same-matter, same-maker overlap.
        if active and not valid_old and windows and not compatible_active:
            result = {**result, "action": "new", "needs_review": review}
            patient_ids = {event.get("patient_id") for event in source_events if event.get("patient_id") is not None}
            about = [entry for entry in result.get("about", []) if entry["person"].casefold() not in {person.casefold() for person in participants}]
            memory = {"memory_id": str(uuid.uuid4()), "patient_id": next(iter(patient_ids)) if len(patient_ids) == 1 else None, "category": result["category"], "summary": result["summary"], "participants": participants, "about": about, "source_events": result["source_events"], "importance": result["importance"], "entities": result["entities"], "topic": result.get("topic"), "emotion": result.get("emotion", "neutral"), "created_at": processed_at, "updated_at": processed_at, "event_starts": [], "time_refs": [], "update_history": [{"at": processed_at, "action": "new", "source_events": result["source_events"]}], "updated_events": [], "needs_review": bool(result.get("needs_review"))}
            _merge_event_times(memory, source_events)
            _apply_time_refs(memory, result, source_events, processed_at)
            memory_store.create(memory)
            continue
        existing["summary"], existing["emotion"] = result["summary"], result.get("emotion", existing.get("emotion", "neutral"))
        existing["participants"] = list(dict.fromkeys(
            person for person in existing.get("participants", []) + participants
            if _name_key(person) != patient_key
        ))
        participant_keys = {person.casefold() for person in existing["participants"]}
        existing["about"] = list({json.dumps(entry, sort_keys=True): entry for entry in existing.get("about", []) + result.get("about", []) if entry["person"].casefold() not in participant_keys}.values())
        existing["source_events"] = list(dict.fromkeys(existing["source_events"] + result["source_events"]))
        _merge_event_times(existing, source_events)
        _apply_time_refs(existing, result, source_events, processed_at)
        existing["updated_at"] = processed_at
        existing["update_history"] = [
            *existing.get("update_history", []),
            {
                "at": processed_at,
                "action": "merge",
                    # The LLM response is validated before normal operation.
                    # Keep direct/internal callers backward-compatible.
                    "what_was_updated": str(result.get("what_was_updated") or result["summary"]).strip(),
                "source_events": result["source_events"],
            },
        ]
        existing["updated_events"] = [
            *existing.get("updated_events", []),
            {
                # The actual conversation time comes directly from the newest
                # source event. The LLM never supplies or calculates it.
                "event_time": max(_iso(event["end"]) for event in source_events),
                "what_was_updated": str(result.get("what_was_updated") or result["summary"]).strip(),
            },
        ]
        memory_store.update(existing)

class MemoryConsolidator:
    """Provider-agnostic runner. Pass a callable that accepts a prompt and returns JSON."""
    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store
        init_db()
        with get_connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS memory_consolidation_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS consolidated_events (event_id TEXT PRIMARY KEY)")
            conn.execute("""CREATE TABLE IF NOT EXISTS memory_consolidation_audit (
                batch_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, event_ids TEXT NOT NULL,
                request_prompt TEXT NOT NULL, response_text TEXT, status TEXT NOT NULL,
                error TEXT)""")
            conn.execute("INSERT OR IGNORE INTO memory_consolidation_meta VALUES ('last_batch_time', ?)", (utc_now().isoformat(),))
            conn.execute("INSERT OR IGNORE INTO memory_consolidation_meta VALUES ('automatic_batching_paused', '0')")
            # Migrate batches processed before llm_status was introduced.
            conn.execute("UPDATE events SET llm_status='processed' WHERE event_id IN (SELECT event_id FROM consolidated_events)")
    def pending_closed_events(self) -> list[dict]:
        return [event for event in list_events("closed") if event.get("llm_status") == "pending"]
    def _last_batch_time(self):
        with get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM memory_consolidation_meta WHERE key='last_batch_time'"
            ).fetchone()
            if row is None:
                value = utc_now().isoformat()
                conn.execute(
                    "INSERT INTO memory_consolidation_meta (key, value) VALUES ('last_batch_time', ?)",
                    (value,),
                )
            else:
                value = row["value"]
        return _dt(value) if value else None
    def automatic_batching_paused(self) -> bool:
        with get_connection() as conn:
            row = conn.execute("SELECT value FROM memory_consolidation_meta WHERE key='automatic_batching_paused'").fetchone()
        return row is not None and row["value"] == "1"
    def set_automatic_batching_paused(self, paused: bool) -> None:
        """Persist automatic-batch state without changing events or memories.

        On resume, start a fresh five-minute window so queued events are not sent
        immediately after a pause.
        """
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO memory_consolidation_meta (key, value) VALUES ('automatic_batching_paused', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                ('1' if paused else '0',),
            )
            if not paused:
                conn.execute(
                    "INSERT INTO memory_consolidation_meta (key, value) VALUES ('last_batch_time', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (utc_now().isoformat(),),
                )
    def run_if_due(self, llm_call: Callable[[str], str | list[dict]], now: datetime | None = None):
        if self.automatic_batching_paused():
            return None
        pending = self.pending_closed_events()
        return self.run(llm_call, pending, now) if should_run_batch(self._last_batch_time(), len(pending), now) else None
    def run(self, llm_call: Callable[[str], str | list[dict]], events: list[dict] | None = None, now: datetime | None = None) -> list[dict]:
        events = events if events is not None else self.pending_closed_events()
        if any(e["status"] != "closed" for e in events): raise ValueError("Only closed events may be consolidated")
        if not events: return []
        applied_at = now or utc_now()
        existing, new_events, aliases = build_batch_context(events, self.memory_store, applied_at)
        prompt = PROMPT_TEMPLATE.format(
            today=applied_at.astimezone(TZ).strftime("%Y-%m-%d (%A) %H:%M"),
            patient_context=_patient_context(events),
            existing_block=existing,
            events_block=new_events,
        )
        response = llm_call(prompt)
        response_text = response if isinstance(response, str) else json.dumps(response)
        try:
            allowed_participants = {
                person for event in events for person in event.get("participants", [])
            }
            patient_name = _patient_name(events)
            if patient_name:
                allowed_participants.add(patient_name)
            results = _normalize_aliases(
                parse_llm_json_response(response), aliases, allowed_participants,
                {event["event_id"]: event.get("participants", []) for event in events},
                patient_name,
            )
            results = _validate_results(results, {e["event_id"] for e in events})
            for result in results:
                if result["action"] == "merge":
                    _get_valid_merge_target(result, self.memory_store)
        except Exception as error:
            self._record_audit(prompt, response_text, events, applied_at, "failed", str(error))
            raise
        # Old source events may be purged after consolidation.  Merge logic uses
        # the durable memory's own event ranges and TimeRefs, never reloads them.
        apply_batch_response(results, {e["event_id"]: e for e in events}, self.memory_store, applied_at)
        with get_connection() as conn:
            conn.executemany("INSERT OR IGNORE INTO consolidated_events VALUES (?)", ((e["event_id"],) for e in events))
            conn.executemany("UPDATE events SET llm_status='processed', consumed=1 WHERE event_id=?", ((e["event_id"],) for e in events))
            conn.execute("UPDATE memory_consolidation_meta SET value=? WHERE key='last_batch_time'", (applied_at.isoformat(),))
            conn.execute("UPDATE grouping_meta SET value='0' WHERE key='unconsumed_event_count'")
        self._record_audit(prompt, response_text, events, applied_at, "success")
        return results

    @staticmethod
    def _record_audit(prompt: str, response_text: str, events: list[dict], created_at: datetime, status: str, error: str | None = None) -> None:
        with get_connection() as conn:
            conn.execute("INSERT INTO memory_consolidation_audit VALUES (?, ?, ?, ?, ?, ?, ?)", (str(uuid.uuid4()), created_at.isoformat(), json.dumps([event["event_id"] for event in events]), prompt, response_text, status, error))
