# orchestrator.py
from dotenv import load_dotenv
load_dotenv()

import os, re, json
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, TypedDict, Mapping

# -------------------- MODELS & PROMPTS --------------------
from prompts_and_llm import memory_db_embedder, small_generative_model, fallback_model  # :contentReference[oaicite:7]{index=7}
small_model = small_generative_model()
reasoning_model = fallback_model()
memory_embedder = memory_db_embedder()

# -------------------- RETRIEVER (portable) ----------------
from retriever import make_retriever
retriever_db = make_retriever()  # replaces eager code in your top block

# -------------------- DB HELPERS --------------------------
from db import (query_user_details, upsert_user_details, update_user_details, load_summary, save_summary,
                select_relevant_entities, search_semantic_notes, recall_episodes, upsert_entities,
                upsert_semantic_notes, upsert_episode, recent_episodes, ensure_full_memory_schema
) 

# -------------------- LLM UTILS --------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

def trim_to_token_budget(text: str, max_tokens: int) -> str:
    words = text.split()
    return text if len(words) <= max_tokens else " ".join(words[:max_tokens])

# -------------------- MEMORY: SUMMARY --------------------
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the dialog for future astrology guidance. "
               "Max 120 words. Capture problem, predictions with dates (dd-mm-yyyy), remedies, preferences, outcomes. Be factual."),
    ("human", "Previous summary:\n{prev}\n\nNew messages:\n{window}\n\nReturn updated summary only.")
])
def llm_update_summary(prev: str, messages_window: List[Any], model) -> str:
    window_text = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages_window])
    chain = summary_prompt | model | StrOutputParser()
    return chain.invoke({"prev": prev or "", "window": window_text})

# -------------------- MEMORY: ENTITIES -------------------
TOPIC_KEYWORDS = ["marriage", "career", "finance", "health", "education", "travel", "family",
                  "property", "legal/disputes", "spirituality", "vehicles", "love/relationships",
                  "separation/divorce", "childbirth/fertility", "visa/PR", "business", "inheritance/ancestral",
                  "loans/debts", "court/police", "enemies/disputes", "mental_health", "surgery/accidents",
                  "property/vastu/renovation", "investment", "pilgrimage", "spiritual_protection",
                  "gemstones_remedies", "friendship/social", "inlaws_family", "education_abroad",
                  "taxation/finance", "government_career", "arts_sports_career", "parents_health",
                  "second_marriage", "relocation/move", "trading/speculation", "documents/paperwork",
                  "timing/muhurta", "pets/animals"]

entity_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Extract compact, factual entities from the text. Choose ONE topic from: {topic_list} that is closed to the text. "
     "Return JSON array with objects (type,key,value,topic). "
     "Allowed types: person, preference, constraint, remedy, window, goal, event, location, occupation, outcome, contact, other. "
     "Dates dd-mm-yyyy. For time windows, emit separate rows start_date/end_date with type='window'. Max 5 rows. Output JSON only."),
    ("human", "{text}")
])

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        l = s.find("["); r = s.rfind("]")
        if l != -1 and r != -1 and r > l:
            try: return json.loads(s[l:r+1])
            except Exception: pass
    return []

def extract_entities(question: str, answer: str, model, topic: Optional[str] = None) -> List[dict]:
    text_in = (question or "") + "\n\n" + (answer or "")
    chain = entity_prompt | model | StrOutputParser()
    out = chain.invoke({"text": text_in, "topic_list": ", ".join(TOPIC_KEYWORDS)})
    data = _safe_json_loads(out)
    cleaned: List[dict] = []
    if isinstance(data, list):
        for e in data:
            t = (e.get("type") or "").strip().lower()
            k = (e.get("key") or "").strip()
            v = (e.get("value") or "").strip()
            top = (e.get("topic") or (topic or "")).strip()
            if t and (k or v):
                cleaned.append({"type": t, "key": k, "value": v, "topic": top})
    return cleaned

# -------------------- MEMORY: RENDER ---------------------
MAX_MEMORY_TOKENS = 300
def render_memory(user_id: str, question: str, topic: Optional[str] = None) -> str:
    summary = load_summary(user_id) or ""
    ents    = select_relevant_entities(user_id, topic=topic, limit=4)
    notes   = search_semantic_notes(memory_embedder, user_id, question, k=4)
    eps     = recall_episodes(memory_embedder, user_id, question, k=3, dist_cutoff=0.40)

    parts = []
    if topic: parts.append(f"Current focus: {topic}.")
    if eps:
        lines = []
        for e in eps:
            dt = e["asked_at"].strftime("%d-%m-%Y")
            lines.append(f"[{dt}] Q: {e['question']} | Advice: {e['digest']}")
        parts.append("Episode recall: " + " || ".join(lines))
    if summary: parts.append(f"Summary: {summary}")
    if ents:    parts.append("Facts: " + "; ".join(ents))
    if notes:   parts.append("Notes: " + " | ".join(notes))
    return trim_to_token_budget(" ".join(parts), MAX_MEMORY_TOKENS)

# -------------------- POST-TURN UPDATE -------------------
notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "Write 1–3 single-sentence factual notes for future astrology advice. Each <= 30 words. No duplication/opinions. Return each on a new line."),
    ("human", "{text}")
])
def extract_salient_notes(question: str, answer: str, model) -> List[str]:
    chain = notes_prompt | model | StrOutputParser()
    text_out = chain.invoke({"text": question + "\n\n" + answer})
    notes = [s.strip("-• ").strip() for s in text_out.split("\n") if s.strip()]
    return [(" ".join(n.split()[:40])) for n in notes][:3]

def post_answer_update(user_id: str, question: str, answer: str, topic: Optional[str]):
    prev = load_summary(user_id)
    new_summary = llm_update_summary(prev, [HumanMessage(content=question), AIMessage(content=answer)], small_model)
    save_summary(user_id, new_summary)
    ents  = extract_entities(question, answer, small_model, topic=topic)
    upsert_entities(user_id, ents)
    notes = extract_salient_notes(question, answer, small_model)
    upsert_semantic_notes(memory_embedder, user_id, notes)
    upsert_episode(memory_embedder, user_id, question, answer, topic=topic, notes=notes)

# -------------------- TOOLS ------------------------------
from astrologer import input_for_LLM  # :contentReference[oaicite:11]{index=11}

class MemorySearchArgs(BaseModel):
    query: Optional[str] = Field(None)
    k: int = Field(3, ge=1, le=10)
    mode: str = Field("auto", description="auto|recent|semantic")

def make_memory_search_tool(user_id: str):
    @tool("memory_search", args_schema=MemorySearchArgs)
    def memory_search(query: Optional[str] = None, k: int = 3, mode: str = "auto") -> str:
        """Search prior conversation turns by recency or semantic similarity.

        Args:
            query: Free-text to search; if None and mode='auto', returns recent episodes.
            k: Number of episodes to return (1–10).
            mode: 'auto' | 'recent' | 'semantic'.

        Returns:
            JSON string: {"episodes":[{"asked_at":"dd-mm-yyyy","question":"...","digest":"..."}]}
        """
        if mode == "recent" or (mode == "auto" and not (query and query.strip())):
            eps = recent_episodes(user_id, n=k)
        else:
            eps = recall_episodes(memory_embedder, user_id, query or "", k=k, dist_cutoff=0.50)
            if not eps:
                eps = recent_episodes(user_id, n=k)
        out = []
        for e in eps:
            out.append({
                "asked_at": e["asked_at"].strftime("%d-%m-%Y"),
                "question": e["question"],
                "digest": e.get("digest") or e.get("answer_digest", "")
            })
        return json.dumps({"episodes": out})
    return memory_search

class AstroSearchArgs(BaseModel):
    birth_date_iso: Optional[str] = None
    birth_time_str: Optional[str] = None
    birth_place: Optional[str] = None
    current_place: Optional[str] = None
    include_transit_events: bool = True
    transit_span_years: int = 3
    use_passed_details: bool = False  # for 'other' subject

def astro_query_tool(user_id: str):
    @tool("astro_search", args_schema=AstroSearchArgs)
    def astro_search(
        birth_date_iso: Optional[str] = None,
        birth_time_str: Optional[str] = None,
        birth_place: Optional[str] = None,
        current_place: Optional[str] = None,
        include_transit_events: bool = True,
        transit_span_years: int = 3,
        use_passed_details: bool = False,
    ) -> str:
        """Create or fetch astrology charts for a user and return a JSON payload.

        Behavior:
          - If use_passed_details=True and all four fields are provided, build charts directly (no DB write).
          - Else, read user_details from DB. If missing any field, return {"status":"request_details", ...}.
          - If details exist (or provided), build charts and return {"status":"charts_created", ...}.

        Args:
            birth_date_iso: 'YYYY-MM-DD'
            birth_time_str: 'HH:MM' 24h
            birth_place: 'City, State[, Country]'
            current_place: 'City, State[, Country]'
            include_transit_events: Include transit calculations
            transit_span_years: Years of transit window
            use_passed_details: Treat details as external subject; skip DB

        Returns:
            JSON string with 'status' plus 'charts' on success, or an error/request message.
        """
        def _json(obj: Dict[str, Any]) -> str:
            return json.dumps(obj, default=str)

        def _make_charts(bd: str, bt: str, bp: str, cp: str) -> str:
            charts = input_for_LLM(bd, bt, bp, cp, include_transit_events=include_transit_events, transit_span_years=transit_span_years)
            if not charts:
                return _json({"status": "error", "message": "Chart engine failed with provided details."})  # CHANGED: guard None
            return _json({
                "status": "charts_created",
                "used_details": {
                    "birth_date_iso": bd, "birth_time_str": bt, "birth_place": bp, "current_place": cp
                },
                "username": user_id,
                "charts": charts
            })

        # Shortcut for 'other'
        if use_passed_details and all([birth_date_iso, birth_time_str, birth_place, current_place]):
            return _make_charts(birth_date_iso, birth_time_str, birth_place, current_place)

        details = query_user_details(user_id)

        # No DB row yet → require all 4 fields
        if not details:
            missing = []
            if not birth_date_iso:  missing.append("birth_date_iso")
            if not birth_time_str:  missing.append("birth_time_str")
            if not birth_place:     missing.append("birth_place")
            if not current_place:   missing.append("current_place")
            if missing:
                return _json({
                    "status": "request_details",
                    "missing": missing,
                    "message": (
                        "Namaste 🙏. I don’t have your birth details yet. "
                        "Could you kindly share them in the following format:\n"
                        "- Birth date (YYYY-MM-DD)\n"
                        "- Birth time (HH:MM, 24-hour)\n"
                        "- Birth place ('City, State/Country')\n"
                        "- Current place ('City, State/Country')"
                    )
                })
            # Persist and create
            upsert_user_details(user_id, birth_date_iso, birth_time_str, birth_place, current_place)
            return _make_charts(birth_date_iso, birth_time_str, birth_place, current_place)

        # DB exists → allow overrides, and persist updated current_place if changed
        bd_iso = details["birth_date"].isoformat()
        bt_str = details["birth_time"].strftime("%H:%M")
        bp     = details["birth_place"]
        cp     = details["current_place"]

        if birth_date_iso: bd_iso = birth_date_iso
        if birth_time_str: bt_str = birth_time_str
        if birth_place:    bp     = birth_place
        if current_place and current_place != cp:
            cp = current_place
            # CHANGED: persist location update for subject=self
            try: update_user_details(user_id, current_place=cp)
            except Exception: pass

        return _make_charts(bd_iso, bt_str, bp, cp)

    return astro_search

# -------------------- CHART SELECTOR ---------------------
JOB_KEYS = {
    "ascendant_longitude", "rasi_chart", "navamsa_chart", "dasamsa_chart",
    "transit_chart", "transit_positions", "mahadasha", "antardasha",
    "natal_strengths", "transit_events", "influence_now",
    "current_md", "current_ad", "birth_datetime", "today_datetime"
}
def pick_relevant_chart_data(chart: Mapping[str, Any] | None) -> Dict[str, Any]:
    return {} if not chart else {k: chart[k] for k in JOB_KEYS if k in chart}

# -------------------- DETAIL PARSER/ASKERS ----------------
details_extract_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract birth details from the user's message and identify the subject.\n"
     "Return ONLY a single JSON object with EXACTLY these keys:\n"
     "  subject, relation, birth_date_iso, birth_time_str, birth_place, current_place\n"
     "Rules:\n"
     "- subject is 'self' or 'other'. If 'other', set relation to: daughter, son, wife, husband, friend, mother, father, relative, family member, or unknown.\n"
     "- For any field that is missing, set its value to null.\n"
     "- Dates must be 'YYYY-MM-DD'. Times must be 'HH:MM' 24h. Places are free text like 'City, State[, Country]'.\n"
     "- Do not add extra keys. Do not include explanations. Output JSON ONLY."
    ),

    # Example 1
    ("human", "birth date: 1992-10-15, birth time: 20:30, birth place: Kodinar, gujarat, current place: colchester, united kingdom"),
    ("ai", '{{"subject":"self","relation":"unknown","birth_date_iso":"1992-10-15","birth_time_str":"20:30","birth_place":"Kodinar, gujarat","current_place":"colchester, united kingdom"}}'),

    # Example 2
    ("human", "for my daughter: 2004-01-05 at 06:25, born in Pune, Maharashtra, now in Toronto, Canada"),
    ("ai", '{{"subject":"other","relation":"daughter","birth_date_iso":"2004-01-05","birth_time_str":"06:25","birth_place":"Pune, Maharashtra","current_place":"Toronto, Canada"}}'),

    # Example 3 (missing current place)
    ("human", "DOB 1987-03-09, time 04:10, birthplace: Jaipur, Rajasthan"),
    ("ai", '{{"subject":"self","relation":"unknown","birth_date_iso":"1987-03-09","birth_time_str":"04:10","birth_place":"Jaipur, Rajasthan","current_place":null}}'),

    # Example 4 (missing time)
    ("human", "hi, i was born 1990-12-01 in Indore, MP. Current city Bengaluru, India. (don’t remember the time)."),
    ("ai", '{{"subject":"self","relation":"unknown","birth_date_iso":"1990-12-01","birth_time_str":null,"birth_place":"Indore, MP","current_place":"Bengaluru, India"}}'),

    # Example 5 (other, unknown relation)
    ("human", "please check for my relative — dob 1975-07-22; time 13:05; place: Kochi, Kerala; currently: Dubai, UAE"),
    ("ai", '{{"subject":"other","relation":"relative","birth_date_iso":"1975-07-22","birth_time_str":"13:05","birth_place":"Kochi, Kerala","current_place":"Dubai, UAE"}}'),

    # Actual input
    ("human", "{text}")
])

import json, re

# Reuse (or keep) the regex helpers — they boost accuracy
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TIME_RE = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)\b")
_PLACE_RE = re.compile(
    r"\b(birth\s*place|birthplace)\s*[:=\-]\s*([^\n,]+(?:,\s*[^\n,]+){0,2})\b|"
    r"\b(current\s*place|current\s*location)\s*[:=\-]\s*([^\n,]+(?:,\s*[^\n,]+){0,2})\b",
    re.IGNORECASE
)

def _clean_place(value: str) -> str:
    # remove any accidental “, current place: ...” or “, current location: ...”
    return re.sub(r",\s*current\s+(place|location)\b.*$", "", value, flags=re.IGNORECASE).strip()

def _extract_with_regex(text: str) -> dict:
    out = {
        "subject": None, "relation": None,
        "birth_date_iso": None, "birth_time_str": None,
        "birth_place": None, "current_place": None,
    }
    m = _DATE_RE.search(text);  m and (out.__setitem__("birth_date_iso", f"{m.group(1)}-{m.group(2)}-{m.group(3)}"))
    m = _TIME_RE.search(text);  m and (out.__setitem__("birth_time_str", f"{m.group(1)}:{m.group(2)}"))
    for m in _PLACE_RE.finditer(text):
        birth_label, birth_val, current_label, current_val = m.groups()
        if birth_val and not out["birth_place"]:     out["birth_place"] = _clean_place(birth_val.strip())
        if current_val and not out["current_place"]: out["current_place"] = current_val.strip()
    # subject heuristic
    subj = "other" if re.search(r"\b(my|for my)\s+(son|daughter|wife|husband|friend|mother|father|relative|family member)\b", text, re.I) else "self"
    out["subject"] = subj
    if subj == "other":
        m = re.search(r"\b(my|for my)\s+(son|daughter|wife|husband|friend|mother|father|relative|family member)\b", text, re.I)
        out["relation"] = (m.group(2).lower() if m else "unknown")
    return out

def _coerce_schema(d: dict) -> dict:
    """Ensure exact keys + normalization; drop everything else."""
    allowed = {"subject","relation","birth_date_iso","birth_time_str","birth_place","current_place"}
    out = {k: d.get(k) if k in d else None for k in allowed}
    # normalize subject/relation
    out["subject"] = out["subject"] if out["subject"] in {"self","other"} else "self"
    out["relation"] = (out.get("relation") or "unknown")
    # trim whitespace
    for k in ("birth_date_iso","birth_time_str","birth_place","current_place","relation"):
        if isinstance(out.get(k), str):
            out[k] = out[k].strip()
            if out[k] == "": out[k] = None
    return out

def parse_details_from_text(text: str) -> dict:
    """
    Few-shot + deterministic:
    1) regex pass (fast, robust)
    2) few-shot LLM fill for anything missing
    3) strict schema coerce
    """
    # 1) regex pass
    rx = _extract_with_regex(text)

    # 2) few-shot LLM (temperature 0) to fill gaps
    need_llm = any(rx[k] is None for k in ["birth_date_iso","birth_time_str","birth_place","current_place"])
    if need_llm:
        llm_zero = small_model.bind(temperature=0)  # force determinism here
        raw = (details_extract_prompt | llm_zero | StrOutputParser()).invoke({"text": text})
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        # merge: prefer regex where present; use LLM for gaps
        merged = {
            "subject": rx["subject"] or (data.get("subject") if data.get("subject") in {"self","other"} else "self"),
            "relation": rx["relation"] or (data.get("relation") or "unknown"),
            "birth_date_iso": rx["birth_date_iso"] or data.get("birth_date_iso"),
            "birth_time_str": rx["birth_time_str"] or data.get("birth_time_str"),
            "birth_place": rx["birth_place"] or data.get("birth_place"),
            "current_place": rx["current_place"] or data.get("current_place"),
        }
    else:
        merged = rx

    # 3) strict schema
    return _coerce_schema(merged)

ask_missing_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ask politely for all missing details in one concise sentence. No bullets. "
     "Start with 'Namaste' and be humble. If subject='other', include the relation word. "
     "Specify formats clearly."),
    ("human",
     "Subject: {subject}; Relation: {relation}; Missing: {missing}\n"
     "Formats: birth_date_iso=YYYY-MM-DD, birth_time_str=HH:MM (24h), "
     "birth_place='City, State/Country', current_place='City, State/Country'.\n"
     "Write the question:")
])

def render_missing_question(subject: str, relation: str, missing: list[str]) -> str:
    return (ask_missing_prompt | small_model | StrOutputParser()).invoke({
        "subject": subject, "relation": relation, "missing": ", ".join(missing)
    })

# -------------------- ANALYSIS PROMPT --------------------
analyze_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise and humble Vedic astrology assistant.\n\n"
     "You have access to three sources of context:\n"
     "1) USER ASTRO CHARTS (chart_data): natal + divisional + dasha + transits .\n"
     "2) RETRIEVER KNOWLEDGE BASE (astro_data): authoritative explanations of planets, houses, dashas, remedies, poojas, gemstones, yantras, etc.\n"
     "3) USER CONTEXT MEMORY (memory): previous conversations and summaries.\n\n"

     "### Your answering process:\n"
     "Step 1. Analyse the user's chart_data directly in relation to the question. "
     "Always mention relevant chart factors explicitly (ascendant, Moon sign, planets, house lords, dashas, antardashas, divisional charts, transits). "
     "Explain why the user is experiencing the issue or asking the question.\n"
     "Step 2. If you do not know how to interpret a chart feature, consult astro_data (retriever results) and use it to guide your analysis. "
     "Always tie explanations back to the user's actual chart placements.\n"
     "Step 3. If the user asks for remedies, pooja, gemstones, or yantras, consult astro_data to propose appropriate remedies. "
     "Align remedies to the specific planets and houses in the chart_data causing the issue. "
     "Be concrete and detailed: include mantras, practices, temple worship, pooja/havan options, gemstone details (stone, finger, metal, day, mantra), and yantra worship. "
     "Explain briefly why the remedy helps (e.g., 'calms Rahu's instability').\n"
     "Step 4. Structure the answer in 2–4 paragraphs. First paragraph(s): chart-based analysis tied to the question. "
     "Second paragraph(s): answer user's question based on chart analysis. "
     "Last paragraph(s): if requested, give guidance, remedies, or outlook, otherwise skip.\n\n"

     "### Formatting rules:\n"
     "- Write in natural flowing paragraphs (no bullet points, no headings).\n"
     "- Use dates in dd-mm-yyyy format when discussing timing.\n"
     "- Mention explicitly which chart factors were considered.\n"
     "- When offering remedies, phrase them in a devotional yet practical tone.\n\n"

     "If the user's query is vague (e.g., 'what to do now?'), you may consult memory_search tool to recall past conversations, "
     "then answer in context, tying chart_data with memory recall."
    ),

    ("human",
     "USER CONTEXT MEMORY:\n{memory}\n\n"
     "USER ASTRO CHARTS (chart_data):\n{chart_data}\n\n"
     "ADDITIONAL ASTRO LOGIC (astro_data):\n{astro_data}\n\n"
     "USER QUESTION:\n{question}")
])

# -------------------- GRAPH-ish ORCHESTRATION -------------
# (Kept simple for Streamlit or any sync call style)

DATE_RE = re.compile(r"\b\d{2}-\d{2}-\d{4}\b")

def _missing_fields_from_db_or_msg(user_id: str, parsed: dict) -> tuple[list[str], dict, str, str, bool]:
    subject = parsed["subject"]
    relation = parsed["relation"]
    is_other = subject == "other"
    if is_other:
        merged = {
            "birth_date_iso": parsed["birth_date_iso"],
            "birth_time_str": parsed["birth_time_str"],
            "birth_place": parsed["birth_place"],
            "current_place": parsed["current_place"],
        }
    else:
        db = query_user_details(user_id)
        if db:
            merged = {
                "birth_date_iso": db["birth_date"].isoformat(),
                "birth_time_str": db["birth_time"].strftime("%H:%M"),
                "birth_place": db["birth_place"],
                "current_place": db["current_place"],
            }
            for k in list(merged.keys()):
                if parsed.get(k): merged[k] = parsed[k]
        else:
            merged = {
                "birth_date_iso": parsed["birth_date_iso"],
                "birth_time_str": parsed["birth_time_str"],
                "birth_place": parsed["birth_place"],
                "current_place": parsed["current_place"],
            }
    needed = ["birth_date_iso","birth_time_str","birth_place","current_place"]
    missing = [k for k in needed if not merged.get(k)]
    return missing, merged, subject, relation, is_other

def _join_docs_trim(docs, max_tokens=1200):
    parts = []
    for d in docs or []:
        parts.append(getattr(d, "page_content", str(d)).replace("\n"," "))
    return trim_to_token_budget("\n\n".join(parts), max_tokens)

def _build_astro_context(q: str) -> str:
    docs = retriever_db.get_relevant_documents(q)
    return _join_docs_trim(docs, max_tokens=1200)

def _render_memory_block(user_id: str, q: str) -> str:
    return render_memory(user_id, q, topic=None)

def _should_force_memory_search(user_id: str, q: str, ans: str) -> bool:
    q_l = q.lower()
    asked_timing = any(w in q_l for w in ["when", "window", "period", "timeline", "date"])
    has_dates = bool(DATE_RE.search(ans or ""))
    vague = any(p in q_l for p in ["what to do", "what should i do", "this", "that", "these", "those", "above", "earlier", "previous", "last time", "now"])
    eps = recall_episodes(memory_embedder, user_id, q, k=3, dist_cutoff=0.40)
    no_recall = (len(eps) == 0)
    return (asked_timing and not has_dates) or (vague and no_recall)

def _batch_upload_conversation(user_id: str, items: List[Tuple[str,str]]):
    """
    CHANGED: simple "batch" uploader — chunks to episodes table.
    items: list of (question, answer_digest/full).
    """
    B = 10
    for i in range(0, len(items), B):
        chunk = items[i:i+B]
        for q, a in chunk:
            upsert_episode(memory_embedder, user_id, q, a, topic=None, notes=[])

# -------------------- Public API --------------------------
def on_app_load(user_id: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Step 1–3: called on login/reload.
    - Checks user_details
    - If complete → builds charts and returns greeting
    - If missing → returns None charts and a one-line ask for missing fields
    """
    ensure_full_memory_schema()  # CHANGED: ensure tables exist

    details = query_user_details(user_id)
    if not details:
        missing_all = ["birth_date_iso","birth_time_str","birth_place","current_place"]
        ask = render_missing_question("self", "unknown", missing_all)
        return None, f"{ask}"

    # Coerce DB record
    bd = details["birth_date"].isoformat()
    bt = details["birth_time"].strftime("%H:%M")
    bp = details["birth_place"]
    cp = details["current_place"]

    charts = input_for_LLM(bd, bt, bp, cp, include_transit_events=True, transit_span_years=3)  # :contentReference[oaicite:12]{index=12}
    if not charts:
        return None, "I couldn’t build your charts due to a technical error. Please verify your birth details."
    chart_bits = pick_relevant_chart_data(charts)
    return chart_bits, "Great! Your charts are ready. How may I help you today?"

def _looks_like_details_only(text: str) -> bool:
    t = text.lower()
    has_dt = bool(_DATE_RE.search(t)) or bool(_TIME_RE.search(t))
    mentions_fields = any(w in t for w in [
        "birth date", "birthtime", "birth time", "birth place", "birthplace",
        "current place", "current location"
    ])
    is_question = "?" in t or any(w in t for w in ["should", "can", "when", "why", "what", "how", "advise", "suggest"])
    return (has_dt or mentions_fields) and not is_question

def ask(user_id: str, question: str) -> str:
    """
    Step 4–9: main single-turn entry.
    - Parses/collects details, creates charts if needed (and persists for self)
    - Answers with tools; forces memory_search for vague queries
    - Saves memory; uploads conversation batch-wise
    """
    ensure_full_memory_schema()

    # 0) Did the user already have details before this message?
    had_details_before = bool(query_user_details(user_id))

    # 1) Build astro_data + memory
    astro_data = _build_astro_context(question)
    memory = _render_memory_block(user_id, question)
    chart_data: Dict[str, Any] = {}

    # 2) Ensure charts or ask for missing
    parsed = parse_details_from_text(question)
    missing, merged, subject, relation, is_other = _missing_fields_from_db_or_msg(user_id, parsed)
    if missing:
        return render_missing_question(subject, relation, missing)

    # 3) Call astro tool
    astro_tool = astro_query_tool(user_id)
    args = {
        "birth_date_iso": merged["birth_date_iso"],
        "birth_time_str": merged["birth_time_str"],
        "birth_place": merged["birth_place"],
        "current_place": merged["current_place"],
        "include_transit_events": True,
        "transit_span_years": 3,
        "use_passed_details": bool(is_other)
    }
    tool_result = astro_tool.invoke(args)
    try:
        payload = json.loads(tool_result)
    except Exception:
        payload = {}

    if payload.get("status") == "request_details":
        return payload.get("message") or "Namaste 🙏. I still need some details in the requested format."

    if payload.get("status") == "charts_created":
        charts = payload.get("charts") or {}
        chart_data = pick_relevant_chart_data(charts)

        # --- NEW: greet and end turn if we just created/updated charts or message looks like details-only
        if (not had_details_before) or _looks_like_details_only(question):
            return "Thank you 🙏. Your birth chart has been prepared. How may I help you today?"

    # 4) Answer with tools (memory_search available)
    mem_tool  = make_memory_search_tool(user_id)
    llm = small_model.bind_tools([mem_tool, astro_query_tool(user_id)])
    msgs = analyze_prompt.format_messages(
        memory=memory, chart_data=chart_data, astro_data=astro_data, question=question
    )
    ai = llm.invoke(msgs)
    if getattr(ai, "tool_calls", None):
        tool_msgs = []
        for tc in ai.tool_calls:
            if tc["name"] == "memory_search":
                result = mem_tool.invoke(tc["args"] or {})
                tool_msgs.append(ToolMessage(content=result, name="memory_search", tool_call_id=tc["id"]))
        final = llm.invoke(msgs + [ai] + tool_msgs)
        answer = final.content
    else:
        answer = ai.content

    # 5) Deterministic vagueness → force memory search
    if _should_force_memory_search(user_id, question, answer):
        forced = mem_tool.invoke({"query": None, "k": 3, "mode": "recent"})
        episodes = json.loads(forced).get("episodes", [])
        if episodes:
            recent_line = f"Recalling your recent topic: {episodes[0]['question']} — {episodes[0]['digest']}"
            answer = f"{recent_line}\n\n{answer}"

    # 6) Fallback reasoning if timing requested but no explicit dates
    asked_timing = any(w in question.lower() for w in ["when","window","period","timeline","date"])
    if asked_timing and not DATE_RE.search(answer or ""):
        better = (analyze_prompt | reasoning_model | StrOutputParser()).invoke({
            "astro_data": astro_data, "chart_data": chart_data, "question": question, "memory": memory
        })
        if better and len(better) > len(answer):
            answer = better

    # 7) Memory writes
    post_answer_update(user_id, question, answer, topic=None)

    # 8) Batch upload current conversation (simple chunking to episodes)
    _batch_upload_conversation(user_id, [(question, answer)])

    return answer