"""
llm_explainer.py
Optional "Deeper AI explanation" for the Heart Health Dashboard.

How it stays safe (RAG = retrieval-augmented generation):
  1. The rule-based engine and the EBM/SHAP model stay the base. The AI only EXPLAINS them.
  2. The AI may only use the numbered evidence snippets below (E1, E2, ...).
  3. Every point must cite an evidence id. Uncited / unknown-id / invented-number replies are rejected.
  4. If the API is off, fails, or the reply fails the checks, the app shows the normal rule-based text.
  5. Only a short, anonymous summary is sent (no name, no income, no education, no exact weight/height).
"""
import os
import re

DEFAULT_MODEL = "claude-sonnet-5-5"   # Claude (paid). Change in secrets: LLM_MODEL
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"   # Groq (free tier). Change in secrets: GROQ_MODEL
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"   # Gemini (free tier). Change in secrets: GEMINI_MODEL

# ------------------------------------------------------------------
# EVIDENCE LIBRARY  (text taken from the app's own reviewed recommendation library)
# TODO (Petro): add the exact page URL for each source before submission, and re-check wording.
# ------------------------------------------------------------------
EVIDENCE = {
    "E1": {"topic": "PhysActivity", "source": "NZ Ministry of Health",
           "text": "At least 150 minutes of moderate-intensity physical activity per week is recommended for heart health."},
    "E2": {"topic": "Smoker", "source": "Quitline NZ",
           "text": "Quitline NZ (0800 778 778) provides free, confidential support to quit smoking. Blood pressure begins to fall within 20 minutes of quitting."},
    "E3": {"topic": "HvyAlcoholConsump", "source": "NZ Health Promotion Agency",
           "text": "Low-risk drinking guidance used in this tool: no more than 14 drinks a week for men and 7 drinks a week for women."},
    "E4": {"topic": "Fruits", "source": "NZ '5+ A Day'",
           "text": "The NZ '5+ A Day' guideline recommends at least 2 servings of fruit daily."},
    "E5": {"topic": "Veggies", "source": "NZ '5+ A Day'",
           "text": "The NZ '5+ A Day' guideline recommends at least 5 servings of vegetables daily."},
    "E6": {"topic": "HighBP", "source": "Heart Foundation NZ",
           "text": "To help blood pressure: eat less salt (under 5g a day, about 1 teaspoon) and stay active (150+ minutes a week). Target is 120/80 mmHg or below."},
    "E7": {"topic": "HighChol", "source": "Heart Foundation NZ",
           "text": "To help cholesterol: swap butter and fatty meat for nuts, olive oil and oats. Target is under 5.5 mmol/L."},
    "E8": {"topic": "BMI_high", "source": "Heart Foundation NZ",
           "text": "A healthy BMI range is 18.5-24.9. Weight is best discussed with your GP alongside other factors."},
    "E9": {"topic": "CholCheck", "source": "Heart Foundation NZ",
           "text": "A cholesterol check is recommended, especially from age 45 onward."},
    "E10": {"topic": "NoDocbcCost", "source": "Healthline NZ",
            "text": "If cost is a barrier to seeing a doctor, community health services and budget GP options may be available. Healthline NZ (0800 611 116) can help find local options."},
    "E11": {"topic": "Managed", "source": "This tool (reviewed rule library)",
            "text": "For stroke, diabetes and difficulty walking, ongoing management with your healthcare provider is the appropriate path, rather than a lifestyle change alone."},
    "E12": {"topic": "General", "source": "This tool",
            "text": "The result is a dataset-based likelihood estimate from a large health survey. It is not a diagnosis and not a validated clinical risk score. Please discuss it with your GP."},
}

# plain-English names for the model's feature names
FEATURE_NAMES = {
    "HighBP": "high blood pressure", "HighChol": "high cholesterol", "CholCheck": "cholesterol check",
    "Stroke": "stroke history", "Diabetes": "diabetes status", "Sex": "sex", "Age": "age group",
    "Smoker": "smoking history", "PhysActivity": "physical activity", "Fruits": "fruit intake",
    "Veggies": "vegetable intake", "HvyAlcoholConsump": "heavy alcohol use", "BMI": "BMI",
    "GenHlth": "self-rated general health", "MentHlth": "days of poor mental health",
    "PhysHlth": "days of heart-related symptoms", "DiffWalk": "difficulty walking",
    "AnyHealthcare": "health coverage", "NoDocbcCost": "cost barrier to seeing a doctor",
    "Education": "education level", "Income": "income level",
}


def pick_evidence(candidate_risks, supportive_flags):
    """Return the evidence ids that match this user. E12 is always included."""
    ids = []
    wanted = set(candidate_risks) | set(supportive_flags)
    for eid, e in EVIDENCE.items():
        if e["topic"] in wanted:
            ids.append(eid)
    if "E12" not in ids:
        ids.append("E12")
    return ids


def _advice_ids(evidence_ids):
    """Guideline snippets the AI may cite. E12 (general disclaimer) is added by the app, not cited by the AI."""
    return [i for i in evidence_ids if i != "E12"]


FACTOR_ANSWER_KEYS = {
    "Age": "age group", "HighBP": "high blood pressure (doctor told)", "HighChol": "high cholesterol (doctor told)",
    "Smoker": "ever smoked 100+ cigarettes", "PhysActivity": "physical activity level",
    "Fruits": "fruit servings per day", "Veggies": "vegetable servings per day",
    "HvyAlcoholConsump": "heavy alcohol", "BMI": "BMI band", "Diabetes": "diabetes", "Stroke": "stroke history",
    "GenHlth": "general health", "Sex": "sex", "CholCheck": "cholesterol checked in past 5 years",
    "DiffWalk": "serious difficulty walking", "NoDocbcCost": "cost stopped a doctor visit in past year",
}


def _answer_quality(f, val):
    """Return 'unhealthy', 'healthy' or None for the person's answer (simple, conservative rules)."""
    if val is None:
        return None
    v = str(val)
    if f == "PhysActivity":
        return "unhealthy" if v == "Low" else ("healthy" if v == "High" else None)
    if f in ("Fruits", "Veggies"):
        try:
            return "unhealthy" if float(v) == 0 else None
        except ValueError:
            return None
    if f == "CholCheck":
        return "unhealthy" if v == "No" else "healthy"
    if f == "Diabetes":
        return "healthy" if v == "No" else "unhealthy"
    if f == "GenHlth":
        return "unhealthy" if v in ("Fair", "Poor") else "healthy"
    if f in ("Smoker", "HighBP", "HighChol", "Stroke", "HvyAlcoholConsump", "DiffWalk", "NoDocbcCost"):
        return "unhealthy" if v == "Yes" else "healthy"
    return None


def _factor_lines(summary):
    """One line per top factor: name, the person's answer (if shared), direction and strength."""
    factors = summary["top_factors"]
    biggest = max((abs(v) for _, v in factors), default=0) or 1
    answers = summary.get("answers", {})
    lines = []
    for f, v in factors:
        key = FACTOR_ANSWER_KEYS.get(f)
        val = answers.get(key) if key else None
        shown = f"your answer: {val}" if val is not None else "your answer: not shared"
        ratio = abs(v) / biggest
        strength = "strongest effect" if ratio >= 0.999 else ("clear effect" if ratio >= 0.4 else "small effect")
        direction = "PUSHED the estimate UP" if v > 0 else "PULLED the estimate DOWN"
        q = _answer_quality(f, val)
        note = ""
        if q == "unhealthy" and v < 0:
            note = (" [NOTE: this answer is NOT healthy, even though it lowered the estimate. "
                    "Say clearly it does NOT make this habit or condition good or protective, and that the person should still look after it.]")
        elif q == "healthy" and v > 0:
            note = (" [NOTE: this answer is healthy. Say it is still good for the heart; the estimate is only rough.]")
        lines.append(f"- {FEATURE_NAMES.get(f, f)} ({shown}): {direction}, {strength}{note}")
    return "\n".join(lines)


def build_prompt(summary, evidence_ids):
    evidence_ids = _advice_ids(evidence_ids)
    ev = "\n".join(f"[{i}] ({EVIDENCE[i]['source']}) {EVIDENCE[i]['text']}" for i in evidence_ids)
    top = _factor_lines(summary)
    answers = "\n".join(f"- {k}: {v}" for k, v in summary["answers"].items())
    system = (
        "You explain results from a heart-health AWARENESS tool to a member of the public in New Zealand, "
        "in simple, kind, plain English (reading age about 12). You are NOT a doctor. The reader knows you are an AI and should not trust you 100%, "
        "so be genuinely helpful and practical, not vague.\n"
        "IMPORTANT: Do NOT mention datasets, surveys, models, data patterns, SHAP or how the estimate was calculated. "
        "Explain everything using general, real-world heart-health knowledge, like a friendly health educator would.\n"
        "Write exactly five sections, each title on its own line in bold exactly as shown:\n"
        "**What your result means** - 2 to 3 sentences: the tool gives an awareness estimate (lower / moderate / higher) based on the person's answers; "
        "say what that level means in simple words; it is not a diagnosis and the best next step is a chat with a GP.\n"
        "**What stood out in your answers** - one short paragraph or bullet for EACH top factor. For each: name it and the person's answer (only if shown); "
        "say it was one of the things that raised (UP) or lowered (DOWN) their estimate the most; "
        "then explain in real-world terms what this factor is, how it affects the heart and blood vessels, and why it matters. "
        "If it is something a person cannot change (age, sex, past stroke) say so kindly and say it is best looked after with their GP. "
        "Follow any [NOTE] on a factor exactly. NEVER say an unhealthy habit (low activity, no fruit or vegetables, smoking, no cholesterol check) protects the person or is good for them. "
        "No [E] tags in this section.\n"
        "**Ideas from NZ guidelines** - use ONLY the numbered EVIDENCE. Every sentence must end with its evidence id in square brackets, e.g. [E1]. "
        "Link each idea to the person's answers. If there is no EVIDENCE write: 'No extra guideline ideas apply from your answers.' with no tag.\n"
        "**More ideas (AI general knowledge)** - 3 to 4 practical, safe, everyday ideas that fit THIS person's answers and top factors "
        "(for example simple food swaps, building walking into the day, sleep, managing stress, keeping up regular check-ups). "
        "No numbers or percentages here. No [E] tags here.\n"
        "**Questions to ask your GP** - 2 to 3 short questions tailored to the person's answers. No [E] tags.\n"
        "RULES:\n"
        "0. NEVER state or guess a value for the person unless it appears in the factor list or PERSON'S ANSWERS.\n"
        "1. Never diagnose. Never promise a result. Never say a change will lower their risk by any amount.\n"
        "2. Never give percentages or statistics. Numbers only from the EVIDENCE, the RESULT or the person's answers.\n"
        "3. Never mention medicines, doses, supplements or stopping/starting any treatment. Say to ask the GP about that.\n"
        "4. If the person has a stroke history, diabetes, or difficulty walking, say ongoing care is best led by their GP or health team.\n"
        "5. Be respectful and non-judgemental. Do not blame the person.\n"
        "6. Max 450 words. Do not add an urgent-help message or a closing disclaimer (the app adds them)."
    )
    user = (
        f"RESULT\n- Likelihood estimate: {summary['likelihood_pct']:.1f}% ({summary['tier']})\n\n"
        f"TOP FACTORS FROM THE MODEL (SHAP)\n{top}\n\n"
        f"PERSON'S ANSWERS (anonymous)\n{answers}\n\n"
        f"EVIDENCE\n{ev if ev else '(none)'}\n"
    )
    return system, user


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


GUIDE_HEAD = re.compile(r"ideas from nz guidelines", re.I)
MORE_HEAD = re.compile(r"more ideas \(ai general knowledge\)", re.I)
MED_WORDS = re.compile(r"\b(aspirin|statins?|metformin|insulin|warfarin|dose|doses|dosage|mg|tablets?|pills?|supplements?|stop taking|start taking|prescri\w*)\b", re.I)


def sanitize_tags(reply):
    """Guideline tags [E#] are only valid inside 'Ideas from NZ guidelines'. Remove them anywhere else."""
    g = GUIDE_HEAD.search(reply)
    if not g:
        return re.sub(r"\s*\[E\d+\]", "", reply)
    m = MORE_HEAD.search(reply, g.end())
    end = m.start() if m else len(reply)
    before, mid, after = reply[:g.end()], reply[g.end():end], reply[end:]
    strip = lambda t: re.sub(r"\s*\[E\d+\]", "", t)
    return strip(before) + mid + strip(after)


def validate(reply, evidence_ids, summary):
    """Return (ok, reason). Rejects unknown/missing citations, invented numbers, medicine talk."""
    advice = _advice_ids(evidence_ids)
    cited = set(re.findall(r"\[(E\d+)\]", reply))
    if not cited.issubset(set(advice)):
        return False, "unknown evidence id"
    if advice and not cited:
        return False, "no citations"
    if MED_WORDS.search(reply):
        return False, "mentions medicines or doses"
    allowed = set()
    for i in advice:
        allowed |= _numbers(EVIDENCE[i]["text"])
    allowed |= _numbers(f"{summary['likelihood_pct']:.1f} {summary['likelihood_pct']:.0f}")
    # numbers from the person's own answers (e.g. age group 18-24, servings per day) are allowed too
    allowed |= _numbers(" ".join(str(v) for v in summary.get("answers", {}).values()))
    body = re.sub(r"\[E\d+\]", "", reply)
    # list numbers like "1." or "2)" at the start of a line are not health figures
    body = re.sub(r"(?m)^\s*(?:[-*]\s*)?\d+[.)]\s+", "", body)
    # 1) Only inside "Ideas from NZ guidelines": every number must come from the evidence / result / answers
    m = GUIDE_HEAD.search(body)
    if m:
        m2 = MORE_HEAD.search(body, m.end())
        gpart = body[m.end(): m2.start() if m2 else len(body)]
        extra = _numbers(gpart) - allowed
        if extra:
            return False, f"numbers not in evidence: {sorted(extra)}"
    # 2) Whole reply: no invented percentages
    pct_extra = {n for n in re.findall(r"(\d+(?:\.\d+)?)\s*%", body)} - allowed
    if pct_extra:
        return False, f"percentages not in evidence: {sorted(pct_extra)}"
    bad = re.search(r"(will|would)\s+(lower|reduce|cut)[^.]*\d+\s*%", body, re.I)
    if bad:
        return False, "promise of risk reduction"
    return True, "ok"


def _secret(secrets, name):
    if secrets is None:
        return None
    try:
        return secrets.get(name)
    except Exception:  # no secrets file etc.
        return None


def _get_key(secrets=None):
    return os.environ.get("ANTHROPIC_API_KEY") or _secret(secrets, "ANTHROPIC_API_KEY")


def _get_gemini_key(secrets=None):
    return os.environ.get("GEMINI_API_KEY") or _secret(secrets, "GEMINI_API_KEY")


def _get_groq_key(secrets=None):
    return os.environ.get("GROQ_API_KEY") or _secret(secrets, "GROQ_API_KEY")


def _call_groq(key, model, system, user, http=None):
    """Groq API (OpenAI-compatible, free tier). Plain HTTPS via requests."""
    if http is None:
        import requests as http
    payload = {"model": model, "temperature": 0.2, "max_tokens": 2000,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "gpt-oss" in model:
        payload["reasoning_effort"] = "low"   # thinking models: keep thinking short so the answer is not cut off
    r = http.post("https://api.groq.com/openai/v1/chat/completions",
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                  json=payload, timeout=60)
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def _call_gemini(key, model, system, user, http=None):
    """Free Google Gemini API (AI Studio key). Uses plain HTTPS, no extra package except requests."""
    if http is None:
        import requests as http
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1500},
    }
    r = http.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=40)
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


LABEL_MEANING = "*Source: your answers and this tool's estimate.*"
LABEL_FACTORS = "*Source: AI general knowledge about each factor. It may contain mistakes.*"
LABEL_ADVICE = "*Source: NZ guidelines, shown after each point.*"
LABEL_MORE = "*AI general knowledge, not from a guideline source. It may contain mistakes.*"
LABEL_GP = "*General questions only, not medical advice.*"
URGENT_NOTE = ("**When to get help**\n"
               "*If you have chest pain or pressure, trouble breathing, sudden weakness or numbness, trouble speaking, or feel very unwell, "
               "call 111 now. For other health questions you can call Healthline free on 0800 611 116.*")


def add_section_labels(text):
    """Put a visible source label under each section title (done by the app, not the AI)."""
    out = []
    for line in text.split("\n"):
        out.append(line)
        low = line.lower()
        if len(line) < 70:
            if "what your result means" in low:
                out.append(LABEL_MEANING)
            elif "what stood out in your answers" in low:
                out.append(LABEL_FACTORS)
            elif "ideas from nz guidelines" in low:
                out.append(LABEL_ADVICE)
            elif "more ideas (ai general knowledge)" in low:
                out.append(LABEL_MORE)
            elif "questions to ask your gp" in low:
                out.append(LABEL_GP)
    return "\n".join(out)


def get_ai_explanation(summary, evidence_ids, secrets=None, client=None, http=None):
    """
    Returns {ok, text, reason}. Never raises.
    Provider order: Groq (GROQ_API_KEY), Gemini (GEMINI_API_KEY), then Claude (ANTHROPIC_API_KEY).
    `client` / `http` can be passed in for testing.
    """
    try:
        system, user = build_prompt(summary, evidence_ids)
        gkey = _get_gemini_key(secrets)
        qkey = _get_groq_key(secrets)
        if client is None and qkey:
            model = os.environ.get("GROQ_MODEL") or _secret(secrets, "GROQ_MODEL") or DEFAULT_GROQ_MODEL
            text = _call_groq(qkey, model, system, user, http=http)
        elif client is None and gkey:
            model = os.environ.get("GEMINI_MODEL") or _secret(secrets, "GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            text = _call_gemini(gkey, model, system, user, http=http)
        else:
            if client is None:
                key = _get_key(secrets)
                if not key:
                    return {"ok": False, "text": "", "reason": "AI explanation is not set up (no API key)."}
                import anthropic
                client = anthropic.Anthropic(api_key=key)
            model = os.environ.get("LLM_MODEL") or _secret(secrets, "LLM_MODEL") or DEFAULT_MODEL
            resp = client.messages.create(
                model=model, max_tokens=1200, temperature=0.2,
                system=system, messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        text = sanitize_tags(text)
        ok, reason = validate(text, evidence_ids, summary)
        if not ok:
            return {"ok": False, "text": "", "reason": f"AI reply failed safety check ({reason}). Start of reply: {text[:150]!r}"}
        text = add_section_labels(text)
        text += "\n\n" + URGENT_NOTE
        text += "\n\n*" + EVIDENCE["E12"]["text"] + "*"
        return {"ok": True, "text": text, "reason": "ok"}
    except Exception as e:
        resp_obj = getattr(e, "response", None)
        code = getattr(resp_obj, "status_code", "")
        detail = ""
        try:
            detail = resp_obj.text[:200]
        except Exception:
            pass
        return {"ok": False, "text": "", "reason": f"AI explanation unavailable ({type(e).__name__} {code}) {detail}"}


def render_sources(text, evidence_ids):
    """Replace [E1] tags with readable source names and list them."""
    used = sorted(set(re.findall(r"\[(E\d+)\]", text)), key=lambda x: int(x[1:]))
    nice = text
    for i in used:
        nice = nice.replace(f"[{i}]", f"[{i}: {EVIDENCE[i]['source']}]")
    src = [f"{i}: {EVIDENCE[i]['source']}" for i in used]
    return nice, src
