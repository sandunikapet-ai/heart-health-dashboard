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
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"   # Groq (free tier). Change in secrets: GROQ_MODEL
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


def build_prompt(summary, evidence_ids):
    ev = "\n".join(f"[{i}] ({EVIDENCE[i]['source']}) {EVIDENCE[i]['text']}" for i in evidence_ids)
    top = "\n".join(f"- {FEATURE_NAMES.get(f, f)}: {'pushed the estimate UP' if v > 0 else 'pulled the estimate DOWN'}"
                    for f, v in summary["top_factors"])
    answers = "\n".join(f"- {k}: {v}" for k, v in summary["answers"].items())
    system = (
        "You explain results from a heart-health AWARENESS tool to a member of the public in New Zealand, "
        "in simple, kind, plain English (reading age about 12). You are NOT a doctor.\n"
        "RULES:\n"
        "1. Use ONLY the numbered EVIDENCE below for any health guidance. Do not use outside knowledge.\n"
        "2. Every guidance sentence must end with its evidence id in square brackets, e.g. [E1].\n"
        "3. Explain why the top factors mattered for THIS result, using only the factor list given.\n"
        "4. Never diagnose. Never say a change will lower their risk by any amount. Never give new numbers "
        "(only numbers that appear in the EVIDENCE or the RESULT).\n"
        "5. If something is not covered by the evidence, say 'This tool does not have guidance on that. "
        "Please ask your GP.'\n"
        "6. The factors are patterns in survey data, not proof of cause.\n"
        "7. Be respectful and non-judgemental. Do not blame the person.\n"
        "8. Max 220 words. Use short paragraphs or bullets. End with a line suggesting they talk to their GP."
    )
    user = (
        f"RESULT\n- Likelihood estimate: {summary['likelihood_pct']:.1f}% ({summary['tier']})\n\n"
        f"TOP FACTORS FROM THE MODEL (SHAP)\n{top}\n\n"
        f"PERSON'S ANSWERS (anonymous)\n{answers}\n\n"
        f"EVIDENCE\n{ev}\n"
    )
    return system, user


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def validate(reply, evidence_ids, summary):
    """Return (ok, reason). Rejects unknown citations, uncited guidance, invented numbers."""
    cited = set(re.findall(r"\[(E\d+)\]", reply))
    if not cited:
        return False, "no citations"
    if not cited.issubset(set(evidence_ids)):
        return False, "unknown evidence id"
    allowed = set()
    for i in evidence_ids:
        allowed |= _numbers(EVIDENCE[i]["text"])
    allowed |= _numbers(f"{summary['likelihood_pct']:.1f} {summary['likelihood_pct']:.0f}")
    body = re.sub(r"\[E\d+\]", "", reply)
    extra = _numbers(body) - allowed
    if extra:
        return False, f"numbers not in evidence: {sorted(extra)}"
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
    r = http.post("https://api.groq.com/openai/v1/chat/completions",
                  headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                  json={"model": model, "temperature": 0.2, "max_tokens": 500,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                  timeout=40)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(key, model, system, user, http=None):
    """Free Google Gemini API (AI Studio key). Uses plain HTTPS, no extra package except requests."""
    if http is None:
        import requests as http
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
    }
    r = http.post(url, headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=body, timeout=40)
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


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
                model=model, max_tokens=500, temperature=0.2,
                system=system, messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        ok, reason = validate(text, evidence_ids, summary)
        if not ok:
            return {"ok": False, "text": "", "reason": f"AI reply failed safety check ({reason})."}
        return {"ok": True, "text": text, "reason": "ok"}
    except Exception as e:
        return {"ok": False, "text": "", "reason": f"AI explanation unavailable ({type(e).__name__})."}


def render_sources(text, evidence_ids):
    """Replace [E1] tags with readable source names and list them."""
    used = sorted(set(re.findall(r"\[(E\d+)\]", text)), key=lambda x: int(x[1:]))
    nice = text
    for i in used:
        nice = nice.replace(f"[{i}]", f"[{i}: {EVIDENCE[i]['source']}]")
    src = [f"{i}: {EVIDENCE[i]['source']}" for i in used]
    return nice, src
