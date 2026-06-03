from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Override via GEN_MODEL env var or .env file.
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5.5")

GRAMMAR_SYLLABUS: list[tuple[str, str]] = [
    ("A1", "Subject pronouns and the verb être"),
    ("A1", "The verb avoir and expressing age"),
    ("A1", "Definite and indefinite articles, grammatical gender"),
    ("A1", "Present tense of regular -er verbs"),
    ("A1", "Negation with ne ... pas"),
    ("A1", "Plural of nouns"),
    ("A1", "Asking questions: intonation, est-ce que, inversion"),
    ("A2", "Present tense of -ir and -re verbs"),
    ("A2", "Passé composé with avoir"),
    ("A2", "Passé composé with être and agreement"),
    ("A2", "The imparfait and its uses"),
    ("A2", "Futur proche (aller + infinitive)"),
    ("A2", "Reflexive (pronominal) verbs"),
    ("A2", "Adjective agreement and placement"),
    ("A2", "Possessive and demonstrative adjectives"),
    ("A2", "Partitive articles: du, de la, des"),
    ("B1", "Direct and indirect object pronouns"),
    ("B1", "The pronouns y and en"),
    ("B1", "Relative pronouns: qui, que, dont, où"),
    ("B1", "The simple future tense"),
    ("B1", "The present conditional"),
    ("B1", "The present subjunctive: formation and triggers"),
    ("B2", "Past subjunctive and concordance des temps"),
    ("B2", "The passive voice"),
    ("B2", "Reported (indirect) speech"),
    ("B2", "Complex relative pronouns: lequel, auquel, duquel"),
]

VOCAB_SYLLABUS: list[tuple[str, str, int]] = [
    ("A1", "greetings and politeness", 15),
    ("A1", "family members", 15),
    ("A1", "numbers and time", 20),
    ("A1", "food and drink", 25),
    ("A1", "common everyday verbs", 25),
    ("A1", "colours and basic adjectives", 15),
    ("A2", "house and furniture", 20),
    ("A2", "city, places and directions", 20),
    ("A2", "work and professions", 20),
    ("A2", "travel and transport", 20),
    ("A2", "weather and seasons", 15),
    ("A2", "emotions and feelings", 15),
    ("B1", "abstract and academic vocabulary", 25),
    ("B1", "technology and media", 20),
]

_GRAMMAR_SYSTEM = (
    "You are a senior French (FLE) curriculum author writing reference material "
    "for a language-tutoring system. You write accurate, idiomatic French and "
    "you never invent forms. Every conjugation and agreement must be correct. "
    "If unsure about a form, omit it rather than guess."
)

_VOCAB_SYSTEM = (
    "You are a French (FLE) lexicographer building vocabulary entries for a "
    "tutoring system. Genders and spellings must be correct. Output strictly "
    "valid JSON and nothing else."
)

VALID_POS = {"noun", "verb", "adjective", "adverb", "preposition",
             "interjection", "pronoun"}


def _grammar_prompt(level: str, topic: str, explanation_lang: str) -> str:
    return (
        f"Write ONE self-contained grammar reference section on: {topic}.\n"
        f"CEFR level: {level}. Explanations in {explanation_lang}.\n\n"
        "FORMAT — return Markdown only, no preamble, exactly:\n"
        f"## <concise rule title> ({level})\n"
        "<2–5 sentence clear explanation>\n"
        "<if relevant, a full conjugation or a short bullet list of forms>\n"
        "Exemples : <3 to 5 correct, natural example sentences>\n\n"
        "Rules: keep each example grammatically correct; cover the core of the "
        "topic, not edge cases; do not add headers other than the one ## line."
    )


def _vocab_prompt(level: str, theme: str, n: int) -> str:
    return (
        f"Produce {n} French vocabulary entries for the theme '{theme}' at CEFR "
        f"level {level}. Return a JSON object: {{\"entries\": [ ... ]}} where each "
        "entry has EXACTLY these keys:\n"
        '  "term"        : the word WITH its article if it is a noun (e.g. "la maison"),\n'
        '  "definition"  : short English gloss,\n'
        '  "examples"    : array of 1-2 correct French example sentences,\n'
        '  "cefr_level"  : "' + level + '",\n'
        '  "pos"         : one of noun|verb|adjective|adverb|preposition|interjection|pronoun,\n'
        '  "gender"      : "m" | "f" | "" (empty unless it is a noun),\n'
        f'  "topic"       : a short slug for "{theme}".\n'
        "Make terms distinct and genuinely useful at this level. JSON only."
    )


def _client():
    from openai import OpenAI
    return OpenAI()


def _chat(client, system: str, user: str, json_mode: bool) -> str:
    kwargs = dict(
        model=GEN_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _valid_vocab_entry(e: dict) -> bool:
    keys = {"term", "definition", "examples", "cefr_level", "pos", "gender", "topic"}
    return (
        not (keys - e.keys())
        and e["pos"] in VALID_POS
        and e["gender"] in ("m", "f", "")
        and isinstance(e["examples"], list)
        and e["examples"]
    )


def generate_grammar(out_dir: Path, explanation_lang: str) -> int:
    """Generates all grammar sections from GRAMMAR_SYLLABUS and writes them as a
    single Markdown file under out_dir/grammar/generated.md. Returns section count."""
    client = _client()
    sections, written = [], 0
    for level, topic in GRAMMAR_SYLLABUS:
        md = _chat(client, _GRAMMAR_SYSTEM, _grammar_prompt(level, topic, explanation_lang),
                   json_mode=False)
        if "##" in md:
            sections.append(md.strip())
            written += 1
            print(f"[grammar] {level}  {topic}")
        else:
            print(f"[grammar] SKIPPED (no heading) {level} {topic}")
        time.sleep(0.2)
    out = out_dir / "grammar" / "generated.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(sections), encoding="utf-8")
    print(f"[grammar] wrote {written} sections -> {out}")
    return written


def generate_vocab(out_dir: Path) -> int:
    """Generates all vocabulary entries from VOCAB_SYLLABUS, deduplicates by term,
    and writes them as out_dir/vocab.json. Returns entry count."""
    client = _client()
    seen, entries = set(), []
    for level, theme, n in VOCAB_SYLLABUS:
        raw = _chat(client, _VOCAB_SYSTEM, _vocab_prompt(level, theme, n), json_mode=True)
        try:
            items = json.loads(raw).get("entries", [])
        except json.JSONDecodeError:
            print(f"[vocab] BAD JSON for {theme}; skipping")
            continue
        kept = 0
        for e in items:
            if _valid_vocab_entry(e) and e["term"].lower() not in seen:
                seen.add(e["term"].lower())
                entries.append(e)
                kept += 1
        print(f"[vocab] {level}  {theme}: kept {kept}/{len(items)}")
        time.sleep(0.2)
    out = out_dir / "vocab.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[vocab] wrote {len(entries)} entries -> {out}")
    return len(entries)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the French curriculum with OpenAI")
    p.add_argument("--lang", default="fr")
    p.add_argument("--out", default="./curriculum/fr")
    p.add_argument("--explanation-lang", default="French",
                   help="language of grammar explanations (e.g. French or English)")
    p.add_argument("--skip-grammar", action="store_true")
    p.add_argument("--skip-vocab", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    if not args.skip_grammar:
        generate_grammar(out_dir, args.explanation_lang)
    if not args.skip_vocab:
        generate_vocab(out_dir)
    print("Done. Next: run ingest.py on the generated files.")
