"""Prompts + JSON schemas used by adjudication strategies (fresh copy).

Duplicated from src/adjudicate.py on purpose: this tree is the experimentation
sandbox; changes here must NOT affect the v1–v4 code that other pipelines rely
on. Add new prompts freely; do not import from src/adjudicate.py.
"""
from __future__ import annotations

NAME_FIELDS = {"interpreted_first_name", "interpreted_last_name"}
FIELD_LABEL = {
    "interpreted_first_name": "given name (first name)",
    "interpreted_last_name": "surname",
}

NUDGE_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "confidence"]}
DEFEND_SCHEMA = {"type": "object", "properties": {
    "argument": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["argument", "confidence"]}
DECIDE_SCHEMA = {"type": "object", "properties": {
    "final_name": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "reasoning": {"type": "string"}},
    "required": ["final_name", "confidence"]}
OPENARG_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"},
    "argument": {"type": "string"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}},
    "required": ["name", "argument", "confidence"]}
OPENREBUT_SCHEMA = {"type": "object", "properties": {
    "final_name": {"type": "string"},
    "persuaded": {"type": "boolean"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
    "reasoning": {"type": "string"}},
    "required": ["final_name", "persuaded", "confidence"]}


def nudge_prompt(label, a, b):
    return (f"You are re-examining ONE handwritten {label} on an 1850 U.S. Census "
            "page (a single ruled row). Two transcribers read this "
            f"{label} as '{a}' and '{b}'. Study the handwriting letter by letter and "
            "report the correct reading. Transcribe the EXACT letters written — do "
            "NOT prefer the more common or modern form of a name over what is "
            "actually on the page (e.g. if it is written 'Chls', report 'Chls', not "
            "the more familiar 'Chas'). Return JSON: name (your reading of the "
            f"{label}), confidence (HIGH/MEDIUM/LOW). No prose outside the JSON.")


def defend_prompt(label, own, other):
    return (f"Examine ONE handwritten {label} on this single-row 1850 U.S. Census "
            f"image. You previously read it as '{own}'; another transcriber read "
            f"'{other}'. Make the strongest LETTER-BY-LETTER case for why '{own}' is "
            "the correct reading of what is ACTUALLY written (cite specific letter "
            "shapes, especially the first capital). Judge by the exact letters on "
            "the page, not by which is a more familiar name. Return JSON: argument "
            f"(1-3 sentences), confidence (HIGH/MEDIUM/LOW that '{own}' is correct). "
            "No prose outside the JSON.")


def decide_prompt(label, a, arg_a, b, arg_b):
    return (f"Decide the correct reading of ONE handwritten {label} on this 1850 "
            "U.S. Census row. EXACTLY TWO readings are proposed, each argued:\n"
            f"  A) '{a}' — {arg_a}\n"
            f"  B) '{b}' — {arg_b}\n"
            "Look at the image and choose which of these two matches the exact "
            f"letters written. Your answer MUST be either '{a}' or '{b}', verbatim — "
            "do NOT invent a third reading. Return JSON: final_name (either "
            f"'{a}' or '{b}'), confidence (HIGH/MEDIUM/LOW), reasoning (1 sentence). "
            "No prose outside the JSON.")


def openarg_prompt(label, a, b):
    return (f"Two transcribers proposed '{a}' and '{b}' for this handwritten {label} "
            "on an 1850 U.S. Census row, but they could NOT agree — so neither may "
            f"be correct. Read the {label} fresh from the image, letter by letter, "
            "and give your best reading (it may be one of those two OR a different "
            "name entirely), plus a 1-2 sentence argument citing SPECIFIC letter "
            "shapes. Transcribe the EXACT letters written, not the most familiar "
            "name. Return JSON: name, argument, confidence (HIGH/MEDIUM/LOW). No "
            "prose outside the JSON.")


def openrebut_prompt(label, other_name, other_arg):
    return (f"Re-examine this handwritten {label} on an 1850 U.S. Census row. "
            f"Another expert transcriber reads it as '{other_name}' and argues: "
            f"\"{other_arg}\" Look again at the image, focusing on the features they "
            "cite. Give your final reading — agree with them, keep your own, or a "
            "different reading if the letters show something else. Return JSON: "
            "final_name, persuaded (true if you adopted their reading), confidence "
            "(HIGH/MEDIUM/LOW), reasoning (1 sentence). No prose outside the JSON.")


def openread_prompt(label):
    return (f"Read the {label} written in this image — a single handwritten name "
            "from an 1850 U.S. Census. What does it say? Transcribe the EXACT "
            "letters as written (do not prefer a more familiar or common name), "
            "and INCLUDE any middle initial written as part of a given name "
            "(e.g. 'Milo J.', not just 'Milo'). Return JSON: name, confidence "
            "(HIGH/MEDIUM/LOW). No prose outside the JSON.")
