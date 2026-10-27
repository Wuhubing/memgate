"""Sentence templates. All English; no template may contain a memory value (checked by validate_world)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Attr:
    name: str          # key suffix
    vtype: str         # value type for ValueFactory.draw
    category: str
    np: str            # noun phrase used in questions / updates, may contain {P}
    record: str        # record sentence, contains {v} and maybe {P}
    topic: str         # chat topic word; "{P}" means the project name


PROJECT_ATTRS = [
    Attr("deadline", "date", "schedule", "the deadline for Project {P}", "Project {P}'s deadline is {v}.", "{P}"),
    Attr("contact", "person", "contact", "the point of contact for Project {P}",
         "The point of contact for Project {P} is {v}.", "{P}"),
]
PREFERENCE_ATTRS = [
    Attr("style", "style", "preference", "my preferred writing style", "The user's preferred writing style is {v}.",
         "writing style"),
    Attr("tool", "tool", "preference", "the note-taking app I use", "The user takes notes in {v}.", "note-taking"),
    Attr("airline", "airline", "preference", "my preferred airline", "The user's preferred airline is {v}.",
         "flight"),
    Attr("theme", "theme", "preference", "my editor theme", "The user's editor theme is {v}.", "editor"),
]
TRIVIA_ATTRS = [
    Attr("pet", "pet", "trivia", "my cat's name", "The user's cat is named {v}.", "cat"),
    Attr("cafe", "cafe", "trivia", "my favorite coffee spot", "The user's favorite coffee spot is {v}.", "coffee"),
    Attr("gym", "gym", "trivia", "my gym", "The user works out at {v}.", "gym"),
    Attr("street", "street", "trivia", "the street I grew up on", "The user grew up on {v}.", "childhood"),
]

QUERY = [
    "What is {np}?",
    "Can you remind me what {np} is?",
    "Quick check: what's {np}?",
    "I forgot {np}. What is it?",
    "Do you know {np}?",
    "Tell me {np}, please.",
    "What did we say {np} was?",
    "Remind me of {np}.",
    "Sorry, what is {np} again?",
]

UPDATE = [
    "Heads up: {np} is now {v}.",
    "Small change, {np} is {v} from now on.",
    "FYI, {np} changed to {v}.",
    "Please note that {np} is now {v}.",
    "Just so you know, {np} is {v} now.",
    "Going forward, {np} is {v}.",
]

CHAT_TOPIC = [
    "Can you write a short opening line for a meeting about {T}?",
    "Help me phrase a polite reminder to the people involved with {T}.",
    "Give me three name ideas for a newsletter about {T}.",
    "Draft a one-sentence status note about {T}.",
    "Write a two-line haiku that mentions {T}.",
    "Suggest a friendly subject line for an email about {T}.",
    "What is a good icebreaker question related to {T}?",
    "Turn this into a cheerful sentence: 'more news about {T} soon'.",
    "Give me a tip for staying organised with {T}.",
    "Write a short thank-you note that mentions {T}.",
]

CHAT_PLAIN = [
    "Can you tighten this sentence: 'We will circle back on that at some point soon.'",
    "Give me a synonym for 'important' that sounds less formal.",
    "How do I politely decline a meeting invite?",
    "Suggest a title for a short talk about teamwork.",
    "What's a good way to start a weekly update email?",
    "Rewrite 'pls send asap' so it sounds professional.",
    "Give me two ideas for a quick team-building activity.",
    "How can I make a slide deck easier to read?",
    "What is a concise way to say 'at this point in time'?",
    "Write a one-line welcome message for a new colleague.",
    "How do I ask for feedback without sounding needy?",
    "Give me a short motivational quote for a Monday.",
]


def fill(template: str, **kw) -> str:
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", v)
    return out
