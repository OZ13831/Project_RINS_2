from difflib import get_close_matches

INTENT_PATTERNS = {
    "BARRELS": [
        "inspect the barrels",
        "inspect barrels",
        "check the barrels",
        "check barrels",
        "look at the barrels",
        "barrels"
    ],
    "RINGS": [
        "count the rings",
        "count rings",
        "rings",
        "count the ring"
    ],
    "ANOMALY_RED": [
        "detect anomalies in the red cell",
        "inspect anomalies in the red cell",
        "red cell",
        "anomalies red cell",
        "detect red cell anomalies"
    ],
    "ANOMALY_GREEN": [
        "detect anomalies in the green cell",
        "inspect anomalies in the green cell",
        "green cell",
        "anomalies green cell",
        "detect green cell anomalies"
    ],
    "YES": [
        "yes",
        "yeah",
        "correct",
        "i am sure",
        "im sure",
        "sure"
    ],
    "NO": [
        "no",
        "not really",
        "nope"
    ],
    "NOTHING": [
        "nothing",
        "do nothing",
        "no task"
    ]
}


def exact_match_intent(text: str) -> str | None:
    for intent, patterns in INTENT_PATTERNS.items():
        if text in patterns:
            return intent
    return None

def keyword_match_intent(text: str) -> str | None:
    words = set(text.split())

    if "barrels" in words or "barrel" in words:
        return "BARRELS"

    if "rings" in words or ("count" in words and "ring" in words):
        return "RINGS"

    if "red" in words and "anomaly" in words:
        return "ANOMALY_RED"

    if "green" in words and "anomaly" in words:
        return "ANOMALY_GREEN"

    if text in {"yes", "sure", "correct"} or ("yes" in words):
        return "YES"

    if text in {"no", "nope"} or ("not" in words and "really" in words):
        return "NO"

    if "nothing" in words:
        return "NOTHING"

    return None


def fuzzy_match_intent(text: str, cutoff: float = 0.75) -> str | None:
    all_patterns = {}
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            all_patterns[p] = intent

    matches = get_close_matches(text, all_patterns.keys(), n=1, cutoff=cutoff)
    if matches:
        return all_patterns[matches[0]]
    return None


def get_intent(text: str) -> tuple[str | None, bool | None]:
    intent = exact_match_intent(text)
    if intent is not None:
        return intent, _extract_affirmation(intent)

    intent = keyword_match_intent(text)
    if intent is not None:
        return intent, _extract_affirmation(intent)

    intent = fuzzy_match_intent(text)
    return intent, _extract_affirmation(intent)


def _extract_affirmation(intent: str | None) -> bool | None:
    if intent == "YES":
        return True
    if intent == "NO":
        return False
    return None
