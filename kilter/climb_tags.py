"""Style tags read from a climb's name and its setter's description, and their condition vector.

The Kilter data has no style tags of its own (its one characteristic is "no_match"), so these
come from the text setters wrote. ``extract`` marks a tag when one of its terms appears and is
not negated. A tag means the setter mentioned it, so a missing tag does not mean the climb lacks
it. data/export_climb_tags.py writes every climb's tags to data/training/climb_tags.csv.gz, and
a model with ``model.climb_tags`` set gets them as a multi-hot vector next to grade and angle.

Rules:

- Text is lowercased and matched on whole words.
- A term in the description always counts. Names are creative ("Pinch of salt", "Power Nap"),
  so a term marked ``name=False`` counts only in the description.
- A term is negated when a negator (no, not, without, don't, never, avoid, sin, sans, pas,
  kein, ohne) is among the 3 words before it in the same clause with no rule word (match,
  feet, legs, hands, kickboard) in between, or it is followed by "off",
  "not allowed", "not required", "banned" or "forbidden". A negated term does not tag the
  climb. A negated dynamic term ("no dyno", "no jumping") tags it ``static`` instead.
- Each tag lists phrases that never count, such as "achilles heel" or "tension board".
"""
import csv
import gzip
import hashlib
import re
from pathlib import Path

import numpy as np

# Each tag: terms as (regex, counts in names), and phrases that never count.
TAG_RULES = {
    "campus": ([(r"campus(?:ing|ed|y)?", True), (r"footless", True), (r"no feet", True), (r"feet ?less", True)],
               [r"campus universitario"]),
    "dynamic": ([(r"dyn(?:o+|os|amic|amically|amo)", True), (r"dino?s?", False), (r"jump(?:s|y|ing|ed)?", True),
                 (r"lunge[sy]?", True), (r"dead ?-?points?", True), (r"pogo(?:s|ing)?", True), (r"throw(?:s|ing|n)?", False),
                 (r"double ?-?clutch", True), (r"jet[ée]s?", False), (r"salto", False), (r"spr(?:ung|ingen)", False)],
                [r"jump ?start", r"throw ?back"]),
    "coordination": ([(r"coord(?:ination|o|y)?", True), (r"paddl(?:e|es|ing)", True), (r"run(?:ning)? ?(?:and|&|n|-) ?jump", True),
                      (r"parkour", True), (r"comp(?:etition)? ?-?style", True), (r"double ?-?clutch", True)], []),
    "static": ([(r"static(?:ally)?", True)], []),
    "cut_loose": ([(r"cut(?:ting|s)? ?(?:loose|feet|out)", True), (r"feet ?cut", True), (r"swing(?:s|y|ing)?", False),
                   (r"barn ?-?door(?:s|ing)?", True)], []),
    "crossover": ([(r"cross ?-?(?:over|through)s?", True), (r"cross(?:es|ing)? (?:to|into|over|under|with|left|right|hand)", True),
                   (r"cross(?:es|ing|y)?", False)], [r"cross ?-?country", r"cross ?fit", r"criss ?-?cross"]),
    "bump": ([(r"bump(?:s|ing|ed|y)?", False)], []),
    "lock_off": ([(r"lock ?-?offs?", True), (r"locky", True)], []),
    "traverse": ([(r"trav(?:e)?rs(?:e|es|ing|al|ée|ee)", True), (r"travese", True)], []),
    "high_feet": ([(r"high ?-?(?:feet|foot|step|steps|stepping)", True), (r"rock ?-?over", True)], []),
    "technical": ([(r"technical", True), (r"techy", True), (r"balanc(?:e|ey|y)", False), (r"body ?-?position(?:ing)?", True),
                   (r"smear(?:s|ing|y)?", True)], []),
    "drop_knee": ([(r"drop ?-?knee(?:s|d)?", True), (r"knee ?-?drop", True), (r"egyptian", True)], []),
    "heel_hook": ([(r"heel(?:s|ed|ing|y)?", True), (r"heel ?-?hook(?:s|ing)?", True), (r"talon(?:s)?", False), (r"tal[oó]n", False)],
                  [r"achilles'? heel", r"high ?-?heels?", r"height heels?", r"over heels"]),
    "toe_hook": ([(r"toe ?-?hook(?:s|ing|ed)?", True), (r"toe ?-?cam(?:s|ming)?", True), (r"bicycle", True)], []),
    "compression": ([(r"compression", True), (r"compy", True), (r"compress(?:ing|ion)?", True), (r"squeez(?:e|y|ing)", True)],
                    [r"lemon ?-?squeez", r"squeeze the cheeks", r"squeezy", r"compression (?:shirt|sleeve|socks?)"]),
    "body_tension": ([(r"(?:body ?-?)?tension(?:y)?", True)], [r"tension ?-?board", r"surface tension", r"(?:sexual|mental|racial) tension"]),
    "gaston": ([(r"gast[oó]n(?:s|ing)?", True)], []),
    "undercling": ([(r"under ?-?cling(?:s|y|ing)?", True), (r"undies", True), (r"invers[ée]e?s?", False), (r"untergriff", False)], []),
    "pinch": ([(r"pinch(?:es|y|ing|er|ers)?", True), (r"pinces?", False), (r"pinzas?", False), (r"zangen?", False)],
              [r"pinch me", r"pinch [oi]f salt", r"pinch hitt"]),
    "jug": ([(r"jugg?(?:s|y|ies|gy)?", True), (r"bacs?", False), (r"henkel", False)], []),
    # Crimp also covers the fingery tag: both are about finger strength on small edges.
    "crimp": ([(r"crimp(?:s|y|er|ers|ing|ed)?", True), (r"edgy", True), (r"edges", False), (r"r[ée]glettes?", False),
               (r"regletas?", False), (r"leisten?", False), (r"finger(?:s|y)?", False), (r"fingery", True)], []),
    "reachy": ([(r"reach(?:y|es|ing)?", True), (r"wing ?span", True), (r"span(?:ny)?", False), (r"morpho", True),
                (r"(?:big|long|huge) moves?", True),
                (r"tall (?:people|climbers?|folks|persons?|ones)", True), (r"if (?:you'?re |you are )?(?:not )?tall", True)], []),
    "short_friendly": ([(r"short(?:ies|y)", True), (r"short ?-?(?:friendly|people|climbers?|folks|persons?|kings?|queens?)", True),
                        (r"for (?:the )?short", True)], []),
    "powerful": ([(r"power(?:ful|y|full)?", False)], [r"power ?-?endurance", r"power ?nap", r"power (?:of|from)\b"]),
    "endurance": ([(r"endurance", True), (r"endurancy", True), (r"pump(?:y|ed|ing)?", False), (r"power ?-?endurance", True),
                   (r"4 ?x ?4s?", True), (r"resistance", True), (r"stamina", True)],
                  [r"least resistance"]),
    "shouldery": ([(r"shoulder(?:s|y|ing)?", False), (r"shouldery", True)], []),
}
TAGS = list(TAG_RULES)
NEGATORS = r"(?:no|not|without|don'?t|do not|never|zero|avoid|won'?t|wasn'?t|isn'?t|aren'?t|sin|sans|pas|kein|keine|ohne|nicht)"
# A negator stops applying at a rule word: in "no matching, just jump" or "no legs paddle dyno"
# the "no" belongs to matching or legs, not to the move.
NEGATION_BREAKS = {"match", "matching", "matches", "matchy", "matchies", "feet", "foot", "legs", "hands", "hand",
                   "kicker", "kickboard", "kick", "board", "just", "but"}
AFTER_NEGATION = re.compile(r"^\W{0,3}(?:off\b|not (?:allowed|required|needed|necessary)|banned|forbidden|discouraged|is off)", re.I)
# "Hard for short people" means the climb is reachy, not short-friendly.
SHORT_PENALTY = re.compile(r"(?:hard|harder|difficult|tough|bad|rough|sorry|rip|struggle|impossible|warning|beware)\W+(?:\w+\W+){0,3}$", re.I)
SHORT_PENALTY_AFTER = re.compile(r"^\W*(?:\w+\W+){0,1}(?:nightmare|beware|problem|struggle|hell)", re.I)
CLAUSE = re.compile(r"[.!?;,\n]|\bbut\b|\bthen\b", re.I)
DYNAMIC_TERMS = {pattern for pattern, _ in TAG_RULES["dynamic"][0]}

_COMPILED = {tag: ([(re.compile(rf"(?<![\w'-]){pattern}(?![\w-])", re.I), in_name, pattern) for pattern, in_name in terms],
                   [re.compile(rf"\b{pattern}", re.I) for pattern in excluded])
             for tag, (terms, excluded) in TAG_RULES.items()}


def _negated(text, start, end):
    clause_start = max((m.end() for m in CLAUSE.finditer(text, 0, start)), default=0)
    before = [word.lower() for word in re.findall(r"[\w']+", text[clause_start:start])[-3:]]
    for i, word in enumerate(before):
        if re.fullmatch(NEGATORS, word, re.I) and not NEGATION_BREAKS.intersection(before[i + 1:]):
            return True
    return bool(AFTER_NEGATION.match(text[end:end + 24]))


def _excluded(text, start, end, patterns):
    window = text[max(0, start - 16):end + 16]
    return any(p.search(window) for p in patterns)


def extract(name, description):
    """Tags for one climb, as {tag: evidence text}. Either argument may be None."""
    found = {}
    for field, text in (("description", description or ""), ("name", name or "")):
        if not text.strip():
            continue
        for tag, (terms, excluded) in _COMPILED.items():
            for regex, in_name, pattern in terms:
                if field == "name" and not in_name:
                    continue
                for match in regex.finditer(text):
                    start, end = match.span()
                    if _excluded(text, start, end, excluded):
                        continue
                    target = tag
                    if _negated(text, start, end):
                        # "No dyno" says the climb is static; any other negated term says nothing.
                        if tag == "dynamic" and pattern in DYNAMIC_TERMS:
                            target = "static"
                        else:
                            continue
                    if tag == "short_friendly" and (SHORT_PENALTY.search(text[max(0, start - 40):start])
                                                    or SHORT_PENALTY_AFTER.match(text[end:end + 30])):
                        target = "reachy"
                    found.setdefault(target, f"{field}: {text[max(0, start - 30):end + 30].strip()}")
    return found


def tag_vector(tags):
    return np.array([tag in tags for tag in TAGS], dtype=np.float32)


def read_tag_file(path):
    """{climb_uuid: multi-hot vector} from data/training/climb_tags.csv.gz."""
    with gzip.open(path, "rt", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames[1:] != TAGS:
            raise ValueError(f"{path} has tags {reader.fieldnames[1:]}, expected {TAGS}")
        return {row["climb_uuid"]: np.array([int(row[tag]) for tag in TAGS], dtype=np.float32) for row in reader}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def attach_tags(arrays, path):
    """Append every row's tag vector to its conditions, in TAGS order. Every climb must be in the file."""
    tags = read_tag_file(path)
    missing = [uuid for uuid in arrays["uuids"][:1000] if uuid not in tags]
    if missing:
        raise ValueError(f"{len(missing)} climbs have no tags in {path}, e.g. {missing[0]}")
    matrix = np.stack([tags[uuid] for uuid in arrays["uuids"]])
    return {**arrays, "conditions": np.concatenate([arrays["conditions"][:, :2], matrix], axis=1).astype(np.float32)}
