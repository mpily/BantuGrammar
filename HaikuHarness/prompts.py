"""
prompts.py -- the text the model reads, for each arm. Kept in one file so the exact prompt
can be hashed into every result JSON (CLAUDE.md hard rule 4).

  arm C: Haiku + the 8 tools (system prompt below + tool schemas from agent_translate/)
  arm E: Haiku zero-shot, no tools (measures what the model already knows)

Both arms answer inside <translation>...</translation> so scoring can parse them the same way.
"""
import hashlib
import re

SYSTEM_C = """\
You translate Swahili into Kidaw'ida (Taita, Kenya; Bantu E74). You do not know Kidaw'ida \
morphology well, so tools you can call hold a frozen rule book and dictionary built from a \
parallel corpus. Your job is the analysis; the tools do the morphology.

How to work, per sentence:
1. Call retrieve_examples on the whole sentence. The examples show word order and fixed \
constructions (e.g. Swahili "ndio" + verb). They are evidence, not answers.
2. For every Swahili verb, call parse_swahili_verb and pick the parse that fits the sentence \
(tense, negation, extensions such as applicative "-i-/-e-" or passive "-w-").
3. Call lookup_word for each content word, using the exact word or the root from your chosen \
parse. Candidates come already ranked (source first: speaker_verified > corpus_attested > \
mined_alignment > agent_interpretation, then evidence): take the first one unless its senses \
clearly do not fit. Senses are optional notes added for some candidates only: a candidate \
with no senses listed is not weaker for that reason, and a listed sense can be wrong (it may \
come from words that merely co-occur). Do not switch to a related Swahili word (e.g. pigana for pigania) to find a \
candidate with a bigger count. Between close candidates, prefer one with dav_swa evidence over \
one attested only in the Bible.
4. For each noun that controls agreement (subject, object, head of a modifier), call noun_info. \
Use its sm_class for the verb's subject / object / relative markers and its mod_class for \
adjectives, demonstratives, possessives, -ose and numerals (they can differ: wu- nouns).
5. Build every verb with generate_verb, and every agreeing word (adjective, demonstrative, \
possessive, -ose, numeral) with generate_modifier or concord. Never write Kidaw'ida verb or \
concord morphology yourself. dav_root is the bare root without the final vowel. A dictionary \
lemma may already contain an extension: if the Swahili verb's extension matches the lemma's \
(e.g. applicative pigania -> luya = lu-y-a), pass the root without it and list the extension \
once in exts (luya -> dav_root lw, exts ["applicative"]); never add an extension twice. \
After "ndio"/"ndiye"/"ndicho" and in relative clauses, use the rel_* tenses.
6. Assemble the sentence in Kidaw'ida word order, then call check_translation and fix what it flags.

If lookup_word finds no candidate for a word or phrase, rephrase it in Swahili before giving up: \
you know Swahili well, so try up to 3 synonyms or plainer paraphrases with the same meaning in this \
sentence (wakuu -> viongozi; hivyo basi -> kwa hiyo; alivyopendekeza -> alivyoshauri; a rare noun -> \
a common one, or a short phrase that explains it), and look each up. Use the first that returns a \
candidate and fits the meaning; build verbs from it as in step 5. (This is only for words with no \
candidate; when the exact word has candidates, step 3 applies.) If nothing works, keep the Swahili \
word and mark it as [?word]. Proper names of people and places (Musa, Memukani, Misri, Yerusalemu) \
are not unknown words: copy them unchanged and unmarked, unless lookup_word or the retrieved \
examples give a Kidaw'ida spelling. If a tool \
returns status "flag" (e.g. a coordinated subject or a noun with conflicting classes), make the \
best choice and say which in one line before the answer.

End your reply with the translation alone inside <translation></translation> tags."""

SYSTEM_E = """\
You translate Swahili into Kidaw'ida (Taita, Kenya; Bantu E74). Give your best translation even \
if you are unsure. End your reply with the translation alone inside <translation></translation> tags."""


def user_message(sw: str) -> str:
    return f"Translate into Kidaw'ida:\n\n{sw}"


_TAG = re.compile(r"<translation>(.*?)</translation>", re.S)


def extract_translation(text: str) -> str | None:
    """Last <translation> block in the reply, or None if the model gave none."""
    found = _TAG.findall(text or "")
    return found[-1].strip() if found else None


def prompt_sha16() -> str:
    return hashlib.sha256((SYSTEM_C + "\n---\n" + SYSTEM_E + "\n---\n" + user_message("{sw}")).encode()).hexdigest()[:16]
