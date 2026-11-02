"""Research entry point for the rule-book flagger. The canonical code lives in
hf_space/rulebook_flagger.py (shared with the Space; not forked)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hf_space"))
from rulebook_flagger import *  # noqa: E402,F401,F403
from rulebook_flagger import candidate_stems, flag, is_established_shared, norm, resources  # noqa: E402,F401
