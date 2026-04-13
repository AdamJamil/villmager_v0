"""Cost analysis for different prompt caching strategies.

Uses actual character descriptions, prompt templates, and two parameter sets:

  --uniform    Theoretical: uniform-random location choice, 10 turns (original model)
  --observed   Calibrated against the actual day 1 transcript

Each parameter set is evaluated under three caching regimes:
  1. No caching
  2. Current scheme — shared context prefix cached across parallel villager
     rating calls; models cross-turn prefix sharing (only the delta history
     line is a cache write, the rest hits the previous turn's cached prefix)
  3. Character-first — character description cached across sequential turns

Run:
  python cost_analysis.py                    # side-by-side comparison
  python cost_analysis.py --day 5            # project to day 5 memory growth
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import VILLAGERS, LOCATIONS, Villager, Location
from prompts import (
    system_prompt,
    location_choice_user,
    rating_shared,
    rating_character,
    speaking_shared,
    speaking_character,
    memory_save_user,
)


# ── Token estimation ──────────────────────────────────────────────────────

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD = 4


def count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + MESSAGE_OVERHEAD


# ── Pricing (per 1M tokens) ──────────────────────────────────────────────

@dataclass(frozen=True)
class ModelPricing:
    input: float
    output: float
    cache_write: float
    cache_read: float

SONNET = ModelPricing(input=3.00, output=15.00, cache_write=3.75, cache_read=0.30)
OPUS   = ModelPricing(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50)


def cost(tokens: int, rate: float) -> float:
    return tokens * rate / 1_000_000


# ── Simulation parameters ────────────────────────────────────────────────

N_VILLAGERS = len(VILLAGERS)  # 8
N_LOCATIONS = len(LOCATIONS)  # 5
PHASES = 3


@dataclass(frozen=True)
class ConversationGroup:
    """Expected conversations per day with a specific group size."""
    size: int
    per_day: float
    turns: int


@dataclass(frozen=True)
class SimParams:
    """All tunable assumptions in one place."""
    name: str
    conversations: list[ConversationGroup]
    out_location: int
    out_rating: int
    out_speaking: int
    out_memory: int

    @property
    def total_convs_per_day(self) -> float:
        return sum(c.per_day for c in self.conversations)

    @property
    def total_villagers_in_convs_per_day(self) -> float:
        return sum(c.size * c.per_day for c in self.conversations)


def _uniform_conversations() -> list[ConversationGroup]:
    """Derive conversation distribution from the binomial occupancy model."""
    p = 1 / N_LOCATIONS
    groups = []
    for k in range(2, N_VILLAGERS + 1):
        pmf = comb(N_VILLAGERS, k) * (p ** k) * ((1 - p) ** (N_VILLAGERS - k))
        per_day = PHASES * N_LOCATIONS * pmf
        if per_day:
            groups.append(ConversationGroup(size=k, per_day=per_day, turns=10))
    return groups


# Observed from transcripts/2026-04-08_121900/day_1.md:
#
#   Morning:   K&C (5 villagers, 20 turns)
#   Afternoon: K&C (3, 20 turns) + Academy (3, 20 turns)
#   Evening:   K&C (5, 20 turns) + M&M (2, 10 turns)
#
# The system has a hard cap of 20 turns per conversation (10 for pairs).
# No villager gave a -1 rating in the day 1 transcript.
#
# NOTE: The transcript interleaves parallel conversations under a single
# section header (e.g., Afternoon shows K&C and Academy turns mixed together
# under "The Calloway Academy — Afternoon").  Splitting by character location
# confirms each conversation independently hits exactly 20/10 turns.
#
# Location distribution: K&C 54%, Academy 21%, M&M 17%, Meadow 8%, Square 0%
# Villagers cluster at "home" locations (Sable→K&C, Wren→Academy, Aldric→M&M).
#
# Output token measurements from transcript (chars/4 + 4):
#   Speaking: avg ~76 tok (min 39, max 157) across 90 turns
#   Memory:   avg ~214 tok (min 29, max 276) across 18 saves
#
# For the observed parameter set we encode the actual conversation outcomes
# directly rather than deriving them from a distribution model.

def _observed_conversations() -> list[ConversationGroup]:
    """Actual day 1 conversations, used as the expected-value model.

    We express them as "per day" counts so the rest of the cost engine
    works unchanged.  A single day's sample is obviously noisy, but it
    exposes the structural biases the uniform model misses entirely.
    """
    return [
        # Morning K&C (5p, 20t) + Evening K&C (5p, 20t)
        ConversationGroup(size=5, per_day=2, turns=20),
        # Afternoon K&C (3p, 20t) + Afternoon Academy (3p, 20t)
        ConversationGroup(size=3, per_day=2, turns=20),
        # Evening M&M (2p, 10t)
        ConversationGroup(size=2, per_day=1, turns=10),
    ]


UNIFORM_PARAMS = SimParams(
    name="Uniform-random (original model)",
    conversations=_uniform_conversations(),
    out_location=10,
    out_rating=5,
    out_speaking=60,
    out_memory=80,
)

OBSERVED_PARAMS = SimParams(
    name="Observed (day 1 transcript)",
    conversations=_observed_conversations(),
    out_location=10,
    out_rating=5,
    out_speaking=77,   # actual avg ~77 across all transcripts
    out_memory=250,    # actual avg ~249 across days 2-3
)


# ── Token measurement ────────────────────────────────────────────────────

# Real dialogue from the Morning K&C conversation (transcript 2026-04-08_121900,
# day 1).  Used to measure shared-prefix growth per turn and history token
# counts.  The transcript interleaves parallel conversations under a single
# section header — these 20 lines are the K&C-only turns.
SAMPLE_HISTORY_LINES = [
    'Sable Dunmore: *wipes down the bar* Morning, everyone. Coffee\'s on, porridge is thick, and I\'ve got eggs if Ham\'s hens cooperated this week. *flashes a gold-toothed grin at Ham*',
    'Pell Arenway: *bursts through the door slightly out of breath, satchel sliding off one shoulder* Morning, morning — Ham, your hens might not have cooperated but I passed your goats on the way here and they were very enthusiastic about seeing me.',
    'Sable Dunmore: *pours a mug of coffee so dark it barely reflects light, slides it across to Pell* The brown one\'s name is Duchess, and she eats everything — letters, shirttails, hope. Don\'t take it personal.',
    'Ham Birch: *settles onto a stool with the unhurried weight of a man who\'s already been up four hours* Hens are fine. Got a dozen and a half in the crate outside. And Pell — Duchess doesn\'t even like me that much.',
    'Pell Arenway: *wraps both hands around the mug* Duchess loves me and you know it, Ham. She follows me the whole length of your fence line every time.',
    'Sable Dunmore: *cracks eggs onto the griddle with one hand* Orla\'s here, Pell, you don\'t need to go up anywhere. She\'s been nursing that same cup of tea in the corner for twenty minutes.',
    'Ham Birch: She doesn\'t follow you, Pell. She\'s guarding the fence line. There\'s a difference.',
    'Pell Arenway: *grins over the rim of the mug* I\'ll take that note for Wren, sure. And Ham — guarding, following, whatever you need to tell yourself.',
    'Ham Birch: *finishes a bite of eggs* Pell, I\'ve had that goat since she was born. She bit me last Thursday. She\'s never once bit you. That\'s not love, that\'s spite.',
    'Sable Dunmore: *laughs, a real one* Ham, that\'s the most poetic thing you\'ve ever said in my establishment. I\'m naming a dish after Duchess now.',
    'Pell Arenway: Spite is love, Ham. Ask anyone who\'s been married.',
    'Sable Dunmore: *snorts* Pell, you\'ve never been married a day in your life.',
    'Sable Dunmore: *leans on the bar* You don\'t need to be married to know about spite. You deliver enough other people\'s letters.',
    'Sable Dunmore: *slides a plate of eggs toward Ham without being asked* Pell, I\'ve got a note for Wren if you\'re passing her way later.',
    'Ham Birch: *pulls the plate closer, nods once at Sable in thanks, picks up a fork*',
    'Sable Dunmore: *sets the rag down* Denn. I\'m serious. You\'ve been sitting there like a fence post with feelings. What\'ll it be?',
    'Sable Dunmore: *drums fingers on the bar, pushes off and starts pouring a coffee without being asked* On the house. Now you owe me conversation.',
    'Sable Dunmore: *glances at the clock* Two silent people I can handle. Three and I start talking to myself.',
    'Sable Dunmore: *refills Ham\'s coffee without asking* You know what, fine. I\'ll take the silence. But I\'m putting The Quiet Table on the menu tonight.',
    'Sable Dunmore: *moves back toward center, lets the quiet sit for just a second* The Quiet Table — it\'s just cold soup and disappointment.',
]


def _make_fake_memories(day: int, count: int) -> list[str]:
    """Generate realistic-length fake memories for token measurement.

    Actual day-1 memories from the transcript average ~880 chars / ~224 tokens.
    The template below is calibrated to that length and mirrors the real structure:
    narrative summary → updated impressions → appointments.
    """
    template = (
        "[Day {d}, {p}] At The Kettle & Crow with Sable, Ham, and Pell. "
        "Sable made a lamb and root thing with Ham's parsnips and too much rosemary, "
        "but it worked because the parsnips were sweet enough to fight back. Ham "
        "revealed he'd left them in the ground two extra weeks. Margot spent half "
        "the evening threatening Denn about the drying rack spacing until his silence "
        "became a binding verbal contract.\n\n"
        "**Updated impressions:**\n"
        "- *Sable* — generous, sharp, gets away with too much rosemary. Named a dish "
        "after Ham's goat.\n"
        "- *Ham* — quietly extraordinary. Two extra weeks, said like it was nothing.\n"
        "- *Margot* — sharper than she lets on about herb sugar profiles.\n\n"
        "**Appointments:** Thursday morning — bring both honey jars to Sable's for "
        "the chamomile cake. Margot insists on smelling them side by side."
    )
    mems = []
    for d in range(1, day + 1):
        for p in ["morning", "afternoon", "evening"]:
            mems.append(template.format(d=d, p=p))
            if len(mems) >= count:
                return mems
    return mems


def measure_tokens(day: int, avg_turns: int):
    """Measure token sizes using real prompt templates.

    avg_turns controls how many history entries we sample for the shared
    prefix measurement.
    """
    villager = VILLAGERS[0]
    n_memories = max(0, (day - 1) * PHASES)
    fake_memories = _make_fake_memories(day, n_memories)
    villager_with_mem = Villager(
        name=villager.name,
        description=villager.description,
        memories=fake_memories,
    )

    sys_tokens = count_tokens(system_prompt())

    loc_user = location_choice_user(villager_with_mem, day, "morning", LOCATIONS)
    loc_user_tokens = count_tokens(loc_user)

    sample_lines = SAMPLE_HISTORY_LINES

    all_names = ["Pell Arenway", "Sable Dunmore", "Denn Corvale", "Orla Fenn", "Ham Birch"]
    location = LOCATIONS[0]

    shared_tokens_by_turn: list[int] = []
    for t in range(avg_turns):
        history = sample_lines[:t]
        shared_text = rating_shared(day, "morning", location, all_names, history)
        shared_tokens_by_turn.append(count_tokens(shared_text))

    full_history = sample_lines[:avg_turns]
    mem_user = memory_save_user(villager_with_mem, day, "morning", location, full_history)
    mem_user_tokens = count_tokens(mem_user)

    all_rating_tokens = []
    all_speaking_tokens = []
    for v in VILLAGERS:
        v_with_mem = Villager(name=v.name, description=v.description, memories=fake_memories)
        all_rating_tokens.append(count_tokens(rating_character(v_with_mem)))
        all_speaking_tokens.append(count_tokens(speaking_character(v_with_mem)))

    return {
        "sys_tokens": sys_tokens,
        "loc_user_tokens": loc_user_tokens,
        "shared_tokens_by_turn": shared_tokens_by_turn,
        "avg_shared": sum(shared_tokens_by_turn) // len(shared_tokens_by_turn),
        "rating_char_tokens": sum(all_rating_tokens) // len(all_rating_tokens),
        "speaking_char_tokens": sum(all_speaking_tokens) // len(all_speaking_tokens),
        "mem_user_tokens": mem_user_tokens,
        "avg_char_tokens": sum(all_rating_tokens) // len(all_rating_tokens),
        "min_char_tokens": min(all_rating_tokens),
        "max_char_tokens": max(all_rating_tokens),
    }


# ── Cost model ────────────────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    location: float = 0.0
    rating_input: float = 0.0
    rating_output: float = 0.0
    speaking_input: float = 0.0
    speaking_output: float = 0.0
    memory: float = 0.0

    @property
    def rating(self) -> float:
        return self.rating_input + self.rating_output

    @property
    def speaking(self) -> float:
        return self.speaking_input + self.speaking_output

    @property
    def total(self) -> float:
        return self.location + self.rating + self.speaking + self.memory


def compute_costs(
    m: dict,
    params: SimParams,
) -> tuple[CostBreakdown, CostBreakdown, CostBreakdown]:
    """Compute costs for (no_cache, current_scheme, char_first)."""
    nc = CostBreakdown()
    cs = CostBreakdown()
    cf = CostBreakdown()

    sys_tok = m["sys_tokens"]
    char_rating = m["avg_char_tokens"]
    char_speaking = m["speaking_char_tokens"]

    # ── Location selection: always 24 Sonnet calls, no caching ──
    loc_input = PHASES * N_VILLAGERS * (sys_tok + m["loc_user_tokens"])
    loc_output = PHASES * N_VILLAGERS * params.out_location
    loc_cost_val = cost(loc_input, SONNET.input) + cost(loc_output, SONNET.output)
    nc.location = cs.location = cf.location = loc_cost_val

    # ── Rating + Speaking: iterate conversation groups ──
    for group in params.conversations:
        k = group.size
        cpd = group.per_day
        turns = group.turns

        # We need shared_tokens_by_turn up to `turns` entries.
        # If our measurement has fewer, extrapolate linearly.
        shared_by_turn = m["shared_tokens_by_turn"]
        if turns > len(shared_by_turn):
            # Extrapolate: token growth per turn from measured data
            if len(shared_by_turn) >= 2:
                growth = (shared_by_turn[-1] - shared_by_turn[0]) / (len(shared_by_turn) - 1)
            else:
                growth = 40  # fallback ~40 tokens per turn
            shared_by_turn = list(shared_by_turn) + [
                shared_by_turn[-1] + int(growth * (i + 1))
                for i in range(turns - len(shared_by_turn))
            ]

        for t in range(turns):
            shared_tok = shared_by_turn[t]
            full_rating_in = sys_tok + shared_tok + char_rating
            full_speaking_in = sys_tok + shared_tok + char_speaking

            # Rating (Sonnet, k calls per turn)
            nc.rating_input += cpd * k * cost(full_rating_in, SONNET.input)

            cached_prefix = sys_tok + shared_tok
            # Cross-turn prefix sharing: Anthropic caches match the longest
            # common prefix.  At turn t>0 the first call's prefix overlaps
            # the previous turn's cached prefix; only the delta (one new
            # history line) is a cache write.  Calls 2..k hit the full
            # prefix cached by call 1 in this turn.
            if t == 0:
                prefix_read = 0
                prefix_write = cached_prefix
            else:
                prev_prefix = sys_tok + shared_by_turn[t - 1]
                prefix_read = prev_prefix
                prefix_write = cached_prefix - prev_prefix
            cs.rating_input += cpd * (
                cost(prefix_read, SONNET.cache_read)
                + cost(prefix_write, SONNET.cache_write)
                + (k - 1) * cost(cached_prefix, SONNET.cache_read)
                + k * cost(char_rating, SONNET.input)
            )

            cached_char = sys_tok + char_rating
            if t == 0:
                cf.rating_input += cpd * k * (
                    cost(cached_char, SONNET.cache_write)
                    + cost(shared_tok, SONNET.input)
                )
            else:
                cf.rating_input += cpd * k * (
                    cost(cached_char, SONNET.cache_read)
                    + cost(shared_tok, SONNET.input)
                )

            out_r = cpd * k * cost(params.out_rating, SONNET.output)
            nc.rating_output += out_r
            cs.rating_output += out_r
            cf.rating_output += out_r

            # Speaking (Opus, 1 call per turn)
            nc.speaking_input += cpd * cost(full_speaking_in, OPUS.input)

            # Only 1 Opus call per turn, so no cross-call prefix sharing.
            # The system prompt cache from Sonnet rating calls does not
            # carry over to Opus — caches are per-model.
            cs.speaking_input += cpd * cost(full_speaking_in, OPUS.input)

            cached_speak = sys_tok + char_speaking
            if t == 0:
                cf.speaking_input += cpd * (
                    cost(cached_speak, OPUS.cache_write)
                    + cost(shared_tok, OPUS.input)
                )
            else:
                cf.speaking_input += cpd * (
                    cost(cached_speak, OPUS.cache_read)
                    + cost(shared_tok, OPUS.input)
                )

            out_s = cpd * cost(params.out_speaking, OPUS.output)
            nc.speaking_output += out_s
            cs.speaking_output += out_s
            cf.speaking_output += out_s

    # ── Memory saves: Sonnet, no caching ──
    vill_in_conv = params.total_villagers_in_convs_per_day
    mem_total_in = vill_in_conv * (sys_tok + m["mem_user_tokens"])
    mem_total_out = vill_in_conv * params.out_memory
    mem_cost_val = cost(mem_total_in, SONNET.input) + cost(mem_total_out, SONNET.output)
    nc.memory = cs.memory = cf.memory = mem_cost_val

    return nc, cs, cf


# ── Reporting ─────────────────────────────────────────────────────────────

def pct(part: float, whole: float) -> str:
    if whole == 0:
        return "  -  "
    return f"{part / whole * 100:5.1f}%"


def fmt_dollar(v: float) -> str:
    return f"${v:.2f}"


def print_params_report(
    label: str,
    params: SimParams,
    m: dict,
    nc: CostBreakdown,
    cs: CostBreakdown,
    cf: CostBreakdown,
):
    avg_group = (params.total_villagers_in_convs_per_day
                 / params.total_convs_per_day)
    total_rating = sum(g.size * g.turns * g.per_day for g in params.conversations)
    total_speaking = sum(g.turns * g.per_day for g in params.conversations)

    print(f"\n{'─' * 72}")
    print(f"  {label}: {params.name}")
    print(f"{'─' * 72}")
    print(f"  Conversations/day:  {params.total_convs_per_day:>6.2f}      "
          f"Avg group size:  {avg_group:.2f}")
    print(f"  Rating calls:       {total_rating:>6.0f}      "
          f"Speaking calls:  {total_speaking:.0f}")
    print(f"  Memory save calls:  {params.total_villagers_in_convs_per_day:>6.0f}      "
          f"Output tok/speech: {params.out_speaking}")
    print(f"  Output tok/memory:  {params.out_memory:>6}")
    print()
    print(f"  {'':24s} {'No cache':>10s} {'Current':>10s} {'Char-first':>10s}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Location  (Sonnet)':24s} {fmt_dollar(nc.location):>10s} "
          f"{fmt_dollar(cs.location):>10s} {fmt_dollar(cf.location):>10s}")
    print(f"  {'Rating    (Sonnet)':24s} {fmt_dollar(nc.rating):>10s} "
          f"{fmt_dollar(cs.rating):>10s} {fmt_dollar(cf.rating):>10s}")
    print(f"  {'Speaking  (Opus)':24s} {fmt_dollar(nc.speaking):>10s} "
          f"{fmt_dollar(cs.speaking):>10s} {fmt_dollar(cf.speaking):>10s}")
    print(f"  {'Memory    (Sonnet)':24s} {fmt_dollar(nc.memory):>10s} "
          f"{fmt_dollar(cs.memory):>10s} {fmt_dollar(cf.memory):>10s}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'TOTAL':24s} {fmt_dollar(nc.total):>10s} "
          f"{fmt_dollar(cs.total):>10s} {fmt_dollar(cf.total):>10s}")
    print()
    cs_save = nc.total - cs.total
    cf_save = nc.total - cf.total
    print(f"  Current saves:     {fmt_dollar(cs_save):>7s}/day  ({pct(cs_save, nc.total)})")
    print(f"  Char-first saves:  {fmt_dollar(cf_save):>7s}/day  ({pct(cf_save, nc.total)})")


def print_component_breakdown(day: int, params: SimParams, max_turns: int):
    """Break each prompt type into its constituent sections and price them."""
    from prompts import _format_memories, _location_list

    villager = VILLAGERS[0]
    n_memories = max(0, (day - 1) * PHASES)
    fake_memories = _make_fake_memories(day, n_memories)
    v = Villager(name=villager.name, description=villager.description, memories=fake_memories)

    location = LOCATIONS[0]
    all_names = ["Pell Arenway", "Sable Dunmore", "Denn Corvale", "Orla Fenn", "Ham Birch"]

    sample_lines = SAMPLE_HISTORY_LINES

    # ── Measure each prompt component individually ──

    sys_text = system_prompt()
    desc_text = v.description
    desc_header = "Here is a description of your character:\n"
    desc_footer = "\nNever contradict it, but feel free to extend it where appropriate.\n\n"
    mem_header = "Here are your memories:\n"
    mem_text = _format_memories(v)

    sys_tok = count_tokens(sys_text)
    desc_tok = count_tokens(desc_header + desc_text + desc_footer)
    mem_tok = count_tokens(mem_header + mem_text)

    # Also measure across all 8 villagers for description range
    all_desc_toks = []
    for vx in VILLAGERS:
        vx2 = Villager(name=vx.name, description=vx.description, memories=fake_memories)
        all_desc_toks.append(count_tokens(desc_header + vx2.description + desc_footer))
    avg_desc_tok = sum(all_desc_toks) // len(all_desc_toks)

    # Rating components
    name_line = f"You are {v.name}.\n\n"
    name_tok = count_tokens(name_line)
    rating_instr = (
        "Output a single integer from 0 to 10 ranking how badly you want to speak "
        "right now. This will determine who speaks next. Alternatively, output -1 "
        "to leave the location and end your time here. Output one integer and "
        "nothing else."
    )
    rating_instr_tok = count_tokens(rating_instr)

    # Speaking components
    speaking_instr = (
        "It is your turn to speak now. Have a realistic conversation with those "
        "present. Keep it short, only a sentence or two, unless the other people "
        "expect you to say more. Aim for a grounded improvisational style — "
        "everything should be deeply realistic and a slow-burn."
    )
    speaking_instr_tok = count_tokens(speaking_instr)

    # Shared prefix components
    names_str = ", ".join(all_names)
    shared_header_text = (
        f"It is morning on day {day}.\n"
        f"You are currently at {location.name}.\n"
        f"The following villmagers are here: {names_str}.\n\n"
    )
    shared_header_tok = count_tokens(shared_header_text)

    # History at midpoint and endpoint
    mid_turn = max_turns // 2
    history_mid = sample_lines[:mid_turn]
    history_full = sample_lines[:max_turns]
    history_mid_text = "Here is the conversation so far:\n" + "\n".join(history_mid)
    history_full_text = "Here is the conversation so far:\n" + "\n".join(history_full)
    history_mid_tok = count_tokens(history_mid_text)
    history_full_tok = count_tokens(history_full_text)
    history_avg_tok = (count_tokens("Nobody has spoken yet.") + history_full_tok) // 2

    # Location choice components
    loc_instr = (
        f"It is morning on day {day}. Choose a location to go to.\n"
        "Only say the name of the location and nothing else.\n\n"
    )
    loc_instr_tok = count_tokens(loc_instr)
    loc_list_text = "Locations:\n" + _location_list(LOCATIONS)
    loc_list_tok = count_tokens(loc_list_text)

    # Memory save components
    mem_save_context = f"It is the end of the morning on day {day}. You were at {location.name}.\n\n"
    mem_save_context_tok = count_tokens(mem_save_context)
    mem_save_instr = (
        "Summarize the conversation into a couple sentences to recall for the "
        "future. Make sure to record future appointments (both day and time of day), and "
        "updated opinions on the villmagers you talked to."
    )
    mem_save_instr_tok = count_tokens(mem_save_instr)
    # The conversation history included in memory save is the full conversation
    mem_history_tok = history_full_tok

    # ── Call counts from params ──
    loc_calls = PHASES * N_VILLAGERS  # 24
    total_rating_calls = sum(g.size * g.turns * g.per_day for g in params.conversations)
    total_speaking_calls = sum(g.turns * g.per_day for g in params.conversations)
    total_mem_calls = params.total_villagers_in_convs_per_day
    avg_turns_weighted = sum(g.turns * g.per_day for g in params.conversations) / params.total_convs_per_day

    # ── Compute per-component costs ──
    # For a single call, cost of a component = tokens * rate/1M
    # For all calls of that type, multiply by call count

    def component_row(label, tokens, calls, pricing, indent="    "):
        total_tokens = tokens * calls
        dollars = cost(int(total_tokens), pricing)
        return (label, tokens, calls, total_tokens, dollars)

    rows = []
    grand_total = 0.0

    # -- Location choice (Sonnet) --
    rows.append(("HEADER", "LOCATION CHOICE (Sonnet)", "", "", "", ""))
    for label, tok in [
        ("system prompt", sys_tok),
        ("description + header", avg_desc_tok),
        ("memories", mem_tok),
        ("instruction", loc_instr_tok),
        ("location list", loc_list_tok),
    ]:
        r = component_row(label, tok, loc_calls, SONNET.input)
        rows.append(("INPUT", *r))
        grand_total += r[4]
    r = component_row("output", params.out_location, loc_calls, SONNET.output)
    rows.append(("OUTPUT", *r))
    grand_total += r[4]

    # -- Rating (Sonnet) --
    rows.append(("HEADER", "DESIRE RATING (Sonnet)", "", "", "", ""))
    rows.append(("NOTE", f"  {total_rating_calls:.0f} calls = "
                 f"{params.total_convs_per_day:.0f} convos x {avg_turns_weighted:.0f} turns x "
                 f"avg {params.total_villagers_in_convs_per_day/params.total_convs_per_day:.1f} villagers",
                 "", "", "", ""))
    for label, tok in [
        ("system prompt", sys_tok),
        ("shared: header (time/loc/names)", shared_header_tok),
        ("shared: conversation history", history_avg_tok),
        ("per-char: name line", name_tok),
        ("per-char: description + header", avg_desc_tok),
        ("per-char: memories", mem_tok),
        ("per-char: rating instruction", rating_instr_tok),
    ]:
        r = component_row(label, tok, total_rating_calls, SONNET.input)
        rows.append(("INPUT", *r))
        grand_total += r[4]
    r = component_row("output", params.out_rating, total_rating_calls, SONNET.output)
    rows.append(("OUTPUT", *r))
    grand_total += r[4]

    # -- Speaking (Opus) --
    rows.append(("HEADER", "SPEAKING (Opus)", "", "", "", ""))
    rows.append(("NOTE", f"  {total_speaking_calls:.0f} calls = "
                 f"{params.total_convs_per_day:.0f} convos x {avg_turns_weighted:.0f} turns x 1 speaker",
                 "", "", "", ""))
    for label, tok in [
        ("system prompt", sys_tok),
        ("shared: header (time/loc/names)", shared_header_tok),
        ("shared: conversation history", history_avg_tok),
        ("per-char: name line", name_tok),
        ("per-char: description + header", avg_desc_tok),
        ("per-char: memories", mem_tok),
        ("per-char: speaking instruction", speaking_instr_tok),
    ]:
        r = component_row(label, tok, total_speaking_calls, OPUS.input)
        rows.append(("INPUT", *r))
        grand_total += r[4]
    r = component_row("output", params.out_speaking, total_speaking_calls, OPUS.output)
    rows.append(("OUTPUT", *r))
    grand_total += r[4]

    # -- Memory save (Sonnet) --
    rows.append(("HEADER", "MEMORY SAVE (Sonnet)", "", "", "", ""))
    for label, tok in [
        ("system prompt", sys_tok),
        ("description + header", avg_desc_tok),
        ("current memories", mem_tok),
        ("phase/location context", mem_save_context_tok),
        ("conversation history", mem_history_tok),
        ("instruction", mem_save_instr_tok),
    ]:
        r = component_row(label, tok, total_mem_calls, SONNET.input)
        rows.append(("INPUT", *r))
        grand_total += r[4]
    r = component_row("output", params.out_memory, total_mem_calls, SONNET.output)
    rows.append(("OUTPUT", *r))
    grand_total += r[4]

    # ── Print ──
    print(f"\n{'=' * 84}")
    print(f"  PROMPT COMPONENT BREAKDOWN — Day {day}, {params.name}")
    print(f"  Memories per villager: ~{n_memories}")
    print(f"{'=' * 84}")
    print()
    print(f"  {'Component':<36s} {'tok/call':>8s} {'calls':>7s} {'total tok':>10s} {'cost':>10s} {'% total':>8s}")
    print(f"  {'─'*36} {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*8}")

    for row in rows:
        if row[0] == "HEADER":
            print(f"\n  {row[1]}")
            continue
        if row[0] == "NOTE":
            print(f"  {row[1]}")
            continue
        kind, label, tok, calls, total_tok, dollars = row
        pct_str = f"{dollars/grand_total*100:6.1f}%" if grand_total > 0 else ""
        if kind == "OUTPUT":
            print(f"    {'→ ' + label:<34s} {tok:>8} {calls:>7.0f} {total_tok:>10.0f} ${dollars:>8.4f} {pct_str:>8s}")
        else:
            print(f"    {label:<36s} {tok:>8} {calls:>7.0f} {total_tok:>10.0f} ${dollars:>8.4f} {pct_str:>8s}")

    print(f"\n  {'─'*36} {'─'*8} {'─'*7} {'─'*10} {'─'*10} {'─'*8}")
    print(f"  {'GRAND TOTAL':<36s} {'':>8s} {'':>7s} {'':>10s} ${grand_total:>8.4f} {'100.0%':>8s}")

    # ── Aggregate by component type across all call types ──
    agg: dict[str, float] = {}
    for row in rows:
        if row[0] in ("HEADER", "NOTE"):
            continue
        _, label, _, _, _, dollars = row
        # Normalize label
        if "system prompt" in label:
            key = "System prompt"
        elif "description" in label:
            key = "Character description"
        elif "memories" in label or "current memories" in label:
            key = "Memories"
        elif "conversation history" in label:
            key = "Conversation history"
        elif "instruction" in label:
            key = "Instruction text"
        elif "header" in label and "shared" in label:
            key = "Shared header (time/loc/names)"
        elif "location list" in label:
            key = "Location list"
        elif "context" in label:
            key = "Phase/location context"
        elif "name line" in label:
            key = "Name line"
        elif "output" in label:
            key = "Output tokens"
        else:
            key = label
        agg[key] = agg.get(key, 0) + dollars

    print(f"\n  {'─'*84}")
    print(f"  AGGREGATED BY COMPONENT (across all call types)")
    print(f"  {'─'*84}")
    for key, dollars in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"    {key:<36s} ${dollars:>8.4f}  {pct(dollars, grand_total):>8s}")
    print(f"    {'─'*36} {'─'*10} {'─'*8}")
    print(f"    {'TOTAL':<36s} ${grand_total:>8.4f}  {'100.0%':>8s}")


def print_report(day: int):
    # Use the max turns from any param set for token measurement
    max_turns_needed = max(
        max(g.turns for g in p.conversations)
        for p in [UNIFORM_PARAMS, OBSERVED_PARAMS]
    )
    m = measure_tokens(day, max_turns_needed)

    print("=" * 72)
    print(f"  VILLMAGE COST ANALYSIS — Day {day}")
    print(f"  Memories per villager: ~{max(0, (day - 1) * PHASES)}")
    print("=" * 72)

    # Token measurements
    print(f"\n  Token measurements (from actual prompts + descriptions):")
    print(f"    System prompt:        {m['sys_tokens']:>4} tokens")
    print(f"    Location choice user: {m['loc_user_tokens']:>4} tokens")
    print(f"    Shared prefix (avg):  {m['avg_shared']:>4} tokens "
          f"(turn 1: {m['shared_tokens_by_turn'][0]}, "
          f"turn {max_turns_needed}: {m['shared_tokens_by_turn'][-1]})")
    print(f"    Per-char rating:      {m['avg_char_tokens']:>4} tokens "
          f"(range: {m['min_char_tokens']}–{m['max_char_tokens']})")
    print(f"    Per-char speaking:    {m['speaking_char_tokens']:>4} tokens")
    print(f"    Memory save user:     {m['mem_user_tokens']:>4} tokens")

    # Component breakdown for observed params
    print_component_breakdown(day, OBSERVED_PARAMS, max_turns_needed)

    # Both parameter sets
    for params in [UNIFORM_PARAMS, OBSERVED_PARAMS]:
        nc, cs, cf = compute_costs(m, params)
        print_params_report(
            "MODEL" if params is UNIFORM_PARAMS else "ACTUAL",
            params, m, nc, cs, cf,
        )

    # Side-by-side delta
    nc_u, cs_u, cf_u = compute_costs(m, UNIFORM_PARAMS)
    nc_o, cs_o, cf_o = compute_costs(m, OBSERVED_PARAMS)
    print(f"\n{'─' * 72}")
    print(f"  Uniform vs Observed (no-cache baseline)")
    print(f"{'─' * 72}")
    print(f"  {'':24s} {'Uniform':>10s} {'Observed':>10s} {'Ratio':>10s}")
    print(f"  {'─'*24} {'─'*10} {'─'*10} {'─'*10}")
    for label, u, o in [
        ("Total (no cache)", nc_u.total, nc_o.total),
        ("Total (current)", cs_u.total, cs_o.total),
        ("Total (char-first)", cf_u.total, cf_o.total),
    ]:
        ratio = o / u if u > 0 else 0
        print(f"  {label:24s} {fmt_dollar(u):>10s} {fmt_dollar(o):>10s} {ratio:>9.2f}x")

    # What drove the difference
    print(f"\n  Key assumption mismatches (day 1 actual vs uniform model):")
    u_convs = UNIFORM_PARAMS.total_convs_per_day
    o_convs = OBSERVED_PARAMS.total_convs_per_day
    u_avg_group = UNIFORM_PARAMS.total_villagers_in_convs_per_day / u_convs
    o_avg_group = OBSERVED_PARAMS.total_villagers_in_convs_per_day / o_convs
    u_avg_turns = sum(g.turns * g.per_day for g in UNIFORM_PARAMS.conversations) / u_convs
    o_avg_turns = sum(g.turns * g.per_day for g in OBSERVED_PARAMS.conversations) / o_convs
    print(f"    {'':30s} {'Uniform':>10s} {'Observed':>10s}")
    print(f"    {'Conversations/day':30s} {u_convs:>10.1f} {o_convs:>10.0f}")
    print(f"    {'Avg group size':30s} {u_avg_group:>10.2f} {o_avg_group:>10.2f}")
    print(f"    {'Avg turns/conversation':30s} {u_avg_turns:>10.1f} {o_avg_turns:>10.1f}")
    print(f"    {'Speech output tokens':30s} "
          f"{UNIFORM_PARAMS.out_speaking:>10} {OBSERVED_PARAMS.out_speaking:>10}")
    print(f"    {'Memory output tokens':30s} "
          f"{UNIFORM_PARAMS.out_memory:>10} {OBSERVED_PARAMS.out_memory:>10}")
    print(f"    {'Location distribution':30s} {'uniform':>10s} {'K&C=54%':>10s}")

    # Multi-day projection
    print(f"\n{'─' * 72}")
    print(f"  Multi-day projection (observed params, no cache)")
    print(f"{'─' * 72}")
    print(f"  {'Day':>5s}  {'No cache':>10s}  {'Current':>10s}  {'Char-first':>10s}")
    for d in [1, 3, 5, 10]:
        md = measure_tokens(d, max_turns_needed)
        n, c, f = compute_costs(md, OBSERVED_PARAMS)
        print(f"  {d:>5d}  {fmt_dollar(n.total):>10s}  {fmt_dollar(c.total):>10s}  {fmt_dollar(f.total):>10s}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Villmage caching cost analysis")
    parser.add_argument("--day", type=int, default=1,
                        help="Simulation day (affects memory count)")
    args = parser.parse_args()
    print_report(args.day)
