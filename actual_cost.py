"""Reconstruct the exact day 1 cost from the transcript.

Replays every API call that simulation.py would have made, using the
actual character descriptions, the actual speeches from the transcript,
and the actual memory saves — building prompts exactly as the code does
and pricing each one.

Run:  python actual_cost.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import VILLAGERS, LOCATIONS
from prompts import (
    system_prompt,
    location_choice_user,
    rating_shared,
    rating_character,
    speaking_shared,
    speaking_character,
    memory_save_user,
)

from cost_analysis.pricing import count_tokens as tok, cost as price, SONNET, OPUS
from cost_analysis.transcript import (
    parse_speeches as _parse_speeches,
    parse_memories as _parse_memories,
    parse_assignments,
)

# Pricing shortcuts for backward compat
SONNET_IN = SONNET.input
SONNET_OUT = SONNET.output
SONNET_CW = SONNET.cache_write
SONNET_CR = SONNET.cache_read
OPUS_IN = OPUS.input
OPUS_OUT = OPUS.output
OPUS_CW = OPUS.cache_write
OPUS_CR = OPUS.cache_read

# ── Parse transcript ──────────────────────────────────────────────────
TRANSCRIPT = Path("transcripts/2026-04-08_121900/day_1.md").read_text()

# Villager lookup
V = {v.name: v for v in VILLAGERS}
L = {loc.name: loc for loc in LOCATIONS}
SYS = system_prompt()
SYS_TOK = tok(SYS)

# Location assignments parsed from transcript
ASSIGNMENTS = parse_assignments(TRANSCRIPT)


def parse_speeches():
    return _parse_speeches(TRANSCRIPT, ASSIGNMENTS)


def parse_memories():
    return _parse_memories(TRANSCRIPT)

# ── Replay simulation ────────────────────────────────────────────────

def main():
    convos = parse_speeches()
    all_memories = parse_memories()
    mem_iter = iter(all_memories)

    # Track each villager's memory state (starts empty)
    memories = {name: [] for name in V}

    # Accumulate costs per call
    calls = []  # list of dicts

    def log(call_type, model, phase, location, villager,
            sys_tokens, user_tokens, out_tokens, cached_tokens=0, turn=None):
        calls.append({
            "type": call_type,
            "model": model,
            "phase": phase,
            "location": location,
            "villager": villager,
            "turn": turn,
            "sys_tok": sys_tokens,
            "user_tok": user_tokens,
            "out_tok": out_tokens,
            "cached_tok": cached_tokens,
        })

    for phase in ["morning", "afternoon", "evening"]:
        # ── 1. Location selection: 8 Sonnet calls ──
        for name, villager in V.items():
            user_msg = location_choice_user(villager, 1, phase, LOCATIONS)
            log("location", "sonnet", phase, "", name,
                SYS_TOK, tok(user_msg), 10)

        # ── 2. Solo villagers: no LLM call, just append memory ──
        for loc_name, names in ASSIGNMENTS[phase].items():
            if len(names) == 1:
                solo = names[0]
                memories[solo].append(f"[Day 1, {phase}] Nobody else appeared at {loc_name}.")

        # ── 3. Conversations ──
        for loc_name, names in ASSIGNMENTS[phase].items():
            if len(names) < 2:
                continue

            location = L[loc_name]
            present = list(names)
            history = []
            turn_speeches = convos.get((phase, loc_name), [])

            for turn_idx, (speaker, score, speech_text) in enumerate(turn_speeches):
                turn_num = turn_idx + 1

                # 3a. Rating calls — all present villagers
                shared = rating_shared(1, phase, location, present, history)
                shared_tok = tok(shared)

                for name in present:
                    char = rating_character(V[name])
                    char_tok = tok(char)
                    # Current caching: shared prefix is cached across
                    # parallel calls within this turn
                    log("rating", "sonnet", phase, loc_name, name,
                        SYS_TOK, shared_tok + char_tok, 5,
                        cached_tokens=SYS_TOK + shared_tok,
                        turn=turn_num)

                # 3b. Speaking call — Opus, 1 call
                speak_shared = speaking_shared(
                    1, phase, location, present, history)
                speak_char = speaking_character(V[speaker])
                speak_shared_tok = tok(speak_shared)
                speak_char_tok = tok(speak_char)
                speech_out_tok = tok(speech_text)

                log("speaking", "opus", phase, loc_name, speaker,
                    SYS_TOK, speak_shared_tok + speak_char_tok,
                    speech_out_tok,
                    cached_tokens=SYS_TOK,
                    turn=turn_num)

                # 3c. Append to history
                history.append(f"{speaker}: {speech_text}")

            # ── 4. Memory saves ──
            for name in names:
                user_msg = memory_save_user(V[name], 1, phase, location, history)
                # Get the actual memory text from transcript
                try:
                    mem_name, mem_text = next(mem_iter)
                    mem_out_tok = tok(mem_text)
                except StopIteration:
                    mem_out_tok = 200  # fallback

                log("memory", "sonnet", phase, loc_name, name,
                    SYS_TOK, tok(user_msg), mem_out_tok)

                memories[name].append(f"[Day 1, {phase}] {mem_text[:200]}")

        # Update villager memory state for next phase
        for name, mems in memories.items():
            V[name].memories = list(mems)

    # ── Compute costs ────────────────────────────────────────────────

    # No-cache cost: every token at full input price
    # Current-cache cost: shared prefix at cache rates within a turn
    for c in calls:
        inp = c["sys_tok"] + c["user_tok"]
        out = c["out_tok"]
        if c["model"] == "sonnet":
            c["cost_nocache"] = price(inp, SONNET_IN) + price(out, SONNET_OUT)
            # Current scheme: cached_tokens get write/read treatment
            cached = c["cached_tok"]
            uncached = inp - cached
            # First call in a turn pays write, rest pay read
            # We mark this per-call; we'll adjust below
            c["cost_current_raw_write"] = (
                price(cached, SONNET_CW) + price(uncached, SONNET_IN)
                + price(out, SONNET_OUT)
            )
            c["cost_current_raw_read"] = (
                price(cached, SONNET_CR) + price(uncached, SONNET_IN)
                + price(out, SONNET_OUT)
            )
        else:  # opus
            c["cost_nocache"] = price(inp, OPUS_IN) + price(out, OPUS_OUT)
            cached = c["cached_tok"]
            uncached = inp - cached
            c["cost_current_raw_write"] = (
                price(cached, OPUS_CW) + price(uncached, OPUS_IN)
                + price(out, OPUS_OUT)
            )
            c["cost_current_raw_read"] = (
                price(cached, OPUS_CR) + price(uncached, OPUS_IN)
                + price(out, OPUS_OUT)
            )

    # For "current" caching: within each (phase, location, turn),
    # the first rating call pays write, the rest pay read.
    # Speaking calls: only sys cached (1 call, always write on sys).
    # Location + memory: no caching.
    for c in calls:
        if c["type"] == "rating":
            # Check if this is the first rating call for this turn
            key = (c["phase"], c["location"], c["turn"])
            first = not any(
                prev["type"] == "rating"
                and (prev["phase"], prev["location"], prev["turn"]) == key
                and prev is not c
                and calls.index(prev) < calls.index(c)
                for prev in calls
            )
            c["cost_current"] = c["cost_current_raw_write"] if first else c["cost_current_raw_read"]
        elif c["type"] == "speaking":
            c["cost_current"] = c["cost_current_raw_write"]
        else:
            c["cost_current"] = c["cost_nocache"]

    # ── Print results ────────────────────────────────────────────────

    print("=" * 90)
    print("  ACTUAL DAY 1 COST — Reconstructed from transcript")
    print("=" * 90)

    # Summary by type
    print(f"\n  {'Call type':<20s} {'Count':>6s} {'Avg in':>8s} {'Avg out':>8s} "
          f"{'No cache':>10s} {'Current':>10s}")
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")

    for ctype in ["location", "rating", "speaking", "memory"]:
        subset = [c for c in calls if c["type"] == ctype]
        if not subset:
            continue
        count = len(subset)
        avg_in = sum(c["sys_tok"] + c["user_tok"] for c in subset) / count
        avg_out = sum(c["out_tok"] for c in subset) / count
        total_nc = sum(c["cost_nocache"] for c in subset)
        total_cur = sum(c["cost_current"] for c in subset)
        model = subset[0]["model"]
        label = f"{ctype} ({model})"
        print(f"  {label:<20s} {count:>6d} {avg_in:>8.0f} {avg_out:>8.0f} "
              f"${total_nc:>9.4f} ${total_cur:>9.4f}")

    total_nc = sum(c["cost_nocache"] for c in calls)
    total_cur = sum(c["cost_current"] for c in calls)
    count = len(calls)
    print(f"  {'─'*20} {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    print(f"  {'TOTAL':<20s} {count:>6d} {'':>8s} {'':>8s} "
          f"${total_nc:>9.4f} ${total_cur:>9.4f}")
    print(f"\n  Current scheme saves ${total_nc - total_cur:.4f} "
          f"({(total_nc - total_cur)/total_nc*100:.1f}%)")

    # ── Per-phase breakdown ──
    print(f"\n  {'Phase':<12s} {'Location':<28s} {'Vill':>4s} {'Turns':>5s} "
          f"{'Rating':>8s} {'Speaking':>8s} {'Memory':>8s} {'Total':>8s}")
    print(f"  {'─'*12} {'─'*28} {'─'*4} {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for phase in ["morning", "afternoon", "evening"]:
        # Location selection cost for this phase
        loc_cost = sum(c["cost_nocache"] for c in calls
                       if c["type"] == "location" and c["phase"] == phase)

        for loc_name, names in ASSIGNMENTS[phase].items():
            if len(names) < 2:
                continue
            subset = [c for c in calls
                      if c["phase"] == phase and c["location"] == loc_name]
            rating_cost = sum(c["cost_nocache"] for c in subset if c["type"] == "rating")
            speak_cost = sum(c["cost_nocache"] for c in subset if c["type"] == "speaking")
            mem_cost = sum(c["cost_nocache"] for c in subset if c["type"] == "memory")
            turns = max((c["turn"] or 0) for c in subset if c["turn"])
            total = rating_cost + speak_cost + mem_cost
            print(f"  {phase:<12s} {loc_name:<28s} {len(names):>4d} {turns:>5d} "
                  f"${rating_cost:>7.4f} ${speak_cost:>7.4f} ${mem_cost:>7.4f} ${total:>7.4f}")

        print(f"  {phase:<12s} {'location selection':<28s} {'8':>4s} {'':>5s} "
              f"{'':>8s} {'':>8s} {'':>8s} ${loc_cost:>7.4f}")

    # ── Token growth through the longest conversation ──
    print(f"\n  Token growth: Morning K&C (5 villagers, longest conversation)")
    print(f"  {'Turn':>6s} {'Shared':>8s} {'Char(avg)':>10s} {'Total/rating':>13s} "
          f"{'Speak out':>10s} {'Turn cost':>10s} {'Cumul':>10s}")
    print(f"  {'─'*6} {'─'*8} {'─'*10} {'─'*13} {'─'*10} {'─'*10} {'─'*10}")

    morning_kc = [c for c in calls
                  if c["phase"] == "morning" and c["location"] == "The Kettle & Crow"]
    cumul = 0
    for t in range(1, 21):
        turn_calls = [c for c in morning_kc if c["turn"] == t]
        if not turn_calls:
            break
        ratings = [c for c in turn_calls if c["type"] == "rating"]
        speaks = [c for c in turn_calls if c["type"] == "speaking"]
        if not ratings:
            break

        # The shared prefix is the same for all rating calls in this turn
        # It's the cached portion minus sys
        shared = ratings[0]["cached_tok"] - SYS_TOK
        avg_char = sum(c["user_tok"] - shared for c in ratings) // len(ratings)
        total_per_rating = SYS_TOK + shared + avg_char
        speak_out = speaks[0]["out_tok"] if speaks else 0

        turn_cost = sum(c["cost_nocache"] for c in turn_calls)
        cumul += turn_cost

        print(f"  {t:>6d} {shared:>8d} {avg_char:>10d} {total_per_rating:>13d} "
              f"{speak_out:>10d} ${turn_cost:>9.4f} ${cumul:>9.4f}")

    # ── Input vs output cost split ──
    print(f"\n  Input vs output cost (no cache):")
    total_in_cost = 0
    total_out_cost = 0
    for c in calls:
        inp = c["sys_tok"] + c["user_tok"]
        out = c["out_tok"]
        if c["model"] == "sonnet":
            total_in_cost += price(inp, SONNET_IN)
            total_out_cost += price(out, SONNET_OUT)
        else:
            total_in_cost += price(inp, OPUS_IN)
            total_out_cost += price(out, OPUS_OUT)
    print(f"    Input:  ${total_in_cost:.4f}  ({total_in_cost/total_nc*100:.0f}%)")
    print(f"    Output: ${total_out_cost:.4f}  ({total_out_cost/total_nc*100:.0f}%)")
    print(f"    Total:  ${total_nc:.4f}")

    # ── Total tokens ──
    total_in_tok = sum(c["sys_tok"] + c["user_tok"] for c in calls)
    total_out_tok = sum(c["out_tok"] for c in calls)
    print(f"\n  Total tokens:")
    print(f"    Input:  {total_in_tok:>10,d}")
    print(f"    Output: {total_out_tok:>10,d}")
    print(f"    Total:  {total_in_tok + total_out_tok:>10,d}")


if __name__ == "__main__":
    main()
