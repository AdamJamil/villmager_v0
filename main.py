"""Entry point for the Villmage simulation."""

import argparse
from typing import Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

import trio

from data import VILLAGERS, LOCATIONS
from simulation import run_phase
from output import Output

import traceback

PHASES = ["morning", "afternoon", "evening"]


async def run_simulation(days: int):
    out = Output()
    day: int
    phase: str
    try:
        for day in range(1, days + 1):
            out.day_start(day)
            location_tracker: Dict[Tuple[str, str], str] = {}

            for phase in PHASES:
                await run_phase(VILLAGERS, LOCATIONS, day, phase, out, location_tracker)

            out.day_end(day, VILLAGERS, location_tracker)
    except Exception as e:
        print(f"Failed on day {day} in the {phase} with excpetion:")
        print(traceback.format_exception(e))
    finally:
        out.close()


def main():
    parser = argparse.ArgumentParser(description="Villmage — a village life simulation")
    parser.add_argument("--days", type=int, default=1, help="Number of days to simulate (default: 1)")
    args = parser.parse_args()
    trio.run(run_simulation, args.days)


if __name__ == "__main__":
    main()
