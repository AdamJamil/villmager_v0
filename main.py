"""Entry point for the Villmage simulation."""

import click

from dotenv import load_dotenv
load_dotenv()

import trio

from data import VILLAGERS, LOCATIONS
from simulation import run_phase
from output import Output

PHASES = ["morning", "afternoon", "evening"]


async def run_simulation(days: int):
    out = Output()
    try:
        for day in range(1, days + 1):
            out.day_start(day)
            location_tracker: dict[tuple[str, str], str] = {}

            for phase in PHASES:
                await run_phase(VILLAGERS, LOCATIONS, day, phase, out, location_tracker)

            out.day_end(day, VILLAGERS, location_tracker)
    finally:
        out.close()


@click.command()
@click.option("--days", default=1, type=int, help="Number of days to simulate (default: 1)")
def main(days: int):
    trio.run(run_simulation, days)


if __name__ == "__main__":
    main()
