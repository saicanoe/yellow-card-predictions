import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "lineups_raw.txt"
OUTPUT_PATH = DATA_DIR / "lineups.csv"
OUTPUT_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Team",
    "Player",
    "Position",
    "Starter",
]


def normalize_whitespace(value: str) -> str:
    """Collapse repeated whitespace and trim a pasted text value."""
    return re.sub(r"\s+", " ", value.strip())


def parse_match_line(value: str) -> tuple[str, str]:
    """Parse MATCH: Home vs Away into home and away team names."""
    teams = re.split(r"\s+vs\s+", value, maxsplit=1, flags=re.IGNORECASE)
    if len(teams) != 2:
        raise ValueError(f"Invalid MATCH line: {value}")
    return normalize_whitespace(teams[0]), normalize_whitespace(teams[1])


def parse_player_line(line: str) -> tuple[str, str]:
    """Parse Player, Position rows."""
    if "," not in line:
        raise ValueError(f"Invalid player line, expected 'Player, Position': {line}")
    player, position = line.split(",", 1)
    return normalize_whitespace(player), normalize_whitespace(position).upper()


def parse_raw_lineups(raw_text: str) -> list[dict]:
    """
    Parse data/lineups_raw.txt into data/lineups.csv rows.

    Expected paste format:
    MATCH: Brentford vs West Ham
    DATE: 2026-05-04
    TEAM: Brentford
    Nathan Collins, CB
    Christian Norgaard, DM
    TEAM: West Ham
    Tomas Soucek, CM
    """
    rows = []
    home_team = ""
    away_team = ""
    match_date = ""
    current_team = ""

    for line_number, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = normalize_whitespace(raw_line)
        if not line or line.startswith("#"):
            continue

        upper_line = line.upper()
        if upper_line.startswith("MATCH:"):
            home_team, away_team = parse_match_line(line.split(":", 1)[1])
            current_team = ""
            continue

        if upper_line.startswith("DATE:"):
            match_date = normalize_whitespace(line.split(":", 1)[1])
            continue

        if upper_line.startswith("TEAM:"):
            current_team = normalize_whitespace(line.split(":", 1)[1])
            continue

        if not home_team or not away_team or not match_date or not current_team:
            raise ValueError(
                f"Player row before MATCH, DATE, and TEAM context at line {line_number}: {line}"
            )

        player, position = parse_player_line(line)
        rows.append(
            {
                "Date": match_date,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Team": current_team,
                "Player": player,
                "Position": position,
                "Starter": "TRUE",
            }
        )

    return rows


def import_lineups() -> list[dict]:
    """Read raw pasted lineups and write normalized CSV output."""
    if not RAW_PATH.exists():
        RAW_PATH.write_text(
            "# Paste lineups here using MATCH/DATE/TEAM headers.\n",
            encoding="utf-8",
        )

    rows = parse_raw_lineups(RAW_PATH.read_text(encoding="utf-8-sig"))
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    rows = import_lineups()
    print(f"Imported lineup count: {len(rows)}")
    print("Saved:")
    print("data/lineups.csv")


if __name__ == "__main__":
    main()
