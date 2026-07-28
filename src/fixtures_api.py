import csv
import io
import sys
from datetime import date, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = DATA_DIR / "upcoming_fixtures.csv"
OUTPUT_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "Div"]
FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"


def season_code(today: date) -> str:
    """Return football-data.co.uk's season code, such as 2526."""
    start_year = today.year if today.month >= 7 else today.year - 1
    end_year = start_year + 1
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def parse_match_date(value: str):
    """Parse common football-data date formats."""
    value = str(value).strip()
    for date_format in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    return None


def fetch_football_data_epl_csv(today: date) -> str:
    """Fetch the current-season EPL CSV from football-data.co.uk."""
    url = f"{FOOTBALL_DATA_BASE_URL}/{season_code(today)}/E0.csv"
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8-sig")


def parse_upcoming_epl_fixtures(csv_text: str, today: date) -> list[dict]:
    """Extract future EPL fixtures and normalize to CardCast's fixture schema."""
    rows = []
    seen = set()
    reader = csv.DictReader(io.StringIO(csv_text))

    for row in reader:
        match_date = parse_match_date(row.get("Date", ""))
        home_team = str(row.get("HomeTeam", "")).strip()
        away_team = str(row.get("AwayTeam", "")).strip()
        division = str(row.get("Div", "E0")).strip() or "E0"

        if division != "E0" or not match_date or match_date < today:
            continue
        if not home_team or not away_team:
            continue

        key = (match_date.isoformat(), home_team, away_team)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "Date": match_date.strftime("%d/%m/%Y"),
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "Div": "E0",
            }
        )

    return sorted(rows, key=lambda item: (item["Date"], item["HomeTeam"], item["AwayTeam"]))


def write_fixtures(rows: list[dict]):
    """Write normalized upcoming fixtures to data/upcoming_fixtures.csv."""
    DATA_DIR.mkdir(exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_fixture_rows(today: date) -> list[dict]:
    """Keep existing future EPL fixtures if the remote fetch is unavailable."""
    if not OUTPUT_PATH.exists():
        return []

    with OUTPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        rows = []
        seen = set()
        for row in reader:
            match_date = parse_match_date(row.get("Date", ""))
            if row.get("Div", "").strip() != "E0" or not match_date or match_date < today:
                continue
            home_team = str(row.get("HomeTeam", "")).strip()
            away_team = str(row.get("AwayTeam", "")).strip()
            if not home_team or not away_team:
                continue
            key = (match_date.isoformat(), home_team, away_team)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Date": match_date.strftime("%d/%m/%Y"),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "Div": "E0",
                }
            )
    return rows


def load_existing_rows_without_date_filter() -> list[dict]:
    """Read the existing fixture file as a last-resort preservation fallback."""
    if not OUTPUT_PATH.exists():
        return []

    with OUTPUT_PATH.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        rows = []
        seen = set()
        for row in reader:
            match_date = parse_match_date(row.get("Date", ""))
            home_team = str(row.get("HomeTeam", "")).strip()
            away_team = str(row.get("AwayTeam", "")).strip()
            division = str(row.get("Div", "")).strip()
            if division != "E0" or not match_date or not home_team or not away_team:
                continue
            key = (match_date.isoformat(), home_team, away_team)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "Date": match_date.strftime("%d/%m/%Y"),
                    "HomeTeam": home_team,
                    "AwayTeam": away_team,
                    "Div": "E0",
                }
            )
    return rows


def import_upcoming_fixtures(today: date | None = None) -> list[dict]:
    """Fetch, normalize, and save upcoming EPL fixtures."""
    today = today or date.today()
    try:
        previous_rows = load_existing_rows_without_date_filter()
        csv_text = fetch_football_data_epl_csv(today)
        rows = parse_upcoming_epl_fixtures(csv_text, today)
        if not rows and previous_rows:
            print("Fixture source returned 0 future EPL rows; preserving existing CSV.")
            rows = previous_rows
        write_fixtures(rows)
        print("Fixture source: football-data.co.uk")
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        rows = load_existing_fixture_rows(today)
        if rows:
            print(f"Fixture source: existing CSV fallback ({error})")
        else:
            print(f"Fixture fetch failed and no existing future EPL fixtures found: {error}")
            write_fixtures(rows)

    print(f"Imported fixture count: {len(rows)}")
    print("Saved:")
    print("data/upcoming_fixtures.csv")
    return rows


def main():
    import_upcoming_fixtures()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Fixture import failed: {error}", file=sys.stderr)
        sys.exit(1)
