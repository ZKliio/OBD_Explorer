import sqlite3
import json

DB_PATH = "../OBD.db"

def parse_inner_json(val):
    """Safely parse a JSON string from the DB field."""
    if not val:
        return {}
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return {}

def score_match(target, candidate):
    """
    Score a candidate row against the target JSON.
    Higher score = better match.
    Priority order:
    1. response_id (3 points)
    2. request_id (2 points)
    3. command (1 point)
    4. transmit_message (1 point)
    """
    if not target:  # No target JSON provided → score 0
        return 0

    score = 0
    if target.get("response_id") and target.get("response_id") == candidate.get("response_id"):
        score += 3
    if target.get("request_id") and target.get("request_id") == candidate.get("request_id"):
        score += 2
    if target.get("command") and target.get("command") == candidate.get("command"):
        score += 1
    if target.get("transmit_message") and target.get("transmit_message") == candidate.get("transmit_message"):
        score += 1
    return score

def find_closest_matches(manufacturer, model=None, search_field="Pack_SOC", target_json_str=None, top_n=5):
    """
    Find closest matches in the DB for the given manufacturer (required),
    optional model, and optional target JSON in the specified search_field.
    If target_json_str is None or empty, returns all matches with score=0.
    """
    if not manufacturer:
        raise ValueError("manufacturer is required")

    target_json = parse_inner_json(target_json_str) if target_json_str else {}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Build query with optional model filter
    if model:
        cur.execute("""
            SELECT * FROM vehicle_pack_commands
            WHERE manufacturer = ? AND model = ?
        """, (manufacturer, model))
    else:
        cur.execute("""
            SELECT * FROM vehicle_pack_commands
            WHERE manufacturer = ?
        """, (manufacturer,))

    rows = cur.fetchall()
    conn.close()

    matches = []
    for row in rows:
        id, manu, mod, year, country_region, type_, Pack_Voltage, Pack_SOC, Pack_SOH = row

        # Pick the field to compare
        field_map = {
            "Pack_Voltage": Pack_Voltage,
            "Pack_SOC": Pack_SOC,
            "Pack_SOH": Pack_SOH
        }
        candidate_json = parse_inner_json(field_map.get(search_field))

        # Score the match (0 if no target JSON provided)
        score = score_match(target_json, candidate_json)

        matches.append({
            "id": id,
            "manufacturer": manu,
            "model": mod,
            "year": year,
            "match_score": score,
            "field": search_field,
            "candidate_json": candidate_json
        })

    # Sort by score (desc) then id for stability
    matches.sort(key=lambda x: (-x["match_score"], x["id"]))

    return matches[:top_n]

# --- Example usage ---
# Case 1: With target JSON
target_soc = '{"request_id": "7E0", "response_id": "7E8", "command": "221164", "transmit_message": "03 22 11 64 00 00 00 00"}'
results_with_target = find_closest_matches(
    manufacturer="Audi",
    search_field="Pack_SOC",
    target_json_str=target_soc,
    top_n=3
)

# Case 2: Without target JSON (returns all rows for manufacturer)
results_no_target = find_closest_matches(
    manufacturer="BMW",
    search_field="Pack_SOC",
    target_json_str=None,
    top_n=3
)

print("With target JSON:")
for r in results_with_target:
    print(f'Score: {r["match_score"]}, Candidate JSON: {r["candidate_json"]}')

print("\nWithout target JSON:")
for r in results_no_target:
    print(f'Score: {r["match_score"]}, Candidate JSON: {r["candidate_json"]}')
