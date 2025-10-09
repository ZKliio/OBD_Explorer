import sqlite3
import json

DB_PATH = "../OBD.db"

def normalize_row(row):
    """
    Convert a DB row into a fully parsed Python dict:
    - Outer row as dict
    - Inner JSON blobs parsed into dicts if possible
    """
    id, manufacturer, model, year, country_region, type_, Pack_Voltage, Pack_SOC, Pack_SOH = row
    result = {
        "id": id,
        "manufacturer": manufacturer or "",
        "model": model or "",
        "year": year,
        "country_region": country_region or "",
        "type": type_ or "",
        "Pack_Voltage": None,
        "Pack_SOC": None,
        "Pack_SOH": None
    }

    for key, val in [
        ("Pack_Voltage", Pack_Voltage),
        ("Pack_SOC", Pack_SOC),
        ("Pack_SOH", Pack_SOH)
    ]:
        if val:
            try:
                result[key] = json.loads(val)  # parse inner JSON string
            except (json.JSONDecodeError, TypeError):
                result[key] = val  # leave as-is if not valid JSON
        else:
            result[key] = None

    return result

def normalize_input(input_json_string):
    """
    Parse the input JSON string and also parse inner JSON blobs.
    """
    data = json.loads(input_json_string)
    for key in ["Pack_Voltage", "Pack_SOC", "Pack_SOH"]:
        if data.get(key):
            try:
                data[key] = json.loads(data[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return data

def loose_match_in_db(input_json_string):
    """
    Returns True if a logically equivalent row exists in the DB.
    Ignores key order, whitespace, and JSON formatting differences.
    """
    input_dict = normalize_input(input_json_string)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM vehicle_pack_commands")
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        if normalize_row(row) == input_dict:
            return True
    return False

# --- Example usage ---
input_data = {
    "id": 1,
    "manufacturer": "Audi",
    "model": "",
    "year": 0,
    "country_region": "",
    "type": "",
    "Pack_Voltage": '{"request_id": "7E5", "response_id": "7ED", "command": "221E3B", "transmit_message": "03 22 1E 3B 00 00 00 00", "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 4.0, "add": 0}',
    "Pack_SOC": '{"request_id": "7E0", "response_id": "7E8", "command": "221164", "transmit_message": "03 22 11 64 00 00 00 00", "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 100.0, "add": 0}',
    "Pack_SOH": '{"request_id": "7E5", "response_id": "7ED", "command": "2251E0", "transmit_message": "03 22 51 E0 00 00 00 00", "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0.127, "div": 0, "add": -1798.574}'
}

input_json_string = json.dumps(input_data, ensure_ascii=False)

if loose_match_in_db(input_json_string):
    print("✅ Loose match found in database")
else:
    print("❌ No match found")