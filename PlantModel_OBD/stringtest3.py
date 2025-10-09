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
                result[key] = val
        else:
            result[key] = None

    return result

def normalize_input(input_json_string):
    """
    Parse the input JSON string and also parse inner JSON blobs.
    """
    data = json.loads(input_json_string)
    for key in ["Pack_Voltage", "Pack_SOC", "Pack_SOH"]:
        if key in data and data[key]:
            try:
                data[key] = json.loads(data[key])
            except (json.JSONDecodeError, TypeError):
                pass
    return data

def super_loose_match_in_db(input_json_string):
    """
    Returns True if a DB row contains all provided keys/values from the input.
    Ignores key order, whitespace, and JSON formatting differences.
    Allows missing keys in the input.
    """
    input_dict = normalize_input(input_json_string)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM vehicle_pack_commands")
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        db_dict = normalize_row(row)

        # Check if all provided keys in input match DB row
        match = True
        for key, val in input_dict.items():
            if key in ["Pack_Voltage", "Pack_SOC", "Pack_SOH"]:
                # Compare nested dicts if both are dicts
                if isinstance(val, dict) and isinstance(db_dict.get(key), dict):
                    if val != db_dict[key]:
                        match = False
                        break
                else:
                    if val != db_dict.get(key):
                        match = False
                        break
            else:
                if val != db_dict.get(key):
                    match = False
                    break

        if match:
            print("Match found:", db_dict)
            return True
    return False

# --- Example usage ---
input_data = {
    "manufacturer": "Audi",
    "year": 0,
    "Pack_SOC": '{"request_id": "7E0", "response_id": "7E8", "command": "221164", "transmit_message": "03 22 11 64 00 00 00 00", "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 100.0, "add": 0}'
}

input_data = {
    "manufacturer": "Audi",
    "Pack_SOC": '{"request_id": "FC00", "response_id": "FC08", "command": "22028C", "transmit_message": "03 22 02 8C 00 00 00 00", "start_bit": 0, "end_bit": 7, "len": 8, "mul": 100.0, "div": 250.0, "add": 0}'
}


input_json_string = json.dumps(input_data, ensure_ascii=False)

if super_loose_match_in_db(input_json_string):
    print("✅ Match found in database")
else:
    print("❌ No match found")