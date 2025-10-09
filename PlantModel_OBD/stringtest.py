import sqlite3
import json

# --- CONFIG ---
DB_PATH = "../OBD.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- STEP 1: Query vehicle_pack_commands table ---
cur.execute("SELECT * FROM vehicle_pack_commands;")
rows = cur.fetchall()

# --- STEP 2: Convert row to dict ---
def row_to_dict(row):
    id, manufacturer, model, year, country_region, type_, Pack_Voltage, Pack_SOC, Pack_SOH = row
    return {
        "id": id,
        "manufacturer": manufacturer,
        "model": model,
        "year": year,
        "country_region": country_region,
        "type": type_,
        "Pack_Voltage": Pack_Voltage,
        "Pack_SOC": Pack_SOC,
        "Pack_SOH": Pack_SOH
    }

# --- STEP 3: Stringify dict as JSON ---
def stringify_row_dict(row):
    return json.dumps(row_to_dict(row), ensure_ascii=False)


# --- STEP 4: Compare stringified JSON to DB row ---
def check_stringified_in_db(input_json_string):
    """
    Compare the given stringified JSON against all rows in the DB.
    Returns True if an exact match is found, False otherwise.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM vehicle_pack_commands")
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        if stringify_row_dict(row) == input_json_string:
            return True
    return False

if __name__ == "__main__":
    # --- Example usage ---
    input_data = {
        "id": 1,
        "manufacturer": "Audi",
        "model": "",
        "year": 0,
        "country_region": "",
        "type": "",
        "Pack_Voltage": "{\"request_id\": \"7E5\", \"response_id\": \"7ED\", \"command\": \"221E3B\", \"transmit_message\": \"03 22 1E 3B 00 00 00 00\", \"start_bit\": 0, \"end_bit\": 15, \"len\": 16, \"mul\": 0, \"div\": 4.0, \"add\": 0}",
        "Pack_SOC": "{\"request_id\": \"7E0\", \"response_id\": \"7E8\", \"command\": \"221164\", \"transmit_message\": \"03 22 11 64 00 00 00 00\", \"start_bit\": 0, \"end_bit\": 15, \"len\": 16, \"mul\": 0, \"div\": 100.0, \"add\": 0}",
        "Pack_SOH": "{\"request_id\": \"7E5\", \"response_id\": \"7ED\", \"command\": \"2251E0\", \"transmit_message\": \"03 22 51 E0 00 00 00 00\", \"start_bit\": 0, \"end_bit\": 15, \"len\": 16, \"mul\": 0.127, \"div\": 0, \"add\": -1798.574}"
    }

    input_json_string = json.dumps(input_data, ensure_ascii=False)


    if check_stringified_in_db(input_json_string):
        print("✅ Yes, this exact stringified JSON exists in the database.")
    else:
        print("❌ No exact match found in the database.")
