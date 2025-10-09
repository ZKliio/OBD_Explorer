import sqlite3
import csv
import json

# --- CONFIG ---
DB_PATH = "OBD.db"
CSV_PATH = "car_models.csv"

# convert JSON list to empty list if not valid JSON
# converts semicolon-separated string to list then to string
def safe_json_or_list(value):
    try:
        # Try to parse as JSON list first
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # If not JSON, split by semicolon
    if value:
        return ",".join([part.strip() for part in value.split(";") if part.strip()])
    return []

# # Then in the CSV loop:
# car_info[key] = {
#     "year": safe_json_or_list(row['Year']),
#     "country_region": safe_json_or_list(row['CountryRegion']),
#     "type": row['EV_Status'].strip()
# }

# --- STEP 1: Load CSV into a lookup dict ---
def car_info_lookup():
    car_info = {}
    with open(CSV_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['Manufacturer'].strip(), row['Model'].strip())
            car_info[key] = {
                # "year": safe_json_or_list(row['Year']),
                "year": int(row['Year']) if row['Year'] and row['Year'].isdigit() else None,
                "country_region": safe_json_or_list(row['CountryRegion']),
                "type": row['EV_Status'].strip() if row['EV_Status'] is not None else "NOT EV"
            }
    return car_info


        # car_info[key] = {
        #     "year": json.loads(row['Year']) if row['Year'].startswith('[') else [row['Year']],
        #     "country_region": json.loads(row['CountryRegion']) if row['CountryRegion'].startswith('[') else [row['CountryRegion']],
        #     "type": row['EV_Status'].strip()
        # }

# --- STEP 2: Connect to DB ---
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# --- STEP 3: Create new table ---
cur.execute("DROP TABLE IF EXISTS vehicle_pack_commands;")

cur.execute("""
CREATE TABLE IF NOT EXISTS vehicle_pack_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer TEXT NOT NULL,
    model TEXT,
    year INTEGER,
    country_region JSON,
    type TEXT,
    Pack_Voltage JSON,
    Pack_SOC JSON,
    Pack_SOH JSON
);
""")

# --- STEP 4: Query signals table for Pack_Voltage ---
def query_parameter(parameter_name):
    cur.execute("""
    SELECT manufacturer, model, request_id, response_id, command, transmit_message, start_bit, end_bit, len, mul, div, "add"
    FROM key_signals
    WHERE key_parameter = ?
    ORDER BY manufacturer;
    """, (parameter_name,))
    return cur.fetchall()

# rows = query_parameter('Pack_Voltage')

# cur.execute("""
# SELECT manufacturer, model, request_id, response_id, command, transmit_message, start_bit, end_bit, len, mul, div, 'add'
# FROM key_signals
# WHERE key_parameter = 'Pack_Voltage'
# ORDER BY manufacturer;
# """)

# rows = cur.fetchall()

# --- STEP 5: Insert into new table ---
def insert_parameter_data(rows, parameter_name):
    # Retrieves manufacturer, model, year, country_region, type from CSV
    car_info = car_info_lookup()

    for manufacturer, model, request_id, response_id, command, transmit_message, start_bit, end_bit, len, mul, div, add in rows:
        # handle NULL model
        key = (manufacturer.strip(), model.strip() if model else "")
        # look up car info
        if car_info.get(key):
            # year_json = json.dumps(car_info[key]["year"])
            year_json = car_info[key]["year"] 

            # country_list = safe_json_or_list(row['CountryRegion'])  # returns a Python list
            # country_json = json.dumps(country_list)  # dumps the list directly
    
            country_json = json.dumps(car_info[key]["country_region"])
            print('1', manufacturer, model, country_json)
            print(manufacturer, model, car_info[key]["country_region"])
            type_val = "EV" if car_info[key]["type"] == "EV" else "NOT EV"
        # manufacturer, not specific model
        else:
            year_json = 0
            country_json = ""
            type_val = ""
    
        parameter_json = json.dumps({
            "request_id": request_id,
            "response_id": response_id,
            "command": command,
            "transmit_message": transmit_message,
            "start_bit": 0 if start_bit is None else start_bit,
            "end_bit": end_bit,
            "len": len,
            "mul": 0 if mul is None else mul,
            "div": 0 if div is None else div,
            "add": 0 if add is None else add
        })

        # sql statement separated because {parameter_name} cannot be directly parsed in cur.execute()
        # sql = f"""
        # INSERT INTO vehicle_pack_commands
        # (manufacturer, model, year, country_region, type, {parameter_name})
        # VALUES (?, ?, ?, ?, ?, ?)
        # """

        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicle_unique
        ON vehicle_pack_commands (manufacturer, model, year, country_region, type);
            """)
        
        sql = f"""
        INSERT INTO vehicle_pack_commands
        (manufacturer, model, year, country_region, type, {parameter_name})
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(manufacturer, model, year, country_region, type)
        DO UPDATE SET {parameter_name} = excluded.{parameter_name};
        """

        cur.execute(sql, (manufacturer, model, year_json, country_json, type_val, parameter_json))


key_parameters = ['Pack_Voltage', 'Pack_SOC', 'Pack_SOH', 'Pack_Temperature', 'Pack_Current']

# for parameter_name in key_parameters:
for i in range(3):  # Only process the first 3 parameters for demonstration
    parameter_name = key_parameters[i]
    rows = query_parameter(parameter_name)  # Fetch rows for the parameter
    insert_parameter_data(rows, parameter_name) # Insert data into the new table pack_table


conn.commit()
conn.close()

print("✅ vehicle_pack_commands table created and populated.")