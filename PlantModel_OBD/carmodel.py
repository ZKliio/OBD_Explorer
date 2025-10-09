import sqlite3
import json
from collections import defaultdict

DB_PATH = "../OBD.db"

def parse_inner_json(val):
    """Parse a JSON‐string blob into a dict (or return {})."""
    if not val:
        return {}
    try:
        return json.loads(val)
    except (TypeError, json.JSONDecodeError):
        return {}

# Class to represent a car
class Car:
    def __init__(self, manufacturer, model, year, country_region, type_):
        self.manufacturer = manufacturer
        self.model = model
        self.year = year
        self.country_region = country_region
        self.type = type_
        self.commands = defaultdict(list)

    def add_command(self, field_name, command_blob):
        cmd = parse_inner_json(command_blob)
        if cmd:
            self.commands[field_name].append(cmd)

    def has_command(self, field_name, target_json_str):
        """
        Return True if target_json_str matches any stored command under field_name.
        Comparison is done by stringifying both dicts with sorted keys.
        """
        # Parse the input JSON string into a dict
        target_cmd = parse_inner_json(target_json_str)

        # Canonical JSON string for the target
        target_str = json.dumps(target_cmd, sort_keys=True, ensure_ascii=False)

        # Compare against each stored command (dict) by stringifying it
        for cmd in self.commands.get(field_name, []):
            cmd_str = json.dumps(cmd, sort_keys=True, ensure_ascii=False)
            print(f'cmd_str: {cmd_str}\n, target_str: {target_str}')
            if cmd_str == target_str:
                return True
        return False

    def to_dict(self):
        """Return a serializable dict representation (including commands)."""
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "year": self.year,
            "country_region": self.country_region,
            "type": self.type,
            "commands": dict(self.commands)
        }

    def __repr__(self):
        return (f"<Car {self.manufacturer} {self.model or '[no model]'} "
                f"({self.year}) cmds={sum(len(v) for v in self.commands.values())}>")

# End of Class

# Function to load cars from DB for given manufacturers (Plant Model), as a list

def load_car(manufacturer, model=None):
    """
    Load Car objects from the DB for a given manufacturer (and optional model).
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if model:
        cur.execute("""
            SELECT manufacturer, model, year, country_region, type,
                   Pack_Voltage, Pack_SOC, Pack_SOH
            FROM vehicle_pack_commands
            WHERE manufacturer = ? AND model like ?
        """, (manufacturer, model))
    else:
        cur.execute("""
            SELECT manufacturer, model, year, country_region, type,
                   Pack_Voltage, Pack_SOC, Pack_SOH
            FROM vehicle_pack_commands
            WHERE manufacturer like ?
        """, (manufacturer,))


    rows = cur.fetchall()
    conn.close()

    cars = {}
    for manu, mod, year, ctr, typ, volt, soc, soh in rows:
        key = (manu, mod, year, ctr, typ)
        if key not in cars:
            cars[key] = Car(manu, mod, year, ctr, typ)

        car = cars[key]
        car.add_command("Pack_Voltage", volt)
        car.add_command("Pack_SOC", soc)
        car.add_command("Pack_SOH", soh)

    return list(cars.values())


# Checks if a specific command exists within car object specified
# def car_contains_message(manufacturer,
#                          target_json_str,
#                          field_name,
#                          model=None):
#     """
#     Return True if any Car for this manufacturer/model contains the
#     exact target_json_str under field_name.
#     """
#     if not manufacturer:
#         raise ValueError("manufacturer is required")
#     if not field_name:
#         raise ValueError("field_name is required")
#     if not target_json_str:
#         raise ValueError("target_json_str is required")

#     # This creates the car upon calling function, but i might want to retrieve it from car collection and choose it
#     cars = load_cars(manufacturer, model)
#     for car in cars:
#         if car.has_command(field_name, target_json_str):
#             print(f'\n✅ Found {target_json_str}in\nManufacturer: {car.manufacturer} {"" if car.model is None else car.model}\nYear: {car.year}\nCountry:{car.country_region}\nType:{car.type}')
#             return True
#     print(f'❌ Did not find {target_json_str} in any car')
#     return False


def find_matching_parameters(manufacturer, target_json_str, model=None):
    """
    Search all Pack_* fields for the exact target JSON.
    Returns a list of matches with the field name and car details.
    """
    if not manufacturer:
        raise ValueError("manufacturer is required")
    if not target_json_str:
        raise ValueError("target_json_str is required")

    # Canonicalize target JSON
    target_cmd = parse_inner_json(target_json_str)
    target_str = json.dumps(target_cmd, sort_keys=True, ensure_ascii=False)

    matches = []
    cars = load_car(manufacturer, model)
    print(cars)

    # Compares each stored command (dict) by stringifying it
    for car in cars:
        for field_name, cmds in car.commands.items():
            for cmd in cmds:
                cmd_str = json.dumps(cmd, sort_keys=True, ensure_ascii=False)
                # print(f'cmd_str: {cmd_str}\n, target_str: {target_str}')
                # print(type(cmd_str), type(target_str))
                if cmd_str == target_str:
                    matches.append({
                        "manufacturer": car.manufacturer,
                        "model": car.model,
                        "year": car.year,
                        "country_region": car.country_region,
                        "type": car.type,
                        "field": field_name
                    })
    return matches





# --- Example Usage ---

# Audi 
# Audi SOC
# target = '''
# {"request_id":"7E0","response_id":"7E8","command":"221164",
#  "transmit_message":"03 22 11 64 00 00 00 00",
#  "start_bit":0,"end_bit":15,"len":16,"mul":0,"div":100.0,"add":0}
# '''

# Audi Q4 e-tron SOC
target = '''

'''

# Bentley 
# Bentley voltage
# target = '''{"request_id": "7E5", "response_id": "7ED", "command": "221E3B", "transmit_message": "03 22 1E 3B 00 00 00 00", 
#  "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 4.0, "add": 0}'''
# Bentley SOC
# target = '''{"request_id": "7E0", "response_id": "7E8", "command": "221164", "transmit_message": "03 22 11 64 00 00 00 00", 
# "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 100.0, "add": 0}'''
# Bentley SOH
# target = '''{"request_id": "7E5", "response_id": "7ED", "command": "2251E0", "transmit_message": "03 22 51 E0 00 00 00 00", 
# "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0.127, "div": 0, "add": -1798.574}'''

# BMW
# BMW Voltage
# target = '''{"request_id": "6F1", "response_id": "6F9", "command": "22DD68", "transmit_message": "03 22 DD 68 00 00 00 00", 
# "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 100.0, "add": 0}'''
# # BMW SOC
# target = '''{"request_id": "6F1", "response_id": "6F9", "command": "22DDBC", "transmit_message": "03 22 DD BC 00 00 00 00", 
# "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 10.0, "add": 0}'''
# BMW SOH
# target = '''{"request_id": "6F1", "response_id": "6F9", "command": "226335", "transmit_message": "03 22 63 35 00 00 00 00", 
# "start_bit": 24, "end_bit": 31, "len": 8, "mul": 0, "div": 0, "add": 0}'''


# Single Iteration
results = find_matching_parameters(
    manufacturer="Audi",
    model="q4 e tron",
    target_json_str=target
)

# Multiple Iteration

# manufacturers = load_cars("", "")

if results:
    print("Found in:")
    for r in results:
        print(f"Manufacturer: {r['manufacturer']}\nModel: {r['model']}\nYear {r['year']}\nField: {r['field']}\n")
else:
    
    print("error : No matching parameter found.")


# 1. Load and inspect cars
# cars = load_cars("Audi", "")
# car_list = [("Audi",  ""), ("BMW", "")]
# cars = []

# for i, car in enumerate(car_list):
#     manufacturer = car[0]
#     model = car[1]
#     cars +=load_cars(manufacturer, model)


# print(f'Loaded cars: {len(cars)} in total\nList: {cars}')

# # Prints the first car
# # print("First car as dict:\n", json.dumps(cars[0].to_dict(), indent=2, ensure_ascii=False))

# # 2. Check if a specific SOC command exists (stringified comparison)
# target_soc = '''
# {"request_id":"7E0","response_id":"7E8","command":"221164",
#  "transmit_message":"03 22 11 64 00 00 00 00",
#  "start_bit":0,"end_bit":15,"len":16,"mul":0,"div":100.0,"add":0}
# '''

# target_voltage = '''
# {"request_id": "6F1", "response_id": "6F9", "command": "22DD68", "transmit_message": "03 22 DD 68 00 00 00 00", 
# "start_bit": 0, "end_bit": 15, "len": 16, "mul": 0, "div": 100.0, "add": 0} '''


# exists = car_contains_message(
#     manufacturer="BMW",
#     model="",
#     field_name="Pack_Voltage",
#     target_json_str=target_voltage
# )
# print("Command exists?" , exists)