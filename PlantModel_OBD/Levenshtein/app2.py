# experimental

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections import defaultdict
import sqlite3, json, os
import Levenshtein



app = FastAPI()
app.mount("/static", StaticFiles(directory="../static", html=True), name="static")
# Put index.html in ./static
# Access at: http://localhost:3000/static

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],            # includes OPTIONS
    allow_headers=["*"],            # includes Content-Type, etc.
)

DB_PATH = "../../OBD.db"

class CarIn(BaseModel):
    manufacturer: str
    model: str
    year: int
    country_region: str = ""
    type_: str = ""

class CarOut(CarIn):
    commands: dict
from typing import List

class CheckRequest(BaseModel):
    manufacturer: str
    model: str = None
    target_json_str: str = None
    target_json_arr: List = None
    field  :str = None

class CheckResponse(BaseModel):
    exists: bool
    matches: list[dict]

def parse_inner_json(val):
    if not val:
        return {}
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return {}

class Car:
    def __init__(self, data: CarIn):
        self.manufacturer = data.manufacturer
        self.model = data.model
        self.year = data.year
        self.country_region = data.country_region
        self.type_ = data.type_
        self.commands = defaultdict(list)

    def add_command(self, field_name, blob):
        cmd = parse_inner_json(blob)
        if cmd:
            self.commands[field_name].append(cmd)

    def to_dict(self):
        return {
            "manufacturer": self.manufacturer,
            "model": self.model,
            "year": self.year,
            "country_region": self.country_region,
            "type": self.type_,
            "commands": dict(self.commands)
        }

def load_car(manufacturer, model=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    print(f'Querying for manufacturer="{manufacturer}", model="{model}"')
    if model:
        cur.execute("""
            SELECT manufacturer, model, year, country_region, type,
                   Pack_Voltage, Pack_SOC, Pack_SOH
            FROM vehicle_pack_commands
            WHERE manufacturer LIKE ? AND model LIKE ?
        """, (manufacturer, f"%{model}%"))
    else:
        cur.execute("""
            SELECT manufacturer, model, year, country_region, type,
                   Pack_Voltage, Pack_SOC, Pack_SOH
            FROM vehicle_pack_commands
            WHERE manufacturer LIKE ?
        """, (f"%{manufacturer}%",))
    rows = cur.fetchone()
    conn.close()

    cars_map = {}
    #for r in rows:
    if rows:
        r=rows
        key = tuple(r[:5])
        if key not in cars_map:
            data = CarIn(
                manufacturer=r[0],
                model=r[1],
                year=r[2],
                country_region=r[3],
                type_=r[4]
            )
            cars_map[key] = Car(data)
        car = cars_map[key]
        car.add_command("Pack_Voltage", r[5])
        car.add_command("Pack_SOC", r[6])
        car.add_command("Pack_SOH", r[7])

    return list(cars_map.values())

@app.post("/cars", response_model=CarOut, status_code=201)
def add_car(payload: CarIn):
    # Example of writing back into the database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO vehicle_pack_commands
        (manufacturer, model, year, country_region, type)
        VALUES (?, ?, ?, ?, ?)
    """, (
        payload.manufacturer,
        payload.model,
        payload.year,
        payload.country_region,
        payload.type_
    ))
    conn.commit()
    conn.close()
    # Return an “empty” CarOut with no commands yet
    return {**payload.dict(), "commands": {}}
cars =[]
@app.get("/cars/{manufacturer}/{model}", response_model=list[CarOut])
def get_cars(manufacturer: str, model: str):
    global cars
    cars = load_car(manufacturer.capitalize(), model)
    if not cars:
        raise HTTPException(status_code=404, detail="Car not found")
    return [c.to_dict() for c in cars]


# Levenshtein distance scoring
def get_levenshtein_similarity(string1, string2):
    """
    Calculates the normalized Levenshtein similarity between two strings.
    A score of 1.0 means the strings are identical, 0.0 means they are completely different.
    """
    distance = Levenshtein.distance(string1, string2)
    max_len = max(len(string1), len(string2))
    if max_len == 0:
        return 1.0  # Both strings are empty, so they are 100% similar
    return 1.0 - (distance / max_len)

# Canonicalize JSON to string
def canonicalize_json_to_string(json_data):
    """
    Sorts JSON keys and serializes the object to a consistent, compact string.
    This ensures that the order of keys does not affect the similarity score.
    """
    # Sort keys for consistent string representation and remove whitespace
    return json.dumps(json_data, sort_keys=True, separators=(',', ':'))

# Assuming all other code and imports from your original script are present
@app.post("/check_existence", response_model=CheckResponse)
def find_matching_parameters(req: CheckRequest):
    try:
        # Input Validation and Preprocessing
        if not req.target_json_arr or not isinstance(req.target_json_arr, list) or not req.target_json_arr[0]:
            raise HTTPException(status_code=400, detail="Invalid or empty target_json_arr format")
        
        target_json_data = req.target_json_arr[0]
        # 2.1: Select Target Parameter (req.field) and Canonicalize target command
        target_command_str = canonicalize_json_to_string(target_json_data)
        print(f'Target Command String: {target_command_str}')
        
        # 1.1: Create Unique String Identifier
        req_car_str = (req.manufacturer + (req.model if req.model else "")).lower()
        print(f'Requested Car String: {req_car_str}')

        # Step 1: Vehicle-Specific Prioritization (Initial Plant Model)
        
        # Step 1: Query the entire database to get all known cars and their commands
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        # Query: manufacturer, model, and the requested field command
        cur.execute(f"""
            SELECT manufacturer, model, {req.field}
            FROM vehicle_pack_commands
        """)
        all_db_data = cur.fetchall()
        conn.close()

        # 1.2 & 1.3: Calculate Character-Level Similarity Scores and Generate Initial Priority List
        initial_priority_list: List[Dict[str, Any]] = []
        
        for row in all_db_data:
            db_manufacturer, db_model, db_command_blob = row
            
            # Combine manufacturer and model strings
            db_car_str = (db_manufacturer + db_model).lower()
            
            # Calculate car name similarity (This is Step 1: Vehicle-Specific Prioritization)
            name_similarity = get_levenshtein_similarity(req_car_str, db_car_str)

            # Store the initial score and the command blob
            initial_priority_list.append({
                "manufacturer": db_manufacturer,
                "model": db_model,
                "field": req.field,
                "name_similarity": name_similarity,
                "db_command_blob": db_command_blob,
                "command": None, # Will be filled later
                "similarity_score": 0.0 # Will be the combined score (Step 2.5)
            })
            
        # Sort the initial list by name similarity (Step 1.3)
        initial_priority_list.sort(key=lambda x: x["name_similarity"], reverse=True)
        
        # Step 2: Command-Level Refinement (Iterate through Priority List Top-Ranked First)
        
        refined_list = []
        for item in initial_priority_list:
            # 2.3: Retrieve UDS Command/PID from Ranked Vehicle
            db_command_blob = item.pop("db_command_blob") # Get and remove blob for clean output
            
            if not db_command_blob:
                continue # Skip if the command is empty for this car/field

            try:
                # Parse the JSON blob
                db_command_data = json.loads(db_command_blob)
                
                # 2.4/2.5 part 1: Canonicalize the database command string
                db_command_str = canonicalize_json_to_string(db_command_data)
                print(f'Database Command String: {db_command_str}')
                # Calculate command string similarity
                command_similarity = get_levenshtein_similarity(target_command_str, db_command_str)
                
                # 2.5: Recalculate Dynamic Similarity Score (Combined Score)
                # Weighting command similarity more heavily, as in the original code
                combined_score = (item["name_similarity"] * 0.3) + (command_similarity * 0.7)
                
                item["similarity_score"] = combined_score
                item["command"] = db_command_data
                refined_list.append(item)

            except json.JSONDecodeError:
                continue # Skip rows with malformed JSON

        # 2.6: Re-Sort Priority List Pushing Updated Vehicle Higher
        refined_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 2.7: All Key Parameters Iterated? Yes, because we only had one target field.
        
        # Step 5: Return the top matches (e.g., those with a score > 0.5)
        top_matches = [m for m in refined_list if m["similarity_score"] > 0.5]
        
        print("\n--- Refined Priority List (Python Backend) ---")
        for match in top_matches:
            score = (match["similarity_score"] * 100)
            # print(f"Score: {score:.2f}% | Manufacturer: {match['manufacturer']} | Model: {match['model']} | Field: {match['field']}")
        
        if top_matches:
            # End Result: Prioritized List of UDS Commands for Unknown Vehicle
            
            return {"exists": True, "matches": top_matches[:5]} # Return top 5 matches
        else:
            return {"exists": False, "matches": []}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

'''
@app.post("/check_existence", response_model=CheckResponse)
def find_matching_parameters(req: CheckRequest):
    # Comment this out to test the DB query (global cars)
    cars = load_car(req.manufacturer, req.model)
    print(cars)
    print(vars(cars[0]))
    try:
        matches = []
        print(f'stored car command:{cars[0].commands.get(req.field)}')
        print(f'target_json_arr: {req.target_json_arr}')
        if cars[0].commands.get(req.field) == req.target_json_arr:
            
            matches.append({req.field:cars[0].commands.get(req.field)})
            return {"exists": True, "matches": matches}
        else:
            matches.append({req.field:0})
            return {"exists": False, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''
'''
    #target = json.dumps(parse_inner_json(req.target_json_str), sort_keys=True)
    target = req.target_json_str[0]
    print(target)
    for car in cars:
        for field, cmds in car.commands.items():
            for cmd in cmds:
                #Sprint("cmd:")
                # print(json.dumps(cmd, sort_keys=True))
                if cmd == target:
                    matches.append({**car.to_dict(), "field": field})
    if bool(matches):
        if req.field==field:    
            return {"exists": bool(matches), "matches": matches}
        else:
            return {"exists": False, "matches": matches}
    else:
        return {"exists": False, "matches": matches}
'''
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)

