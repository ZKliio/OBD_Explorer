# Levenhenstein Distance but with appending of parameters at every iteration
# Discrete Logic: Success = append, Failure = next car in priority list
# Priority list is re-sorted after every successful guess

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from collections import defaultdict
from typing import List, Optional
import sqlite3, json
import Levenshtein
import copy # Used for deep copying the priority list data

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static", html=True), name="static")
# Put index.html in ./static
# Access at: http://localhost:8000/static

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

DB_PATH = "../../OBD.db"

# --- Pydantic Models ---
class CarIn(BaseModel):
    manufacturer: str
    model: str
    year: int
    country_region: str = ""
    type_: str = ""

class CarOut(CarIn):
    commands: dict

class CheckRequest(BaseModel):
    manufacturer: str
    model: str
    # target_json_arr is now optional, as we're guessing the commands iteratively
    target_json_arr: Optional[List[dict]] = None 
    
    # The new_car_attributes string is passed to hold the iteratively built "fingerprint"
    new_car_attributes: str = ""

class CheckResponse(BaseModel):
    exists: bool
    matches: list[dict]
    new_car_attributes: str # Return the refined attribute string

# --- Utility Functions ---

def parse_inner_json(val):
    if not val:
        return {}
    try:
        return json.loads(val)
    except json.JSONDecodeError:
        return {}

def get_levenshtein_similarity(string1, string2):
    """
    Calculates the normalized Levenshtein similarity between two strings.
    """
    distance = Levenshtein.distance(string1, string2)
    max_len = max(len(string1), len(string2))
    if max_len == 0:
        return 1.0 
    return 1.0 - (distance / max_len)

def canonicalize_json_to_string(json_data):
    """
    Sorts JSON keys and serializes the object to a consistent, compact string.
    """
    return json.dumps(json_data, sort_keys=True, separators=(',', ':'))

# --- Car Class and DB Loading (Simplified for known car data) ---

class Car:
    def __init__(self, data: CarIn):
        self.manufacturer = data.manufacturer
        self.model = data.model
        self.year = data.year
        self.country_region = data.country_region
        self.type_ = data.type_
        self.commands = defaultdict(list)
        # Unique identifier built from static attributes
        self.static_identifier = (
            f"manufacturer{self.manufacturer.lower()}"
            f"model{self.model.lower()}"
            f"year{self.year}"
            f"country{self.country_region.lower()}"
            f"type{self.type_.lower()}"
        )

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

def load_all_known_cars() -> List[Car]:
    """Loads all unique car entries and their commands from the DB."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT manufacturer, model, year, country_region, type,
               Pack_Voltage, Pack_SOC, Pack_SOH
        FROM vehicle_pack_commands
    """)
    rows = cur.fetchall()
    conn.close()

    cars_map = {}
    for r in rows:
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
        # Only take the first command for simplicity in this model
        if r[5]: car.add_command("Pack_Voltage", r[5])
        if r[6]: car.add_command("Pack_SOC", r[6])
        if r[7]: car.add_command("Pack_SOH", r[7])

    return list(cars_map.values())


# --- Core Logic: The Guessing Attack Strategy (Mapping to Mermaid Flow) ---

def initialize_priority_list(known_cars: List[Car], new_car_str: str) -> List[dict]:
    """
    (1.1 - 1.3) Creates the initial priority list based on static string similarity.
    """
    priority_list = []
    
    # A list of static identifiers for the new car to compare against
    new_car_static_str = (
        f"manufacturer{new_car_str.manufacturer.lower()}"
        f"model{new_car_str.model.lower()}"
    )
    print(f'New Car Static Identifier: {new_car_static_str}')
    
    for car in known_cars:
        # Calculate Character-Level Similarity Scores (1.2)
        initial_similarity = get_levenshtein_similarity(new_car_static_str, car.static_identifier)
        
        priority_list.append({
            "manufacturer": car.manufacturer,
            "model": car.model,
            "static_identifier": car.static_identifier,
            "current_identifier": new_car_static_str, # The growing identifier of the unknown car
            "similarity_score": initial_similarity,
            "commands_found": {}, # Dictionary to store discovered commands
            "db_car_object": car,
        })
    
    # Sort the list by similarity score (1.3)
    priority_list.sort(key=lambda x: x["similarity_score"], reverse=True)
    return priority_list

# ... (imports, models, utility functions remain the same)

def guessing_attack_strategy(new_car_data: CheckRequest) -> CheckResponse:
    """
    Implements the iterative guessing attack with success/failure determined 
    by command message similarity to a provided target.
    """
    known_cars = load_all_known_cars() 
    key_parameters = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"] 
    
    # --- New: Extract and Canonicalize the Target Command ---
    if not new_car_data.target_json_arr or not isinstance(new_car_data.target_json_arr[0], dict):
        raise HTTPException(status_code=400, detail="CheckRequest must contain a valid 'target_json_arr' for comparison.")
    
    # Assuming the target command provided is for the FIRST parameter (Pack_SOC)
    target_command_data = new_car_data.target_json_arr[0]
    target_command_str = canonicalize_json_to_string(target_command_data)
    
    # --- Initialization ---
    current_new_car_str = new_car_data.new_car_attributes
    if not current_new_car_str:
        current_new_car_str = f"manufacturer{new_car_data.manufacturer.lower()}model{new_car_data.model.lower()}"

    priority_list = initialize_priority_list(known_cars, new_car_data)
    refined_priority_list = copy.deepcopy(priority_list)
    commands_found = {}

    # Set a similarity threshold for a successful "guess"
    SUCCESS_THRESHOLD = 0.95 

    # --- Step 2: Command-Level Refinement ---
    for param in key_parameters:
        
        # --- Crucial Assumption ---
        # In a full run, you would need a new target command for EACH parameter.
        # For this demonstration, we assume the single provided target_command_str 
        # is the ground truth for this parameter's command format.
        
        # H: Iterate through Priority List Top-Ranked Vehicle First
        for i, match_data in enumerate(refined_priority_list):
            db_car = match_data["db_car_object"]
            
            # I: Retrieve Transmit Command from Ranked Vehicle
            if db_car.commands.get(param):
                potential_command_json = db_car.commands[param][0] 
                db_command_str = canonicalize_json_to_string(potential_command_json)
                # print(f'Comparing Target: {target_command_str} with DB Command: {db_command_str}')
                
                # --- J: Determine Success/Failure by Comparison ---
                command_similarity = get_levenshtein_similarity(target_command_str, db_command_str)
                print(command_similarity)
                # J -- Success --> K: Append if similarity is high (Near-Perfect Match)
                if command_similarity >= SUCCESS_THRESHOLD:
                    print(f"SUCCESS for {param}: Found high match in {db_car.manufacturer}. Score: {command_similarity:.4f}")
                    
                    if param not in commands_found:
                        commands_found[param] = potential_command_json
                        
                        # K: Append Successful Command String to Identifier
                        current_new_car_str += f"{param}{db_command_str}"
                        
                        # L: Update Similarity Score and Re-Sort Priority List (Full Re-Prioritization)
                        for item in refined_priority_list:
                            # Re-calculate the full identifier for the item being compared against
                            # NOTE: This is complex. We'll simplify to comparing the NEW current_new_car_str 
                            # against the static identifier for re-ranking.
                            new_score = get_levenshtein_similarity(current_new_car_str, item["static_identifier"])
                            
                            item["current_identifier"] = current_new_car_str 
                            item["similarity_score"] = new_score
                            item["commands_found"].update(commands_found)

                        refined_priority_list.sort(key=lambda x: x["similarity_score"], reverse=True)
                        
                        # Break and move to the next parameter
                        break 
                
                # J -- Failure --> N: Move to Next Vehicle in Priority List
                # Implicit: If similarity < SUCCESS_THRESHOLD, the loop continues to the next car (i+1)

        # M: All Key Parameters Tested? (Implicitly moves to next 'param' loop)

    # ... (Final result formatting remains the same)
    final_matches = []
    # Use a minimum threshold for final results display
    DISPLAY_THRESHOLD = 0.5 
    
    for match in refined_priority_list:
        if match["similarity_score"] >= DISPLAY_THRESHOLD:
            final_matches.append({
                "manufacturer": match["manufacturer"],
                "model": match["model"],
                "similarity_score": match["similarity_score"],
                "command": match["commands_found"].get(param, {}), 
                "field": param 
            })

    # Return the top 5
    return CheckResponse(
        exists=bool(final_matches),
        matches=final_matches[:5],
        new_car_attributes=current_new_car_str 
    )
    

# --- FastAPI Endpoints ---

# Get cars by manufacturer and model
@app.get("/cars/{manufacturer}/{model}", response_model=list[CarOut])
def get_cars(manufacturer: str, model: str):
    global cars
    cars = load_car(manufacturer.capitalize(), model)
    if not cars:
        raise HTTPException(status_code=404, detail="Car not found")
    return [c.to_dict() for c in cars]



# (Keep your existing /cars/{manufacturer}/{model} and /cars POST endpoints)

@app.post("/check_existence", response_model=CheckResponse)
def check_existence_endpoint(req: CheckRequest):
    """
    Main entry point for the priority list guessing attack.
    """
    try:
        # A: Start: Input New Unknown Vehicle
        return guessing_attack_strategy(req)
        
    except Exception as e:
        # Catch and handle other unexpected errors
        print(f"Error in guessing attack: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Execution ---
# if __name__ == "__main__":
#     import uvicorn
#     # uvicorn.run(app, host="0.0.0.0", port=3000)
#     print("FastAPI app is ready. Run using: uvicorn your_file_name:app --reload --port 3000")