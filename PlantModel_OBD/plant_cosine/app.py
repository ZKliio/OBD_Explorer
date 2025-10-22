"""
Car Model Server (Plant Model)
Simulates a physical car with randomly initialized field values from the database.
Auto-initializes on startup. Receives test payloads and returns match/no-match responses.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import sqlite3
import random

app = FastAPI(title="Car Model Server (Plant Model)")

# Serve static files
app.mount(
    "/static_model",
    StaticFiles(directory="./static_model", html=True),
    name="static_model",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "../../OBD.db"

# --- Models ---
class TestFieldRequest(BaseModel):
    field: str  # e.g., "Pack_SOC", "Pack_Voltage", "Pack_SOH"
    value: List[Dict[str, Any]]  # The command payload to test

class TestFieldResponse(BaseModel):
    field: str
    matched: bool
    message: str
    expected_value: Optional[List[Dict[str, Any]]] = None  # Optional: show what was expected

class CarModelInfo(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    fields_available: List[str]  # Which fields this car has
    Pack_SOC: Optional[List[Dict[str, Any]]] = None
    Pack_Voltage: Optional[List[Dict[str, Any]]] = None
    Pack_SOH: Optional[List[Dict[str, Any]]] = None

# --- Global state for the current car ---
current_car: Optional[Dict[str, Any]] = None

# --- Helper Functions ---
def parse_json_field(raw_value: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a JSON field from the database and return as list of dicts.
    Returns None if parsing fails or if the value is empty.
    """
    if not raw_value:
        return None
    
    try:
        # Parse if it's a string
        parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        
        # Normalize to list of dicts
        if isinstance(parsed, dict):
            return [parsed]
        elif isinstance(parsed, list):
            # Filter to only dicts
            dict_list = [item for item in parsed if isinstance(item, dict)]
            return dict_list if dict_list else None
        else:
            return None
    except:
        return None

def load_random_car_from_db() -> Dict[str, Any]:
    """
    Load a random car from the database with all its field values FROM THAT SPECIFIC ROW.
    Returns a dict with car metadata and commands.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Get all cars
    cur.execute("""
        SELECT manufacturer, model, year, country_region, type,
               Pack_Voltage, Pack_SOC, Pack_SOH
        FROM vehicle_pack_commands
    """)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        raise Exception("No cars in database")
    
    # Pick ONE random row and use ALL its values
    random_row = random.choice(rows)
    (manufacturer, model, year, country_region, type_,
     raw_voltage, raw_soc, raw_soh) = random_row
    
    # Initialize car data structure
    car_data = {
        "manufacturer": manufacturer,
        "model": model,
        "year": year,
        "country_region": country_region or "",
        "type_": type_ or "",
        "commands": {}
    }
    
    # Parse each field FROM THIS SPECIFIC ROW ONLY
    pack_voltage = parse_json_field(raw_voltage)
    if pack_voltage:
        car_data["commands"]["Pack_Voltage"] = pack_voltage
    
    pack_soc = parse_json_field(raw_soc)
    if pack_soc:
        car_data["commands"]["Pack_SOC"] = pack_soc
    
    pack_soh = parse_json_field(raw_soh)
    if pack_soh:
        car_data["commands"]["Pack_SOH"] = pack_soh
    
    # Debug output
    print(f"   Raw Pack_Voltage from DB: {raw_voltage[:100] if raw_voltage else 'None'}...")
    print(f"   Parsed Pack_Voltage: {pack_voltage}")
    print(f"   Raw Pack_SOC from DB: {raw_soc[:100] if raw_soc else 'None'}...")
    print(f"   Parsed Pack_SOC: {pack_soc}")
    print(f"   Raw Pack_SOH from DB: {raw_soh[:100] if raw_soh else 'None'}...")
    print(f"   Parsed Pack_SOH: {pack_soh}")
    
    return car_data

def normalize_value_for_comparison(value: Any) -> Any:
    """
    Recursively normalize values for comparison:
    - Convert floats that are whole numbers to ints (255.0 -> 255)
    - Preserve original order (no sorting)
    - Handle nested structures
    """
    if isinstance(value, float):
        # If it's a whole number, convert to int
        if value.is_integer():
            return int(value)
        return value
    elif isinstance(value, dict):
        # Recursively normalize dict values, preserve key order
        return {k: normalize_value_for_comparison(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Recursively normalize list items, preserve order
        return [normalize_value_for_comparison(item) for item in value]
    else:
        return value

def normalize_command_value(value: Any) -> str:
    """
    Normalize a command value to a consistent string for comparison.
    Handles both single dicts and lists of dicts.
    Does NOT sort - preserves original order.
    Normalizes floats (255.0 -> 255) for comparison.
    """
    # First normalize the value structure
    normalized = normalize_value_for_comparison(value)
    
    if isinstance(normalized, list):
        # Don't sort - keep original order
        return json.dumps(normalized, separators=(',', ':'))
    elif isinstance(normalized, dict):
        return json.dumps([normalized], separators=(',', ':'))
    else:
        return json.dumps(normalized, separators=(',', ':'))

# --- Startup Event: Auto-Initialize Car ---
@app.on_event("startup")
async def startup_event():
    """
    Auto-initialize a random car when the server starts.
    """
    global current_car
    try:
        current_car = load_random_car_from_db()
        print(f"\n🚗 Car Model Server initialized:")
        print(f"   Car: {current_car['manufacturer']} {current_car['model']} {current_car.get('year', '')}")
        print(f"   Available fields: {list(current_car['commands'].keys())}")
        print(f"   Field counts:")
        for field, value in current_car['commands'].items():
            print(f"      {field}: {len(value) if isinstance(value, list) else 1} command(s)")
    except Exception as e:
        print(f"❌ Failed to initialize car: {str(e)}")
        import traceback
        traceback.print_exc()

# --- Endpoints ---

@app.get("/car-info")
async def get_car_info():
    """
    Get information about the current initialized car.
    """
    if not current_car:
        raise HTTPException(500, detail="Car not initialized. Server may have failed to start properly.")
    
    return CarModelInfo(
        manufacturer=current_car["manufacturer"],
        model=current_car["model"],
        year=current_car["year"],
        country_region=current_car["country_region"],
        type_=current_car["type_"],
        fields_available=list(current_car["commands"].keys()),
        Pack_SOC=current_car["commands"].get("Pack_SOC"),
        Pack_Voltage=current_car["commands"].get("Pack_Voltage"),
        Pack_SOH=current_car["commands"].get("Pack_SOH")
    )

@app.post("/test-field", response_model=TestFieldResponse)
async def test_field(request: TestFieldRequest):
    """
    Test if a given field value matches the current car's field value.
    Returns matched=True if the values match, False otherwise.
    """
    if not current_car:
        raise HTTPException(500, detail="Car not initialized. Server may have failed to start properly.")
    
    field = request.field
    test_value = request.value
    
    # Debug logging
    print(f"\n🔍 Testing field: {field}")
    print(f"   Test value type: {type(test_value)}")
    print(f"   Test value: {test_value[:2] if isinstance(test_value, list) and len(test_value) > 2 else test_value}")
    
    # Check if the car has this field
    if field not in current_car["commands"]:
        print(f"   ❌ Field '{field}' not available in current car")
        print(f"   Available fields: {list(current_car['commands'].keys())}")
        return TestFieldResponse(
            field=field,
            matched=False,
            message=f"Field '{field}' not available in current car"
        )
    
    # Get the actual value for this field
    actual_value = current_car["commands"][field]
    print(f"   Actual value type: {type(actual_value)}")
    print(f"   Actual value: {actual_value[:2] if isinstance(actual_value, list) and len(actual_value) > 2 else actual_value}")
    
    # Normalize both values for comparison
    test_value_str = normalize_command_value(test_value)
    actual_value_str = normalize_command_value(actual_value)
    
    print(f"   Test normalized: {test_value_str[:100]}...")
    print(f"   Actual normalized: {actual_value_str[:100]}...")
    
    matched = test_value_str == actual_value_str
    
    print(f"   {'✅ MATCH!' if matched else '❌ No match'}")
    
    return TestFieldResponse(
        field=field,
        matched=matched,
        message="Match!" if matched else "No match",
        expected_value=actual_value if not matched else None
    )

@app.post("/reset")
async def reset_car():
    """
    Reset and reinitialize with a new random car.
    """
    global current_car
    try:
        current_car = load_random_car_from_db()
        print(f"\n🔄 Car reset:")
        print(f"   New car: {current_car['manufacturer']} {current_car['model']} {current_car.get('year', '')}")
        print(f"   Available fields: {list(current_car['commands'].keys())}")
        
        return {
            "status": "reset",
            "message": f"New car initialized: {current_car['manufacturer']} {current_car['model']}",
            "car_info": CarModelInfo(
                manufacturer=current_car["manufacturer"],
                model=current_car["model"],
                year=current_car["year"],
                country_region=current_car["country_region"],
                type_=current_car["type_"],
                fields_available=list(current_car["commands"].keys()),
                Pack_SOC=current_car["commands"].get("Pack_SOC"),
                Pack_Voltage=current_car["commands"].get("Pack_Voltage"),
                Pack_SOH=current_car["commands"].get("Pack_SOH")
            )
        }
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to reset car: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)