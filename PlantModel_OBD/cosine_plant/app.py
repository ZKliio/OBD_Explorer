"""
Attack Server (Main Server)
Handles user input and orchestrates iterative attacks against the Car Model Server.
Fetches target car info from Car Model Server (which auto-initializes).
NOW WITH RE-SORTING: After each field match, re-calculates similarity and re-sorts candidate list.
"""
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import httpx

app = FastAPI(title="Attack Server (Main)")

# Serve static files
app.mount(
    "/static",
    StaticFiles(directory="./static", html=True),
    name="static",
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
CAR_MODEL_SERVER_URL = "http://localhost:8001"  # Car Model Server URL

# --- Models ---
class IterativeAttackRequest(BaseModel):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None

class FieldTestAttempt(BaseModel):
    field: str
    candidate_car: str  # e.g., "Toyota Prius 2020"
    candidate_rank: int
    test_value: str  # JSON string of the command value tested
    matched: bool
    message: str

class IterationDetail(BaseModel):
    field: str
    attempts: List[FieldTestAttempt]
    final_status: str  # "matched" or "not_found"
    matched_car: Optional[str] = None
    matched_value: Optional[str] = None
    list_resorted: bool  # NEW: indicates if list was re-sorted before this field
    resorted_list: Optional[List[Dict[str, Any]]] = None  # Top 20 after re-sorting

class IterativeAttackResponse(BaseModel):
    target_car_info: Dict[str, Any]  # Info about the target car from Car Model Server
    initial_identifier: str  # e.g., "hyundai_ioniq10"
    initial_sorted_list: List[Dict[str, Any]]  # Top 20 initial matches
    iterations: List[IterationDetail]  # Detailed log of each field's attempts
    final_string: str  # The accumulated identifier string
    final_matched_fields: Dict[str, bool]  # Which fields were successfully matched
    attack_summary: Dict[str, Any]  # Overall attack statistics

# --- Domain & Loaders ---
class Car:
    def __init__(self, manufacturer: str, model: str, year: int = None, 
                 country_region: str = "", type_: str = ""):
        self.manufacturer = manufacturer
        self.model = model
        self.year = year
        self.country_region = country_region
        self.type_ = type_
        self.commands: Dict[str, List[Dict[str, Any]]] = {}

    def add_command(self, field: str, raw_json: Any):
        """
        Accept a JSON‐string or already parsed dict/list and normalize it
        into one or more dicts under self.commands[field].
        """
        parsed = (
            json.loads(raw_json)
            if isinstance(raw_json, str)
            else raw_json
        )

        cmd_list: List[Dict[str, Any]]
        if isinstance(parsed, dict):
            cmd_list = [parsed]
        elif isinstance(parsed, list):
            cmd_list = [p for p in parsed if isinstance(p, dict)]
        else:
            return

        self.commands.setdefault(field, []).extend(cmd_list)

def load_all_known_cars() -> List[Car]:
    """
    Loads all unique car entries and their commands from the DB.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT manufacturer, model, year, country_region, type,
               Pack_Voltage, Pack_SOC, Pack_SOH
        FROM vehicle_pack_commands
    """)
    rows = cur.fetchall()
    conn.close()
    
    cars_map: Dict[Tuple[str, str, int, str, str], Car] = {}

    for (manufacturer, model, year, country_region, type_,
         raw_vol, raw_soc, raw_soh) in rows:
        key = (manufacturer, model, year, country_region, type_)

        if key not in cars_map:
            cars_map[key] = Car(manufacturer, model, year, country_region, type_)

        car = cars_map[key]

        if raw_vol:
            car.add_command("Pack_Voltage", raw_vol)
        if raw_soc:
            car.add_command("Pack_SOC", raw_soc)
        if raw_soh:
            car.add_command("Pack_SOH", raw_soh)

    return list(cars_map.values())

def canonical_identifier(manufacturer: Optional[str] = None, 
                        model: Optional[str] = None,
                        year: Optional[int] = None, 
                        country_region: Optional[str] = None, 
                        type_: Optional[str] = None) -> str:
    """
    Creates a consistent canonical identifier for car with all attributes.
    If no manufacturer/model provided, returns "unknown" for brute force attack.
    """
    if not manufacturer and not model:
        identifier = "unknown"
    else:
        m = (manufacturer or "").strip().lower().replace(" ", "")
        mod = (model or "").strip().lower().replace(" ", "")
        
        if m and mod:
            identifier = f"{m}_{mod}"
        elif m:
            identifier = m
        elif mod:
            identifier = mod
        else:
            identifier = "unknown"

    if year is not None:
        identifier += f"_{year}"
    if country_region:
        identifier += f"_{country_region.strip().lower().replace(' ', '')}"
    if type_:
        identifier += f"_{type_.strip().lower().replace(' ', '')}"
    
    return identifier

def stringify_commands(cmd_list: List[Dict[str, Any]]) -> str:
    """
    Convert list of commands to a consistent string representation.
    Preserves original order and does NOT sort keys.
    """
    if not cmd_list:
        return ""
    
    # Keep original order, don't sort commands or keys
    return json.dumps(cmd_list, separators=(',', ':'))

def build_car_identifier_with_commands(car: Car, fields_to_include: List[str]) -> str:
    """
    Build a complete identifier string for a car including specified command fields.
    This mirrors what current_identifier looks like during the attack.
    """
    base_id = canonical_identifier(
        car.manufacturer,
        car.model,
        car.year,
        car.country_region,
        car.type_
    )
    
    for field in fields_to_include:
        if field in car.commands and car.commands[field]:
            cmd_str = stringify_commands(car.commands[field])
            base_id += f"_{field}:{cmd_str}"
    
    return base_id

def calculate_similarity_scores(target_identifier: str, 
                                car_identifiers: List[str]) -> np.ndarray:
    """
    Calculate cosine similarity between target identifier and all car identifiers.
    Returns array of similarity scores.
    """
    all_ids = car_identifiers + [target_identifier]
    
    # Try TF-IDF vectorization
    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=1,
            max_df=1.0
        )
        vectorizer.fit(all_ids)
        target_vec = vectorizer.transform([target_identifier])
        car_vecs = vectorizer.transform(car_identifiers)
        similarities = cosine_similarity(target_vec, car_vecs).flatten()
        return similarities
    except:
        # Fallback to uniform similarity if vectorization fails
        return np.ones(len(car_identifiers))

# --- Car Model Server Communication ---
async def get_target_car_info():
    """
    Fetch the current target car info from the Car Model Server.
    The Car Model Server auto-initializes on startup.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CAR_MODEL_SERVER_URL}/car-info")
        response.raise_for_status()
        return response.json()

async def test_field_with_car_model(field: str, value: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Test a field value against the Car Model Server.
    Returns the response indicating match or no match.
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAR_MODEL_SERVER_URL}/test-field",
            json={"field": field, "value": value}
        )
        response.raise_for_status()
        return response.json()

# --- Main Attack Endpoint ---
@app.post("/iterative-attack", response_model=IterativeAttackResponse)
async def iterative_guessing_attack(request: IterativeAttackRequest):
    """
    Enhanced iterative attack with RE-SORTING after each successful field match.
    
    PROCESS:
    1. Fetch target car info from Car Model Server (already initialized)
    2. Create initial identifier from user input (manufacturer, model, year, etc.)
    3. Generate initial sorted list based on similarity to initial identifier
    4. For each field (Pack_SOC, Pack_Voltage, Pack_SOH):
       a. Test candidates in current sorted order
       b. On match: Update current_identifier, RE-CALCULATE similarities, RE-SORT list
       c. Continue to next field with newly sorted list
    5. Return detailed log showing how list order changed with each match
    """
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]
    
    # Step 1: Fetch target car info from Car Model Server
    try:
        target_car_info = await get_target_car_info()
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to fetch target car info: {str(e)}")
    
    # Step 2: Create initial identifier from user input
    initial_identifier = canonical_identifier(
        request.manufacturer,
        request.model,
        request.year,
        request.country_region,
        request.type_
    )
    
    # Step 3: Load all known cars
    known_cars = load_all_known_cars()
    if not known_cars:
        raise HTTPException(500, detail="No cars in database")
    
    # Create base car identifiers (name only, no commands yet)
    car_base_identifiers = []
    for car in known_cars:
        car_id = canonical_identifier(
            car.manufacturer,
            car.model,
            car.year,
            car.country_region,
            car.type_
        )
        car_base_identifiers.append(car_id)
    
    # Calculate initial similarities
    initial_similarities = calculate_similarity_scores(initial_identifier, car_base_identifiers)
    
    # Create initial match list
    initial_matches = []
    for idx, car in enumerate(known_cars):
        initial_matches.append({
            "car": car,
            "car_base_identifier": car_base_identifiers[idx],
            "similarity": float(initial_similarities[idx]),
            "rank": 0
        })
    
    # Sort by initial similarity
    initial_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Assign initial ranks
    for i, match in enumerate(initial_matches):
        match["rank"] = i + 1
    
    # Save initial top 20 for response
    initial_list_info = [
        {
            "rank": match["rank"],
            "manufacturer": match["car"].manufacturer,
            "model": match["car"].model,
            "year": match["car"].year,
            "similarity": match["similarity"],
            "identifier": match["car_base_identifier"]
        }
        for match in initial_matches[:20]
    ]
    
    # Step 4: Iterative field testing with re-sorting
    current_identifier = initial_identifier
    matched_fields_so_far = []  # Track which fields have been matched
    iterations = []
    final_matched_fields = {}
    total_attempts = 0
    
    for field_idx, field in enumerate(ALL_PARAMS):
        field_attempts = []
        field_matched = False
        matched_car_name = None
        matched_value = None
        list_was_resorted = field_idx > 0 and len(matched_fields_so_far) > 0
        
        # Capture the current sorted list (top 20) before testing this field
        current_sorted_list = [
            {
                "rank": match["rank"],
                "manufacturer": match["car"].manufacturer,
                "model": match["car"].model,
                "year": match["car"].year,
                "similarity": match["similarity"],
                "identifier": build_car_identifier_with_commands(match["car"], matched_fields_so_far)
            }
            for match in initial_matches[:20]
        ] if list_was_resorted else None
        
        # Test each candidate in current order
        for match_info in initial_matches:
            car = match_info["car"]
            
            # Check if this car has the field
            if field not in car.commands or not car.commands[field]:
                continue
            
            # Get the field value to test
            test_value = car.commands[field]
            test_value_str = stringify_commands(test_value)
            
            # Test with Car Model Server
            total_attempts += 1
            try:
                test_response = await test_field_with_car_model(field, test_value)
                matched = test_response["matched"]
                
                # Log this attempt
                attempt = FieldTestAttempt(
                    field=field,
                    candidate_car=f"{car.manufacturer} {car.model} {car.year or ''}".strip(),
                    candidate_rank=match_info["rank"],
                    test_value=test_value_str,
                    matched=matched,
                    message=test_response["message"]
                )
                field_attempts.append(attempt)
                
                # If matched, update identifier and RE-SORT for next field
                if matched:
                    field_matched = True
                    matched_car_name = f"{car.manufacturer} {car.model}"
                    matched_value = test_value_str
                    
                    # Update current identifier (cumulative)
                    current_identifier += f"_{field}:{test_value_str}"
                    matched_fields_so_far.append(field)
                    
                    # RE-SORT: Recalculate similarities based on updated identifier
                    # Build full identifiers for all cars including matched fields
                    updated_car_identifiers = []
                    for c in known_cars:
                        full_car_id = build_car_identifier_with_commands(c, matched_fields_so_far)
                        updated_car_identifiers.append(full_car_id)
                    
                    # Recalculate similarities
                    new_similarities = calculate_similarity_scores(
                        current_identifier, 
                        updated_car_identifiers
                    )
                    
                    # Update similarity scores in initial_matches
                    for idx, match in enumerate(initial_matches):
                        match["similarity"] = float(new_similarities[idx])
                    
                    # RE-SORT by new similarities
                    initial_matches.sort(key=lambda x: x["similarity"], reverse=True)
                    
                    # Update ranks
                    for i, match in enumerate(initial_matches):
                        match["rank"] = i + 1
                    
                    break
                
            except Exception as e:
                # Log error but continue
                attempt = FieldTestAttempt(
                    field=field,
                    candidate_car=f"{car.manufacturer} {car.model} {car.year or ''}".strip(),
                    candidate_rank=match_info["rank"],
                    test_value=test_value_str,
                    matched=False,
                    message=f"Error: {str(e)}"
                )
                field_attempts.append(attempt)
        
        # Log iteration results for this field
        iteration_detail = IterationDetail(
            field=field,
            attempts=field_attempts,
            final_status="matched" if field_matched else "not_found",
            matched_car=matched_car_name,
            matched_value=matched_value,
            list_resorted=list_was_resorted,
            resorted_list=current_sorted_list
        )
        iterations.append(iteration_detail)
        final_matched_fields[field] = field_matched
    
    # Calculate attack statistics
    matched_count = sum(1 for matched in final_matched_fields.values() if matched)
    attack_summary = {
        "total_fields": len(ALL_PARAMS),
        "fields_matched": matched_count,
        "fields_not_found": len(ALL_PARAMS) - matched_count,
        "total_attempts": total_attempts,
        "success_rate": f"{matched_count}/{len(ALL_PARAMS)}",
        "average_attempts_per_field": total_attempts / len(ALL_PARAMS) if ALL_PARAMS else 0
    }
    
    return IterativeAttackResponse(
        target_car_info=target_car_info,
        initial_identifier=initial_identifier,
        initial_sorted_list=initial_list_info,
        iterations=iterations,
        final_string=current_identifier,
        final_matched_fields=final_matched_fields,
        attack_summary=attack_summary
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)