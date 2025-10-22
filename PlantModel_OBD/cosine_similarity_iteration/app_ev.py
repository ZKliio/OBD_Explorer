# Final working version with enhanced iterative attack endpoint
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
import hashlib

app = FastAPI()

# # Serve ./static/index.html at /static
app.mount(
    "/static_evl",
    StaticFiles(directory="./static_evl", html=True),
    name="static_evl",
)

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "../../OBD.db"  # replace with your real DB path

# --- Models ----------------------------------------------------------------
# Request model (input from frontend side)
class CheckRequest(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    new_car_attributes: Optional[str] = None
    target_json_arr: List[Dict[str, Any]]
    field: Optional[str] = None

# For iterative attack
class IterativeAttackRequest(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None

class CarWithCommands(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    Pack_SOC: Optional[List[Dict[str, Any]]] = None
    Pack_Voltage: Optional[List[Dict[str, Any]]] = None
    Pack_SOH: Optional[List[Dict[str, Any]]] = None

# Response models
class Match(BaseModel):
    similarity_score: float
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    commands_found: Dict[str, Dict[str, Any]]

# Enhanced Response Model with final_string and similarity_evolution
class IterativeAttackResponse(BaseModel):
    target_car: CarWithCommands
    matches: List[Match]
    best_match_identifier: str
    final_string: str  # EXPLICITLY ADDED for vehicle testing
    final_similarity_score: str
    commands_matched: Dict[str, bool]
    matched_field_values: Dict[str, str]  # Track actual field values
    initial_sorted_list: List[Dict[str, Any]]  # Show the initial sorted list
    iterations: List[Dict[str, Any]]
    similarity_evolution: List[Dict[str, Any]]  # Track how rankings change after each iteration
    unknown_car_info: Optional[Dict[str, Any]] = None  # Info about unknown car handling

# --- Domain & Loaders ------------------------------------------------------
class Car:
    def __init__(self, manufacturer: str, model: str, year: int = None, 
                 country_region: str = "", type_: str = ""):
        self.manufacturer = manufacturer
        self.model = model
        self.year = year
        self.country_region = country_region
        self.type_ = type_
        # each key is a field like "Pack_SOC", value is a list of JSON dicts
        self.commands: Dict[str, List[Dict[str, Any]]] = {}

    def add_command(self, field: str, raw_json: Any):
        """
        Accept a JSON‐string or already parsed dict/list and normalize it
        into one or more dicts under self.commands[field].
        """
        # parse if it's a string
        parsed = (
            json.loads(raw_json)
            if isinstance(raw_json, str)
            else raw_json
        )

        # unify into a list of dicts
        cmd_list: List[Dict[str, Any]]
        if isinstance(parsed, dict):
            cmd_list = [parsed]
        elif isinstance(parsed, list):
            cmd_list = [p for p in parsed if isinstance(p, dict)]
        else:
            # unrecognized format
            return

        # append each dict
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
    
    # map (manufacturer, model, year, country_region, type) to each unique Car
    cars_map: Dict[Tuple[str, str, int, str, str], Car] = {}

    for (
        manufacturer,
        model,
        year,
        country_region,
        type_,
        raw_vol,
        raw_soc,
        raw_soh,
    ) in rows:
        key = (manufacturer, model, year, country_region, type_)

        # Create Car if not exists in cars_map
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

def get_specific_car(manufacturer: str, model: str, year: Optional[int] = None,
                    country_region: Optional[str] = None, 
                    type_: Optional[str] = None) -> Optional[Car]:
    """
    Get a specific car from database matching all provided criteria.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Build dynamic query based on provided fields
    query = """
        SELECT manufacturer, model, year, country_region, type,
               Pack_Voltage, Pack_SOC, Pack_SOH
        FROM vehicle_pack_commands
        WHERE LOWER(manufacturer) = LOWER(?) AND LOWER(model) = LOWER(?)
    """
    params = [manufacturer, model]
    
    if year is not None:
        query += " AND year = ?"
        params.append(year)
    if country_region:
        query += " AND LOWER(country_region) = LOWER(?)"
        params.append(country_region)
    if type_:
        query += " AND LOWER(type) = LOWER(?)"
        params.append(type_)
    
    query += " LIMIT 1"
    
    cur.execute(query, params)
    row = cur.fetchone()
    conn.close()
    
    if not row:
        return None
    
    (manufacturer, model, year, country_region, type_,
     raw_vol, raw_soc, raw_soh) = row
    
    car = Car(manufacturer, model, year, country_region, type_)
    
    if raw_vol:
        car.add_command("Pack_Voltage", raw_vol)
    if raw_soc:
        car.add_command("Pack_SOC", raw_soc)
    if raw_soh:
        car.add_command("Pack_SOH", raw_soh)
    
    return car

# --- Utils -----------------------------------------------------------------

def canonical_identifier(manufacturer: str, model: str,
                        year: Optional[int] = None, 
                        country_region: Optional[str] = None, 
                        type_: Optional[str] = None) -> str:
    """
    Creates a consistent canonical identifier for car with all attributes.
    Strip whitespace, lowercase, and remove ALL spaces.
    """
    m = manufacturer.strip().lower().replace(" ", "")
    mod = model.strip().lower().replace(" ", "")
    identifier = f"{m}_{mod}"

    if year is not None:
        identifier += f"_{year}"
    if country_region:
        identifier += f"_{country_region.strip().lower().replace(' ', '')}"
    if type_:
        identifier += f"_{type_.strip().lower().replace(' ', '')}"
    
    return identifier

def stringify_commands(cmd_list: List[Dict[str, Any]], use_hash: bool = False) -> str:
    """
    Convert list of commands to a string representation.
    If use_hash=True, create a shorter hash for the final string.
    CHANGED: Default behavior now returns full JSON string, not hash
    """
    if not cmd_list:
        return ""
    
    # Sort commands for consistent ordering
    sorted_cmds = sorted(cmd_list, key=lambda x: json.dumps(x, sort_keys=True))
    full_string = json.dumps(sorted_cmds, sort_keys=True, separators=(',', ':'))
    
    if use_hash:
        # Create a shorter hash representation if explicitly requested
        return hashlib.md5(full_string.encode()).hexdigest()[:8]
    
    return full_string

# --- Enhanced Iterative Attack Endpoint ------------------------------------

@app.get("/car/{manufacturer}/{model}", response_model=CarWithCommands)
async def get_car_commands(
    manufacturer: str,
    model: str,
    year: Optional[int] = None,
    country_region: Optional[str] = None,
    type_: Optional[str] = None
):
    """
    Get a specific car and its commands from the database.
    This serves as the "plant model" for the iterative attack.
    """
    car = get_specific_car(manufacturer, model, year, country_region, type_)
    
    if not car:
        raise HTTPException(404, detail=f"Car not found: {manufacturer} {model}")
    
    return CarWithCommands(
        manufacturer=car.manufacturer,
        model=car.model,
        year=car.year,
        country_region=car.country_region,
        type_=car.type_,
        Pack_SOC=car.commands.get("Pack_SOC", []),
        Pack_Voltage=car.commands.get("Pack_Voltage", []),
        Pack_SOH=car.commands.get("Pack_SOH", [])
    )

@app.post("/iterative-attack", response_model=IterativeAttackResponse)
async def iterative_guessing_attack(
    request: IterativeAttackRequest,
    success_threshold: float = 0.80,
    use_hash_in_final: bool = False,
    top_n_strings: int = 10  # Number of final strings to generate for all cars
):
    """
    Enhanced iterative guessing attack with explicit final string generation.
    For ALL cars (known and unknown): generates multiple final strings using field values from ranked matches.
    
    STEPS:
    1. Initialize car object with manufacturer, model, year, country/region
    2. Create initial string identifier
    3. Vectorize and compare to DB rows, create sorted list by similarity
    4. Generate multiple final strings using values from ranked list
    5. Return final_strings list with detailed field source tracking
    """
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]
    
    # Step 1: Try to load the target car (plant model)
    target_car = get_specific_car(
        request.manufacturer, 
        request.model, 
        request.year,
        request.country_region,
        request.type_
    )
    
    is_unknown_car = target_car is None
    unknown_car_info = None
    
    # Load all known cars for comparison
    known_cars = load_all_known_cars()
    if not known_cars:
        raise HTTPException(500, detail="No cars in database")
    
    # Step 2: Create initial identifier string
    initial_identifier = canonical_identifier(
        request.manufacturer, 
        request.model,
        request.year,
        request.country_region,
        request.type_
    )
    
    # Step 3: Vectorize and compare car identifiers to create initial sorted list
    car_identifiers = []
    car_id_map = {}  # Map identifier back to car object
    
    for car in known_cars:
        car_id = canonical_identifier(
            car.manufacturer, 
            car.model, 
            car.year,
            car.country_region,
            car.type_
        )
        car_identifiers.append(car_id)
        car_id_map[car_id] = car
    
    # Add the target car identifier for vectorization
    all_car_ids = car_identifiers + [initial_identifier]
    
    # Create TF-IDF vectorizer for car names
    name_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_df=1.0
    )
    
    try:
        name_vectorizer.fit(all_car_ids)
        target_vec = name_vectorizer.transform([initial_identifier])
        car_vecs = name_vectorizer.transform(car_identifiers)
        
        # Calculate similarities
        name_similarities = cosine_similarity(target_vec, car_vecs).flatten()
    except:
        # Fallback to weighted matching if vectorization fails
        name_similarities = np.zeros(len(car_identifiers))
        for i, car_id in enumerate(car_identifiers):
            score = 0.0
            if request.manufacturer and request.manufacturer.lower() in car_id.lower():
                score += 0.45
            if request.model and request.model.lower() in car_id.lower():
                score += 0.35
            if request.year and str(request.year) in car_id:
                score += 0.1
            if request.country_region and request.country_region.lower() in car_id.lower():
                score += 0.05
            if request.type_ and request.type_.lower() in car_id.lower():
                score += 0.05
            name_similarities[i] = score
    
    # Step 4: Create initial sorted list of cars (HIGHEST TO LOWEST)
    initial_matches = []
    for idx, car in enumerate(known_cars):
        initial_matches.append({
            "car": car,
            "car_identifier": car_identifiers[idx],
            "initial_similarity": float(name_similarities[idx])
        })
    
    # Sort by initial similarity (highest to lowest)
    initial_matches.sort(key=lambda x: x["initial_similarity"], reverse=True)
    
    # UNIFIED LOGIC FOR BOTH KNOWN AND UNKNOWN CARS
    # Generate multiple final strings using field values from ranked matches
    final_strings_list = []
    field_combinations = []
    
    # Track similarity evolution through iterations
    similarity_evolution = []
    
    # For each position in the ranked list, create a final string
    for primary_idx in range(min(top_n_strings, len(initial_matches))):
        final_string_parts = [initial_identifier]
        field_sources = {}
        
        # For each parameter, try to get value from primary match first
        for param in ALL_PARAMS:
            value_found = False
            
            # Start from primary_idx and go down the list to find a value
            for fallback_idx in range(primary_idx, len(initial_matches)):
                car = initial_matches[fallback_idx]["car"]
                
                if param in car.commands and car.commands[param]:
                    # Found a value for this field
                    cmd_string = stringify_commands(car.commands[param], use_hash=use_hash_in_final)
                    final_string_parts.append(f"{param}:{cmd_string}")
                    
                    field_sources[param] = {
                        "from_car": f"{car.manufacturer} {car.model}",
                        "rank": fallback_idx + 1,
                        "value": cmd_string,
                        "car_identifier": initial_matches[fallback_idx]["car_identifier"]
                    }
                    value_found = True
                    break
            
            if not value_found:
                # No car in the list has this field
                field_sources[param] = {
                    "from_car": "none",
                    "rank": -1,
                    "value": "not_found"
                }
        
        # Build the final string for this combination
        final_string = "_".join(final_string_parts)
        
        final_strings_list.append({
            "final_string": final_string,
            "primary_match": f"{initial_matches[primary_idx]['car'].manufacturer} {initial_matches[primary_idx]['car'].model}",
            "primary_rank": primary_idx + 1,
            "field_sources": field_sources
        })
        
        field_combinations.append(field_sources)
    
    # === SIMILARITY EVOLUTION TRACKING ===
    # Track how similarities change as we add each parameter to the primary string
    # Use the first generated string (primary_idx=0) for tracking evolution
    if final_strings_list:
        evolution_string_parts = [initial_identifier]
        previous_rankings = {car_identifiers[i]: i for i in range(len(car_identifiers))}
        
        # Initial state (just car name/identifier)
        similarity_evolution.append({
            "iteration": 0,
            "parameter_added": "initial_identifier",
            "current_string": initial_identifier,
            "top_matches": [
                {
                    "rank": i + 1,
                    "manufacturer": initial_matches[i]["car"].manufacturer,
                    "model": initial_matches[i]["car"].model,
                    "year": initial_matches[i]["car"].year,
                    "similarity_score": initial_matches[i]["initial_similarity"],
                    "rank_change": 0
                }
                for i in range(min(20, len(initial_matches)))
            ]
        })
        
        # For each parameter in the primary string, recalculate similarities
        iteration_num = 1
        for param in ALL_PARAMS:
            # Add this parameter to the evolution string if available
            if param in final_strings_list[0]["field_sources"]:
                source_info = final_strings_list[0]["field_sources"][param]
                if source_info["value"] != "not_found":
                    evolution_string_parts.append(f"{param}:{source_info['value']}")
            
            current_evolution_string = "_".join(evolution_string_parts)
            
            # Recalculate similarities with the updated string
            all_strings_for_comparison = car_identifiers + [current_evolution_string]
            
            try:
                iteration_vectorizer = TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=1,
                    max_df=1.0
                )
                iteration_vectorizer.fit(all_strings_for_comparison)
                iteration_target_vec = iteration_vectorizer.transform([current_evolution_string])
                iteration_car_vecs = iteration_vectorizer.transform(car_identifiers)
                
                iteration_similarities = cosine_similarity(iteration_target_vec, iteration_car_vecs).flatten()
            except:
                # Fallback if vectorization fails
                iteration_similarities = name_similarities.copy()
            
            # Create new rankings based on current similarities
            iteration_matches = []
            for idx, car in enumerate(known_cars):
                iteration_matches.append({
                    "car": car,
                    "car_identifier": car_identifiers[idx],
                    "similarity": float(iteration_similarities[idx]),
                    "previous_rank": previous_rankings.get(car_identifiers[idx], -1)
                })
            
            # Sort by current similarity
            iteration_matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Calculate rank changes
            current_rankings = {match["car_identifier"]: i for i, match in enumerate(iteration_matches)}
            
            top_matches_info = []
            for i, match in enumerate(iteration_matches[:20]):
                prev_rank = match["previous_rank"]
                current_rank = i
                rank_change = prev_rank - current_rank if prev_rank != -1 else 0
                
                top_matches_info.append({
                    "rank": current_rank + 1,
                    "manufacturer": match["car"].manufacturer,
                    "model": match["car"].model,
                    "year": match["car"].year,
                    "similarity_score": match["similarity"],
                    "previous_rank": prev_rank + 1 if prev_rank != -1 else None,
                    "rank_change": rank_change  # Positive means moved up, negative means moved down
                })
            
            similarity_evolution.append({
                "iteration": iteration_num,
                "parameter_added": param,
                "current_string": current_evolution_string,
                "top_matches": top_matches_info
            })
            
            # Update previous rankings for next iteration
            previous_rankings = current_rankings
            iteration_num += 1
    
    # If unknown car, create synthetic target
    if is_unknown_car:
        # Create synthetic target using first match's values (for compatibility)
        synthetic_commands = {}
        for param in ALL_PARAMS:
            for match_info in initial_matches:
                car = match_info["car"]
                if param in car.commands and car.commands[param]:
                    synthetic_commands[param] = car.commands[param]
                    break
        
        target_car = Car(
            manufacturer=request.manufacturer,
            model=request.model,
            year=request.year,
            country_region=request.country_region or "",
            type_=request.type_ or ""
        )
        target_car.commands = synthetic_commands
        
        unknown_car_info = {
            "status": "unknown_car",
            "message": f"Car not found in database: {request.manufacturer} {request.model}",
            "final_strings_generated": len(final_strings_list),
            "final_strings_list": final_strings_list,
            "closest_match": {
                "manufacturer": initial_matches[0]["car"].manufacturer,
                "model": initial_matches[0]["car"].model,
                "year": initial_matches[0]["car"].year,
                "similarity": initial_matches[0]["initial_similarity"]
            } if initial_matches else None
        }
    else:
        # Known car - add info about the actual car and string generation
        unknown_car_info = {
            "status": "known_car",
            "message": f"Car found in database: {request.manufacturer} {request.model}",
            "final_strings_generated": len(final_strings_list),
            "final_strings_list": final_strings_list,
            "actual_car_fields": {
                param: stringify_commands(target_car.commands.get(param, []), use_hash=use_hash_in_final)
                for param in ALL_PARAMS
                if param in target_car.commands and target_car.commands[param]
            }
        }
    
    # Use the first generated string as the primary final_string
    final_string = final_strings_list[0]["final_string"] if final_strings_list else initial_identifier
    
    # Track iterations for known cars (enhanced with comparison to actual values)
    iterations = []
    commands_matched = {}
    matched_field_values = {}
    
    if not is_unknown_car:
        # For known cars, also track which strings match the actual car's fields
        for param in ALL_PARAMS:
            target_cmds = target_car.commands.get(param, [])
            
            if not target_cmds:
                iterations.append({
                    "parameter": param,
                    "status": "no_target_command",
                    "matched_car": None,
                    "similarity_score": 0.0,
                    "note": "Field not available for this car"
                })
                continue
            
            target_cmd_string = stringify_commands(target_cmds, use_hash=use_hash_in_final)
            
            # Check which generated strings have the correct value for this field
            matching_strings = []
            for i, fs in enumerate(final_strings_list):
                if param in fs["field_sources"] and fs["field_sources"][param]["value"] == target_cmd_string:
                    matching_strings.append(i + 1)  # 1-indexed for readability
            
            iterations.append({
                "parameter": param,
                "actual_value": target_cmd_string,
                "matching_string_indices": matching_strings,
                "total_strings_with_correct_value": len(matching_strings),
                "note": f"Found in {len(matching_strings)}/{len(final_strings_list)} generated strings"
            })
            
            # Check if first string has correct value
            if final_strings_list and param in final_strings_list[0]["field_sources"]:
                if final_strings_list[0]["field_sources"][param]["value"] == target_cmd_string:
                    commands_matched[param] = True
                    matched_field_values[param] = target_cmd_string
                else:
                    commands_matched[param] = False
            else:
                commands_matched[param] = False
    else:
        # For unknown cars, use simplified tracking
        for param in ALL_PARAMS:
            if final_strings_list and param in final_strings_list[0]["field_sources"]:
                source_info = final_strings_list[0]["field_sources"][param]
                if source_info["value"] != "not_found":
                    commands_matched[param] = True
                    matched_field_values[param] = source_info["value"]
                else:
                    commands_matched[param] = False
            else:
                commands_matched[param] = False
    
    # Calculate final similarity score
    matched_count = sum(1 for matched in commands_matched.values() if matched)
    final_score = f'{matched_count}/{len(ALL_PARAMS)}' if ALL_PARAMS else '0/0'
    
    # Build final matches list with enhanced information
    final_matches = []
    for match_info in initial_matches[:10]:
        car = match_info["car"]
        commands_found = {}
        
        # For known cars, check against actual values
        if not is_unknown_car:
            for param in ALL_PARAMS:
                car_cmds = car.commands.get(param, [])
                target_cmds = target_car.commands.get(param, [])
                if car_cmds and target_cmds:
                    car_cmd_string = stringify_commands(car_cmds, use_hash=use_hash_in_final)
                    target_cmd_string = stringify_commands(target_cmds, use_hash=use_hash_in_final)
                    if car_cmd_string == target_cmd_string:
                        commands_found[param] = {
                            "matched": True,
                            "value": target_cmd_string
                        }
        else:
            # For unknown cars, just show what commands the car has
            for param in ALL_PARAMS:
                if param in car.commands and car.commands[param]:
                    commands_found[param] = {
                        "available": True,
                        "value": stringify_commands(car.commands[param], use_hash=use_hash_in_final)
                    }
        
        command_match_rate = len(commands_found) / len(ALL_PARAMS) if ALL_PARAMS else 0
        overall_score = (0.4 * match_info["initial_similarity"] + 0.6 * command_match_rate)
        
        final_matches.append(
            Match(
                manufacturer=car.manufacturer,
                model=car.model,
                year=car.year,
                country_region=car.country_region,
                type_=car.type_,
                similarity_score=float(overall_score),
                commands_found=commands_found
            )
        )
    
    final_matches.sort(key=lambda m: m.similarity_score, reverse=True)
    
    # Create initial sorted list for visibility
    initial_list_info = [
        {
            "rank": i + 1,
            "manufacturer": match["car"].manufacturer,
            "model": match["car"].model,
            "year": match["car"].year,
            "initial_similarity": match["initial_similarity"],
            "identifier": match["car_identifier"]
        }
        for i, match in enumerate(initial_matches[:20])
    ]
    
    return IterativeAttackResponse(
        target_car=CarWithCommands(
            manufacturer=target_car.manufacturer,
            model=target_car.model,
            year=target_car.year,
            country_region=target_car.country_region,
            type_=target_car.type_,
            Pack_SOC=target_car.commands.get("Pack_SOC", []),
            Pack_Voltage=target_car.commands.get("Pack_Voltage", []),
            Pack_SOH=target_car.commands.get("Pack_SOH", [])
        ),
        best_match_identifier=final_string,
        final_string=final_string,
        final_similarity_score=final_score,
        commands_matched=commands_matched,
        matched_field_values=matched_field_values,
        initial_sorted_list=initial_list_info,
        iterations=iterations,
        similarity_evolution=similarity_evolution,  # NEW: Shows how rankings change after each parameter
        matches=final_matches,
        unknown_car_info=unknown_car_info  # Now contains final_strings_list for both known and unknown
    )