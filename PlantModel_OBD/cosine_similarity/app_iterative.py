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

app = FastAPI()

# # Serve ./static/index.html at /static
app.mount(
    "/static",
    StaticFiles(directory="./static", html=True),
    name="static",
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# # override the directory index
# @app.get("/static/")
# async def custom_index():
#     return FileResponse("static/index1.html")


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

# Response Models returned in a list
class CheckResponse(BaseModel):
    exists: bool
    matches: List[Match]
    new_car_attributes: str

class IterativeAttackResponse(BaseModel):
    target_car: CarWithCommands
    matches: List[Match]
    best_match_identifier: str
    final_similarity_score: float
    commands_matched: Dict[str, bool]
    iterations: List[Dict[str, Any]]


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
    Expects columns:
      manufacturer | model | year | country_region | type |
      Pack_Voltage | Pack_SOC | Pack_SOH
    where the last three are JSON blobs (object or array).
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

def canonicalize_json_to_string_sorted(obj: Dict[str, Any]) -> str:
    """
    Deterministic JSON→string by sorting keys and stripping whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

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

def extract_command_signature(cmd: Dict[str, Any]) -> str:
    """
    Extract only the core OBD-II identifying fields: request_id, response_id, and command.
    These three fields uniquely identify what data is being requested.
    """
    # Only use the essential OBD-II fields
    essential_fields = ['request_id', 'response_id', 'command']
    
    # Extract only essential fields
    signature_parts = []
    for field in essential_fields:
        if field in cmd:
            value = str(cmd[field]).strip().upper()  # Uppercase for consistency
            signature_parts.append(f"{field}:{value}")
    
    # Create a simple signature: "request_id:7E4|response_id:7EC|command:220105"
    return "|".join(signature_parts)

def normalize_command(cmd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a command dict to ensure consistent comparison.
    Converts numeric values to consistent format.
    """
    normalized = {}
    for k, v in cmd.items():
        if isinstance(v, (int, float)):
            # Normalize numbers to avoid floating point comparison issues
            normalized[k] = float(v)
        else:
            normalized[k] = v
    return normalized

# --- Endpoints --------------------------------------------------------------

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
):
    """
    Perform an iterative guessing attack:
    1. Load the target car and its commands
    2. Try to match each command iteratively
    3. Build up the identifier as matches are found
    """
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]
    
    # 1) Load the target car (plant model)
    target_car = get_specific_car(
        request.manufacturer, 
        request.model, 
        request.year,
        request.country_region,
        request.type_
    )
    
    if not target_car:
        raise HTTPException(404, detail=f"Target car not found: {request.manufacturer} {request.model}")
    
    # 2) Load all known cars for comparison
    known_cars = load_all_known_cars()
    if not known_cars:
        raise HTTPException(500, detail="No cars in database")
    
    # 3) Initialize the identifier with the base car name
    current_identifier = canonical_identifier(request.manufacturer, request.model)
    
    # 4) Track iterations and results
    iterations = []
    commands_matched = {param: False for param in ALL_PARAMS}
    
    # 5) Build the TF-IDF vocabulary
    all_texts = []
    
    # Add car identifiers
    for car in known_cars:
        car_id = canonical_identifier(car.manufacturer, car.model, car.year)
        all_texts.append(car_id)
    
    # Add the query car ID
    all_texts.append(current_identifier)
    
    # Add all command signatures
    for car in known_cars:
        for param, cmd_list in car.commands.items():
            for cmd in cmd_list:
                all_texts.append(extract_command_signature(cmd))
    
    # Add target command signatures
    for param in ALL_PARAMS:
        if param in target_car.commands:
            for cmd in target_car.commands[param]:
                all_texts.append(extract_command_signature(cmd))
    
    # Remove duplicates
    all_texts = list(set(all_texts))
    
    if len(all_texts) < 2:
        raise HTTPException(500, detail="Insufficient data for comparison")
    
    vectorizer = TfidfVectorizer(
        analyzer="char", 
        ngram_range=(2, 4),
        min_df=1,
        max_df=1.0
    )
    vectorizer.fit(all_texts)
    
    # 6) Iterate through each parameter
    for param in ALL_PARAMS:
        target_cmds = target_car.commands.get(param, [])
        if not target_cmds:
            iterations.append({
                "parameter": param,
                "status": "no_target_command",
                "similarity_scores": []
            })
            continue
        
        # Use first command as target
        target_cmd = target_cmds[0]
        target_sig = extract_command_signature(target_cmd)
        target_vec = vectorizer.transform([target_sig])
        
        # Compare against all known cars
        best_matches = []
        for car in known_cars:
            car_cmds = car.commands.get(param, [])
            if not car_cmds:
                continue
            
            for cmd in car_cmds:
                cmd_sig = extract_command_signature(cmd)
                cmd_vec = vectorizer.transform([cmd_sig])
                score = cosine_similarity(target_vec, cmd_vec).flatten()[0]
                
                best_matches.append({
                    "car": f"{car.manufacturer} {car.model} {car.year}",
                    "command_signature": cmd_sig,
                    "score": float(score)
                })
        
        # Sort by score
        best_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Check if we found a good match
        if best_matches and best_matches[0]["score"] >= success_threshold:
            commands_matched[param] = True
            # Append to identifier
            current_identifier += f"_{param[:3]}_{hash(target_sig) % 10000:04d}"
            
            iterations.append({
                "parameter": param,
                "status": "matched",
                "best_score": best_matches[0]["score"],
                "matched_command": best_matches[0]["command_signature"],
                "updated_identifier": current_identifier,
                "top_matches": best_matches[:3]
            })
        else:
            iterations.append({
                "parameter": param,
                "status": "no_match",
                "best_score": best_matches[0]["score"] if best_matches else 0.0,
                "top_matches": best_matches[:3] if best_matches else []
            })
    
    # 7) Calculate final similarity score
    matched_count = sum(1 for matched in commands_matched.values() if matched)
    final_score = matched_count / len(ALL_PARAMS)
    
     # --- Build a list of Match objects just like in guessing_attack ---
    matches: List[Match] = []
    for car in known_cars:
        # figure out which params this car “won” in the iterative loop
        commands_found: Dict[str, Any] = {}
        scores: List[float] = []

        for param in ALL_PARAMS:
            # did we ever mark this param as matched?
            if commands_matched[param]:
                # find the top match for this param in `iterations`
                # (we stored top_matches per param earlier)
                for it in iterations:
                    if it["parameter"] == param and it["status"] == "matched":
                        # take the first/top match
                        top = it["top_matches"][0]
                        # only record if this car shows up
                        if top["car"].startswith(f"{car.manufacturer} {car.model}"):
                            # commands_found[param] = top["command_signature"]
                            # scores.append(top["score"])
                            commands_found[param] = {
                                "command_signature": top["command_signature"],
                                "score": top["score"]
                                }
                        break
            else:
                scores.append(0.0)

        if scores:
            avg_score = float(np.mean(scores))
        else:
            avg_score = 0.0

        overall = min(
            1.0,
            0.5 * avg_score
            + 0.5 * final_score  # you can tweak these weights
        )

        # only include if we matched anything or scored high enough
        display_threshold = 0.8
        if commands_found or overall >= display_threshold:
            matches.append(
                Match(
                    manufacturer=car.manufacturer,
                    model=car.model,
                    year=car.year,
                    country_region=car.country_region,
                    type_=car.type_,
                    similarity_score=overall,
                    commands_found=commands_found,
                )
            )

    # sort descending
    matches.sort(key=lambda m: m.similarity_score, reverse=True)



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
        best_match_identifier=current_identifier,
        final_similarity_score=final_score,
        commands_matched=commands_matched,
        iterations=iterations,
        matches=matches,
    )

@app.post("/guessing-attack", response_model=CheckResponse)
async def guessing_attack_strategy(
    new_car_data: CheckRequest,
    success_threshold: float = 0.80,
    display_threshold: float = 0.50,
):
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]

    # 1) Decide which params to test
    if new_car_data.field:
        if new_car_data.field not in ALL_PARAMS:
            raise HTTPException(400, detail=f"Unknown field '{new_car_data.field}'.")
        if len(new_car_data.target_json_arr) != 1:
            raise HTTPException(
                400,
                detail="When specifying 'field', 'target_json_arr' must have exactly one object.",
            )
        key_params = [new_car_data.field]
    else:
        if len(new_car_data.target_json_arr) != len(ALL_PARAMS):
            raise HTTPException(
                400,
                detail=(
                    f"Omitting 'field' requires {len(ALL_PARAMS)} targets—"
                    "one for each of Pack_SOC, Pack_Voltage, Pack_SOH."
                ),
            )
        key_params = ALL_PARAMS

    # 2) Extract command signatures for targets
    target_cmd_strs = {}
    for p, j in zip(key_params, new_car_data.target_json_arr):
        target_cmd_strs[p] = extract_command_signature(j)

    # 3) Load known cars
    known_cars = load_all_known_cars()
    if not known_cars:
        base_id = new_car_data.new_car_attributes or canonical_identifier(
            new_car_data.manufacturer, new_car_data.model, new_car_data.year
        )
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    # 4) Create canonical IDs
    car_identifiers = []
    for car in known_cars:
        car_id = canonical_identifier(car.manufacturer, car.model, car.year)
        car_identifiers.append(car_id)

    # 5) Create the query car's canonical ID
    query_car_id = canonical_identifier(
        new_car_data.manufacturer, 
        new_car_data.model,
        new_car_data.year
    )

    # 6) Build TF-IDF on car IDs and command signatures
    all_texts = []
    
    # Add car identifiers
    all_texts.extend(car_identifiers)
    all_texts.append(query_car_id)
    
    # Add all known command signatures
    for car in known_cars:
        for param, cmd_list in car.commands.items():
            for cmd in cmd_list:
                all_texts.append(extract_command_signature(cmd))
    
    # Add target command signatures
    all_texts.extend(target_cmd_strs.values())

    # Remove duplicates
    all_texts = list(set(all_texts))

    if len(all_texts) < 2:
        base_id = new_car_data.new_car_attributes or query_car_id
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char", 
            ngram_range=(2, 4),
            min_df=1,
            max_df=1.0
        )
        vectorizer.fit(all_texts)
    except ValueError:
        base_id = new_car_data.new_car_attributes or query_car_id
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    # 7) Calculate initial similarity based on car names
    query_vec = vectorizer.transform([query_car_id])
    car_id_vecs = vectorizer.transform(car_identifiers)
    name_similarities = cosine_similarity(query_vec, car_id_vecs).flatten()

    # 8) Calculate command similarities and build matches
    matches = []
    
    for idx, car in enumerate(known_cars):
        commands_found = {}
        command_scores = []
        
        # Check each parameter
        for param in key_params:
            target_vec = vectorizer.transform([target_cmd_strs[param]])
            
            # Get commands from this car for this parameter
            car_cmds = car.commands.get(param, [])
            
            if car_cmds:
                # Find the best matching command
                best_score = 0
                best_cmd = None
                
                for cmd in car_cmds:
                    cmd_sig = extract_command_signature(cmd)
                    cmd_vec = vectorizer.transform([cmd_sig])
                    score = cosine_similarity(target_vec, cmd_vec).flatten()[0]
                    
                    if score > best_score:
                        best_score = score
                        best_cmd = cmd
                
                # Check if it's an exact match or close enough
                if best_score >= success_threshold:
                    commands_found[param] = best_cmd
                    command_scores.append(best_score)
                else:
                    # Penalize for not having a good match
                    command_scores.append(best_score * 0.5)
            else:
                # Penalize for missing this parameter
                command_scores.append(0.0)
        
        # Calculate overall similarity score
        if command_scores:
            avg_cmd_score = np.mean(command_scores)
            # Bonus for finding exact matches
            match_bonus = len(commands_found) / len(key_params) * 0.2
            overall_score = (0.3 * name_similarities[idx] + 
                           0.5 * avg_cmd_score + 
                           match_bonus)
            # Ensure score doesn't exceed 1.0
            overall_score = min(1.0, overall_score)
        else:
            overall_score = name_similarities[idx] * 0.3
        
        if overall_score >= display_threshold or commands_found:
            matches.append(Match(
                manufacturer=car.manufacturer,
                model=car.model,
                year=car.year,
                country_region=car.country_region,
                type_=car.type_,
                similarity_score=float(overall_score),
                commands_found=commands_found
            ))
    
    # Sort matches by score
    matches.sort(key=lambda m: m.similarity_score, reverse=True)
    
    # Generate new car attributes
    if matches and matches[0].commands_found:
        new_attr = query_car_id
        for param, cmd in matches[0].commands_found.items():
            cmd_sig = extract_command_signature(cmd)
            new_attr += f"_{param[:3]}_{hash(cmd_sig) % 10000:04d}"
    else:
        new_attr = new_car_data.new_car_attributes or query_car_id
    
    return CheckResponse(
        exists=len(matches) > 0,
        matches=matches,
        new_car_attributes=new_attr
    )