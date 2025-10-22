from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Serve ./static/index.html at /static
app.mount(
    "/static",
    StaticFiles(directory="./static", html=True),
    name="static",
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
class CheckRequest(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    new_car_attributes: Optional[str] = None
    target_json_arr: List[Dict[str, Any]]
    field: Optional[str] = None

class IterativeAttackRequest(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    use_progressive: Optional[bool] = True  # New flag for progressive matching

class CarWithCommands(BaseModel):
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    Pack_SOC: Optional[List[Dict[str, Any]]] = None
    Pack_Voltage: Optional[List[Dict[str, Any]]] = None
    Pack_SOH: Optional[List[Dict[str, Any]]] = None

class IterativeAttackResponse(BaseModel):
    target_car: CarWithCommands
    best_match_identifier: str
    final_similarity_score: float
    commands_matched: Dict[str, bool]
    iterations: List[Dict[str, Any]]
    candidate_pool_sizes: List[int]  # Track how candidate pool shrinks

class Match(BaseModel):
    similarity_score: float
    manufacturer: str
    model: str
    year: Optional[int] = None
    country_region: Optional[str] = None
    type_: Optional[str] = None
    commands_found: Dict[str, Dict[str, Any]]

class CheckResponse(BaseModel):
    exists: bool
    matches: List[Match]
    new_car_attributes: str

# --- Domain & Loaders ------------------------------------------------------
class Car:
    def __init__(self, manufacturer: str, model: str, year: int = None, 
                 country_region: str = "", type_: str = ""):
        self.manufacturer = manufacturer
        self.model = model
        self.year = year
        self.country_region = country_region
        self.type_ = type_
        self.commands: Dict[str, List[Dict[str, Any]]] = {}
        self.identifier = ""  # Store the car's identifier

    def add_command(self, field: str, raw_json: Any):
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

        if key not in cars_map:
            car = Car(manufacturer, model, year, country_region, type_)
            car.identifier = canonical_identifier(manufacturer, model, year, country_region, type_)
            cars_map[key] = car

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
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
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
    car.identifier = canonical_identifier(manufacturer, model, year, country_region, type_)
    
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
    essential_fields = ['request_id', 'response_id', 'command']
    signature_parts = []
    for field in essential_fields:
        if field in cmd:
            value = str(cmd[field]).strip().upper()
            signature_parts.append(f"{field}:{value}")
    return "|".join(signature_parts)

def build_identifier_with_commands(base_id: str, commands: Dict[str, str]) -> str:
    """Build identifier with command hashes appended."""
    identifier = base_id
    for param, sig in commands.items():
        identifier += f"_{param[:3]}_{hash(sig) % 10000:04d}"
    return identifier

# --- Endpoints --------------------------------------------------------------

@app.get("/car/{manufacturer}/{model}", response_model=CarWithCommands)
async def get_car_commands(
    manufacturer: str,
    model: str,
    year: Optional[int] = None,
    country_region: Optional[str] = None,
    type_: Optional[str] = None
):
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
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]
    
    # Load the target car
    target_car = get_specific_car(
        request.manufacturer, 
        request.model, 
        request.year,
        request.country_region,
        request.type_
    )
    
    if not target_car:
        raise HTTPException(404, detail=f"Target car not found: {request.manufacturer} {request.model}")
    
    # Load all known cars
    known_cars = load_all_known_cars()
    if not known_cars:
        raise HTTPException(500, detail="No cars in database")
    
    # Initialize tracking
    current_identifier = canonical_identifier(request.manufacturer, request.model)
    iterations = []
    commands_matched = {param: False for param in ALL_PARAMS}
    matched_signatures = {}  # Store matched command signatures
    candidate_pool_sizes = [len(known_cars)]  # Track candidate pool size
    
    # Build TF-IDF vocabulary
    all_texts = []
    for car in known_cars:
        all_texts.append(car.identifier)
    all_texts.append(current_identifier)
    
    for car in known_cars:
        for param, cmd_list in car.commands.items():
            for cmd in cmd_list:
                all_texts.append(extract_command_signature(cmd))
    
    for param in ALL_PARAMS:
        if param in target_car.commands:
            for cmd in target_car.commands[param]:
                all_texts.append(extract_command_signature(cmd))
    
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
    
    # Candidate pool for progressive matching
    candidate_cars = known_cars.copy() if request.use_progressive else None
    
    # Iterate through each parameter
    for param_idx, param in enumerate(ALL_PARAMS):
        target_cmds = target_car.commands.get(param, [])
        if not target_cmds:
            iterations.append({
                "parameter": param,
                "status": "no_target_command",
                "candidate_pool_size": len(candidate_cars) if candidate_cars else len(known_cars)
            })
            continue
        
        target_cmd = target_cmds[0]
        target_sig = extract_command_signature(target_cmd)
        target_vec = vectorizer.transform([target_sig])
        
        # Use progressive candidate pool or all cars
        cars_to_check = candidate_cars if candidate_cars else known_cars
        
        # Find best matches
        best_matches = []
        matching_cars = []  # Cars that have this exact command
        
        for car in cars_to_check:
            car_cmds = car.commands.get(param, [])
            if not car_cmds:
                continue
            
            for cmd in car_cmds:
                cmd_sig = extract_command_signature(cmd)
                cmd_vec = vectorizer.transform([cmd_sig])
                score = cosine_similarity(target_vec, cmd_vec).flatten()[0]
                
                if score >= success_threshold:
                    matching_cars.append(car)
                
                best_matches.append({
                    "car": car,
                    "car_name": f"{car.manufacturer} {car.model} {car.year}",
                    "command_signature": cmd_sig,
                    "score": float(score)
                })
        
        # Sort by score
        best_matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Update state based on match
        if best_matches and best_matches[0]["score"] >= success_threshold:
            commands_matched[param] = True
            matched_signatures[param] = target_sig
            current_identifier = build_identifier_with_commands(
                canonical_identifier(request.manufacturer, request.model),
                matched_signatures
            )
            
            # Progressive narrowing: keep only cars that matched this command
            if request.use_progressive and matching_cars:
                candidate_cars = matching_cars
                candidate_pool_sizes.append(len(candidate_cars))
            
            iterations.append({
                "parameter": param,
                "status": "matched",
                "best_score": best_matches[0]["score"],
                "matched_command": best_matches[0]["command_signature"],
                "updated_identifier": current_identifier,
                "candidate_pool_size": len(candidate_cars) if candidate_cars else len(known_cars),
                "top_matches": [
                    {"car": m["car_name"], "score": m["score"]} 
                    for m in best_matches[:3]
                ]
            })
        else:
            # No good match found
            if request.use_progressive and candidate_cars:
                # Don't narrow pool if no match found
                candidate_pool_sizes.append(len(candidate_cars))
            
            iterations.append({
                "parameter": param,
                "status": "no_match",
                "best_score": best_matches[0]["score"] if best_matches else 0.0,
                "candidate_pool_size": len(candidate_cars) if candidate_cars else len(known_cars),
                "top_matches": [
                    {"car": m["car_name"], "score": m["score"]} 
                    for m in best_matches[:3]
                ] if best_matches else []
            })
    
    # Calculate final similarity score
    matched_count = sum(1 for matched in commands_matched.values() if matched)
    final_score = matched_count / len(ALL_PARAMS)
    
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
        candidate_pool_sizes=candidate_pool_sizes
    )

@app.post("/guessing-attack", response_model=CheckResponse)
async def guessing_attack_strategy(
    new_car_data: CheckRequest,
    success_threshold: float = 0.80,
    display_threshold: float = 0.50,
):
    ALL_PARAMS = ["Pack_SOC", "Pack_Voltage", "Pack_SOH"]

    # Decide which params to test
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

    # Extract command signatures for targets
    target_cmd_strs = {}
    for p, j in zip(key_params, new_car_data.target_json_arr):
        target_cmd_strs[p] = extract_command_signature(j)

    # Load known cars
    known_cars = load_all_known_cars()
    if not known_cars:
        base_id = new_car_data.new_car_attributes or canonical_identifier(
            new_car_data.manufacturer, new_car_data.model, new_car_data.year
        )
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    # Create the query car's canonical ID
    query_car_id = canonical_identifier(
        new_car_data.manufacturer, 
        new_car_data.model,
        new_car_data.year
    )

    # Check if we have a pre-built identifier with command hashes
    if new_car_data.new_car_attributes and "_SOC_" in new_car_data.new_car_attributes:
        # Use the provided identifier for better matching
        query_car_id = new_car_data.new_car_attributes

    # Build TF-IDF vocabulary
    all_texts = []
    car_identifiers = [car.identifier for car in known_cars]
    all_texts.extend(car_identifiers)
    all_texts.append(query_car_id)
    
    for car in known_cars:
        for param, cmd_list in car.commands.items():
            for cmd in cmd_list:
                all_texts.append(extract_command_signature(cmd))
    
    all_texts.extend(target_cmd_strs.values())
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

    # Calculate similarities
    query_vec = vectorizer.transform([query_car_id])
    car_id_vecs = vectorizer.transform(car_identifiers)
    name_similarities = cosine_similarity(query_vec, car_id_vecs).flatten()

    # Calculate command similarities and build matches
    matches = []
    
    for idx, car in enumerate(known_cars):
        commands_found = {}
        command_scores = []
        
        for param in key_params:
            target_vec = vectorizer.transform([target_cmd_strs[param]])
            car_cmds = car.commands.get(param, [])
            
            if car_cmds:
                best_score = 0
                best_cmd = None
                
                for cmd in car_cmds:
                    cmd_sig = extract_command_signature(cmd)
                    cmd_vec = vectorizer.transform([cmd_sig])
                    score = cosine_similarity(target_vec, cmd_vec).flatten()[0]
                    
                    if score > best_score:
                        best_score = score
                        best_cmd = cmd
                
                if best_score >= success_threshold:
                    commands_found[param] = best_cmd
                    command_scores.append(best_score)
                else:
                    command_scores.append(best_score * 0.5)
            else:
                command_scores.append(0.0)
        
        # Calculate overall similarity score
        if command_scores:
            avg_cmd_score = np.mean(command_scores)
            match_bonus = len(commands_found) / len(key_params) * 0.2
            overall_score = (0.3 * name_similarities[idx] + 
                           0.5 * avg_cmd_score + 
                           match_bonus)
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