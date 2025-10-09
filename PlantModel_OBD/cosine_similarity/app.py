from fastapi import FastAPI, HTTPException
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
    new_car_attributes: Optional[str] = None
    target_json_arr: List[Dict[str, Any]]
    field: Optional[str] = None

class Match(BaseModel):
    manufacturer: str
    model: str
    similarity_score: float
    commands_found: Dict[str, Dict[str, Any]]

class CheckResponse(BaseModel):
    exists: bool
    matches: List[Match]
    new_car_attributes: str

# --- Domain & Loaders ------------------------------------------------------
class CarIn(BaseModel):
    manufacturer: str
    model: str
    year: int
    country_region: str = ""
    type_: str = ""

class Car:
    def __init__(self, manufacturer: str, model: str):
        self.manufacturer = manufacturer
        self.model = model
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
            cars_map[key] = Car(manufacturer, model)
        car = cars_map[key]

        if raw_vol:
            car.add_command("Pack_Voltage", raw_vol)
        if raw_soc:
            car.add_command("Pack_SOC", raw_soc)
        if raw_soh:
            car.add_command("Pack_SOH", raw_soh)

    return list(cars_map.values())

# --- Utils -----------------------------------------------------------------

def canonicalize_json_to_string_sorted(obj: Dict[str, Any]) -> str:
    """
    Deterministic JSON→string by sorting keys and stripping whitespace.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def canonical_identifier(manufacturer: str, model: str) -> str:
    """
    Create a consistent canonical identifier for car manufacturer and model.
    Strip whitespace, lowercase, and remove ALL spaces.
    """
    m = manufacturer.strip().lower().replace(" ", "")
    mod = model.strip().lower().replace(" ", "")
    return f"{m}_{mod}"

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

# --- Endpoint --------------------------------------------------------------

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

    # 2) Normalize and canonicalize all provided target commands
    target_commands = {}
    target_cmd_strs = {}
    for p, j in zip(key_params, new_car_data.target_json_arr):
        normalized = normalize_command(j)
        target_commands[p] = normalized
        target_cmd_strs[p] = canonicalize_json_to_string_sorted(normalized)

    # 3) Load known cars and guard empties
    known_cars = load_all_known_cars()
    if not known_cars:
        base_id = new_car_data.new_car_attributes or canonical_identifier(
            new_car_data.manufacturer, new_car_data.model
        )
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    # 4) Create canonical IDs for all known cars
    car_identifiers = []
    for car in known_cars:
        car_id = canonical_identifier(car.manufacturer, car.model)
        car_identifiers.append(car_id)

    # 5) Create the query car's canonical ID
    query_car_id = canonical_identifier(new_car_data.manufacturer, new_car_data.model)

    # 6) Build TF-IDF on car IDs and all command strings
    all_texts = []
    
    # Add car identifiers
    all_texts.extend(car_identifiers)
    all_texts.append(query_car_id)  # Add the query car ID
    
    # Add all known commands (normalized)
    for car in known_cars:
        for param, cmd_list in car.commands.items():
            for cmd in cmd_list:
                normalized = normalize_command(cmd)
                all_texts.append(canonicalize_json_to_string_sorted(normalized))
    
    # Add target commands
    all_texts.extend(target_cmd_strs.values())

    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in all_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)

    if len(unique_texts) < 2:
        base_id = new_car_data.new_car_attributes or query_car_id
        return CheckResponse(exists=False, matches=[], new_car_attributes=base_id)

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char", 
            ngram_range=(2, 4),
            min_df=1,
            max_df=1.0
        )
        vectorizer.fit(unique_texts)
    except ValueError as e:
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
                    normalized = normalize_command(cmd)
                    cmd_str = canonicalize_json_to_string_sorted(normalized)
                    cmd_vec = vectorizer.transform([cmd_str])
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
        # Weight: 30% name similarity, 70% command similarity
        if command_scores:
            avg_cmd_score = np.mean(command_scores)
            # Bonus for finding exact matches
            match_bonus = len(commands_found) / len(key_params) * 0.2
            overall_score = (0.3 * name_similarities[idx] + 
                           0.7 * avg_cmd_score + 
                           match_bonus)
            # Ensure score doesn't exceed 1.0
            overall_score = min(1.0, overall_score)
        else:
            overall_score = name_similarities[idx] * 0.3
        
        if overall_score >= display_threshold or commands_found:
            matches.append(Match(
                manufacturer=car.manufacturer,
                model=car.model,
                similarity_score=float(overall_score),
                commands_found=commands_found
            ))
    
    # Sort matches by score
    matches.sort(key=lambda m: m.similarity_score, reverse=True)
    
    # Generate new car attributes based on best match
    if matches and matches[0].commands_found:
        new_attr = query_car_id
        for param, cmd in matches[0].commands_found.items():
            cmd_str = canonicalize_json_to_string_sorted(normalize_command(cmd))
            # Add a shortened version to avoid extremely long attributes
            new_attr += f"_{param[:3]}_{hash(cmd_str) % 10000:04d}"
    else:
        new_attr = new_car_data.new_car_attributes or query_car_id
    
    return CheckResponse(
        exists=len(matches) > 0,
        matches=matches,
        new_car_attributes=new_attr
    )