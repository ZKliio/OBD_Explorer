# Main purpose
Develop a model for predicting SOH of EV Pack Batteries via OBD data (UDS requests)
Simulating car (plant model) and performing guessing attacks based on current database of transmit messages (attack server) 

## Resources
### OBD.db
Contains all vehicles sourced online 
Tables:
1) vehicle_pack_commands: Already filtered Pack Battery parameters ie. SOC, Voltage, SOH
2) key_signals: signals
3) commands: All commands including non-Pack related ECU transmit commands
4) signals: Signal (commands can have multiple signal values), this provides full information required for full decoding of UDS request
5) sqlite_sequence: Row counts for each table

## Build files
---
### parser.py
py `parser.py` creates OBD.db database
parser.py iterates through JSON files within folder obd_car_files 
formatted in obd explorer database reformatted into sqlite database OBD.db

1) Create database
2) Create table of key parameters "key_signals", key parameters include 
    Pack_SOH, Pack_SOC, Pack_Voltag, Pack_Current, Torque and RPM

### pack_table.py
py `pack_table.py`
1) Creates specific table to store battery pack related commands from key_signals
2) Formats for string comparison strategy

## Run files
PlantModel_OBD/

## Plant Model Server and Attack Server
Various branches for different versions of Attack Server
1) dynamic_scoring: Most updated, uses success counter to prioritize successful matches before re-sorting, add on to dynamic iteration 
2) dynamic_iteration: After each successful match, tag new field to initial identifier string; Re-sorts with new field attached to all strings
3) static_iteration: Uses same initial list for subsequent comparisons (no updates)

## Testing algorithms (not important)
### cosine_similarity/
py `app.py` 
1) stringify rows from database + input 
2) vectorizes strings
3) implements cosine similarity algorithm, score by similarity to input 
4) update priority list (sorted by similarity score)

### Levenshtein/
levenshtein distance 
py `app.py`
1) stringify rows from database + input 
2) implements levenshtein distance algorithm, score by similarity to input string
3) update priority list (sorted by similarity score)

py `app1.py`
1) stringify rows from database + input 
2) compares stringified input (transmit message field) to database string rows
    Discrete Logic: One to One Similarity = append, Else = next car in priority list
    New string is appended with transmit message
3) implements levenshtein distance algorithm, score by similarity to input 
