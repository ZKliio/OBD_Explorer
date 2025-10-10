# Main purpose
Develop a model for predicting SOH of EV Batteries via OBD data (UDS requests)

## Build files
---
py `parser.py` creates OBD.db database
parser.py iterates through JSON files within folder obd_car_files 
formatted in obd explorer database reformatted into sqlite database OBD.db

1) Create database
2) Create table of key parameters "key_signals", key parameters include 
    Pack_SOH, Pack_SOC, Pack_Voltag, Pack_Current, Torque and RPM

py `pack_table.py`
1) Creates specific table to store battery pack related commands from key_signals
2) Formats for string comparison strategy

## Run files
PlantModel_OBD/

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
