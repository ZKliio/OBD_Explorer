DROP TABLE IF EXISTS commands;
DROP TABLE IF EXISTS signals;
DROP TABLE IF EXISTS key_signals;
DROP TABLE IF EXISTS vehicle_pack_commands;

CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer TEXT NOT NULL,
    model TEXT,
    request_id TEXT NOT NULL,
    response_id TEXT NOT NULL,
    command TEXT NOT NULL,
    -- rax TEXT NOT NULL,
    -- fcm1 BOOLEAN NOT NULL,
    transmit_message TEXT NOT NULL,
    frequency REAL NOT NULL,
    signal_ids TEXT NOT NULL,
    signal_paths TEXT NOT NULL,
    suggested_metrics TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id INTEGER,
    manufacturer TEXT NOT NULL,
    model TEXT,
    type TEXT,
    signal_id TEXT NOT NULL,
    name TEXT NOT NULL,
    key_parameter TEXT,
    request_id TEXT NOT NULL,
    response_id TEXT NOT NULL,
    command TEXT NOT NULL,
    transmit_message TEXT NOT NULL,
    -- description TEXT,
    frequency REAL,
    map TEXT,
    start_bit INTEGER,
    end_bit INTEGER,
    len INTEGER NOT NULL,
    min_value REAL,
    max_value REAL,
    mul REAL,
    div REAL,
    "add" REAL,
    path TEXT,
    suggested_metric TEXT,
    unit TEXT,
    FOREIGN KEY (command_id) REFERENCES commands (id)
);

CREATE TABLE key_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_id INTEGER,
    manufacturer TEXT NOT NULL,
    model TEXT,
    type TEXT,
    signal_id TEXT NOT NULL,
    name TEXT NOT NULL,
    key_parameter TEXT,
    request_id TEXT NOT NULL,
    response_id TEXT NOT NULL,
    command TEXT NOT NULL,
    transmit_message TEXT NOT NULL,
    -- description TEXT,
    frequency REAL,
    map TEXT,
    start_bit INTEGER,
    end_bit INTEGER,
    len INTEGER NOT NULL,
    min_value REAL,
    max_value REAL,
    mul REAL,
    div REAL,
    "add" REAL,
    path TEXT,
    suggested_metric TEXT,
    unit TEXT,
    FOREIGN KEY (command_id) REFERENCES commands (id)
);

CREATE TABLE IF NOT EXISTS vehicle_pack_commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer TEXT NOT NULL,
    model TEXT,
    year JSON,
    country_region JSON,
    type TEXT,
    pack_voltage, JSON
);