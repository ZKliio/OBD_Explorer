import sqlite3
import pandas as pd
import os

def export_sqlite_to_excel(db_path, excel_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Get all table names
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # Create Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for table in tables:
            # Read each table into a DataFrame
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)

            # Optional: convert hex-like columns to strings
            for col in df.columns:
                if df[col].dtype == object and df[col].astype(str).str.match(r'^[0-9A-Fa-f]+$').any():
                    df[col] = df[col].astype(str)

            # Write to Excel sheet named after the table
            df.to_excel(writer, sheet_name=table, index=False)

    conn.close()
    print(f"✅ Exported {len(tables)} tables to {excel_path}")

if __name__ == "__main__":
    db_path = os.path.join(os.path.dirname(__file__), 'OBD.db')
    excel_path = os.path.join(os.path.dirname(__file__), 'OBD_data.xlsx')
    export_sqlite_to_excel(db_path, excel_path)
