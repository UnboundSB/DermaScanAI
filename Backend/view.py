import sqlite3

def view_database(db_path):
    print(f"Analyzing Database: {db_path}\n")
    
    try:
        # 1. Establish connection to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 2. Fetch all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        if not tables:
            print("The database is empty or contains no tables.")
            return

        # 3. Loop through each table to print schema and contents
        for table in tables:
            table_name = table[0]
            
            # Skip SQLite internal tables
            if table_name.startswith("sqlite_"):
                continue
                
            print(f"{'='*50}")
            print(f"TABLE: {table_name}")
            print(f"{'='*50}")

            # Get Schema using PRAGMA
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            print("--- SCHEMA ---")
            col_names = []
            for col in columns:
                # col format: (cid, name, type, notnull, dflt_value, pk)
                col_name = col[1]
                col_type = col[2]
                col_names.append(col_name)
                print(f" - {col_name} ({col_type})")

            # Get Data
            print("\n--- CONTENTS ---")
            cursor.execute(f"SELECT * FROM {table_name};")
            rows = cursor.fetchall()

            if not rows:
                print("(Table is empty)\n")
            else:
                # Print column headers
                print(" | ".join(col_names))
                print("-" * 30)
                # Print each row
                for row in rows:
                    print(row)
            print("\n")

    except sqlite3.Error as e:
        print(f"An SQLite error occurred: {e}")
    finally:
        # Always ensure the connection is closed
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    # Pointing to your specific database path
    DATABASE_PATH = r"D:\Projects\DermaScanAI\Backend\derma_history.db"
    view_database(DATABASE_PATH)