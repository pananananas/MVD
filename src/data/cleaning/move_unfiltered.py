from pathlib import Path
from tqdm import tqdm
import sqlite3
import shutil
import os

DATASET_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/renders"
PROCESSING_QUEUE_PATH = "/net/pr2/projects/plgrid/plggtattooai/MeshDatasets/objaverse/queue"
DB_PATH = os.path.join(os.path.dirname(DATASET_PATH), "processing_status.db")

os.makedirs(PROCESSING_QUEUE_PATH, exist_ok=True)

def setup_database():
    if not os.path.exists(DB_PATH):
        print(f"Database file does not exist at {DB_PATH}. Creating new database.")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS samples (
            path TEXT PRIMARY KEY,
            processed BOOLEAN,
            is_useful BOOLEAN,
            prompt TEXT,
            error TEXT,
            processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
    else:
        conn = sqlite3.connect(DB_PATH)
    
    return conn

def move_unprocessed_files():
    conn = setup_database()
    cursor = conn.cursor()
    
    zip_files = sorted([str(f) for f in Path(DATASET_PATH).glob("*.zip")])
    print(f"Found {len(zip_files)} total zip files in dataset directory")
    
    for zip_path in zip_files:
        cursor.execute("INSERT OR IGNORE INTO samples (path, processed) VALUES (?, ?)", 
                      (zip_path, False))
    conn.commit()
    
    cursor.execute("SELECT path FROM samples WHERE processed = 0")
    unprocessed = [row[0] for row in cursor.fetchall()]
    print(f"Found {len(unprocessed)} unprocessed samples out of {len(zip_files)} total")

    moved_count = 0
    for zip_path in tqdm(unprocessed, desc="Moving unprocessed files"):
        if os.path.exists(zip_path):
            filename = os.path.basename(zip_path)
            destination = os.path.join(PROCESSING_QUEUE_PATH, filename)

            new_path = destination
            
            shutil.move(zip_path, destination)
            
            cursor.execute("UPDATE samples SET path = ? WHERE path = ?", (new_path, zip_path))
            
            moved_count += 1
    
    conn.commit()
    print(f"Moved {moved_count} unprocessed files to {PROCESSING_QUEUE_PATH}")
    conn.close()

if __name__ == "__main__":
    move_unprocessed_files()