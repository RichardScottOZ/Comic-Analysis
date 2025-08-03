#!/usr/bin/env python3
"""
Organize comic analysis results into a scalable data structure.
Converts individual JSON files into organized databases and vector storage.
"""

import json
import sqlite3
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
from datetime import datetime
import hashlib

def extract_comic_info(filename):
    """Extract comic ID and page number from filename."""
    # Expected format: comic_id_page.json
    match = re.match(r'([a-f0-9]+)_(\d+)\.json', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def parse_json_file(file_path):
    """Parse a JSON analysis file and extract structured data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract basic info
        comic_id, page_number = extract_comic_info(file_path.name)
        
        # Count elements
        panel_count = len(data.get('panels', []))
        dialogue_count = sum(len(panel.get('speakers', [])) for panel in data.get('panels', []))
        character_count = len(data.get('summary', {}).get('characters', []))
        
        # Extract all dialogue
        all_dialogue = []
        for panel in data.get('panels', []):
            panel_num = panel.get('panel_number', 0)
            for speaker in panel.get('speakers', []):
                all_dialogue.append({
                    'panel_number': panel_num,
                    'character': speaker.get('character', ''),
                    'dialogue': speaker.get('dialogue', ''),
                    'speech_type': speaker.get('speech_type', '')
                })
        
        # Extract all characters
        characters = data.get('summary', {}).get('characters', [])
        
        return {
            'comic_id': comic_id,
            'page_number': page_number,
            'overall_summary': data.get('overall_summary', ''),
            'panel_count': panel_count,
            'dialogue_count': dialogue_count,
            'character_count': character_count,
            'panels': data.get('panels', []),
            'dialogue': all_dialogue,
            'characters': characters,
            'summary': data.get('summary', {}),
            'raw_data': data
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def create_sqlite_database(db_path):
    """Create SQLite database with proper schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Comic pages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comic_pages (
            id TEXT PRIMARY KEY,
            comic_id TEXT,
            page_number TEXT,
            file_path TEXT,
            analysis_date TEXT,
            model_used TEXT,
            processing_time REAL,
            overall_summary TEXT,
            character_count INTEGER,
            panel_count INTEGER,
            dialogue_count INTEGER,
            raw_json TEXT
        )
    ''')
    
    # Panels table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS panels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id TEXT,
            panel_number INTEGER,
            caption TEXT,
            description TEXT,
            key_elements TEXT,
            actions TEXT,
            FOREIGN KEY (page_id) REFERENCES comic_pages (id)
        )
    ''')
    
    # Dialogue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dialogue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id TEXT,
            panel_number INTEGER,
            character_name TEXT,
            dialogue_text TEXT,
            speech_type TEXT,
            FOREIGN KEY (page_id) REFERENCES comic_pages (id)
        )
    ''')
    
    # Characters table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            page_id TEXT,
            character_name TEXT,
            character_type TEXT,
            FOREIGN KEY (page_id) REFERENCES comic_pages (id)
        )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_comic_pages_comic_id ON comic_pages(comic_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_panels_page_id ON panels(page_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_dialogue_page_id ON dialogue(page_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_characters_page_id ON characters(page_id)')
    
    conn.commit()
    return conn

def insert_comic_page(conn, data, file_path, model_used="unknown"):
    """Insert comic page data into database."""
    cursor = conn.cursor()
    
    page_id = f"{data['comic_id']}_{data['page_number']}"
    
    # Insert main page record
    cursor.execute('''
        INSERT OR REPLACE INTO comic_pages 
        (id, comic_id, page_number, file_path, analysis_date, model_used, 
         overall_summary, character_count, panel_count, dialogue_count, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        page_id,
        data['comic_id'],
        data['page_number'],
        str(file_path),
        datetime.now().isoformat(),
        model_used,
        data['overall_summary'],
        data['character_count'],
        data['panel_count'],
        data['dialogue_count'],
        json.dumps(data['raw_data'])
    ))
    
    # Insert panels
    for panel in data['panels']:
        cursor.execute('''
            INSERT INTO panels (page_id, panel_number, caption, description, key_elements, actions)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            page_id,
            panel.get('panel_number', 0),
            panel.get('caption', ''),
            panel.get('description', ''),
            json.dumps(panel.get('key_elements', [])),
            json.dumps(panel.get('actions', []))
        ))
    
    # Insert dialogue
    for dialogue in data['dialogue']:
        cursor.execute('''
            INSERT INTO dialogue (page_id, panel_number, character_name, dialogue_text, speech_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            page_id,
            dialogue['panel_number'],
            dialogue['character'],
            dialogue['dialogue'],
            dialogue['speech_type']
        ))
    
    # Insert characters
    for character in data['characters']:
        cursor.execute('''
            INSERT INTO characters (page_id, character_name, character_type)
            VALUES (?, ?, ?)
        ''', (
            page_id,
            character,
            'main'  # Default type
        ))
    
    conn.commit()

def create_summary_statistics(conn):
    """Create summary statistics from the database."""
    cursor = conn.cursor()
    
    # Basic counts
    cursor.execute('SELECT COUNT(*) FROM comic_pages')
    total_pages = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM panels')
    total_panels = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM dialogue')
    total_dialogue = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT character_name) FROM characters')
    unique_characters = cursor.fetchone()[0]
    
    # Average panels per page
    cursor.execute('SELECT AVG(panel_count) FROM comic_pages')
    avg_panels = cursor.fetchone()[0]
    
    # Most common characters
    cursor.execute('''
        SELECT character_name, COUNT(*) as count 
        FROM characters 
        GROUP BY character_name 
        ORDER BY count DESC 
        LIMIT 10
    ''')
    top_characters = cursor.fetchall()
    
    # Comics with most pages
    cursor.execute('''
        SELECT comic_id, COUNT(*) as page_count 
        FROM comic_pages 
        GROUP BY comic_id 
        ORDER BY page_count DESC 
        LIMIT 10
    ''')
    top_comics = cursor.fetchall()
    
    return {
        'total_pages': total_pages,
        'total_panels': total_panels,
        'total_dialogue': total_dialogue,
        'unique_characters': unique_characters,
        'avg_panels_per_page': avg_panels,
        'top_characters': top_characters,
        'top_comics': top_comics
    }

def main():
    parser = argparse.ArgumentParser(description='Organize comic analysis results into database')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing JSON analysis files')
    parser.add_argument('--output-db', type=str, default='comic_analysis.db',
                       help='Output SQLite database path')
    parser.add_argument('--model-used', type=str, default='qwen/qwen2.5-vl-32b-instruct:free',
                       help='Model used for analysis')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Find all JSON files
    json_files = list(input_dir.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    # Create database
    print(f"Creating database: {args.output_db}")
    conn = create_sqlite_database(args.output_db)
    
    # Process files
    processed = 0
    errors = 0
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            data = parse_json_file(json_file)
            if data:
                insert_comic_page(conn, data, json_file, args.model_used)
                processed += 1
            else:
                errors += 1
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            errors += 1
    
    # Create summary statistics
    print("\nGenerating summary statistics...")
    stats = create_summary_statistics(conn)
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total pages: {stats['total_pages']}")
    print(f"Total panels: {stats['total_panels']}")
    print(f"Total dialogue: {stats['total_dialogue']}")
    print(f"Unique characters: {stats['unique_characters']}")
    print(f"Average panels per page: {stats['avg_panels_per_page']:.2f}")
    
    print(f"\n=== Top Characters ===")
    for char, count in stats['top_characters']:
        print(f"  {char}: {count} appearances")
    
    print(f"\n=== Top Comics ===")
    for comic, pages in stats['top_comics']:
        print(f"  {comic}: {pages} pages")
    
    conn.close()
    print(f"\nDatabase saved to: {args.output_db}")

if __name__ == "__main__":
    main() 