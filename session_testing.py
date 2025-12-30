import sqlite3
import json
from datetime import datetime
import argparse
import os

DB_PATH = "./data/sessions.db"

def print_section_header(title):
    """Prints a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def pretty_print_json(data):
    """Prints a dictionary as nicely formatted JSON."""
    if not data:
        print("  <No data>")
        return
    # Use a custom encoder to handle datetime objects if they exist
    print(json.dumps(data, indent=2, default=str))


def inspect_session(session_id: str):
    """
    Connects to the session database and prints the details for a specific session.
    """
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            if not row:
                print(f"Error: No session found with ID '{session_id}'")
                return

            # --- Print Session Details ---
            print_section_header("SESSION OVERVIEW")
            print(f"  Session ID: {row['session_id']}")
            print(f"  Last Updated: {datetime.fromisoformat(row['updated_at']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Conversation Complete: {'Yes' if row['conversation_complete'] else 'No'}")

            # --- Parse and Print Metadata ---
            metadata = json.loads(row['metadata'])
            print_section_header("METADATA (Conversation State)")
            pretty_print_json(metadata)

            # --- Parse and Print User Data ---
            user_data = json.loads(row['user_data'])
            
            print_section_header("DEMOGRAPHICS")
            pretty_print_json(user_data.get("demographics"))

            print_section_header("AI GENERATED QUESTIONS & RESPONSES")
            pretty_print_json(user_data.get("ai_generated_responses"))
            
            print_section_header("AI COUNTERVIEW (Phase 4)")
            pretty_print_json(user_data.get("phase4"))

            print_section_header("USER REFLECTION (Phase 5)")
            pretty_print_json(user_data.get("phase5"))


    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON data from the database: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def list_sessions():
    """Lists all available sessions in the database."""
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file not found at '{DB_PATH}'")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            # Fetch the 20 most recently updated sessions
            cursor = conn.execute("SELECT session_id, updated_at, conversation_complete FROM sessions ORDER BY updated_at DESC LIMIT 20")
            rows = cursor.fetchall()

            if not rows:
                print("No sessions found in the database.")
                return
            
            print_section_header("RECENT SESSIONS")
            print(f"{'SESSION ID':<45} {'LAST UPDATED':<25} {'COMPLETE'}")
            print("-" * 80)
            
            for row in rows:
                session_id = row['session_id']
                updated_at = datetime.fromisoformat(row['updated_at']).strftime('%Y-%m-%d %H:%M:%S')
                complete = "Yes" if row['conversation_complete'] else "No"
                print(f"{session_id:<45} {updated_at:<25} {complete}")
            
            print("\nTo inspect a session, run: python inspect_session.py <session_id>")

    except sqlite3.Error as e:
        print(f"Database error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect user conversation sessions in the SQLite database.")
    parser.add_argument("session_id", nargs='?', default=None, help="The ID of the session to inspect. If omitted, lists all recent sessions.")
    
    args = parser.parse_args()
    
    if args.session_id:
        inspect_session(args.session_id)
    else:
        list_sessions()