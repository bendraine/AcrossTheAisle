import json
from pathlib import Path

def load_user_demographics(user_id: str):
    demographics_path = Path("data/demographics.json")
    if not demographics_path.exists():
        raise FileNotFoundError("Demographics file not found at data/demographics.json")
    
    with open(demographics_path, "r") as f:
        all_users = json.load(f)
    
    if user_id not in all_users:
        raise KeyError(f"User ID {user_id} not found in demographics file.")
    
    return all_users[user_id]
