import json
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import sqlite3
import threading
import logging
from pathlib import Path

@dataclass
class ConversationSession:
    """Enhanced session data structure for AI-generated questions"""
    session_id: str
    step: int = 0
    user_data: Dict[str, Any] = None
    conversation_complete: bool = False
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.user_data is None:
            self.user_data = {
                "demographics": {},
                "ai_generated_responses": {},  # New format for AI questions
                "generated_questions": [],     # Store the generated questions
                "phase4": {},  # AI Counterview
                "phase5": {},  # Post-Reflection
                "bonus": {}    # Feedback & Loop
            }
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {
                "current_question_index": -1,  # Track which AI question we're on
                "demographics_complete": False,
                "questions_generated": False,
                "value_profile_generated": False
            }

class SessionManager:
    """Persistent session manager using SQLite"""
    
    def __init__(self, db_path: str = "./data/sessions.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._init_db()
        
        # Clean up old sessions periodically
        self._cleanup_old_sessions()
    
    def _init_db(self):
        """Initialize SQLite database with proper schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        step INTEGER NOT NULL DEFAULT 0,
                        user_data TEXT NOT NULL DEFAULT '{}',
                        conversation_complete BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{}'
                    )
                """)
                
                # Create index for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sessions_updated_at 
                    ON sessions (updated_at)
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than max_age_hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM sessions WHERE updated_at < ? AND conversation_complete = TRUE",
                    (cutoff_time,)
                )
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    self.logger.info(f"Cleaned up {deleted} old sessions")
                    
        except Exception as e:
            self.logger.error(f"Error cleaning up sessions: {e}")
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a session by ID"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        return ConversationSession(
                            session_id=row['session_id'],
                            step=row['step'],
                            user_data=json.loads(row['user_data']),
                            conversation_complete=bool(row['conversation_complete']),
                            created_at=datetime.fromisoformat(row['created_at']),
                            updated_at=datetime.fromisoformat(row['updated_at']),
                            metadata=json.loads(row['metadata'])
                        )
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def create_session(self, session_id: str) -> ConversationSession:
        """Create a new session"""
        session = ConversationSession(session_id=session_id)
        self.save_session(session)
        return session
    
    def save_session(self, session: ConversationSession) -> bool:
        """Save or update a session"""
        try:
            session.updated_at = datetime.now()
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions 
                        (session_id, step, user_data, conversation_complete, created_at, updated_at, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        session.session_id,
                        session.step,
                        json.dumps(session.user_data),
                        session.conversation_complete,
                        session.created_at.isoformat(),
                        session.updated_at.isoformat(),
                        json.dumps(session.metadata)
                    ))
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error saving session {session.session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM sessions WHERE session_id = ?",
                        (session_id,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self.logger.error(f"Error deleting session {session_id}: {e}")
            return False
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about sessions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Total sessions
                total = conn.execute("SELECT COUNT(*) as count FROM sessions").fetchone()['count']
                
                # Completed sessions
                completed = conn.execute(
                    "SELECT COUNT(*) as count FROM sessions WHERE conversation_complete = TRUE"
                ).fetchone()['count']
                
                # Sessions by step
                steps = {}
                step_data = conn.execute(
                    "SELECT step, COUNT(*) as count FROM sessions GROUP BY step ORDER BY step"
                ).fetchall()
                
                for row in step_data:
                    steps[row['step']] = row['count']
                
                # Recent activity (last 24 hours)
                recent_cutoff = datetime.now() - timedelta(hours=24)
                recent = conn.execute(
                    "SELECT COUNT(*) as count FROM sessions WHERE updated_at > ?",
                    (recent_cutoff,)
                ).fetchone()['count']
                
                return {
                    "total_sessions": total,
                    "completed_sessions": completed,
                    "completion_rate": completed / total if total > 0 else 0,
                    "sessions_by_step": steps,
                    "recent_activity_24h": recent
                }
                
        except Exception as e:
            self.logger.error(f"Error getting session stats: {e}")
            return {"error": str(e)}

# Global session manager instance
_session_manager = SessionManager()

def get_or_create_session(session_id: str) -> ConversationSession:
    """Get existing session or create new one"""
    session = _session_manager.get_session(session_id)
    if session is None:
        session = _session_manager.create_session(session_id)
    return session

def save_session(session: ConversationSession) -> bool:
    """Save session to persistent storage"""
    return _session_manager.save_session(session)

def delete_session(session_id: str) -> bool:
    """Delete a session"""
    return _session_manager.delete_session(session_id)

def get_session_stats() -> Dict[str, Any]:
    """Get session statistics"""
    return _session_manager.get_session_stats()

# Input validation helpers
def validate_age(age_str: str) -> Optional[int]:
    """Validate and parse age input"""
    try:
        age = int(age_str.strip())
        if 13 <= age <= 120:  # Reasonable age range
            return age
        return None
    except (ValueError, AttributeError):
        return None

def validate_confidence_score(score_str: str) -> Optional[int]:
    """Validate and parse confidence score (0-10)"""
    try:
        score = int(score_str.strip())
        if 0 <= score <= 10:
            return score
        return None
    except (ValueError, AttributeError):
        return None

def validate_non_empty_string(text: str, min_length: int = 1, max_length: int = 1000) -> bool:
    """Validate that text is non-empty and within length limits"""
    if not isinstance(text, str):
        return False
    text = text.strip()
    return min_length <= len(text) <= max_length

def sanitize_text_input(text: str) -> str:
    """Basic text sanitization"""
    if not isinstance(text, str):
        return ""
    # Remove excessive whitespace and limit length
    text = ' '.join(text.split())
    return text[:2000]  # Reasonable limit for user input

# Session state helpers
def is_demographics_complete(session: ConversationSession) -> bool:
    """Check if all required demographics are collected"""
    required_demo_fields = [
        "political_orientation", "libertarian_or_authoritarian", "left_or_right",
        "conservative_or_progressive", "individualist_or_collectivist",
        "socioeconomic_status", "job_sector", "religion", "location",
        "rural_or_urban", "red_or_blue_state", "red_or_blue_city"
    ]
    
    demographics = session.user_data.get("demographics", {})
    return all(field in demographics and demographics[field] for field in required_demo_fields)

def get_current_ai_question(session: ConversationSession) -> Optional[Dict[str, Any]]:
    """Get the current AI-generated question based on session state"""
    current_index = session.metadata.get("current_question_index", -1)
    questions = session.user_data.get("generated_questions", [])
    
    if 0 <= current_index < len(questions):
        return questions[current_index]
    return None

def advance_to_next_question(session: ConversationSession) -> bool:
    """Advance to the next AI-generated question. Returns True if there are more questions."""
    current_index = session.metadata.get("current_question_index", -1)
    questions = session.user_data.get("generated_questions", [])
    
    next_index = current_index + 1
    session.metadata["current_question_index"] = next_index
    
    return next_index < len(questions)

def store_question_response(session: ConversationSession, response: str, follow_ups: Dict[str, str] = None):
    """Store response to current AI question"""
    current_index = session.metadata.get("current_question_index", -1)
    questions = session.user_data.get("generated_questions", [])
    
    if 0 <= current_index < len(questions):
        question_data = questions[current_index]
        question_id = f"question_{current_index + 1}"
        
        response_data = {
            "question": question_data['question_text'],
            "category": question_data['category'],
            "response": response,
            "rationale": question_data.get('rationale', ''),
            "expected_insights": question_data.get('expected_insights', '')
        }
        
        if follow_ups:
            response_data.update(follow_ups)
        
        session.user_data["ai_generated_responses"][question_id] = response_data