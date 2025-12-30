import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass

load_dotenv()

@dataclass
class ValueProfile:
    """Simplified value profile focused on actionable insights"""
    individual_vs_collective: float  # 0 = collective focus, 1 = individual focus
    government_trust: float  # 0 = low trust, 1 = high trust
    change_orientation: float  # 0 = preserve status quo, 1 = embrace change
    primary_concerns: list  # List of main concerns mentioned
    confidence: float  # Overall confidence in the profile (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'individual_vs_collective': self.individual_vs_collective,
            'government_trust': self.government_trust,
            'change_orientation': self.change_orientation,
            'primary_concerns': self.primary_concerns,
            'confidence': self.confidence
        }

class ValueProfiler:
    """Simplified value profiler that focuses on key dimensions for counterarguments"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Test the connection
            self.client.models.list()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def analyze_user_data(self, user_data: Dict[str, Any]) -> Optional[ValueProfile]:
        """
        Analyze user data to extract key value dimensions
        Returns None if analysis fails
        """
        try:
            # Extract relevant text from user responses
            text_content = self._extract_text_content(user_data)
            
            if not text_content.strip():
                self.logger.warning("No substantial text content found in user data")
                return None
            
            # Generate value profile using GPT
            profile_data = self._generate_profile(text_content)
            
            if profile_data:
                return ValueProfile(
                    individual_vs_collective=profile_data.get('individual_vs_collective', 0.5),
                    government_trust=profile_data.get('government_trust', 0.5),
                    change_orientation=profile_data.get('change_orientation', 0.5),
                    primary_concerns=profile_data.get('primary_concerns', []),
                    confidence=profile_data.get('confidence', 0.5)
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing user data: {e}")
            return None
    
    def _extract_text_content(self, user_data: Dict[str, Any]) -> str:
  
        """Extract all relevant text from dynamically generated Q&A responses"""
    
        text_parts = []
        
        ai_responses = user_data.get("ai_generated_responses", {})
        if not ai_responses:
            self.logger.warning("No AI-generated responses found in user data")
            return ""
        
        for _, response_data in ai_responses.items():
            question = response_data.get("question", "")
            response = response_data.get("response", "")
            category = response_data.get("category", "")
            
            if response.strip():
                if category:
                    text_parts.append(f"[{category}] Q: {question}")
                else:
                    text_parts.append(f"Q: {question}")
                text_parts.append(f"A: {response}")
                
                # Handle any dynamic follow-ups
                for key, value in response_data.items():
                    if key.startswith("follow_up_") and value.strip():
                        text_parts.append(f"Follow-up: {value}")
        
        return "\n".join(text_parts)

    
    def _generate_profile(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Use GPT to analyze text and extract value dimensions"""
        
        system_prompt = """You are a political psychology analyst. Analyze the user's text to identify their core value orientations on three key dimensions. Respond with ONLY a JSON object.

            Key Dimensions:
            1. individual_vs_collective: How much they emphasize individual responsibility/freedom (1.0) vs collective action/community responsibility (0.0)
            2. government_trust: Their trust in government institutions and effectiveness (0.0 = very low, 1.0 = very high)  
            3. change_orientation: Their attitude toward change (0.0 = preserve current systems, 1.0 = embrace reform/change)

            Also extract:
            - primary_concerns: List of 2-4 main issues/values they care about (e.g., "economic security", "individual freedom")
            - confidence: Your confidence in this analysis (0.0-1.0)

            Analysis Guidelines:
            - Base your assessment on the actual content of their responses
            - Look for patterns across multiple answers, not just single statements
            - Pay attention to the reasoning behind their positions, not just the positions themselves
            - Consider the language they use (e.g., "personal responsibility" vs "systemic issues")
            - Note any nuanced or contradictory views that suggest moderate positions

            Individual vs Collective Indicators:
            - HIGH (0.8-1.0): Emphasizes personal responsibility, market solutions, individual rights, self-reliance
            - MODERATE (0.4-0.6): Balances individual and collective concerns, context-dependent solutions
            - LOW (0.0-0.2): Emphasizes systemic solutions, collective action, government programs, community responsibility

            Government Trust Indicators:
            - HIGH (0.8-1.0): Supports government programs, believes in effective regulation, trusts institutions
            - MODERATE (0.4-0.6): Mixed views on government effectiveness, supports some programs but not others
            - LOW (0.0-0.2): Skeptical of government efficiency, prefers private solutions, mentions waste/corruption

            Change Orientation Indicators:
            - HIGH (0.8-1.0): Supports reform, mentions need for adaptation, embraces innovation
            - MODERATE (0.4-0.6): Open to some changes but values stability, incremental improvement
            - LOW (0.0-0.2): Values tradition, stability, proven methods, skeptical of change

            Return only this JSON format:
            {
            "individual_vs_collective": 0.0-1.0,
            "government_trust": 0.0-1.0, 
            "change_orientation": 0.0-1.0,
            "primary_concerns": ["concern1", "concern2"],
            "confidence": 0.0-1.0
            }"""
                                
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use cheaper model for this simple task
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this user's responses:\n\n{text_content}"}
                ],
                temperature=0,
                max_tokens=300,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Handle potential JSON formatting issues
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Parse JSON response
            try:
                profile_data = json.loads(content)
                
                # Validate the response has required fields
                required_fields = ['individual_vs_collective', 'government_trust', 'change_orientation', 'primary_concerns', 'confidence']
                if all(field in profile_data for field in required_fields):
                    return profile_data
                else:
                    self.logger.error(f"Missing required fields in value profile response")
                    return None
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON from value profiler: {content}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            return None

# Global profiler instance
_value_profiler = ValueProfiler()

def value_profiler(user_data: Dict[str, Any]) -> Optional[ValueProfile]:
    """
    Main interface function for value profiling
    Returns ValueProfile object or None if analysis fails
    """
    return _value_profiler.analyze_user_data(user_data)

def get_profile_summary(profile: ValueProfile) -> str:
    """Generate a human-readable summary of the value profile"""
    if not profile:
        return "Unable to generate value profile"
    
    summary_parts = []
    
    # Individual vs collective
    if profile.individual_vs_collective > 0.7:
        summary_parts.append("emphasizes individual responsibility and personal freedom")
    elif profile.individual_vs_collective < 0.3:
        summary_parts.append("values collective action and community responsibility")
    else:
        summary_parts.append("balances individual and collective concerns")
    
    # Government trust
    if profile.government_trust > 0.7:
        summary_parts.append("has high trust in government institutions")
    elif profile.government_trust < 0.3:
        summary_parts.append("is skeptical of government effectiveness")
    else:
        summary_parts.append("has moderate trust in government")
    
    # Change orientation
    if profile.change_orientation > 0.7:
        summary_parts.append("supports reform and change")
    elif profile.change_orientation < 0.3:
        summary_parts.append("prefers preserving current systems")
    else:
        summary_parts.append("is open to some changes")
    
    # Primary concerns
    if profile.primary_concerns:
        concerns_text = ", ".join(profile.primary_concerns)
        summary_parts.append(f"is primarily concerned with {concerns_text}")
    
    return f"This user {', '.join(summary_parts)}."