from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import json
from typing import Dict, Any, List, Optional
import re

load_dotenv()

class DynamicQuestionGenerator:
    """
    AI-powered question generator that creates personalized questionnaires
    based on user demographics to elicit optimal information for counterargument generation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Test the connection
            self.client.models.list()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def generate_personalized_questions(self, demographics: Dict[str, Any], topic: str = "political issue") -> List[Dict[str, Any]]:
        """
        Generate a personalized questionnaire based on user demographics
        
        Args:
            demographics: User demographic information
            topic: The topic area for generating contextually relevant questions
            
        Returns:
            List of question dictionaries with metadata
        """
        try:
            demographic_context = self._build_demographic_context(demographics)
            
            system_prompt = self._get_system_prompt()
            user_prompt = self._build_user_prompt(demographic_context, topic)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.9,
                max_tokens=10000,
                top_p=0.92,
                frequency_penalty=0.5,
                presence_penalty=1.0,
                response_format={ "type": "json_object" }  
            )  
            
            # Parse the JSON response
            questions_data = self._parse_questions_response(response.choices[0].message.content)               
                
            # Validate and structure questions
            structured_questions = self._structure_questions(questions_data)
            
            self.logger.info(f"Generated {len(structured_questions)} personalized questions")
            return structured_questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {e}")
            return self._get_fallback_questions()
    
    def _get_system_prompt(self) -> str:
        """Expert-level system prompt for question generation"""
        return """You are an expert in political psychology, survey design, and empathy-building conversations. Your task is to generate a personalized questionnaire that will:

            1. **Elicit genuine viewpoints** - Get users to articulate their position authentically
            2. **Reveal underlying values** - Understand what moral foundations and life experiences drive their beliefs
            3. **Assess cognitive flexibility** - Gauge their openness to alternative perspectives
            4. **Prime for empathy** - Prepare them to genuinely consider opposing viewpoints
            5. **Optimize for counterargument effectiveness** - Gather information that enables the most persuasive, respectful counterarguments

            ## Core Psychological Principles:
            - **Cognitive Dissonance Reduction**: Frame questions to reduce defensiveness and threat perception
            - **Value-Based Reasoning**: Identify which moral foundations (care/harm, fairness, loyalty, authority, sanctity, liberty) are most salient
            - **Perspective-Taking Priming**: Use questions that naturally encourage considering multiple viewpoints
            - **Social Identity Awareness**: Understand how group membership influences political reasoning
            - **Motivated Reasoning Detection**: Identify where emotion vs. logic primarily drives their position

            ## Question Design Guidelines:

            ### Demographic Responsiveness:
            - **Socioeconomic Status**: Higher SES → focus on policy nuance, systemic effects; Lower SES → focus on lived experience, practical impacts
            - **Geographic Context**: Urban → emphasize efficiency, diversity; Rural → emphasize tradition, local impact, personal responsibility
            - **Political Orientation**: 
            - Conservatives → frame around tradition, security, local control, unintended consequences
            - Liberals → frame around progress, equality, systemic solutions, marginalized groups
            - Libertarians → focus on individual liberty, government overreach, market solutions
            - **Religious Context**: Incorporate moral reasoning styles and community values appropriately
            - **Professional Context**: Use relevant analogies and concerns from their work sector

            ### Question Categories (Generate ~5-7 total):
            1. **Position Articulation** (1-2 questions): Get clear, nuanced statement of their view
            2. **Value Foundation Identification** (1-2 questions): Understand moral reasoning behind position
            3. **Experience Grounding** (1 question): Connect position to personal/observed experiences
            4. **Flexibility Assessment** (1 question): Gauge openness to alternative evidence/perspectives
            5. **Empathy Priming** (1-2 questions): Encourage perspective-taking before counterargument

            ### Linguistic Optimization:
            - Use **their demographic's preferred terminology** and avoid triggering words
            - Match **complexity level** to education/professional background
            - Incorporate **cultural references** and **value language** that resonates
            - Use **open-ended formats** that invite storytelling and personal reflection
            - Frame questions to **minimize social desirability bias**

            ### Advanced Techniques:
            - **Gradient questioning**: Move from easier to more challenging reflection
            - **Assumption surfacing**: Help them recognize unstated beliefs
            - **Counterfactual thinking**: "What if" scenarios based on their background
            - **Value hierarchy probing**: When values conflict, which takes priority?
            - **Source credibility mapping**: What evidence would they find most convincing?

            Respond ONLY with a valid JSON object in this exact format:
            {
            "questions": [
                {
                "id": 1,
                "category": "position_articulation|value_foundation|experience_grounding|flexibility_assessment|empathy_priming",
                "question_text": "The actual question to ask the user",
                "rationale": "Why this question is effective for this demographic profile",
                "expected_insights": "What key information this will reveal for counterargument generation",
                "follow_up_prompts": ["Optional clarifying questions if needed"]
                }
            ]
            }
            CRITICAL REQUIREMENT: Your entire response must be a single, raw, and perfectly-formed JSON object. Do not include any explanatory text, comments, markdown, or any characters outside of the main JSON structure. Ensure all strings are enclosed in double quotes and are properly escaped.
"""

    def _build_user_prompt(self, demographic_context: str, topic: str) -> str:
        """Build the user prompt with demographic context"""
        return f"""Generate a personalized questionnaire for this user profile:

            ## User Demographics:
            {demographic_context}

            ## Topic Context:
            {topic}

            ## Requirements:
            - Generate 5-7 strategically designed questions
            - Optimize for this specific demographic profile
            - Questions should feel natural and respectful to someone with this background
            - Focus on gathering information that will enable the most effective, empathetic counterargument
            - Ensure questions progress logically from easier self-reflection to deeper value exploration

            ## Key Considerations:
            - What terminology and framing will resonate most with this demographic?
            - What values and concerns are likely most salient for someone with this background?
            - What types of evidence and reasoning will they find most credible?
            - How can we prime them for genuine empathy and perspective-taking?
            - What potential blind spots or assumptions might they have based on their demographics?

            Generate questions that would feel personally relevant and intellectually engaging to someone with this exact demographic profile."""

    def _build_demographic_context(self, demographics: Dict[str, Any]) -> str:
        """Build a readable context summary from demographics"""
        context_parts = []
        
        # Political orientation
        if demographics.get("political_orientation"):
            context_parts.append(f"Political Orientation: {demographics['political_orientation']}")
        
        if demographics.get("left_or_right"):
            context_parts.append(f"Left/Right: {demographics['left_or_right']}")
            
        if demographics.get("conservative_or_progressive"):
            context_parts.append(f"Conservative/Progressive: {demographics['conservative_or_progressive']}")
        
        if demographics.get("libertarian_or_authoritarian"):
            context_parts.append(f"Libertarian/Authoritarian: {demographics['libertarian_or_authoritarian']}")
            
        if demographics.get("individualist_or_collectivist"):
            context_parts.append(f"Individual/Collective: {demographics['individualist_or_collectivist']}")
        
        # Socioeconomic
        if demographics.get("socioeconomic_status"):
            context_parts.append(f"Socioeconomic Status: {demographics['socioeconomic_status']}")
            
        if demographics.get("job_sector"):
            context_parts.append(f"Job Sector: {demographics['job_sector']}")
        
        # Geographic
        if demographics.get("location"):
            context_parts.append(f"Location: {demographics['location']}")
            
        if demographics.get("rural_or_urban"):
            context_parts.append(f"Rural/Urban: {demographics['rural_or_urban']}")
            
        if demographics.get("red_or_blue_state"):
            context_parts.append(f"State Political Lean: {demographics['red_or_blue_state']}")
            
        if demographics.get("red_or_blue_city"):
            context_parts.append(f"City Political Lean: {demographics['red_or_blue_city']}")
        
        # Cultural
        if demographics.get("religion"):
            context_parts.append(f"Religion: {demographics['religion']}")
        
        return "\n".join(context_parts) if context_parts else "Limited demographic information available"
    
    def _parse_questions_response(self, response_text: str) -> Dict[str, Any]:
        
        """Parse and repair JSON response from the AI"""
        # Debug raw response
        cleaned = response_text.strip()
        print("RAW GPT RESPONSE:", cleaned)

        # Remove markdown fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        # Try parsing directly
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to extract first {...} block
            braces_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if braces_match:
                candidate = braces_match.group(0)

                # Fix common issues:
                candidate = candidate.replace("“", "\"").replace("”", "\"")  # smart quotes → normal
                candidate = candidate.replace("\n", " ")  # flatten newlines
                candidate = re.sub(r",\s*}", "}", candidate)  # remove trailing commas
                candidate = re.sub(r",\s*]", "]", candidate)

                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as inner_e:
                    print("CLEANED CANDIDATE STILL FAILED:", candidate)
                    raise inner_e

            raise ValueError("Invalid JSON response from AI")

    
    def _structure_questions(self, questions_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Structure and validate the questions"""
        questions = questions_data.get('questions', [])
        
        structured = []
        for q in questions:
            raw_text = q.get('question_text', '')
            if isinstance(raw_text, str) and raw_text.strip():
                # Clean the text of any non-printable characters
                # This regex removes characters that are not standard letters, numbers,
                # punctuation, or whitespace.
                cleaned_text = re.sub(r'[^\x20-\x7E\n\r\t]', '', raw_text).strip()
                
                if cleaned_text:
                    structured_q = {
                        'id': q.get('id', len(structured) + 1),
                        'category': q.get('category', 'general'),
                        'question_text': cleaned_text, # Use the cleaned text
                        'rationale': q.get('rationale', ''),
                        'expected_insights': q.get('expected_insights', ''),
                        'follow_up_prompts': q.get('follow_up_prompts', [])
                    }
                    structured.append(structured_q)
        
        return structured
    
    def _get_fallback_questions(self) -> List[Dict[str, Any]]:
        """Fallback questions if AI generation fails"""
        return [
            {
                'id': 1,
                'category': 'position_articulation',
                'question_text': 'What is your current position on this issue, and how would you explain it to someone who disagrees?',
                'rationale': 'Gets clear position statement',
                'expected_insights': 'Core viewpoint and reasoning style',
                'follow_up_prompts': []
            },
            {
                'id': 2,
                'category': 'value_foundation',
                'question_text': 'What values or principles are most important to you when thinking about this issue?',
                'rationale': 'Identifies moral foundations',
                'expected_insights': 'Underlying value system',
                'follow_up_prompts': []
            },
            {
                'id': 3,
                'category': 'experience_grounding',
                'question_text': 'Has a personal experience or something you witnessed influenced your view on this?',
                'rationale': 'Connects to personal experience',
                'expected_insights': 'Emotional and experiential basis',
                'follow_up_prompts': []
            },
            {
                'id': 4,
                'category': 'flexibility_assessment',
                'question_text': 'What kind of evidence or argument might make you reconsider your position?',
                'rationale': 'Assesses openness to change',
                'expected_insights': 'Cognitive flexibility and persuasion targets',
                'follow_up_prompts': []
            },
            {
                'id': 5,
                'category': 'empathy_priming',
                'question_text': 'Why do you think reasonable people might disagree with your position?',
                'rationale': 'Primes perspective-taking',
                'expected_insights': 'Empathy capacity and understanding of opposition',
                'follow_up_prompts': []
            }
        ]

# Global instance
_question_generator = DynamicQuestionGenerator()

def generate_personalized_questions(demographics: Dict[str, Any], topic: str = "political issue") -> List[Dict[str, Any]]:
    """
    Main interface function for generating personalized questions
    
    Args:
        demographics: User demographic data
        topic: Topic context for questions
        
    Returns:
        List of personalized question dictionaries
    """
    return _question_generator.generate_personalized_questions(demographics, topic)