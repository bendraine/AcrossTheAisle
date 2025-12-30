from flask import Flask, request, jsonify, render_template
import os
import logging
from typing import Dict, Any
import uuid
import traceback
from load_dg import load_user_demographics
import json

# Import our enhanced modules
from response_handler import get_ai_counterargument, evaluate_response
from value_identifier import value_profiler, get_profile_summary
from document_store import add_pdf_from_docs, get_store_stats
from question_generator import generate_personalized_questions
from session_manager import (
    get_or_create_session, save_session, get_session_stats,
    validate_age, validate_confidence_score, validate_non_empty_string, sanitize_text_input,
    is_demographics_complete, get_current_ai_question, advance_to_next_question, store_question_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
logger = logging.getLogger(__name__)

def initialize_docs():
    """Initialize document store with better error handling"""
    try:
        docs_folder = "./data/docs"
        if not os.path.exists(docs_folder):
            os.makedirs(docs_folder, exist_ok=True)
            logger.warning(f"Created docs folder at {docs_folder}")
            return

        pdf_files = [f for f in os.listdir(docs_folder) if f.lower().endswith(".pdf")]
        
        if not pdf_files:
            logger.warning("No PDF files found in docs folder")
            return
        
        successful_loads = 0
        for filename in pdf_files:
            try:
                if add_pdf_from_docs(filename):
                    successful_loads += 1
                    logger.info(f"Successfully loaded {filename}")
                else:
                    logger.error(f"Failed to load {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
        
        logger.info(f"✅ Successfully loaded {successful_loads}/{len(pdf_files)} documents")
        
        # Log document store stats
        stats = get_store_stats()
        logger.info(f"Document store stats: {stats}")
        
    except Exception as e:
        logger.error(f"⚠️ Critical error initializing documents: {e}")

# Initialize documents on startup
initialize_docs()

class ConversationHandler:
    """Enhanced conversation handler for AI-generated questions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConversationHandler")
        
        # Define demographic fields for collection
        self.demographic_fields = [
            ("political_orientation", "What is your political orientation? (e.g., Conservative, Liberal, Libertarian, Progressive, Independent)"),
            ("libertarian_or_authoritarian", "On a scale from Libertarian to Authoritarian, where do you lean?"),
            ("left_or_right", "On a scale from Left to Right, where do you position yourself politically?"),
            ("conservative_or_progressive", "Do you consider yourself more Conservative or Progressive?"),
            ("individualist_or_collectivist", "Do you lean more Individualist or Collectivist in your values?"),
            ("socioeconomic_status", "What is your socioeconomic status? (e.g., Working class, Middle class, Upper middle class, etc.)"),
            ("job_sector", "What job sector do you work in? (e.g., Healthcare, Technology, Education, etc.)"),
            ("religion", "What is your religious affiliation or spiritual belief system?"),
            ("location", "What city and state/country are you located in?"),
            ("rural_or_urban", "Do you live in a Rural or Urban area?"),
            ("red_or_blue_state", "Would you consider your state to lean Red (Republican) or Blue (Democratic)?"),
            ("red_or_blue_city", "Would you consider your city to lean Red (Republican) or Blue (Democratic)?")
        ]
    
    def process_message(self, session, message: str) -> Dict[str, Any]:
        """Process a message based on current session state"""
        try:
            # Clean and validate input
            message = sanitize_text_input(message)
            
            # Determine current phase of conversation
            if not session.metadata.get("demographics_complete", False):
                return self._handle_demographics_collection(session, message)
            elif session.metadata.get("current_question_index", -1) < 4:  # 5 questions (0-4)
                return self._handle_dynamic_question_response(session, message)
            elif not session.metadata.get("counterargument_generated", False):
                return self._generate_counterargument(session)
            else:
                return self._handle_reflection_phase(session, message)
            
        except Exception as e:
            self.logger.error(f"Error processing message for session {session.session_id}: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'response': f"I encountered an error processing your response. Please try again. If the problem persists, please refresh and start over.",
                'error': True,
                'conversation_complete': False
            }
    
    def _handle_demographics_collection(self, session, message: str) -> Dict[str, Any]:
        """Handle collection of demographic information"""
        current_demo_index = session.metadata.get("current_demo_index", 0)
        
        # First message - start demographics collection
        if current_demo_index == 0 and not message.strip():
            session.metadata["current_demo_index"] = 0
            field_name, question = self.demographic_fields[0]
            return {
                'response': f"Welcome! I'll help you explore different political perspectives through a personalized conversation. First, I need to understand your background to create the best experience for you.\n\n{question}",
                'conversation_complete': False
            }
        
        # Validate and store current demographic response
        if message.strip() and current_demo_index < len(self.demographic_fields):
            field_name, _ = self.demographic_fields[current_demo_index]
            
            if not validate_non_empty_string(message, 1, 200):
                _, question = self.demographic_fields[current_demo_index]
                return {
                    'response': f"Please provide a valid response. {question}",
                    'error': True,
                    'conversation_complete': False
                }
            
            # Store the demographic data
            session.user_data["demographics"][field_name] = message
            current_demo_index += 1
            session.metadata["current_demo_index"] = current_demo_index
            
            # Check if we've collected all demographics
            if current_demo_index >= len(self.demographic_fields):
                session.metadata["demographics_complete"] = True
                return {
                    'response': "Thank you for providing your background information! I'm now analyzing your profile to generate personalized questions. This may take a moment...",
                    'conversation_complete': False
                }
            else:
                # Ask next demographic question
                field_name, question = self.demographic_fields[current_demo_index]
                return {
                    'response': question,
                    'conversation_complete': False
                }
        
        # Handle invalid response
        _, question = self.demographic_fields[current_demo_index]
        return {
            'response': f"Please provide a valid response. {question}",
            'error': True,
            'conversation_complete': False
        }
    
    def _handle_dynamic_question_response(self, session, message: str) -> Dict[str, Any]:
        """Handle responses and generate next question dynamically"""
        current_index = session.metadata.get("current_question_index", -1)
        
        # First question after demographics
        if current_index == -1:
            if not message.strip():  # Just transitioning from demographics
                next_question = self._generate_dynamic_question(session, 0, "position_articulation")
                session.metadata["current_question_index"] = 0
                return {
                    'response': f"**Question 1 of 5**\n\n{next_question}",
                    'conversation_complete': False
                }
        
        # Store previous response
        if message.strip():
            if not validate_non_empty_string(message, 5, 1000):
                return {
                    'response': "Please provide a more detailed response (at least 5 characters).",
                    'error': True,
                    'conversation_complete': False
                }
            
            question_id = f"question_{current_index + 1}"
            session.user_data["ai_generated_responses"][question_id] = {
                'response': message,
                'question_number': current_index + 1
            }
        
        # Generate next question
        next_index = current_index + 1
        session.metadata["current_question_index"] = next_index
        
        if next_index >= 5:  # Done with questions
            session.metadata["all_questions_answered"] = True
            return {
                'response': "Thank you! I'm now analyzing your answers and generating a personalized counterargument...",
                'conversation_complete': False
            }
        
        # Determine focus for next question
        focus_progression = ["position_articulation", "value_foundation", "experience_grounding", 
                            "flexibility_assessment", "empathy_priming"]
        focus = focus_progression[next_index]
        
        next_question = self._generate_dynamic_question(session, next_index, focus)
        
        return {
            'response': f"**Question {next_index + 1} of 5**\n\n{next_question}",
            'conversation_complete': False
        }

    def _generate_dynamic_question(self, session, question_number: int, focus: str) -> str:
        """Generate question based on conversation so far"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            demographics = session.user_data["demographics"]
            previous_responses = []
            
            # Gather all previous responses
            for resp_data in session.user_data["ai_generated_responses"].values():
                previous_responses.append(resp_data.get('response', ''))
            
            # Build context
            demo_context = "\n".join([f"{k}: {v}" for k, v in demographics.items()])
            conversation_context = "\n\n".join([f"Response {i+1}: {r}" for i, r in enumerate(previous_responses)])
            
            focus_instructions = {
                "position_articulation": "Get their clear position on income tax policy",
                "value_foundation": "Explore underlying values driving their position",
                "experience_grounding": "Ask about personal experiences shaping their view",
                "flexibility_assessment": "Probe their openness to other perspectives",
                "empathy_priming": "Encourage understanding of opposing views"
            }
            
            system_prompt = f"""Generate ONE personalized question about income tax policy.

                Goal: {focus_instructions.get(focus)}

                User Profile:
                {demo_context}

                Previous Responses:
                {conversation_context}

                Requirements:
                - Build naturally on their previous answers
                - Personally relevant to their background
                - Conversational (1-2 sentences max)
                - Don't mention categories or be academic

                Respond with ONLY the question."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}],
                temperature=0.8,
                max_tokens=100
            )
            
            question = response.choices[0].message.content.strip().strip('"')
            self.logger.info(f"Generated dynamic Q{question_number+1}: {question[:50]}...")
            return question
            
        except Exception as e:
            self.logger.error(f"Error generating question: {e}")
            fallbacks = [
                "What's your position on income tax policy?",
                "What values guide your thinking on this?",
                "Has personal experience shaped your view?",
                "What might change your mind?",
                "Why do others disagree with you?"
            ]
            return fallbacks[min(question_number, len(fallbacks)-1)]
        
        
    # def _generate_ai_questions(self, session) -> Dict[str, Any]:
    #     """Generate personalized questions based on demographics"""
    #     try:
    #         self.logger.info(f"Generating AI questions for session {session.session_id}")
            
    #         # Generate personalized questions
    #         personalized_questions = generate_personalized_questions(
    #             session.user_data["demographics"], 
    #             topic="income tax policy"  # Can be made configurable
    #         )
            
    #         if not personalized_questions:
    #             return {
    #                 'response': "I'm having trouble generating personalized questions. Let me ask you some general questions instead.",
    #                 'error': True,
    #                 'conversation_complete': False
    #             }
            
    #         # Store questions and initialize tracking
    #         session.user_data["generated_questions"] = personalized_questions
    #         session.metadata["questions_generated"] = True
    #         session.metadata["current_question_index"] = 0
            
    #         # Ask first question
    #         first_question = personalized_questions[0]
    #         question_text = first_question['question_text']
    #         category_display = first_question['category'].replace('_', ' ').title()
            
    #         response = f"Based on your background, I've generated {len(personalized_questions)} personalized questions to understand your perspective better.\n\n"
    #         response += f"**Question 1 of {len(personalized_questions)}** [{category_display}]\n"
    #         response += question_text
            
    #         self.logger.info(f"Generated {len(personalized_questions)} questions for session {session.session_id}")
    #         return {
    #             'response': response,
    #             'conversation_complete': False
    #         }
            
    #     except Exception as e:
    #         self.logger.error(f"Error generating AI questions: {e}")
    #         return {
    #             'response': "I encountered an error generating personalized questions. Please try refreshing and starting over.",
    #             'error': True,
    #             'conversation_complete': False
    #         }
    
    # def _handle_ai_question_response(self, session, message: str) -> Dict[str, Any]:
    #     """Handle response to AI-generated questions"""
    #     current_question = get_current_ai_question(session)
        
    #     if not current_question:
    #         return {
    #             'response': "Something went wrong with the question flow. Please refresh and start over.",
    #             'error': True,
    #             'conversation_complete': False
    #         }
        
    #     # Validate response
    #     if not validate_non_empty_string(message, 5, 1000):
    #         return {
    #             'response': f"Please provide a more detailed response to: {current_question['question_text']}",
    #             'error': True,
    #             'conversation_complete': False
    #         }
        
    #     # Store the response
    #     store_question_response(session, message)
        
    #     # Check for follow-up questions (simplified - could be enhanced)
    #     current_index = session.metadata["current_question_index"]
    #     follow_up_needed = (
    #         len(message.strip()) < 50 and  # Short response
    #         current_question.get('follow_up_prompts', [])  # Has follow-ups available
    #     )
        
    #     if follow_up_needed and not session.metadata.get(f"asked_followup_{current_index}", False):
    #         # Ask a follow-up question
    #         follow_ups = current_question['follow_up_prompts']
    #         if follow_ups:
    #             session.metadata[f"asked_followup_{current_index}"] = True
    #             return {
    #                 'response': f"Could you elaborate a bit more? {follow_ups[0]}",
    #                 'conversation_complete': False
    #             }
        
    #     # Move to next question
    #     if advance_to_next_question(session):
    #         next_question = get_current_ai_question(session)
    #         current_index = session.metadata["current_question_index"]
    #         total_questions = len(session.user_data["generated_questions"])
    #         category_display = next_question['category'].replace('_', ' ').title()
            
    #         response = f"**Question {current_index + 1} of {total_questions}** [{category_display}]\n"
    #         response += next_question['question_text']
            
    #         return {
    #             'response': response,
    #             'conversation_complete': False
    #         }
    #     else:
    #         # All questions answered, move to counterargument generation
    #         session.metadata["all_questions_answered"] = True
    #         return {
    #             'response': "Thank you for those thoughtful responses! I'm now analyzing your answers and generating a personalized counterargument. This may take a moment...",
    #             'conversation_complete': False
    #         }
    
    def _generate_counterargument(self, session) -> Dict[str, Any]:
        """Generate the AI counterargument based on responses"""
        try:
            self.logger.info("🔄 Generating value profile...")
            
            # Generate value profile from responses
            value_profile = value_profiler(session.user_data)
            
            if value_profile:
                self.logger.info("✅ Value profile generated successfully")
                session.metadata["value_profile_generated"] = True
                
                # Show value profile to user
                profile_summary = get_profile_summary(value_profile)
                profile_confidence = f"{value_profile.confidence:.1%}"
            else:
                self.logger.warning("⚠️ Value profile generation failed")
                profile_summary = "Unable to generate detailed value profile"
                profile_confidence = "Low"
            
            self.logger.info("🔄 Generating AI counterargument...")
            
            # Transform responses for counterargument generator
            transformed_data = self._transform_ai_responses(session.user_data)
            result = get_ai_counterargument(transformed_data, value_profile)
            
            counterargument = result.get('counterargument', 'Unable to generate counterargument')
            sources = result.get('sources', [])
            confidence = result.get('confidence', 0.0)
            
            # Store results
            session.user_data["phase4"]["counterview"] = counterargument
            session.user_data["phase4"]["sources"] = sources
            session.user_data["phase4"]["confidence"] = confidence
            session.user_data["phase4"]["value_profile"] = value_profile.to_dict() if value_profile else None
            
            # Evaluate response quality
            self.logger.info("🔄 Evaluating response quality...")
            evaluation = evaluate_response(counterargument, sources, confidence)
            session.user_data["phase4"]["evaluation"] = evaluation
            
            # Mark counterargument as generated
            session.metadata["counterargument_generated"] = True
            session.metadata["reflection_step"] = 0  # Start reflection phase
            
            # Format response
            response_parts = [
                "=== Value Profile Analysis ===",
                f"**Your Profile:** {profile_summary}",
                f"**Analysis Confidence:** {profile_confidence}",
                "",
                "=== Counterargument ===",
                "Based on your responses and values, here's a thoughtful counterargument:",
                "",
                counterargument
            ]
            
            if sources:
                response_parts.extend([
                    "",
                    "=== Sources Used ===",
                    f"This response drew from {len(sources)} sources:"
                ])
                
                for i, source in enumerate(sources[:3], 1):
                    response_parts.append(f"{i}. {source['document']} (Page {source['page']}, Relevance: {source['relevance_score']:.2f})")
                    response_parts.append(f"   Excerpt: \"{source['excerpt']}\"")
                
                if len(sources) > 3:
                    response_parts.append(f"   ... and {len(sources) - 3} additional sources")
            
            # Add quality evaluation
            if evaluation.get('overall_score'):
                response_parts.extend([
                    "",
                    "=== Quality Assessment ===",
                    f"Overall Quality Score: {evaluation['overall_score']}/5",
                    f"System Confidence: {confidence:.1%}",
                    f"Assessment: {evaluation.get('brief_feedback', 'No detailed feedback available')}"
                ])
            
            response_parts.extend([
                "",
                "Now I'd like to understand your reaction. After reading this counterpoint, please rate your agreement on a scale of 0-10:"
            ])
            
            return {
                'response': "\n".join(response_parts),
                'conversation_complete': False,
                'counterargument_data': {
                    'sources': sources,
                    'confidence': confidence,
                    'evaluation': evaluation,
                    'value_profile': value_profile.to_dict() if value_profile else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error generating counterargument: {e}")
            self.logger.error(traceback.format_exc())
            return {
                'response': "I'm having trouble generating the counterargument right now. This could be due to insufficient source material for your topic, or a temporary system issue. Please try again, or consider exploring a different political issue.",
                'error': True,
                'conversation_complete': False
            }
    
    def _handle_reflection_phase(self, session, message: str) -> Dict[str, Any]:
        """Handle the post-counterargument reflection questions"""
        reflection_step = session.metadata.get("reflection_step", 0)
        
        reflection_questions = [
            ("agreement_score", "After reading the counterpoint, rate your agreement (0-10):", validate_confidence_score),
            ("understanding_score", "Rate your understanding of the other side (0-10):", validate_confidence_score),
            ("discomfort_score", "Rate your discomfort with the counterargument (0-10):", validate_confidence_score),
            ("emotional_response", "How did you feel reading the counterpoint?", lambda x: validate_non_empty_string(x, 5, 500)),
            ("explain_other_side", "Imagine you're explaining the other side's argument to someone else. How would you put it?", lambda x: validate_non_empty_string(x, 10, 500)),
            ("opinion_shift", "Did your opinion shift in any way?", lambda x: validate_non_empty_string(x, 3, 300)),
            ("confidence_change", "Did your confidence in your position increase, decrease, or stay the same?", lambda x: validate_non_empty_string(x, 3, 300)),
            ("continue", "Would you like to explore another issue? (yes/no)", lambda x: x.lower().strip() in ['yes', 'no', 'y', 'n']),
            ("deeper_dive", "Would you like a deeper dive on this issue? (yes/no)", lambda x: x.lower().strip() in ['yes', 'no', 'y', 'n'])
        ]
        
        if reflection_step >= len(reflection_questions):
            # Conversation complete
            session.conversation_complete = True
            return self._generate_final_summary(session)
        
        field_name, question_text, validator = reflection_questions[reflection_step]
        
        # If this is the first reflection question, just ask it
        if reflection_step == 0 and not message.strip():
            return {
                'response': question_text,
                'conversation_complete': False
            }
        
        # Validate the response
        if message.strip():
            if validator(message):
                # Store the response
                if field_name in ["agreement_score", "understanding_score", "discomfort_score"]:
                    session.user_data["phase5"][field_name] = int(message.strip())
                else:
                    session.user_data["phase5"][field_name] = message.strip()
                
                # Move to next question
                session.metadata["reflection_step"] = reflection_step + 1
                
                # Check if we're done
                if session.metadata["reflection_step"] >= len(reflection_questions):
                    session.conversation_complete = True
                    return self._generate_final_summary(session)
                
                # Ask next question
                next_field, next_question, _ = reflection_questions[session.metadata["reflection_step"]]
                return {
                    'response': next_question,
                    'conversation_complete': False
                }
            else:
                # Invalid response, ask again
                return {
                    'response': f"Please provide a valid response. {question_text}",
                    'error': True,
                    'conversation_complete': False
                }
        
        # No message provided, ask the current question
        return {
            'response': question_text,
            'conversation_complete': False
        }
    
    def _generate_final_summary(self, session) -> Dict[str, Any]:
        """Generate final conversation summary"""
        summary_parts = [
            "Thank you for this thoughtful conversation! I hope this helped you see different perspectives.",
            "",
            "=== Conversation Summary ===",
        ]
        
        # Add insights if available
        if session.user_data.get("phase5"):
            phase5 = session.user_data["phase5"]
            if phase5.get("agreement_score") is not None:
                summary_parts.append(f"Your agreement with the counterargument: {phase5['agreement_score']}/10")
            if phase5.get("understanding_score") is not None:
                summary_parts.append(f"Your understanding of the other side: {phase5['understanding_score']}/10")
            if phase5.get("opinion_shift"):
                summary_parts.append(f"Opinion shift: {phase5['opinion_shift']}")
        
        # Add value profile insights if available
        phase4 = session.user_data.get("phase4", {})
        if phase4.get("value_profile"):
            summary_parts.extend([
                "",
                "=== Your Value Profile ===",
                f"Individual vs Collective: {phase4['value_profile']['individual_vs_collective']:.2f}",
                f"Government Trust: {phase4['value_profile']['government_trust']:.2f}",
                f"Change Orientation: {phase4['value_profile']['change_orientation']:.2f}",
                f"Primary Concerns: {', '.join(phase4['value_profile']['primary_concerns'])}"
            ])
        
        summary_parts.extend([
            "",
            "Your responses have been saved for research purposes to help improve political dialogue.",
            "Thank you for contributing to better understanding across political divides!"
        ])
        
        return {
            'response': "\n".join(summary_parts),
            'conversation_complete': True
        }
    
    def _transform_ai_responses(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform AI responses into format expected by counterargument generator"""
        ai_responses = user_data.get("ai_generated_responses", {})
        
        print("\n--- DEBUG: Data being transformed ---")
        print(json.dumps(ai_responses, indent=2))
        # Extract key information from AI responses
        opinion_statement = ""
        reasoning = []
        flexibility_info = ""
        empathy_info = ""
        experience_info = ""
        
        for question_id, response_data in ai_responses.items():
            category = response_data.get("category", "")
            response_text = response_data.get("response", "")
            
            if category == "position_articulation" and not opinion_statement:
                opinion_statement = response_text
            elif category == "value_foundation":
                reasoning.append(response_text)
            elif category == "experience_grounding":
                experience_info = response_text
            elif category == "flexibility_assessment":
                flexibility_info = response_text
            elif category == "empathy_priming":
                empathy_info = response_text
        
        # Create structured data that mimics the original format
        transformed_data = {
            "demographics": user_data["demographics"],
            "phase1": {
                "opinion_statement": opinion_statement,
                "friend_explanation": opinion_statement,
                "reasoning": reasoning if reasoning else ["No specific reasoning provided"],
                "shaping_event": experience_info
            },
            "phase2": {
                "confidence_score": 7,  # Default value
                "flexibility": "It depends" if flexibility_info else "No",
                "flexibility_reason": flexibility_info,
                "mind_change_trigger": flexibility_info
            },
            "phase3": {
                "valid_points_other_side": empathy_info,
                "motivation_other_side": empathy_info,
                "identity_shift": "Maybe",
                "identity_shift_reason": empathy_info
            }
        }
        
        return transformed_data

# Initialize conversation handler
conversation_handler = ConversationHandler()

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint with enhanced error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get or create session
        session = get_or_create_session(session_id)

        # Load demographics if available
        user_demographics = load_user_demographics("user_1")
        if user_demographics:
            session.user_data["demographics"] = user_demographics
            session.metadata["demographics_complete"] = True
        
        # Process the message
        result = conversation_handler.process_message(session, message)
        
        # Save session state
        if not save_session(session):
            logger.error(f"Failed to save session {session_id}")
        
        # Add session_id to response
        result['session_id'] = session_id
        
        # Log conversation progress
        phase = "Demographics"
        if session.metadata.get("demographics_complete"):
            if session.metadata.get("questions_generated"):
                if session.metadata.get("counterargument_generated"):
                    phase = "Reflection"
                else:
                    current_q = session.metadata.get("current_question_index", -1) + 1
                    total_q = len(session.user_data.get("generated_questions", []))
                    phase = f"AI Questions ({current_q}/{total_q})"
            else:
                phase = "Generating Questions"
        
        logger.info(f"Session {session_id[:8]} - {phase} - {'Complete' if session.conversation_complete else 'In Progress'}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.',
            'response': 'I encountered an unexpected error. Please refresh the page and start over.'
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    try:
        return jsonify({
            'document_stats': get_store_stats(),
            'session_stats': get_session_stats()
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to retrieve stats'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Political empathy platform is running'
    })

@app.route('/')
def home():
    """Serve the main page"""
    return render_template('index.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./chroma_store', exist_ok=True)
    
    # Log startup information
    logger.info("🚀 Starting Political Empathy Platform with AI Questions")
    logger.info(f"Document store stats: {get_store_stats()}")
    logger.info("✅ Server ready to handle requests")
    
    app.run(debug=True, port=5000)