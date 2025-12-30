from response_handler import get_ai_counterargument, evaluate_response
from value_identifier import value_profiler
from load_dg import load_user_demographics
from question_generator import generate_personalized_questions

def user_experience():
    user_demographics = load_user_demographics("user_1")
    
    user_data = {
        "demographics": {},
        "ai_generated_responses": {},  # Store responses to AI-generated questions
        "phase4": {},  # AI Counterview placeholder
        "phase5": {},  # Post-Reflection
        "bonus": {}    # Feedback & Loop
    }

    # ---- DEMOGRAPHICS (Keep unchanged as requested) ----
    print("\n--- Demographics ---")
    user_data["demographics"] = {
        "political_orientation": user_demographics.get("Political orientation"),
        "libertarian_or_authoritarian": user_demographics.get("Libertarian or authoritarian"),
        "left_or_right": user_demographics.get("Left or right"),
        "conservative_or_progressive": user_demographics.get("Conservative or progressive"),
        "individualist_or_collectivist": user_demographics.get("Individualist or collectivist"),
        "socioeconomic_status": user_demographics.get("Socio economic status"),
        "job_sector": user_demographics.get("Job sector"),
        "religion": user_demographics.get("religion"),
        "location": user_demographics.get("Location"),
        "rural_or_urban": user_demographics.get("Rural or urban"),
        "red_or_blue_state": user_demographics.get("Red or blue state"),
        "red_or_blue_city": user_demographics.get("Red or blue city")
    }

    # ======================
    # AI-Generated Personalized Questions
    # ======================
    print("\n--- Generating Personalized Questions ---")
    print("Analyzing your background to create tailored questions...")
    
    try:
        # Generate questions based on demographics
        # You can change the topic parameter based on your specific use case
        personalized_questions = generate_personalized_questions(
            user_data["demographics"], 
            topic="income tax policy"  # Adjust this based on your topic
        )
        
        print(f"Generated {len(personalized_questions)} personalized questions based on your profile.\n")
        
        # Ask each generated question
        for i, question_data in enumerate(personalized_questions, 1):
            print(f"--- Question {i} ---")
            print(f"[{question_data['category'].replace('_', ' ').title()}]")
            
            # Ask the main question
            response = input(f"{question_data['question_text']} ")
            
            # Store the response with metadata
            user_data["ai_generated_responses"][f"question_{i}"] = {
                "question": question_data['question_text'],
                "category": question_data['category'],
                "response": response,
                "rationale": question_data['rationale'],
                "expected_insights": question_data['expected_insights']
            }
            
            # Ask follow-up questions if they exist and the response warrants it
            if question_data.get('follow_up_prompts') and response.strip() and len(response.strip()) > 20:
                for j, follow_up in enumerate(question_data['follow_up_prompts'], 1):
                    follow_up_response = input(f"  Follow-up: {follow_up} ")
                    if follow_up_response.strip():
                        user_data["ai_generated_responses"][f"question_{i}"][f"follow_up_{j}"] = follow_up_response
            
            print()  # Add spacing between questions
        
    except Exception as e:
        print(f"Error generating personalized questions: {e}")
        print("Falling back to standard questions...")
        
        # Fallback to basic questions if AI generation fails
        user_data["ai_generated_responses"] = _ask_fallback_questions()

    # ======================
    # Phase 4: AI Counterview 
    # ======================
    print("\n--- Generated Counterview ---")
    print("Analyzing your responses to generate a thoughtful counterargument...")
    
    # Transform AI responses into format expected by counterargument generator
    transformed_data = _transform_responses_for_counterargument(user_data)
    
    user_data["phase4"]["counterview"] = get_ai_counterargument(transformed_data, value_profile={})
    print("AI Counterview:")
    print(user_data["phase4"]["counterview"])

    evaluation = evaluate_response(user_data["phase4"]["counterview"])
    print("\n=== Credibility Evaluation ===\n")
    print(evaluation)

    # ======================
    # Phase 5: Post-Reflection
    # ======================
    print("\n--- Reflection ---")

    # Frame 8: Emotional + Intellectual Response
    user_data["phase5"]["agreement_score"] = int(
        input("After reading the counterpoint, rate your agreement (0-10): ")
    )
    user_data["phase5"]["understanding_score"] = int(
        input("Rate your understanding of the other side (0-10): ")
    )
    user_data["phase5"]["discomfort_score"] = int(
        input("Rate your discomfort (0-10): ")
    )
    user_data["phase5"]["emotional_response"] = input(
        "How did you feel reading the counterpoint? "
    )

    # Frame 9: Explain the Other Side
    user_data["phase5"]["explain_other_side"] = input(
        "Imagine you're explaining the other side's argument to someone else. How would you put it? "
    )

    # Frame 10: Opinion Evolution
    user_data["phase5"]["opinion_shift"] = input("Did your opinion shift in any way? ")
    user_data["phase5"]["confidence_change"] = input(
        "Did your confidence in your position increase, decrease, or stay the same? "
    )

    # ======================
    # Bonus: Feedback & Loop
    # ======================
    print("\n--- Feedback & Next Steps ---")
    user_data["bonus"]["continue"] = input(
        "Would you like to explore another issue? (yes/no): "
    )
    user_data["bonus"]["deeper_dive"] = input(
        "Would you like a deeper dive on this one? (yes/no): "
    )

    return user_data

def _transform_responses_for_counterargument(user_data):
    """
    Transform AI-generated question responses into the format expected by 
    the counterargument generator (mimicking the original phase structure)
    """
    ai_responses = user_data["ai_generated_responses"]
    
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
            "friend_explanation": opinion_statement,  # Use same as opinion for now
            "reasoning": reasoning if reasoning else ["No specific reasoning provided"],
            "shaping_event": experience_info
        },
        "phase2": {
            "confidence_score": 7,  # Default value - could extract from responses
            "flexibility": "It depends" if flexibility_info else "No",
            "flexibility_reason": flexibility_info,
            "mind_change_trigger": flexibility_info
        },
        "phase3": {
            "valid_points_other_side": empathy_info,
            "motivation_other_side": empathy_info,
            "identity_shift": "Maybe",  # Default
            "identity_shift_reason": empathy_info
        }
    }
    
    return transformed_data

def _ask_fallback_questions():
    """
    Fallback questions if AI generation fails
    """
    fallback_responses = {}
    
    questions = [
        ("What is your position on this issue?", "position_articulation"),
        ("What are your main reasons for this position?", "value_foundation"),
        ("What experiences have shaped this view?", "experience_grounding"),
        ("How open are you to changing your mind?", "flexibility_assessment"),
        ("Why might others disagree with you?", "empathy_priming")
    ]
    
    for i, (question, category) in enumerate(questions, 1):
        response = input(f"{question} ")
        fallback_responses[f"question_{i}"] = {
            "question": question,
            "category": category,
            "response": response,
            "rationale": "Fallback question",
            "expected_insights": "Basic response"
        }
    
    return fallback_responses

if __name__ == "__main__":
    # Run the experience
    result = user_experience()
    print("\n--- Session Complete ---")
    print("Thank you for participating in this empathy-building exercise!")