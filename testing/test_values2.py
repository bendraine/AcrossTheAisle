# test_value_profiler.py
from value_identifier import ValueProfile, ValueProfiler

def test_dynamic_responses():
    # Fake user_data with dynamic Q&A
    user_data = {
        "ai_generated_responses": {
            "question_1": {
                "question": "What is your view on income tax?",
                "category": "position_articulation",
                "response": "I think taxes should be lower because people deserve to keep more of their money.",
                "rationale": "Personal freedom focus",
                "expected_insights": "Individualist perspective"
            },
            "question_2": {
                "question": "Why do you hold this view?",
                "category": "value_foundation",
                "response": "Because I believe government wastes money and individuals know best how to spend.",
                "rationale": "Efficiency concern",
                "expected_insights": "Skepticism of government"
            },
            "question_3": {
                "question": "Have your experiences shaped this belief?",
                "category": "experience_grounding",
                "response": "Yes, I once saw a program in my city fail despite high funding.",
                "follow_up_1": "It made me distrust government programs."
            }
        }
    }

    profiler = ValueProfiler()
    text_content = profiler._extract_text_content(user_data)

    print("\n=== Extracted Text Content ===\n")
    print(text_content)

    # Use the correct method from your class
    profile = profiler._generate_profile(user_data)

    print("\n=== Value Profile Object ===\n")
    print(profile)

if __name__ == "__main__":
    test_dynamic_responses()
