# test_value_profiler.py
import json
from value_identifier import value_profiler, get_profile_summary

# Mock user data in the "legacy" format (phase1/phase2/phase3)
sample_user_data_legacy = {
    "phase1": {
        "opinion_statement": "I think taxes should be lower to give people more freedom with their money.",
        "friend_explanation": "Government tends to waste funds, and I believe individuals know best how to spend their money.",
        "reasoning": [
            "Lower taxes encourage economic growth.",
            "People should have more control over their finances."
        ],
        "shaping_event": "My parents struggled with high taxes when running their small business."
    },
    "phase2": {
        "flexibility_reason": "If I saw clear proof that government programs worked efficiently, I might reconsider.",
        "mind_change_trigger": "Evidence of tax-funded programs improving quality of life directly."
    },
    "phase3": {
        "valid_points_other_side": "Taxes help pay for infrastructure and education.",
        "motivation_other_side": "They want fairness and equal opportunity.",
        "identity_shift_reason": "If I grew up in a less privileged environment, I might value redistribution more."
    }
}

# Mock user data in the "ai_generated_responses" format
sample_user_data_ai = {
    "ai_generated_responses": {
        "q1": {
            "question": "Do you think taxes are fair?",
            "response": "I think taxes are too high and unfair to middle-class workers.",
            "category": "taxes"
        },
        "q2": {
            "question": "What role should government play in the economy?",
            "response": "Government should stay small and let markets handle most things.",
            "category": "economy"
        }
    }
}

def run_test(user_data, label):
    print(f"\n=== Running test for: {label} ===")
    profile = value_profiler(user_data)
    if profile:
        print("Raw profile dict:")
        print(json.dumps(profile.to_dict(), indent=2))
        print("\nSummary:")
        print(get_profile_summary(profile))
    else:
        print("❌ Failed to generate profile")

if __name__ == "__main__":
    run_test(sample_user_data_legacy, "Legacy format user data")
    run_test(sample_user_data_ai, "AI-generated responses format")
