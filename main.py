import os
from user_experience import user_experience
from value_identifier import value_profiler
from response_handler import get_ai_counterargument
from document_store import add_pdf_from_docs

def process_user_data(user_data):
    value_profile = value_profiler(user_data)
    ai_response = get_ai_counterargument(user_data, value_profile)
    return ai_response

def main():
    docs_folder = "../data/docs"
    for filename in os.listdir(docs_folder):
        if filename.lower().endswith(".pdf"):
            add_pdf_from_docs(filename)
    user_data = user_experience()
    process_user_data(user_data)

if __name__ == "__main__":
    main()