from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # loads from .env
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

assistant = client.beta.assistants.create(
  name="Financial Analyst Assistant",
  instructions="You are an expert mathmetician. Use you knowledge base to answer questions about these math lecture notes.",
  model="gpt-4o",
  tools=[{"type": "file_search"}],
)

# Create a vector store caled "Financial Statements"
vector_store = client.vector_stores.create(name="Math Lectures")

# Ready the files for upload to OpenAI
file_paths = ["C:/Users/bendr/Downloads/Mth341Lecture14.pdf", "C:/Users/bendr/Downloads/Mth341Lecture13.pdf"]
file_streams = [open(path, "rb") for path in file_paths]

# Use the upload and poll SDK helper to upload the files, add them to the vector store,
# and poll the status of the file batch for completion.
file_batch = client.vector_stores.file_batches.upload_and_poll(
  vector_store_id=vector_store.id, files=file_streams
)

# You can print the status and the file counts of the batch to see the result of this operation.
print(file_batch.status)
print(file_batch.file_counts)

assistant = client.beta.assistants.update(
  assistant_id=assistant.id,
  tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)

# Upload the user provided file to OpenAI
message_file = client.files.create(
  file=open("C:/Users/bendr/Downloads/Mth341Lecture14.pdf", "rb"), purpose="assistants"
)

# Create a thread and attach the file to the message
thread = client.beta.threads.create(
  messages=[
    {
      "role": "user",
      "content": "What is the main topic of the text?",
      # Attach the new file to the message.
      "attachments": [
        { "file_id": message_file.id, "tools": [{"type": "file_search"}] }
      ],
    }
  ]
)

# The thread now has a vector store with that file in its tool resources.
print(thread.tool_resources.file_search)

# Use the create and poll SDK helper to create a run and poll the status of
# the run until it's in a terminal state.

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

message_content = messages[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))











# user_belief = "Raising corporate taxes is bad for the economy."
# user_values = "Wants small government and economic freedom."
# topic = "Corporate Taxes"

# # Template prompt
# system_prompt = (
#     "You are an empathetic political analyst. Your job is to respectfully challenge user viewpoints. "
#     "You must stay calm, avoid sarcasm, and tailor your response based on the user’s beliefs and values. "
#     "Do not agree with the user or re-state their view."
# )

# user_prompt = (
#     f"Context: The user believes '{user_belief}'. They value: {user_values}. "
#     f"Input: '{user_belief}' "
#     "Output: Provide a respectful counterargument, ideally with one economic example or data point."
# )

# response = client.chat.completions.create(
#     model="gpt-4o",
#     messages=[
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": user_prompt}
#     ]
# )

# print(response.choices[0].message.content)



# from flask import Flask, request, jsonify
# import openai

# app = Flask(__name__)

# openai.api_key = "your-api-key-here"

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_input = request.json["message"]
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Be a friendly debate coach."},
#             {"role": "user", "content": user_input}
#         ]
#     )
    
#     return jsonify({"reply": response["choices"][0]["message"]["content"]})