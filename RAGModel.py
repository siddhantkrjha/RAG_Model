import os
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Set your API keys here
openai.api_key = os.getenv("OPENAI_API_KEY", "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")  # Replace with your actual OpenAI API key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "42ed1766-49d4-4fcc-b910-bafbdaeaa2c3")  # Replace with your actual Pinecone API key

# Initialize Pinecone instance (New API)
pc = Pinecone(
    api_key=PINECONE_API_KEY
)

# Define index name and specifications
index_name = "ragmodel-index"

# Check if the index exists, otherwise create it
if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Set dimension to 1536 for OpenAI's text embedding models
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Define your preferred cloud and region
    )

# Connect to the index
index = pc.Index(index_name)

# Sample dataset
data = [
    {"question": "What are your business hours?", "answer": "We are open from 9 AM to 5 PM, Monday to Friday."},
    {"question": "What is your return policy?", "answer": "You can return items within 30 days with a valid receipt."},
    {"question": "Where are you located?", "answer": "We are located at 123 Main Street, City, Country."},
    {"question": "How can I contact customer support?",
     "answer": "You can contact customer support via email at support@example.com."},
    {"question": "Do you offer international shipping?",
     "answer": "Yes, we offer international shipping to selected countries."}
]


# Embedding generation function (updated for the new API)
def generate_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response['data'][0]['embedding']
    except openai.OpenAIError as e:
        print(f"Error generating embedding: {e}")
        return None


# Generate detailed answer using GPT
def generate_answer(query, retrieved_answer):
    prompt = f"Question: {query}\nAnswer: {retrieved_answer}\nGenerate a more detailed response."
    try:
        response = openai.chat_completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response['choices'][0]['message']['content'].strip()
    except openai.OpenAIError as e:
        print(f"Error generating answer: {e}")
        return "Error generating answer."


# Prepare dataset for embedding
texts = [item['question'] + " " + item['answer'] for item in data]
embeddings = [generate_embedding(text) for text in texts]
embeddings = [emb for emb in embeddings if emb is not None]  # Remove None embeddings

# Store the embeddings in Pinecone
for i, embedding in enumerate(embeddings):
    index.upsert([(str(i), embedding)])  # Use upsert to store embeddings


# Query Pinecone for relevant documents
def query_pinecone(query, top_k=1):
    query_embedding = generate_embedding(query)

    if query_embedding is None:
        print("Failed to generate embedding for the query.")
        return None, None  # Early exit if embedding generation fails

    result = index.query(queries=[query_embedding], top_k=top_k)

    # Print result for debugging
    print(f"Query: {query}\nEmbedding: {query_embedding}\nResult: {result}")

    if len(result['matches']) == 0:
        print("No matches found for the query.")
        return None, "No relevant answer found."  # Handle no matches case

    match_id = result['matches'][0]['id']
    return match_id, data[int(match_id)]['answer']


# Test with multiple queries
queries = [
    "What is your return policy?",
    "Where are you located?",
    "How can I contact customer support?",
    "Do you offer international shipping?"
]

for q in queries:
    doc_id, answer = query_pinecone(q)
    if answer is not None:  # Ensure answer is valid
        final_answer = generate_answer(q, answer)
        print(f"Query: {q}\nAnswer: {final_answer}\n")
    else:
        print(f"Query: {q}\nAnswer: No relevant answer found.\n")
