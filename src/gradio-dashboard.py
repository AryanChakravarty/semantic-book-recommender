import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import gradio as gr

# Load environment variables if needed
load_dotenv()

# Initialize Hugging Face embeddings
try:
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error initializing embeddings: {e}")
    raise

# Load books data
try:
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
except Exception as e:
    print(f"Error loading books data: {e}")
    raise

# Initialize vector store as None
db_books = None

def initialize_vector_store():
    """Initialize the vector store if it hasn't been created yet."""
    global db_books
    if db_books is None:
        try:
            print("Starting to load documents...")
            # Load and split the text documents
            raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
            print(f"Loaded {len(raw_documents)} raw documents")
            
            print("Splitting documents into chunks...")
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,  # Set a reasonable chunk size
                chunk_overlap=200  # Add some overlap for context
            )
            documents = text_splitter.split_documents(raw_documents)
            print(f"Split into {len(documents)} chunks")
            
            print("Creating vector store with embeddings (this may take a while)...")
            # Create vector store with Hugging Face embeddings
            db_books = Chroma.from_documents(
                documents, 
                hf_embeddings,
                persist_directory="./chroma_db"  # Add persistence to avoid recomputing
            )
            print("Vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # Ensure vector store is initialized
    if db_books is None:
        initialize_vector_store()
    
    try:
        # Search for similar documents
        recs = db_books.similarity_search(query, k=initial_top_k)

        # Extract ISBNs from search results
        books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]

        # Filter books DataFrame based on ISBNs
        book_recs = books[books["isbn13"].isin(books_list)]

        # Filter by category if needed
        if category != "All":
            book_recs = book_recs[book_recs["simple_categories"] == category]

        # Limit to top recommendations
        book_recs = book_recs.head(final_top_k)

        # Sort by tone/emotion if specified
        if tone and tone != "All":
            tone_mapping = {
                "Happy": "joy",
                "Surprising": "surprise",
                "Angry": "anger",
                "Suspenseful": "fear",
                "Sad": "sadness"
            }
            if tone in tone_mapping:
                book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)

        return book_recs
    except Exception as e:
        print(f"Error in retrieve_semantic_recommendations: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        # Check if authors is a string
        authors_field = row["authors"]
        if isinstance(authors_field, str):
            authors_split = authors_field.split(";")
        else:
            # Handle missing or non-string values
            authors_split = []

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"] if isinstance(row["authors"], str) else "Unknown author"

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results


# Prepare categories and tones for dropdowns
categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter the description of a book",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    try:
        print("Starting application...")
        print("Initializing vector store (this may take several minutes)...")
        # Initialize vector store before launching the dashboard
        initialize_vector_store()
        print("Launching Gradio interface...")
        dashboard.launch(share=False)  # Set share=False for local use only
    except Exception as e:
        print(f"Error launching dashboard: {e}")
        raise

print(gr.__version__)