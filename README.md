# PolicyScout

**An AI-powered Company Policy Assistant using Retrieval-Augmented Generation (RAG).**

PolicyScout allows users to ask questions about company policies and procedures through a conversational interface. It retrieves relevant information from policy documents and uses a cloud-based language model to generate context-grounded answers.

## Features

- **Document Processing:** Extracts text from DOCX and TXT documents.
- **Text Preprocessing:** Normalises extracted content and removes unnecessary artifacts.
- **Intelligent Chunking:** Splits documents into smaller, meaningful sections while preserving context.
- **Semantic Search:** Uses BGE embeddings and FAISS for similarity-based retrieval.
- **Context-Aware Answers:** Uses an LLM to generate answers based on retrieved policy information.
- **Source References:** Displays the document, section, and chunk used to generate an answer.
- **Conversation Interface:** Provides an interactive Streamlit chat interface.
- **Query Logging:** Records questions, retrieved content, and generated answers locally.

## Technology Stack

| Component            | Technology             |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Frontend             | Streamlit              |
| Embedding Model      | BAAI/bge-small-en-v1.5 |
| Vector Search        | FAISS                  |
| LLM Provider         | OpenRouter             |
| LLM Model Route      | openrouter/free        |
| Document Processing  | python-docx            |
| Numerical Processing | NumPy                  |

## Architecture

```text
             Policy Documents
                    |
                    v
             Document Loader
                    |
                    v
              Preprocessing
                    |
                    v
                Chunking
                    |
                    v
             Embedding Model
                    |
                    v
              FAISS Index
                    |
                    |
User Question       |
      |             |
      v             |
Query Embedding    |
      |             |
      v             |
 FAISS Retrieval <--
      |
      v
 Retrieved Context
      |
      v
   OpenRouter
      |
      v
 Generated Answer
      |
      v
 Streamlit Interface
```

## Project Structure

```text
PolicyScout/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embedding_generator.py
│   ├── vector_store.py
│   └── llm.py
│
├── evaluation/
│   └── evaluate.py
│
├── data/                 # Original policy documents
├── extracted_text/       # Extracted document text
├── preprocessed_text/    # Cleaned text
├── output/               # Chunks, embeddings and FAISS index
└── logs/                 # Application query logs
```

The `data`, `extracted_text`, `preprocessed_text`, `output`, and `logs` directories are generated or populated locally. Their contents are excluded from this repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Arock32/PolicyScout.git
cd PolicyScout
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Configuration

PolicyScout uses OpenRouter to access a cloud-based language model.

1. Create an API key through OpenRouter.
2. Copy `.env.example` and rename the copy to `.env`.
3. Add your API key.

Example:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Never commit your actual `.env` file or API key to GitHub.

## Document Preparation

Create the following directories in the project root:

```text
data/
extracted_text/
preprocessed_text/
output/
logs/
```

Place your policy documents inside `data/`.

The current document loader expects these filenames:

- `Claims User Manual.docx`
- `Regularisiation User Manual.docx`
- `Timecard User Manual.docx`
- `Work_Week_Email.txt`

If you use different filenames, update the document list in `src/document_loader.py` accordingly.

## Document Processing Pipeline

Run the following commands from the project root.

### Step 1: Extract document content

```bash
python -m src.document_loader
```

Extracted content is saved in `extracted_text/`.

### Step 2: Preprocess extracted text

```bash
python -m src.preprocessing
```

Cleaned documents are saved in `preprocessed_text/`.

### Step 3: Generate document chunks

```bash
python -m src.chunking
```

The generated chunks are saved in:

```text
output/chunks.json
```

### Step 4: Generate embeddings

```bash
python -m src.embedding_generator
```

The embedding model generates normalised document embeddings.

The output is saved as:

```text
output/embeddings.npz
```

### Step 5: Build the FAISS index

After generating the embeddings, the FAISS index must be created and saved as:

```text
output/policy_index.faiss
```

The index creation functions are available as commented-out code in `src/vector_store.py`.

To build the index:

1. Uncomment the FAISS index creation functions.
2. Execute the index creation workflow using the generated `embeddings.npz` and corresponding `chunks.json`.
3. Verify that the number of indexed vectors matches the number of document chunks.
4. Once the index is successfully created and saved, comment out the index creation code again.

**Note:** FAISS index creation is a one-time operation. During normal application execution, the existing saved index is loaded using `faiss.read_index()` rather than being recreated every time.

Ensure that the embeddings and document chunks maintain the same ordering so that retrieved vectors map to their correct document chunks.

## Running the Application

Once the required chunk, embedding, and FAISS index files are available, launch the application:

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can interact with PolicyScout.

## Evaluation

The repository includes an evaluation script at:

```text
evaluation/evaluate.py
```

To execute it, provide your own `evaluation/evaluation_questions.json` containing a `questions` array with question IDs, questions, and expected answers.

Run:

```bash
python evaluation/evaluate.py
```

The evaluation script generates JSON results and a CSV report locally.

The original evaluation dataset is intentionally not included in this repository.

## License

No license has currently been specified.