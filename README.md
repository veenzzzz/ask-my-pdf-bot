


Ask My PDF Bot --- Retrieval-Augmented Generation (RAG)
A complete PDF question-answering application built with Python,
Streamlit, Sentence Transformers, FAISS, and Google Gemini.

The application lets a user upload a PDF, extract its text, split the
text into smaller chunks, convert those chunks into embeddings, store
the embeddings in a FAISS vector index, retrieve the most relevant
chunks for a question, and ask an LLM to generate an answer grounded in
the retrieved document content.

1. Project Overview
What problem does this project solve?
PDF files often contain a large amount of information that is difficult
to search manually. A normal keyword search may find matching words but
does not understand the meaning of a natural-language question.

This project uses Retrieval-Augmented Generation (RAG).

Instead of asking the LLM to answer from its general knowledge, the
application:

Reads the uploaded PDF.

Extracts the text.

Splits the text into manageable chunks.

Creates an embedding for every chunk.

Stores the embeddings in FAISS.

Converts the user's question into an embedding.

Searches FAISS for the most relevant chunks.

Sends the retrieved context to Gemini.

Generates an answer based on the retrieved document content.

Displays the answer and source information in the Streamlit UI.

High-level flow
                 PDF
                  |
                  v
           PDF Text Loader
                  |
                  v
            Text Chunker
                  |
                  v
        Sentence Transformer
          Embedding Model
                  |
                  v
             Embeddings
                  |
                  v
             FAISS Index
                  |
                  |
User Question ---> Embedding
                  |
                  v
          Similarity Search
                  |
                  v
       Top Relevant Chunks
                  |
                  v
        Context + Question
                  |
                  v
           Google Gemini
                  |
                  v
              Answer
                  |
                  v
           Streamlit UI
2. What is RAG?
RAG = Retrieval-Augmented Generation.

It combines two major operations:

Retrieval --- find relevant information from the document.

Generation --- use an LLM to produce a natural-language answer
from that information.

For example, suppose the PDF says:

FAISS is used as the vector database for semantic retrieval.

The user asks:

What vector database is used?

The system does not need to send the entire PDF to Gemini.

Instead:

Question
   ↓
Create question embedding
   ↓
Search FAISS
   ↓
Retrieve relevant chunk
   ↓
Send chunk + question to Gemini
   ↓
Generate answer
This is the central idea behind the project.

3. Why Not Send the Entire PDF to the LLM?
Sending the entire PDF every time is inefficient.

Problems include:

Large context size

Higher API usage

Slower responses

More irrelevant information

Difficulty handling large documents

Increased chance of the model using unrelated information

RAG solves this by retrieving only the most relevant parts of the
document.

4. Technologies Used
Core Technologies
Technology Purpose

Python Main programming language
Streamlit Web interface
pdfplumber PDF text extraction
Sentence Transformers Text embeddings
all-MiniLM-L6-v2 Embedding model
NumPy Numerical arrays
FAISS Vector similarity search
Google Gemini Answer generation
python-dotenv Environment variable management
Pickle Saving chunk metadata
Virtual environment Dependency isolation

The internship project guide also identifies Python, embeddings, vector
search, LLM generation, Streamlit/Gradio, and FAISS/Qdrant as relevant
technologies for an industry-style PDF RAG system. The guide recommends
starting with PDF extraction, chunking, Sentence Transformers
(all-MiniLM-L6-v2), FAISS, and an LLM such as Gemini.

5. Project Structure
The project is organized approximately as follows:

ask-my-pdf-bot/
│
├── .venv/
│
├── data/
│   └── sample.pdf
│
├── src/
│   ├── __init__.py
│   ├── loader.py
│   ├── pdf_loader.py
│   ├── chunker.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── indexer.py
│   ├── retriever.py
│   ├── llm.py
│   └── rag_pipeline.py
│
├── vectorstore/
│   ├── index.faiss
│   └── chunks.pkl
│
├── app.py
├── check_chunks.py
├── test_loader.py
├── test_indexer.py
├── test_search.py
├── test_retriever.py
├── test_rag.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
6. Explanation of Every Important File
app.py
This is the Streamlit frontend.

It is responsible for:

Displaying the application

Accepting PDF uploads

Processing documents

Showing document information

Accepting user questions

Calling the RAG pipeline

Displaying generated answers

Displaying retrieved sources

Maintaining chat messages in Streamlit session state

The UI is the part the user interacts with directly.

src/loader.py
Responsible for loading PDF files and extracting their text.

Typical flow:

PDF file
   ↓
pdfplumber
   ↓
Page-by-page text
   ↓
Page metadata
Keeping the loader separate makes the project easier to maintain.

src/pdf_loader.py
Contains PDF-specific loading functionality if the project uses the
separate PDF loader abstraction.

The important requirement is that the loader should preserve page
information whenever possible.

Page information is useful because the final answer can identify where
retrieved information came from.

src/chunker.py
Responsible for splitting loaded document text into chunks.

Why chunking is necessary:

A PDF may contain thousands of words. Embedding an entire PDF as one
vector is not useful for precise retrieval.

Instead:

Large document
      ↓
Page text
      ↓
Smaller chunks
      ↓
Embedding for each chunk
The current project successfully produced:

Pages: 20
Chunks: 50
Embeddings: 50
during indexing.

src/text_splitter.py
Contains the text-splitting logic used by the project.

A good chunking strategy tries to keep related information together
while keeping chunks small enough for effective semantic retrieval.

The internship guide specifically recommends document chunking and later
suggests semantic chunking as an upgrade.

src/embeddings.py
Responsible for converting text into vectors.

The project uses:

SentenceTransformer
Model:
all-MiniLM-L6-v2
Example concept:

"FAISS is a vector search library"
                 ↓
        Embedding Model
                 ↓
       [0.12, -0.43, 0.87, ...]
The vector represents the semantic meaning of the text.

The same model is used for:

Document chunks

User questions

This is important because the vectors must exist in the same embedding
space for similarity search to work correctly.

7. What Is an Embedding?
An embedding converts text into a numerical vector.

For example:

Text A:
"FAISS performs vector similarity search"

              ↓

[0.13, -0.28, 0.71, ...]
Another sentence with a similar meaning will have a vector located
relatively close to it.

For example:

"FAISS searches similar embeddings"

              ↓

[0.15, -0.25, 0.69, ...]
The system can therefore compare the vectors mathematically.

8. Why We Need the Same Embedding Model for Questions and Documents
Suppose document chunks are embedded using:

all-MiniLM-L6-v2
The question must also be embedded using:

all-MiniLM-L6-v2
Otherwise the vectors may not be directly comparable in the intended
way.

The retrieval process is:

Document chunk
      ↓
Embedding model
      ↓
Document vector

User question
      ↓
Same embedding model
      ↓
Question vector

Question vector
      ↓
Compare against document vectors
      ↓
Retrieve closest chunks
9. src/vector_store.py
This module handles the FAISS vector database/index.

The project creates a FAISS index for cosine-style similarity by
normalizing embeddings and using an appropriate FAISS similarity index.

The index stores the vectors.

The actual text chunks are stored separately in:

vectorstore/chunks.pkl
This gives the system two related pieces:

index.faiss
    ↓
Vector similarity search

chunks.pkl
    ↓
Original chunk text + metadata
10. What Is FAISS?
FAISS = Facebook AI Similarity Search.

It is a library for efficient similarity search over vectors.

The basic operation is:

Question vector
      ↓
FAISS
      ↓
Find closest document vectors
      ↓
Return top K results
The project successfully created:

FAISS cosine-similarity index created.
Number of vectors: 50
11. src/indexer.py
The indexer combines the ingestion pipeline.

Its responsibility is essentially:

PDF
 ↓
Load
 ↓
Chunk
 ↓
Embed
 ↓
Build FAISS index
 ↓
Save vector store
The successful indexing test produced:

Pages: 20
Chunks: 50
Embeddings: 50
This confirms that the ingestion and indexing stages are working.

12. src/retriever.py
The retriever takes a user's question and finds relevant chunks.

The current implementation:

Creates an embedding for the question.

Searches FAISS.

Retrieves the requested number of results.

Applies a similarity threshold.

Returns only sufficiently relevant results.

The current default configuration uses:

top_k = 5
similarity_threshold = 0.30
Meaning:

Retrieve up to 5 results.

Ignore results with a score below 0.30.

The threshold is configurable.

13. What Does top_k Mean?
If:

top_k = 5
the retriever asks FAISS for up to five relevant chunks.

Example:

Question
   ↓
FAISS search
   ↓
Result 1
Result 2
Result 3
Result 4
Result 5
A higher value gives the LLM more context but can also introduce
irrelevant information.

A smaller value gives less context but may miss useful information.

14. What Is Similarity Score?
A similarity score indicates how closely a retrieved vector matches the
question vector.

Example:

Result 1 → 0.59
Result 2 → 0.49
Result 3 → 0.41
Result 4 → 0.35
Generally, a higher score indicates stronger similarity under the chosen
FAISS metric.

However, the exact useful threshold depends on the embedding model,
data, chunking strategy, and retrieval implementation.

Do not assume that a score such as 0.30 always means "30% correct."

It is a similarity value, not an answer-confidence percentage.

15. src/llm.py
This module is responsible for communicating with the LLM.

The project uses Google Gemini.

The LLM receives:

Question
+
Retrieved document context
+
Optional conversation history
and generates the final response.

16. Environment Variables
The project uses a .env file.

Example:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=YOUR_SUPPORTED_GEMINI_MODEL
Do not commit the real API key to GitHub.

Add .env to .gitignore.

If an API key has ever been exposed in a screenshot, chat, Git
repository, terminal recording, or public repository, revoke/rotate that
key and create a new one.

Never put a real secret inside this README.

17. src/rag_pipeline.py
This is the central RAG orchestration layer.

The expected flow is:

Question
   ↓
Retriever
   ↓
Relevant chunks
   ↓
Build context
   ↓
LLM
   ↓
Answer
A clean RAG pipeline should return information such as:

{
    "answer": answer,
    "sources": results,
    "context": context
}
The source information is important because it allows the UI to show
which document pages contributed to the answer.

18. Complete RAG Pipeline
The whole application can be understood as six major stages.

Stage 1 --- Ingestion
PDF
 ↓
PDF loader
 ↓
Extract text
Stage 2 --- Chunking
Extracted text
 ↓
Text splitter
 ↓
50 chunks
Stage 3 --- Embedding
Chunks
 ↓
Sentence Transformer
 ↓
Embedding vectors
Stage 4 --- Indexing
Embedding vectors
 ↓
FAISS
 ↓
index.faiss
and:

Chunks + metadata
 ↓
chunks.pkl
Stage 5 --- Retrieval
User question
 ↓
Question embedding
 ↓
FAISS similarity search
 ↓
Top relevant chunks
Stage 6 --- Generation
Question
+
Retrieved chunks
 ↓
Gemini
 ↓
Final answer
19. Current Working Test Results
The project has been tested through the following stages.

PDF loading
The test successfully loaded:

Pages loaded: 20
Chunking
Chunks created: 50
Embeddings
Embeddings created: 50
FAISS
FAISS cosine-similarity index created.
Number of vectors: 50
Vector store
Vector store saved.
Retrieval
The search test successfully returned relevant document chunks with page
numbers and similarity scores.

RAG generation
The RAG test successfully generated an answer from retrieved document
context.

This confirms that the main backend pipeline is functioning:

PDF → chunks → embeddings → FAISS → retrieval → Gemini
20. Installation
Step 1 --- Open the project
cd C:\Users\navee\Downloads\ask-my-pdf-bot
Step 2 --- Create virtual environment
If it does not already exist:

python -m venv .venv
Step 3 --- Activate virtual environment
PowerShell:

.\.venv\Scripts\Activate.ps1
You should see:

(.venv)
at the beginning of the terminal line.

21. Install Dependencies
Install the project's dependencies:

python -m pip install -r requirements.txt
If a specific dependency is missing, install it inside the active
.venv.

For example:

python -m pip install pdfplumber
and:

python -m pip install streamlit
The important point is to install packages into the same Python
environment that runs the project.

Verify the Python executable:

python -c "import sys; print(sys.executable)"
It should point to:

...\ask-my-pdf-bot\.venv\Scripts\python.exe
22. Configure Gemini
Create:

.env
in the project root.

Use:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
GEMINI_MODEL=YOUR_SUPPORTED_GEMINI_MODEL
Do not add the real key to Git.

23. Run the Tests in Order
Run the tests in this order.

Test 1 --- PDF Loader
python test_loader.py
Expected purpose:

PDF → extracted pages
Test 2 --- Indexer
python test_indexer.py
Expected pipeline:

PDF
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS
 ↓
Vector store
Successful output should contain information similar to:

Pages: 20
Chunks: 50
Embeddings: 50
Vector store saved.
Test 3 --- Search
python test_search.py
This checks whether FAISS can retrieve relevant chunks.

Expected output includes:

RETRIEVED DOCUMENTS
with:

Page
Similarity Score
Text
Test 4 --- Retriever
python test_retriever.py
This checks the retriever abstraction and similarity threshold.

Test 5 --- Full RAG
python test_rag.py
This checks:

Question
 ↓
Retriever
 ↓
Context
 ↓
Gemini
 ↓
Answer
24. Run the Streamlit Application
After the backend tests work:

streamlit run app.py
Streamlit will display a local URL in the terminal.

Open the displayed URL in your browser.

25. How to Use the Application
Step 1
Open the Streamlit application.

Step 2
Upload one or more PDF files.

Step 3
Click:

Process PDF(s)
The application should:

Load PDF
 ↓
Create chunks
 ↓
Create embeddings
 ↓
Build FAISS index
 ↓
Save vector store
Step 4
Ask a question.

Example:

What technologies are used in the RAG chatbot?
Step 5
The system retrieves relevant chunks.

Step 6
Gemini generates the answer from the retrieved context.

Step 7
The UI displays the answer and source information.

26. Example Query
Question:

What technologies are used in the RAG chatbot?
The system may retrieve pages containing:

Python
FAISS
HuggingFace embeddings
Gemini
Streamlit
Docker
The retrieved context is then provided to the LLM.

The final answer should be based on the document rather than unrelated
general knowledge.

27. Error Handling
Error: ModuleNotFoundError: No module named 'pdfplumber'
Install the package inside the active virtual environment:

python -m pip install pdfplumber
Then run:

python test_indexer.py
again.

Error: No module named 'src.chunker'
Make sure:

src/
└── chunker.py
exists.

If the file has a different name, make the import match the actual
filename.

For example:

from src.text_splitter import ...
if the implementation is actually in text_splitter.py.

Error: cannot import name 'split_documents'
This means the module exists, but it does not contain a function with
that exact name.

Check:

src/chunker.py
and make sure the function imported by indexer.py actually exists.

The function name and import statement must match exactly.

Error: unexpected keyword argument 'chat_history'
This occurs when the caller sends:

chat_history=...
but the RAGPipeline.ask() method does not accept that parameter.

The caller and function definition must use the same interface.

For example:

def ask(self, question, top_k=5, chat_history=None):
If conversation history is not being used, remove the argument from the
caller instead.

Do not mix the two interfaces.

Error: Streamlit missing import
If the editor reports:

Cannot find module 'streamlit'
while Streamlit is installed, first verify the selected Python
interpreter is the project's .venv.

Then run:

python -m pip install streamlit
and:

python -c "import streamlit; print(streamlit.__version__)"
If the command succeeds but the editor still shows the warning, reload
the editor or select the .venv Python interpreter.

28. PDF FontBBox Warnings
During PDF extraction you may see messages similar to:

Could not get FontBBox from font descriptor because None cannot be parsed as 4 floats
If the program continues and successfully reports:

Pages loaded: 20
Chunks created: 50
Embeddings created: 50
then the warnings did not stop the indexing operation.

They are PDF/font parsing warnings produced while reading the document.

They should be investigated if the extracted text is actually missing or
corrupted, but they do not automatically mean that indexing failed.

29. Hugging Face Authentication Warning
You may see:

Warning: You are sending unauthenticated requests to the HF Hub.
If the embedding model loads successfully, this is not necessarily a
project failure.

It means the Hugging Face Hub request is unauthenticated.

For a local model that has already been downloaded/cached, the
application may continue normally.

Authentication can be configured if higher download limits or
authenticated Hub access is required.

30. PaddleOCR/Transformers max_pixels Warning
You may see a warning similar to:

[ERROR] `max_pixels` is part of PaddleOCRVLImageProcessorKwargs,
but not documented.
This is a documentation/schema warning from the installed
Transformers/PaddleOCR-VL components.

If the application continues to run and your PDF pipeline is producing
correct text, embeddings, and search results, this message is not
evidence that the RAG pipeline itself failed.

Do not change unrelated RAG code solely because of this warning.

31. Security --- API Keys
Never commit:

.env
to GitHub.

Your .gitignore should contain:

.env
.venv/
__pycache__/
*.pyc
vectorstore/
If a real API key has been exposed, rotate it before publishing the
project.

Use a placeholder in documentation:

GEMINI_API_KEY=YOUR_GEMINI_API_KEY
Never paste the actual secret into README files.

32. Why We Save a Vector Store
Without a saved vector store, the application would need to repeat:

PDF
 ↓
Chunking
 ↓
Embedding
 ↓
FAISS index creation
every time the application starts.

Saving:

vectorstore/index.faiss
vectorstore/chunks.pkl
allows the application to load the existing index.

That makes repeated querying much faster.

33. Why the Index and Chunks Are Separate
FAISS stores vectors.

The application also needs the original text.

Therefore:

index.faiss
    ↓
Vector IDs / vectors
and:

chunks.pkl
    ↓
Actual text + metadata
are linked by their ordering/identifiers.

When FAISS returns a matching vector, the application uses that result
to retrieve the corresponding chunk.

34. Source Citations
A strong RAG system should not only answer the question.

It should also show where the answer came from.

The project keeps page metadata such as:

Page: 18
A source can therefore be displayed as:

Source:
sample.pdf
Page 18
This is important for:

Trust

Debugging

Verification

Legal/enterprise use cases

Interview demonstrations

The internship project guide specifically identifies document/page
source citations as an industry-use requirement.

35. Preventing Hallucinations
A RAG system should be instructed to answer using the retrieved document
context.

A strong system prompt should communicate the rule:

Answer using the provided document context.
If the answer cannot be found in the context, say that the
information was not found in the provided document.
Do not invent facts.
This does not mathematically guarantee zero hallucinations, but it
reduces unsupported answers and makes the system's behavior more
controlled.

The internship guide lists zero hallucination on out-of-scope questions
as a target for the project.

36. Conversation Memory
The project can maintain chat history using Streamlit session state and
pass relevant history into the RAG pipeline.

Conceptually:

Question 1
   ↓
Answer 1

Question 2
   ↓
Question 1 + Answer 1 + Question 2
   ↓
Answer 2
Conversation memory is an upgrade beyond basic single-question
retrieval.

The internship guide also recommends conversational memory as an
intermediate improvement.

37. Multi-Document Support
A production-oriented version should support:

PDF 1
PDF 2
PDF 3
PDF 4
   ↓
Combined indexing
   ↓
Single searchable vector store
Each chunk should preserve:

document name
page number
chunk text
Then the answer can identify both:

Document
+
Page
for every source.

38. Current Architecture vs Future Upgrades
Current implementation
pdfplumber
    ↓
Custom chunking
    ↓
Sentence Transformers
all-MiniLM-L6-v2
    ↓
FAISS
    ↓
Retriever
    ↓
Gemini
    ↓
Streamlit
Possible upgrades
The internship guide proposes progressively adding:

Better embedding models such as BAAI/bge-small-en-v1.5

Semantic chunking

Conversational memory

Streamlit interface

Hybrid FAISS + BM25 retrieval

Source citations

Multi-document support

Docker

Offline local LLMs

Celery + Redis asynchronous processing

Agentic RAG

HyDE

Re-ranking with cross-encoders

These are upgrades, not requirements for the current working version.

39. Hybrid Retrieval
Basic retrieval uses semantic similarity:

Question
 ↓
Embedding
 ↓
FAISS
Hybrid retrieval combines semantic search with keyword search:

Question
       |
       +------ Semantic Search → FAISS
       |
       +------ Keyword Search → BM25
       |
       v
Combine / Rank results
       |
       v
Best chunks
This can help when a question contains:

Exact names

Product codes

Legal clauses

Technical identifiers

Rare keywords

40. Re-ranking
A future retrieval system can use:

FAISS
 ↓
Top 20 candidates
 ↓
Cross-encoder / re-ranker
 ↓
Best 5 chunks
 ↓
LLM
This can improve precision by using a stronger model to compare the
question with each candidate chunk.

41. Semantic Chunking
Basic chunking may split text according to characters or fixed sizes.

Semantic chunking attempts to keep logically related content together.

For example:

Heading
Paragraph 1
Paragraph 2
Paragraph 3
should ideally remain together when they describe the same concept.

This is one of the recommended upgrades in the internship project guide.

42. Better Embedding Models
The current project uses:

all-MiniLM-L6-v2
A future version can evaluate stronger embedding models such as:

BAAI/bge-small-en-v1.5
The model should be selected based on:

Retrieval quality

Speed

Memory usage

Domain

Hardware

Deployment constraints

Do not change the embedding model without rebuilding the FAISS index
because document vectors and query vectors must be generated in the same
embedding space.

43. Deployment Roadmap
A production-oriented version can eventually be deployed using:

Streamlit
   ↓
Docker
   ↓
AWS EC2
Documents can be stored in:

AWS S3
and asynchronous document processing can later use:

AWS Lambda
Celery
Redis
The internship guide specifically suggests AWS EC2, S3, Lambda, Docker,
and monitoring as possible deployment components.

44. Suggested 8-Week Development Plan
The internship guide provides an eight-week structure.

Week 1 --- Problem and Setup
Define the business problem

Define KPIs

Set up GitHub

Configure Python environment

Week 2 --- Data Processing
PDF extraction

Data cleaning

Chunking

Initial testing

Week 3 --- Model Development
For this RAG project:

Embeddings

FAISS

Retrieval

LLM integration

Week 4 --- Evaluation
Test retrieval quality

Test answer quality

Tune chunking

Tune similarity threshold

Reduce irrelevant results

Week 5 --- Backend
Organize pipeline

Add logging

Improve error handling

Containerize if required

Week 6 --- UI
Streamlit interface

PDF upload

Chat interface

Source display

Week 7 --- Deployment
Docker

AWS

Monitoring

Configuration

Week 8 --- Testing and Documentation
End-to-end testing

Load/stress testing

README

Screenshots

Demo video

Final presentation

The guide explicitly recommends a comprehensive README, architecture
diagrams, a demo video/screenshots, and documentation of design
decisions for final delivery.

45. Testing Checklist
Before considering the project complete, test:

PDF
Upload one PDF

Upload a large PDF

Upload a PDF with multiple pages

Verify page count

Verify extracted text

Chunking
Verify chunk count

Inspect several chunks

Ensure chunks are not empty

Ensure page metadata is retained

Embeddings
Embedding model loads

Embeddings are generated

Embedding dimensions are consistent

FAISS
Index is created

Vector count matches chunk count

Index can be saved

Index can be loaded

Retrieval
Relevant questions return relevant pages

Irrelevant questions return few/no useful results

Similarity threshold works

top_k works

LLM
Gemini API configuration works

Context is passed correctly

Question is passed correctly

Errors are handled

UI
PDF upload works

Processing works

Question input works

Answer displays

Sources display

Chat history behaves correctly

Clear chat works

46. Common Debugging Strategy
When something fails, test each layer independently.

Do not immediately change the entire project.

Use:

test_loader.py
      ↓
test_indexer.py
      ↓
test_search.py
      ↓
test_retriever.py
      ↓
test_rag.py
      ↓
app.py
This makes it clear which layer contains the problem.

For example:

If test_indexer.py fails
Check:

Loader
Chunker
Embeddings
Vector store
If test_search.py fails
Check:

FAISS
Saved index
Chunks
Query embedding
If test_rag.py fails
Check:

Retriever
RAG pipeline
LLM
API configuration
If tests work but Streamlit fails
Check:

app.py
Streamlit state
UI callbacks
Function arguments
Selected Python interpreter
47. Interview Explanation
A concise interview explanation:

I built an Ask My PDF Bot using Retrieval-Augmented Generation. First,
I extract text from uploaded PDFs and split it into smaller chunks. I
generate semantic embeddings for those chunks using Sentence
Transformers and store the vectors in a FAISS index. When a user asks
a question, I embed the question using the same model and perform
similarity search to retrieve the most relevant chunks. I then pass
the question and retrieved context to Google Gemini to generate a
grounded answer. Streamlit provides the user interface, and the
application also keeps page-level source information so the retrieved
content can be traced back to the PDF.

48. One-Line Architecture for Interviews
PDF → Loader → Chunker → Embeddings → FAISS → Retriever → Gemini → Streamlit Answer
49. Key Concepts You Should Be Able to Explain
Before presenting the project, understand these terms:

What is RAG?

What is an embedding?

Why do we need embeddings?

What is Sentence Transformers?

What is all-MiniLM-L6-v2?

What is FAISS?

What is vector similarity search?

What is cosine similarity?

What is chunking?

Why is chunking required?

What is top_k?

What is a similarity threshold?

What is a vector store?

Why save index.faiss?

Why save chunks.pkl?

What is retrieval?

What is generation?

Why use Gemini?

What is hallucination?

How can RAG reduce hallucinations?

Why are source citations useful?

How does chat history work?

How would you support multiple PDFs?

How would you improve retrieval?

How would you deploy the application?

50. Final Architecture
The completed project can be represented as:

                    ┌──────────────────┐
                    │   PDF Document   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PDF Loader     │
                    │   pdfplumber     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Chunking      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Embeddings     │
                    │ all-MiniLM-L6-v2 │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      FAISS       │
                    │   Vector Index   │
                    └────────┬─────────┘
                             │
                             │
User Question ───────────────┘
      │
      ▼
┌─────────────────────┐
│ Question Embedding  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Similarity Search   │
│      Retriever      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Relevant PDF Chunks │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Gemini LLM        │
│ Question + Context  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Streamlit Interface │
│ Answer + Sources    │
└─────────────────────┘
51. Project Completion Criteria
The project can be considered functionally complete when all of the
following work:

[✓] PDF upload
[✓] PDF text extraction
[✓] Page tracking
[✓] Text chunking
[✓] Sentence Transformer embeddings
[✓] FAISS index creation
[✓] Vector store persistence
[✓] Query embedding
[✓] Similarity retrieval
[✓] Similarity threshold
[✓] Gemini answer generation
[✓] Source information
[✓] Streamlit interface
[✓] End-to-end RAG test
[ ] Production deployment
[ ] Advanced hybrid retrieval
[ ] Re-ranking
[ ] Advanced evaluation
[ ] Load/stress testing
The unchecked items are future production/industry upgrades rather than
prerequisites for the current working local application.

52. Final Summary
This project demonstrates the complete foundation of a practical RAG
application:

Data Ingestion
      ↓
Document Processing
      ↓
Chunking
      ↓
Embedding
      ↓
Vector Database
      ↓
Semantic Retrieval
      ↓
LLM Generation
      ↓
Source-Aware Answer
      ↓
User Interface
The most important concept is that the LLM is not directly searching
the PDF.

The retrieval system first finds relevant document information:

PDF → chunks → embeddings → FAISS → relevant chunks
Then the LLM uses those retrieved chunks:

Question + Retrieved Context → Gemini → Answer
That separation between retrieval and generation is the core of
Retrieval-Augmented Generation.

Reference Project Requirements
The internship project guide describes the Ask My PDF Bot as a 6--8 week
expert-level RAG project and identifies document ingestion, chunking,
embeddings, vector search, and LLM generation as the core technical
requirements. It also recommends source citations, conversational
context, multi-document support, hybrid retrieval, semantic chunking,
stronger embeddings, Docker, and optional offline LLM deployment as
progressive upgrades.

For the current implementation, the working local stack is intentionally
simpler:

Python
+
pdfplumber
+
Sentence Transformers
+
FAISS
+
Google Gemini
+
Streamlit
This provides a clear foundation before adding the advanced production
features.