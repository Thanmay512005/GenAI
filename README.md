TravelMate Karnataka — RAG Generative AI Project
TravelMate is a Karnataka-focused travel assistant built as a Retrieval-Augmented Generation (RAG) application. It retrieves relevant passages from a local travel knowledge base using semantic embeddings and ChromaDB, then asks Google Gemini to generate a concise answer grounded in those passages. Retrieved sources are displayed alongside answers.
Features
RAG chatbot with grounded answers and source passages.
Local semantic vector search with ChromaDB cosine similarity.
Sentence Transformers `all-MiniLM-L6-v2` embeddings (384 dimensions).
Expanded district-wise Karnataka travel knowledge base covering all 31 current districts in `data/travel_faq.txt`.
Structured district directory in `data/districts.json`, including destination descriptions, travel themes, suggested durations and planning notes.
Sample multi-day regional itineraries for heritage, coast, hills, wildlife and city trips.
Explore destinations, travel reminders/notices, and traveler question submission.
Admin login for FAQ and notice management.
Persistent JSON data for notices and submitted questions.
Automatic first-run vector index construction.
Extractive response fallback when Gemini generation is unavailable but relevant documents were retrieved.
Architecture
```text
Traveler question
      |
      v
Streamlit UI
      |
      v
SentenceTransformer embedding (384D)
      |
      v
ChromaDB semantic retrieval (top-k)
      |
      v
Retrieved Karnataka travel passages
      |
      v
Grounded prompt + Google Gemini
      |
      v
Generated answer + visible source passages
```
The RAG retrieval stage is essential: the model prompt contains the selected knowledge-base passages and explicitly instructs Gemini not to invent details or use unsupported facts. If no passages are found, the application declines to fabricate an answer.
Requirements
Windows 10/11, Python 3.10–3.12 recommended (Python 3.13 may work depending on available wheels for PyTorch/Sentence Transformers).
Google Gemini API key for generative answers. Semantic search and the extractive fallback can work without a key after dependencies and embedding model are installed.
Internet access on first run to download the embedding model and call Gemini.
Windows setup (PowerShell)
Open this folder in VS Code, then run:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```
Edit `.env` and set `GEMINI_API_KEY` and a private `ADMIN_PASSWORD`.
Start the application:
```powershell
streamlit run app.py
```
Open the local URL shown by Streamlit (normally http://localhost:8501).
If using Python 3.13, try `py -3.13 -m venv .venv`. If a dependency does not have a compatible wheel, install Python 3.11 or 3.12 and recreate the environment.
Tests
Run:
```powershell
python -m unittest discover -s tests -v
```
Tests that need external Gemini or Hugging Face downloads should be mocked; see the local unit tests for FAQ parsing, managers, and grounded generation.
Project structure
```text
TravelMate_Karnataka/
├── app.py
├── document_processing.py
├── notice_manager.py
├── question_manager.py
├── rag_pipeline.py
├── vector_store.py
├── data/
│   ├── travel_faq.txt
│   ├── districts.json
│   ├── notices.json
│   └── submitted_questions.json
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```
District coverage and trip planning
The district guide follows the current 31-district administrative list, including Bengaluru South (the former Ramanagara district; Ramanagara remains its headquarters). District entries describe major attractions and suggested durations, while itinerary FAQs provide sample routes. These are planning suggestions, not live navigation or booking data. Distances, road access, permits, operating hours, weather, ticket prices and transport schedules can change; verify them with official providers before travel.
Important notes
The included data is general educational travel guidance, not a live or official transport/booking feed.
Verify opening times, entry rules, transport schedules, weather, fares, and safety conditions with official providers before travel.
`.env`, vector database files, virtual environments, and secrets are excluded from version control.
Change the default administrator password before sharing or deploying the project.
