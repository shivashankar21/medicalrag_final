
---

# 🏥 Medical RAG System — AI-Powered Medical Question Answering

*A Retrieval-Augmented Generation System using Google Generative AI, ChromaDB & FastAPI*

---

## 📌 Overview

This project is a **Medical RAG (Retrieval Augmented Generation)** system that answers medical questions using verified medical documents (MedQuAD dataset). It uses:

* **Google Generative AI** (text-embedding-004 + Gemini models)
* **ChromaDB** for vector search
* **FastAPI** backend
* **Streamlit UI**

The system retrieves relevant medical snippets, filters them with safety rules, generates an answer, evaluates it, and returns a final safe response.

---

# 🚀 System Functionality Flow

### **1. User asks a medical question**

Via API or Streamlit UI.

---

### **2. The question is embedded**

Using Google’s **text-embedding-004** model:

* Converts text → numerical vector

---

### **3. Retrieve relevant medical documents (snippets)**

Vector DB (ChromaDB) returns **top-k similar documents**.

Example snippet:

```
ID: mq_42
"What is glaucoma?... Glaucoma is an eye disease..."
```

---

### **4. Guardrail Agent checks safety**

* Detects emergency queries (chest pain, suicidal ideation, bleeding)
* Filters irrelevant documents
* Ensures query is safe to answer

Possible outcomes:

* **UNSAFE** → “Seek immediate medical attention”
* **INSUFFICIENT** → Not enough relevant data
* **OK** → Proceed with selected snippets

---

### **5. Generator Agent produces answer**

Uses only the selected snippets (no hallucination allowed).

---

### **6. Evaluator Agent scores the answer**

Checks:

* Does the answer rely only on provided snippets?
* Is it complete?
* Is it consistent?

If score < 0.8 → regenerate with fewer snippets
If score ≥ 0.8 → return final answer

---

### **7. API returns final answer, confidence score, and source IDs**

---

# 📂 Code / Directory Structure

```
project/
│
├── src/
│   ├── agents.py
│   ├── embedder.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── vectorstore.py
│   ├── server.py
│   ├── config.py
│   └── telemetry.py
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   └── medquad.csv
│
└── README.md
```

---

# 🧠 Detailed Code Flow (File-by-file Explanation)

---

# 📌 `src/pipeline.py` — **Main RAG Pipeline**

Handles ingestion + retrieval + guardrail + generation + evaluation.

### **1. Ingest documents into vector DB**

```python
def ingest_contents(contents):
```

* Embeds all medical documents
* Stores them in ChromaDB
* Assigns IDs (`mq_0`, `mq_1`, etc.)

---

### **2. RAG Answer Pipeline**

```python
def rag_answer(query):
```

Steps:

1. Embed user query
2. Retrieve relevant documents → snippets
3. Pass snippets to Guardrail Agent
4. If safe → generate answer
5. Evaluate answer; regenerate if needed
6. Return answer + score + source IDs

This is the heart of the system.

---

# 📌 `src/preprocess.py` — **Load + Clean MedQuAD Dataset**

Loads CSV and prepares each document:

* Detects **question** and **answer** columns
* Creates combined `"content"` = question + answer
* Removes duplicates / empty rows
* Returns a clean dataframe ready for ingestion

---

# 📌 `src/embedder.py` — **Text Embedding**

Uses Google API:

```python
genai.embed_content(model="text-embedding-004")
```

* Batches requests to stay under rate limits
* Returns embeddings for documents or queries

---

# 📌 `src/vectorstore.py` — **ChromaDB Vector Store**

Handles all vector database operations.

### **Key components**

* `create_collection()` → initializes persistent DB
* `add_documents()` → stores embeddings + text + metadata
* `query_embeddings()` → retrieves closest documents

This is where "snippets" come from.

---

# 📌 `src/agents.py` — **Guardrail, Generator & Evaluator Agents**

### **1. Guardrail Agent**

Ensures the system NEVER gives unsafe medical advice.

* Filters relevant snippets
* Detects critical emergencies
* Produces decision:

  * **OK**
  * **UNSAFE**
  * **INSUFFICIENT**

---

### **2. Generator Agent**

Uses only selected snippets to answer the question:

```
"You must ONLY use these snippets..."
```

Prevents hallucinations.

---

### **3. Evaluator Agent**

Scores answer on a scale 0–1:

* Does answer match snippets?
* Is any claim unsupported?
* JSON output: `{score, issues}`

If score < 0.8 → regenerate with fewer snippets.

---

# 📌 `src/server.py` — **FastAPI Backend**

Endpoints:

### **POST `/ask`**

* Accepts user query
* Calls `rag_answer()`
* Logs interaction
* Returns JSON:

```json
{
  "answer": "...",
  "score": 0.92,
  "source_ids": ["mq_42", "mq_51"]
}
```

Runs with:

```
uvicorn src.server:app --reload
```

---

# 📌 `ui/streamlit_app.py` — **Frontend UI**

Simple Streamlit interface:

* Text box for questions
* Sends query to FastAPI
* Displays:

  * Answer
  * Confidence score
  * Source IDs

---

# 📊 Example Response

Input:

```
What are symptoms of glaucoma?
```

Output:

```json
{
  "answer": "Based on sources [mq_42] ...",
  "score": 0.91,
  "source_ids": ["mq_42"]
}
```

---

# ⚙️ Setup Instructions

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Set Google API Key**

Create `.env`:

```
GOOGLE_API_KEY=your_key_here
```

### **3. Start Backend**

```bash
uvicorn src.server:app --reload
```

### **4. Start Streamlit UI**

```bash
streamlit run ui/streamlit_app.py
```

---

# 🛡️ Safety Features

* Emergency detection ("chest pain", "suicidal", "bleeding")
* Answer must be **fully traceable** to retrieved documents
* Evaluator prevents hallucination
* Guardrails block unsafe or irrelevant topics

---

# 📜 License

MIT License

---

# 🙌 Contributing

Pull requests welcome!

---

<img width="1464" height="855" alt="newsletter87-RAG-simple" src="https://github.com/user-attachments/assets/f0596ad2-09e7-4f8c-81a6-ace4d1748971" />
