# 🏥 MedChat — Personalised Cancer RAG Assistant

> An AI-powered chatbot that answers clinical questions about cancer by searching peer-reviewed medical literature **and** a structured oncology knowledge graph — all in plain language.

---

## 📋 Table of Contents

- [What Does This Project Do?](#-what-does-this-project-do)
- [How It Works (Plain English)](#-how-it-works-plain-english)
- [System Architecture](#-system-architecture)
- [Project Files Explained](#-project-files-explained)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration (.env Setup)](#-configuration-env-setup)
- [Running the Project (Step by Step)](#-running-the-project-step-by-step)
- [Using the App](#-using-the-app)
- [Example Questions You Can Ask](#-example-questions-you-can-ask)
- [Troubleshooting](#-troubleshooting)
- [Tech Stack](#-tech-stack)
- [Medical Disclaimer](#-medical-disclaimer)

---

## 🤔 What Does This Project Do?

MedChat is a **Retrieval-Augmented Generation (RAG)** chatbot designed to help cancer patients and clinicians understand medical information quickly and clearly.

You can:
- **Upload a patient report** (PDF or text) and get an instant AI-powered clinical analysis
- **Ask plain-English questions** like *"What foods should I eat while on cisplatin?"* or *"What are the side effects of doxorubicin?"*
- Get answers backed by **peer-reviewed literature** with **inline figures and charts**
- Get **drug interaction warnings** (e.g., "Is it safe to take ibuprofen with methotrexate?")
- Get **personalised dietary advice** based on treatment-related side effects

---

## 🧠 How It Works (Plain English)

Think of MedChat as having two "brains" it searches every time you ask a question:

### Brain 1 — The Literature Library (Vector Database)
Medical PDFs (research papers, clinical guidelines) are converted into a searchable database. When you ask a question, the system finds the most relevant passages using **two search strategies simultaneously**:
- **Dense search** — understands the *meaning* of your question (like Google)
- **Sparse/keyword search (BM25)** — finds exact medical terms (like Ctrl+F)

Both results are merged and re-ranked to give the best 15 passages. Any figures or tables from those papers are displayed inline in the answer.

### Brain 2 — The Knowledge Graph (Neo4j)
A structured medical database of facts like:
- *"Breast Cancer is treated with Paclitaxel"*
- *"Cisplatin causes Nausea"*
- *"Banana helps with Nausea"*
- *"Doxorubicin + Warfarin = severe bleeding risk"*

This is great for **specific, factual lookups**: drug interactions, food recommendations, side effect pathways.

### Putting It Together
Both sources are combined and sent to **Groq's LLaMA 3.3 70B** (a large language model) which writes a clear, structured answer in plain English — with a medical disclaimer at the end.

```
Your Question
     │
     ▼
Entity Extraction          ← Identifies drugs, cancers, foods, side effects in your question
     │
     ├──► Vector Search (Qdrant)     ← Searches medical PDFs
     │         Dense + Sparse + RRF + MMR
     │
     └──► Graph Search (Neo4j)       ← Queries structured oncology facts
               Cypher multi-hop queries
                    │
                    ▼
            Context Fusion
                    │
                    ▼
          Groq LLaMA 3.3 70B         ← Generates the final answer
                    │
                    ▼
         Answer + Sources + Inline Images
```

---

## 🗂 System Architecture

```
medchat/
│
├── 📄 cancer_app_v2.py              ← Streamlit web UI (the app you see)
│
├── 🔍 Cancer_retrieval_v2_visual.py ← Core RAG pipeline (search + LLM)
├── 📥 Cancer_ingestion_v2.py        ← Converts PDFs into searchable chunks
│
├── 🕸 graphrag_integration.py       ← Fuses vector + graph contexts
├── 🔎 graph_retrieval.py            ← Neo4j Cypher query layer
├── 🏷 entity_extractor.py           ← Extracts medical terms from queries
│
├── 🗄 neo4j_client.py               ← Neo4j database connection
├── 📐 graph_schema.py               ← Defines the graph's structure
├── 📦 graph_data_loader.py          ← Loads medical facts into Neo4j
├── 🚀 bootstrap.py                  ← One-time setup: schema + data load
│
├── 📁 pdfs/                         ← Put your medical PDFs here
├── 📁 output/
│   ├── chunks/                      ← Processed text chunks (auto-generated)
│   ├── images/                      ← Extracted figures from papers (auto-generated)
│   └── markdown/                    ← Converted markdown from PDFs (auto-generated)
├── 📁 vector_db/
│   └── qdrantt_store/               ← Local vector database (auto-generated)
│
└── .env                             ← Your API keys (you create this)
```

---

## 📁 Project Files Explained

| File | What it does | When you interact with it |
|------|-------------|--------------------------|
| `cancer_app_v2.py` | The Streamlit web UI — chat interface, sidebar for report upload | Every time you run the app |
| `Cancer_retrieval_v2_visual.py` | The search + answer engine — hybrid retrieval, MMR reranking, Groq LLM call | Runs automatically when you ask questions |
| `Cancer_ingestion_v2.py` | Reads PDFs, extracts text/images, creates chunks, embeds and stores in Qdrant | **Run once** when setting up or adding new PDFs |
| `graphrag_integration.py` | Combines vector search results + graph results into one fused context for the LLM | Runs automatically |
| `graph_retrieval.py` | Translates entity queries into Neo4j Cypher graph queries | Runs automatically |
| `entity_extractor.py` | Reads the user's question and identifies medical entities (cancers, drugs, foods, side effects) | Runs automatically |
| `neo4j_client.py` | Handles the connection to the Neo4j database | Runs automatically |
| `graph_schema.py` | Defines all node types, relationship types, indexes in Neo4j | Used by `bootstrap.py` |
| `graph_data_loader.py` | Contains the medical facts data and loads them into Neo4j | Used by `bootstrap.py` |
| `bootstrap.py` | One-shot script: creates the Neo4j schema and loads all graph data | **Run once** when setting up |

---

## ✅ Prerequisites

Before you start, make sure you have:

| Requirement | Version | How to check |
|-------------|---------|--------------|
| Python | 3.10 or 3.11 | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| A **Groq API key** | Free tier available | [console.groq.com](https://console.groq.com) |
| A **Neo4j instance** | Aura Free or local | [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) |
| ~4 GB disk space | For models and vector DB | — |

> 💡 **Neo4j Aura Free** gives you a free cloud database — no local installation needed. Sign up at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/).

> 💡 **Groq** offers a free API tier with access to LLaMA 3.3 70B. Sign up at [console.groq.com](https://console.groq.com).

---

## 🛠 Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/amukherjee-sudo/Multimodal-GraphRAG-MedChat.git 
cd medchat
```

### Step 2 — Create a virtual environment

This keeps the project's dependencies separate from your system Python.

```bash
# Create the virtual environment
python -m venv venv

# Activate it — Windows:
venv\Scripts\activate

# Activate it — Mac/Linux:
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install the core packages manually:

```bash
pip install streamlit groq langchain langchain-huggingface langchain-qdrant \
            langchain-community qdrant-client neo4j certifi python-dotenv \
            pymupdf pillow tqdm rank-bm25 duckduckgo-search sentence-transformers
```

> ⚠️ **Optional but recommended for better PDF parsing:**
> ```bash
> pip install docling easyocr
> ```
> These are optional — the system will fall back to PyMuPDF if they're not installed.

---

## 🔑 Configuration (.env Setup)

Create a file named `.env` in the root of the project folder (the same folder as `cancer_app_v2.py`):

```bash
# On Mac/Linux:
touch .env

# On Windows, just create a new text file named ".env" (no .txt extension)
```

Open the `.env` file and add the following — replacing the placeholder values with your real credentials:

```env
# ── Groq (LLM provider) ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# ── Neo4j (knowledge graph database) ────────────────────────────
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here
```

**Where to find these values:**

- **GROQ_API_KEY** → Log in to [console.groq.com](https://console.groq.com) → API Keys → Create Key
- **NEO4J_URI** → Your Neo4j Aura dashboard → Connect → Connection URI (starts with `neo4j+s://`)
- **NEO4J_USER** → Usually `neo4j` by default
- **NEO4J_PASSWORD** → Set when you created your Aura instance

> ⚠️ **Never commit your `.env` file to Git.** Add it to `.gitignore`:
> ```bash
> echo ".env" >> .gitignore
> ```

---

## 🚀 Running the Project (Step by Step)

There are **three steps** to get the system running the first time. After the initial setup, you only need Step 3.

---

### Step 1 — Add your medical PDFs (First time only)

Create a folder called `pdfs/` in the project root and copy your medical papers or clinical guidelines into it:

```
medchat/
└── pdfs/
    ├── breast_cancer_guidelines_2023.pdf
    ├── osteosarcoma_treatment.pdf
    └── ...
```

> 📌 The more high-quality PDFs you add, the better the answers will be. Good sources: PubMed full-text articles, NCCN guidelines, WHO cancer reports.

---

### Step 2 — Ingest PDFs into the vector database (First time only)

This step reads all your PDFs, extracts text and images, splits them into chunks, creates embeddings, and stores everything in the local Qdrant database.

```bash
python Cancer_ingestion_v2.py
```

**Expected output:**
```
Processing: breast_cancer_guidelines_2023.pdf ...
  Extracted 142 chunks, 23 images
Processing: osteosarcoma_treatment.pdf ...
  Extracted 87 chunks, 11 images
✅ Ingestion complete. 229 total chunks stored in Qdrant.
```

> ⏱️ This can take **5–20 minutes** depending on the number and size of PDFs. It only needs to run once (or when you add new PDFs).

---

### Step 3 — Set up the Neo4j knowledge graph (First time only)

This step creates the database schema (node types, indexes) and loads all the pre-defined medical facts (drugs, side effects, food relationships, drug interactions) into Neo4j.

```bash
python bootstrap.py
```

**Expected output:**
```
=== ENV DEBUG ===
NEO4J_URI: neo4j+s://xxxx.databases.neo4j.io
NEO4J_USER: neo4j
NEO4J_PASSWORD: SET
=================

INFO — ✅ Connected to Neo4j
INFO — Schema bootstrap complete.
INFO — Loading graph data ...
=== NODE COUNTS ===
  Cancer               6
  Drug                 18
  Food                 35
  SideEffect           20
  ...
🚀 Bootstrap complete. GraphRAG ready.
```

> ⚠️ If you see a connection error, double-check your `.env` values and make sure your Neo4j Aura instance is running (it may have paused after inactivity — log in to the Aura console to resume it).

---

### Step 4 — Launch the app 🎉

```bash
streamlit run cancer_app_v2.py
```

Your browser will automatically open to `http://localhost:8501`. If it doesn't, open that URL manually.

---

### Re-running after initial setup

Once the database and vector store are set up, you only ever need:

```bash
# Activate your virtual environment first
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Then launch the app
streamlit run cancer_app_v2.py
```

---

## 💬 Using the App

### Chatting without a patient report
Type any cancer-related question in the chat input at the bottom of the screen. Press Enter or click Send.

### Uploading a patient report
1. Click the **←** arrow (top-left) to open the sidebar
2. Click **"Upload report (.txt or .pdf)"** and select your file — or paste report text in the text area
3. The AI will **automatically analyse the report** against the clinical literature and provide a structured summary covering diagnosis, stage, treatment plan, key findings, and relevant literature

### Reading the answers
- Answers include **inline images** (survival curves, treatment flowcharts, clinical tables) pulled directly from the papers
- Click **"Sources"** at the bottom of any answer to see which papers were used
- Suggested follow-up questions may appear to guide your next query

---

## 💡 Example Questions You Can Ask

**Side effects and symptoms:**
- *"What are the side effects of cisplatin chemotherapy?"*
- *"Why does methotrexate cause mouth sores?"*
- *"What causes hair loss during chemotherapy?"*

**Food and nutrition:**
- *"What foods help with nausea from chemotherapy?"*
- *"What should I avoid eating when I have mouth sores?"*
- *"What is the best diet for a patient with osteosarcoma on MAP protocol?"*

**Drug interactions:**
- *"Can I take ibuprofen while on methotrexate?"*
- *"Is warfarin safe with doxorubicin?"*
- *"What medications interact with cisplatin?"*

**Clinical questions:**
- *"What is the 5-year survival rate for Stage III breast cancer?"*
- *"What topical drug delivery systems are used in breast cancer treatment?"*
- *"What is the difference between NSCLC and SCLC treatment?"*

**With a patient report uploaded:**
- *"What does my diagnosis mean?"*
- *"What are my treatment options given my HER2 status?"*
- *"What side effects should I expect from my regimen?"*

---

## 🔧 Troubleshooting

### ❌ "No chunk files found" error on startup
You haven't run the ingestion step yet, or the `pdfs/` folder is empty.
```bash
# Add PDFs to the pdfs/ folder, then run:
python Cancer_ingestion_v2.py
```

### ❌ Neo4j connection error / AuthError
- Check your `.env` file — all three values (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`) must be correct
- Log in to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/) and check if your instance is **running** (free instances pause after inactivity — click "Resume")
- Make sure your URI starts with `neo4j+s://` (not `bolt://`)

### ❌ "GROQ_API_KEY not set" or Groq authentication error
- Open your `.env` file and verify `GROQ_API_KEY` is set correctly
- Make sure there are no spaces around the `=` sign: `GROQ_API_KEY=sk-...` ✅ not `GROQ_API_KEY = sk-...` ❌

### ❌ Images not showing ("Image not found on disk")
The image was referenced by the LLM but the file is missing from `output/images/`.
```bash
# Re-run ingestion to regenerate extracted images:
python Cancer_ingestion_v2.py
```

### ❌ Very slow first response
The embedding model (`BAAI/bge-base-en-v1.5`) is being downloaded from HuggingFace on first run (~440 MB). This is a one-time download. Subsequent runs will be fast.

### ❌ Streamlit "module not found" errors
Make sure your virtual environment is activated before running:
```bash
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
```

---

## 🧰 Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Groq — LLaMA 3.3 70B | Generates natural language answers |
| **Embeddings** | BAAI/bge-base-en-v1.5 (HuggingFace) | Converts text into searchable vectors |
| **Vector DB** | Qdrant (local) | Stores and searches embedded text chunks |
| **Sparse Retrieval** | BM25 (rank-bm25) | Keyword-based search over chunks |
| **Graph DB** | Neo4j Aura | Stores structured oncology knowledge |
| **PDF Parsing** | PyMuPDF + Docling (optional) | Extracts text and images from PDFs |
| **UI** | Streamlit | Web chat interface |
| **Retrieval Strategy** | Hybrid (Dense + BM25) → RRF → MMR | Multi-stage ranking for best results |

---

## ⚠️ Medical Disclaimer

> **MedChat is an AI research tool intended for informational purposes only.**
>
> It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified oncologist or healthcare provider before making any clinical decisions. The information provided may not be complete, up-to-date, or applicable to every individual patient's situation.
>
> If you are experiencing a medical emergency, contact your doctor or emergency services immediately.

---

## 📄 License

This project is for educational and research purposes. Please ensure any medical PDFs you ingest comply with their respective copyright licenses.

---

*Built with ❤️ for cancer patients and the clinicians who care for them.*
