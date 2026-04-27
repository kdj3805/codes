# MedChat — Multimodal Healthcare RAG Assistant for Cancer Patients

## Proof of Concept (POC) Document

| **Field**            | **Detail**                                                                 |
|----------------------|----------------------------------------------------------------------------|
| **Project Title**    | MedChat — Multimodal Healthcare RAG Assistant for Cancer Patients          |
| **Document Type**    | Proof of Concept (POC)                                                     |
| **Version**          | 1.0                                                                        |
| **Date**             | March 17, 2026                                                             |
| **Classification**   | Internal — Technical Stakeholders & Leadership                             |
| **Status**           | Active Development                                                         |

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Limitations of Existing Systems](#3-limitations-of-existing-systems)
4. [Proposed Solution](#4-proposed-solution)
5. [System Architecture](#5-system-architecture)
6. [Key Features](#6-key-features)
7. [Tools & Technologies](#7-tools--technologies)
8. [Methodology](#8-methodology)
   - 8.1 [Ingestion](#81-ingestion)
   - 8.2 [Retrieval](#82-retrieval)
   - 8.3 [Generation](#83-generation)
   - 8.4 [Multimodal Processing](#84-multimodal-processing)
   - 8.5 [GraphRAG Extension](#85-graphrag-extension)
9. [Use Cases](#9-use-cases)
10. [Challenges & Solutions](#10-challenges--solutions)
11. [Performance Considerations](#11-performance-considerations)
12. [Scalability & Production Readiness](#12-scalability--production-readiness)
13. [Future Scope](#13-future-scope)
14. [Conclusion](#14-conclusion)
15. [References](#15-references)

---

## 1. Executive Summary

MedChat is a multimodal Retrieval-Augmented Generation (RAG) assistant engineered to support cancer patients in interpreting clinical reports, understanding treatment protocols, and navigating dietary considerations. The system ingests peer-reviewed medical literature, extracts both textual and visual content, and serves context-grounded, citation-backed responses through a conversational interface.

The current implementation comprises a fully operational ingestion pipeline capable of processing heterogeneous PDF formats (scanned, vector, and mixed-layout), a hybrid retrieval system combining dense vector search (Qdrant) with sparse keyword retrieval (BM25) fused via Reciprocal Rank Fusion (RRF) and re-ranked using Maximal Marginal Relevance (MMR), and a generation layer powered by LLaMA 3.3 through the Groq API. The system also incorporates a web search fallback mechanism, a Streamlit-based frontend with inline image rendering, and automated patient report analysis.

A planned extension introduces a GraphRAG architecture backed by Neo4j, enabling structured reasoning over cancer–chemotherapy–nutrition relationships, drug–food interaction detection, and biomarker-driven recommendation capabilities. This extension targets a knowledge graph of 150+ nodes and 200+ relationships, with intelligent query routing across graph, vector, and hybrid retrieval pathways.

This document presents the technical foundation, implemented capabilities, architectural decisions, and forward roadmap for internal stakeholder review.

---

## 2. Problem Statement

Cancer patients routinely face the following challenges when attempting to interpret their clinical information:

1. **Clinical Report Complexity.** Pathology reports, staging documents, and treatment plans are written in dense medical terminology that is inaccessible to most patients without clinical training.

2. **Fragmented Information Landscape.** Relevant medical knowledge—spanning treatment options, prognosis statistics, dietary guidelines, and drug interaction data—is distributed across disparate research papers, clinical databases, and institutional guidelines. Patients lack the tools and expertise to aggregate and cross-reference this information effectively.

3. **Absence of Personalized, Evidence-Based Guidance.** General-purpose search engines and consumer health portals return broad, non-contextualized results that do not account for a patient's specific cancer type, stage, or treatment regimen.

4. **Limited Multimodal Interpretation.** Medical literature frequently relies on diagnostic images, histopathological figures, and statistical charts to convey critical information. Existing patient-facing tools overwhelmingly limit interaction to text, discarding visual context that may be essential for comprehensive understanding.

5. **Trustworthiness and Citation Deficit.** Patients and their caregivers require responses that are not only accurate but also traceable to authoritative sources. Systems that generate responses without explicit source attribution introduce unacceptable risk in a healthcare context.

These factors collectively create a significant gap between the information available in medical literature and the actionable understanding accessible to cancer patients.

---

## 3. Limitations of Existing Systems

| **Limitation**                              | **Description**                                                                                                                                                   |
|---------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Text-Only Retrieval**                     | Most healthcare chatbots and QA systems operate exclusively on textual data, ignoring figures, histopathology images, and diagnostic visuals embedded in literature. |
| **No Source Attribution**                   | Consumer-facing health tools (e.g., symptom checkers, general chatbots) generate responses without citing specific sources, undermining clinical trustworthiness.    |
| **Single Retrieval Strategy**               | Systems relying solely on dense vector retrieval miss keyword-specific matches; those relying solely on keyword search miss semantic equivalences.                   |
| **No Report-Level Analysis**                | Existing tools do not ingest and automatically analyze patient-specific clinical reports against a curated knowledge base.                                           |
| **Static Knowledge Bases**                  | Many systems use fixed FAQ datasets or pre-curated content that cannot be extended with new research without significant re-engineering.                             |
| **No Relational Reasoning**                 | Flat retrieval architectures cannot model or traverse structured relationships, such as drug–drug interactions, drug–food contraindications, or protocol hierarchies.|
| **Generic Response Generation**             | Responses are not tailored to cancer-specific contexts such as staging, biomarker profiles, or treatment-phase-dependent dietary restrictions.                      |

---

## 4. Proposed Solution

MedChat addresses the identified limitations through a vertically integrated, multimodal RAG architecture purpose-built for oncology-related patient support. The solution encompasses:

### 4.1 Current System (Implemented)

- **Multimodal Ingestion Pipeline.** Processes medical research PDFs using Docling as the primary parser with PyMuPDF fallback, handling scanned (OCR), vector, and mixed-layout documents. Extracts text with multi-stage cleaning and normalization, performs header-aware semantic chunking, classifies content by cancer type and content category (clinical, prognosis, statistics), extracts images with multi-layer noise filtering, and grounds images within text via `[IMAGE: filename.png]` tags.

- **Hybrid Retrieval System.** Combines dense retrieval via Qdrant (using BAAI/bge-base-en-v1.5 embeddings [3]) with sparse retrieval via BM25. Retrieval results are fused using Reciprocal Rank Fusion (RRF) and re-ranked with Maximal Marginal Relevance (MMR) to balance relevance and diversity.

- **Context-Grounded Generation.** Employs LLaMA 3.3 via the Groq API to produce responses grounded strictly in retrieved context, with explicit source citations and patient-accessible language.

- **Web Search Fallback.** Activates DuckDuckGo-based web search when the RAG pipeline yields insufficient results, synthesizing external information through the LLM with appropriate safety disclaimers.

- **Automated Patient Report Analysis.** Parses uploaded clinical reports to extract diagnosis, stage, treatment plan, and clinical findings, then cross-references extracted entities against the ingested literature corpus.

- **Interactive Frontend.** A Streamlit-based interface providing chat functionality, PDF upload, inline image rendering, and session-persistent conversation history.

### 4.2 Planned Extension (GraphRAG)

- **Neo4j Knowledge Graph.** Models structured relationships between cancer types, chemotherapy drugs, drug interactions, eating-related side effects, and nutrition guidelines across 150+ nodes and 200+ relationships [1][5].
- **Intelligent Query Routing.** Directs queries to graph-based, vector-based, or hybrid retrieval pathways based on query characteristics.
- **Structured Reasoning.** Supports Cypher-based querying for drug–food–interaction reasoning, protocol-level insights, and biomarker-driven dietary recommendations [2][5].

---

## 5. System Architecture

### 5.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                      │
│                                                                 │
│   ┌──────────────────┐    ┌──────────────────────────────────┐  │
│   │  Text Query       │    │  Patient Report Upload (PDF)     │  │
│   └────────┬─────────┘    └────────────────┬─────────────────┘  │
│            │                               │                    │
│            └───────────┬───────────────────┘                    │
│                        ▼                                        │
│            ┌──────────────────────┐                              │
│            │  Streamlit Frontend  │                              │
│            └──────────┬───────────┘                              │
└───────────────────────┼─────────────────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY PROCESSING LAYER                        │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  Query Analysis & Routing                              │    │
│   │  • Report Detection → Automatic Report Analysis        │    │
│   │  • Query Classification → Retrieval Strategy Selection  │    │
│   └────────────────────────┬───────────────────────────────┘    │
└────────────────────────────┼────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   HYBRID RETRIEVAL SYSTEM                       │
│                                                                 │
│   ┌─────────────────┐  ┌─────────────────┐                     │
│   │ Dense Retrieval  │  │ Sparse Retrieval│                     │
│   │ (Qdrant + BGE)   │  │ (BM25)          │                     │
│   └────────┬────────┘  └────────┬────────┘                     │
│            └────────┬───────────┘                               │
│                     ▼                                           │
│        ┌────────────────────────┐                               │
│        │ Reciprocal Rank Fusion │                               │
│        └───────────┬────────────┘                               │
│                    ▼                                            │
│        ┌────────────────────────┐                               │
│        │ MMR Re-ranking         │                               │
│        └───────────┬────────────┘                               │
└────────────────────┼────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GENERATION LAYER                              │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  Context Builder                                       │    │
│   │  • Chunk Assembly    • Image Tag Injection             │    │
│   │  • Source Mapping    • Context Window Management        │    │
│   └───────────────────────┬────────────────────────────────┘    │
│                           ▼                                     │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  LLM Generation (LLaMA 3.3 via Groq API)              │    │
│   │  • Context-Grounded Response                           │    │
│   │  • Source Citations ([1], [2], etc.)                    │    │
│   │  • Patient-Friendly Language Adaptation                │    │
│   └───────────────────────┬────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FALLBACK LAYER                                │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐    │
│   │  Web Search Fallback (DuckDuckGo)                      │    │
│   │  • Triggered on Low-Confidence RAG Results             │    │
│   │  • LLM Synthesis of External Content                   │    │
│   │  • Safety Disclaimers Appended                         │    │
│   └───────────────────────┬────────────────────────────────┘    │
└───────────────────────────┼─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RESPONSE DELIVERY                             │
│                                                                 │
│   ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│   │ Text Response │  │ Image Rendering│  │ Source Citations  │   │
│   └──────────────┘  └────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Ingestion Pipeline Architecture

```
┌──────────────────────┐
│  Medical Research     │
│  PDFs (Input)         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Docling Parser       │────▶│  PyMuPDF Fallback    │
│  (Primary)            │     │  (On Failure)        │
└──────────┬───────────┘     └──────────┬───────────┘
           └──────────┬────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│              DOCUMENT PROCESSING PIPELINE            │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐ │
│  │ Multi-Stage Text  │  │ Image Extraction          │ │
│  │ Cleaning &        │  │ • Multi-Layer Noise       │ │
│  │ Normalization     │  │   Filtering               │ │
│  └────────┬─────────┘  │ • Caption Extraction       │ │
│           ▼             │   with Fallback            │ │
│  ┌──────────────────┐  │ • Image Grounding          │ │
│  │ Header-Aware      │  │   via [IMAGE] Tags        │ │
│  │ Semantic Chunking │  └──────────┬───────────────┘ │
│  └────────┬─────────┘             │                  │
│           ▼                       ▼                  │
│  ┌──────────────────────────────────────────────┐    │
│  │ Classification Layer                         │    │
│  │ • Cancer Type Classification                 │    │
│  │ • Content Type (Clinical / Prognosis / Stats)│    │
│  └──────────────────────┬───────────────────────┘    │
└─────────────────────────┼────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────┐
│              EMBEDDING & STORAGE                     │
│                                                      │
│  ┌────────────────────┐  ┌────────────────────────┐ │
│  │ Text Embeddings     │  │ Image Caption          │ │
│  │ (BGE-base-en-v1.5)  │  │ Embeddings             │ │
│  └─────────┬──────────┘  └──────────┬─────────────┘ │
│            └──────────┬─────────────┘                │
│                       ▼                              │
│  ┌────────────────────────────────────────────────┐  │
│  │ Qdrant Vector Database                         │  │
│  │ • Chunk Content & Metadata                     │  │
│  │ • Section Hierarchy                            │  │
│  │ • Cancer Type Tags                             │  │
│  │ • Source References                            │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 5.3 GraphRAG Extension Architecture (Planned)

```
┌──────────────────────────────────────────────────────┐
│                NEO4J KNOWLEDGE GRAPH                  │
│                                                       │
│  Nodes (150+):                                        │
│  • Cancer Types    • Chemotherapy Drugs               │
│  • Side Effects    • Nutrition Guidelines             │
│  • Biomarkers      • Food Items                       │
│                                                       │
│  Relationships (200+):                                │
│  • TREATS          • INTERACTS_WITH                   │
│  • CAUSES_SIDE_EFFECT  • RECOMMENDED_FOR              │
│  • CONTRAINDICATED_WITH • MONITORED_BY                │
└───────────────────────┬──────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────┐
│              QUERY ROUTER                             │
│                                                       │
│  ┌─────────┐   ┌──────────┐   ┌────────────────┐    │
│  │ Graph    │   │ Vector   │   │ Hybrid         │    │
│  │ Pathway  │   │ Pathway  │   │ (Graph+Vector) │    │
│  └─────────┘   └──────────┘   └────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## 6. Key Features

### 6.1 Implemented Features

| **Feature**                        | **Description**                                                                                                    |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Multimodal PDF Ingestion**       | Processes scanned, vector, and mixed-layout PDFs with OCR support, extracting both text and images.                 |
| **Semantic Chunking**              | Header-aware chunking preserving section hierarchy and document structure.                                          |
| **Cancer-Type Classification**     | Automatic classification of ingested content by cancer type for targeted retrieval.                                 |
| **Content-Type Classification**    | Categorizes chunks as clinical, prognosis, or statistical content.                                                  |
| **Image Extraction & Grounding**   | Extracts images with noise filtering, associates captions, and grounds images in text via `[IMAGE]` tags.           |
| **Dual Embedding Strategy**        | Generates separate embeddings for text content and image captions using BAAI/bge-base-en-v1.5 [3].                 |
| **Hybrid Retrieval (Dense+Sparse)**| Combines Qdrant dense vector search with BM25 sparse retrieval for comprehensive result coverage [4].               |
| **RRF Fusion**                     | Merges ranked results from dense and sparse retrieval using Reciprocal Rank Fusion.                                 |
| **MMR Re-ranking**                 | Applies Maximal Marginal Relevance to balance relevance with diversity in final results.                            |
| **Context-Grounded Generation**    | LLaMA 3.3 generates responses strictly grounded in retrieved context with explicit citations.                       |
| **Patient Report Analysis**        | Automatically parses uploaded clinical reports, extracting diagnosis, stage, and treatment plan.                     |
| **Literature Cross-Referencing**   | Cross-references report findings against the ingested corpus for evidence-based context.                            |
| **Web Search Fallback**            | DuckDuckGo-based external search with LLM synthesis when internal retrieval is insufficient.                        |
| **Safety Disclaimers**             | Automatically appends medical safety disclaimers to generated responses.                                            |
| **Session-Based Chat History**     | Maintains conversation context within user sessions for coherent multi-turn interactions.                           |

### 6.2 Planned Features (GraphRAG Extension)

| **Feature**                        | **Description**                                                                                                    |
|------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| **Knowledge Graph Construction**   | Neo4j-based graph modeling cancer–drug–nutrition relationships [1][5].                                              |
| **Intelligent Query Routing**      | Automatic routing to graph, vector, or hybrid retrieval pathways based on query semantics.                          |
| **Cypher-Based Querying**          | Structured graph queries for drug interactions, contraindications, and protocol relationships [2][5].               |
| **Drug–Food Interaction Reasoning**| Traversal-based reasoning over drug–food interaction pathways in the knowledge graph.                               |
| **Biomarker-Driven Recommendations**| Personalized recommendations based on biomarker profiles and graph-encoded clinical protocols.                      |

---

## 7. Tools & Technologies

### 7.1 Current Stack

| **Category**          | **Technology**                             | **Purpose**                                              |
|-----------------------|--------------------------------------------|----------------------------------------------------------|
| **Document Parsing**  | Docling                                    | Primary PDF parser with layout analysis                  |
| **Document Parsing**  | PyMuPDF                                    | Fallback PDF parser                                      |
| **OCR**               | Integrated via Docling                     | Scanned PDF text extraction                              |
| **Embedding Model**   | BAAI/bge-base-en-v1.5                      | Text and caption embedding generation [3]                |
| **Vector Database**   | Qdrant                                     | Dense vector storage and similarity search [4]           |
| **Sparse Retrieval**  | BM25 (rank-bm25)                           | Keyword-based sparse retrieval                           |
| **LLM**              | LLaMA 3.3 (via Groq API)                  | Response generation                                      |
| **Web Search**        | DuckDuckGo (duckduckgo-search)             | Fallback external search                                 |
| **Frontend**          | Streamlit                                  | Web-based user interface                                 |
| **Language**          | Python 3.x                                 | Primary implementation language                          |

### 7.2 Planned Stack (GraphRAG Extension)

| **Category**          | **Technology**                             | **Purpose**                                              |
|-----------------------|--------------------------------------------|----------------------------------------------------------|
| **Graph Database**    | Neo4j                                      | Knowledge graph storage and traversal [5]                |
| **Query Language**    | Cypher                                     | Structured graph querying [5]                            |
| **Integration**       | Neo4j Python Driver                        | Graph database connectivity                              |
| **Vector Store**      | Qdrant / FAISS                             | Hybrid retrieval vector component [2]                    |

---

## 8. Methodology

### 8.1 Ingestion

The ingestion pipeline is designed to handle the heterogeneity of medical research PDFs, which vary significantly in format, layout complexity, and content encoding.

**Document Parsing.** Docling serves as the primary parsing engine due to its superior layout analysis capabilities, including table detection, multi-column handling, and figure identification. PyMuPDF is invoked as a fallback when Docling encounters parsing failures—a pattern that ensures robustness across the spectrum of PDF formats encountered in medical literature (scanned documents requiring OCR, vector PDFs with embedded fonts, and mixed-layout documents combining both).

**Text Processing.** Extracted text undergoes a multi-stage cleaning and normalization pipeline that addresses common artifacts in medical PDFs: ligature expansion, encoding normalization, whitespace regularization, header/footer removal, and reference artifact cleanup. This pipeline is specifically tuned for medical text patterns, preserving clinically significant formatting (e.g., measurement units, dosage notations) while removing structural noise.

**Semantic Chunking.** The system employs a header-aware chunking strategy that respects the logical structure of medical papers. Rather than applying fixed-size windowing, chunks are delineated at section boundaries identified through header detection, preserving the semantic coherence of clinical discussions, methodology descriptions, and results sections. Each chunk retains its position within the document's section hierarchy as metadata.

**Classification.** Two classification layers operate on each chunk:
- **Cancer-type classification** tags content with the specific cancer type(s) discussed, enabling cancer-specific retrieval filtering.
- **Content-type classification** categorizes chunks as clinical (treatment protocols, diagnostic criteria), prognostic (survival rates, outcomes data), or statistical (study metrics, population data), supporting content-type-aware retrieval strategies.

**Image Processing.** Images are extracted from PDFs and subjected to a multi-layer noise filtering pipeline that removes decorative elements, blank images, and artifacts below quality thresholds. Captions are extracted using a primary strategy with fallback mechanisms for documents with non-standard caption formatting. Extracted images are grounded within the text through `[IMAGE: filename.png]` tags, enabling the generation layer to include relevant visuals in responses.

### 8.2 Retrieval

The retrieval system implements a hybrid architecture that addresses the complementary strengths and weaknesses of dense and sparse retrieval methods.

**Dense Retrieval.** User queries are embedded using the same BAAI/bge-base-en-v1.5 model [3] used during ingestion, ensuring embedding space consistency. The resulting query vector is searched against the Qdrant vector database [4], which returns the top-k semantically similar chunks based on cosine similarity. This retrieval mode excels at capturing semantic equivalences (e.g., matching "tumor removal" to "surgical excision") that keyword-based methods miss.

**Sparse Retrieval.** Concurrently, the BM25 algorithm performs keyword-based retrieval over the indexed chunk corpus. This retrieval mode ensures that queries containing specific medical terms, drug names, or clinical codes return exact matches that dense retrieval might rank lower due to embedding space generalization.

**Reciprocal Rank Fusion (RRF).** Results from dense and sparse retrieval are combined using RRF, which merges ranked lists by computing a fused score based on the reciprocal of each document's rank in each individual list. This approach avoids the score normalization challenges inherent in direct score combination and produces a balanced ranking that benefits from both retrieval modalities.

**Maximal Marginal Relevance (MMR) Re-ranking.** The fused result set is re-ranked using MMR, which iteratively selects documents that are both relevant to the query and diverse relative to already-selected documents. In the medical context, this ensures that responses draw from multiple perspectives—e.g., treatment options, side effect profiles, and dietary guidelines—rather than redundantly retrieving semantically overlapping chunks from the same source section.

### 8.3 Generation

The generation layer transforms retrieved context into patient-accessible, citation-backed responses.

**Context Assembly.** Retrieved chunks are assembled into a structured context window, ordered by relevance score. Each chunk carries its source metadata (document title, section, page number) for citation mapping. Image tags embedded in chunks are preserved to enable multimodal response generation.

**LLM Inference.** The assembled context is passed to LLaMA 3.3 via the Groq API with a system prompt engineered for:
- **Groundedness:** The model is constrained to generate responses based solely on the provided context, mitigating hallucination risk.
- **Citation:** Source references are formatted as inline citations ([1], [2], etc.) that map to the specific documents from which information was derived.
- **Patient Accessibility:** Medical terminology is accompanied by patient-friendly explanations, and response structure follows a progressive disclosure pattern (summary → detail → sources).

**Safety Mechanisms.** All generated responses include appropriate medical disclaimers advising patients to consult healthcare professionals. The web search fallback path applies additional safety disclaimers when information is sourced externally rather than from the curated literature corpus.

### 8.4 Multimodal Processing

Multimodal capability is a core architectural principle, not an afterthought.

**Image Ingestion.** During PDF processing, the extraction pipeline identifies and isolates figures, diagrams, histopathological images, and statistical charts. A multi-layer filtering pipeline eliminates noise: images below minimum resolution thresholds, decorative elements (logos, borders), and blank or near-blank captures are automatically discarded.

**Caption Association.** Image captions are extracted using spatial proximity analysis (identifying text immediately below or above figure boundaries) with fallback strategies for documents where captions are positioned non-standardly or where figure numbering conventions differ. Extracted captions are embedded using the same BGE model, creating a parallel semantic index for image content.

**Image Grounding.** Each extracted image is assigned a unique identifier and inserted into the corresponding text chunk as an `[IMAGE: filename.png]` tag at its approximate source location. This grounding mechanism allows the LLM to reference specific images in its responses, and the frontend to render them inline.

**Dual Embedding.** Text content and image captions are embedded separately, enabling retrieval to surface relevant images even when the query matches the caption's semantic content rather than the surrounding text. This dual-embedding approach ensures that visual content is a first-class retrieval target.

### 8.5 GraphRAG Extension

> **Note:** The GraphRAG extension described in this section is a planned enhancement. It is not part of the current implementation.

The GraphRAG extension introduces structured relational reasoning capabilities that complement the existing flat retrieval architecture [1][2].

**Knowledge Graph Construction.** A Neo4j-based knowledge graph will encode structured relationships across oncology domains [5]:
- **Cancer Types** linked to applicable chemotherapy protocols, staging criteria, and biomarker profiles.
- **Chemotherapy Drugs** connected to known side effects, drug–drug interactions, and administration protocols.
- **Nutritional Guidelines** mapped to specific cancer types, treatment phases, and drug interactions.
- **Food Items** linked to drug interaction profiles (contraindications, absorption effects).

The target graph encompasses 150+ nodes and 200+ relationships, constructed from curated medical ontologies and verified clinical guidelines.

**Query Routing.** An intelligent query router will classify incoming queries and direct them to the optimal retrieval pathway:
- **Graph Pathway:** For relationship queries (e.g., "What foods should I avoid while taking cisplatin?") where Cypher traversals over the knowledge graph yield precise, structured answers.
- **Vector Pathway:** For open-ended clinical queries (e.g., "What are the latest treatment approaches for stage III breast cancer?") where dense retrieval over the literature corpus is more appropriate.
- **Hybrid Pathway:** For queries requiring both structured reasoning and literature-backed evidence (e.g., "Explain the dietary restrictions during carboplatin treatment and the evidence behind them").

**Cypher-Based Reasoning.** The graph layer supports structured queries expressed in Neo4j's Cypher query language [5], enabling:
- Multi-hop drug interaction detection (Drug A → Interaction → Drug B → Side Effect).
- Protocol-level treatment pathway querying.
- Biomarker-driven recommendation generation based on graph-encoded clinical evidence.

---

## 9. Use Cases

### Use Case 1: Clinical Report Interpretation

| **Field**       | **Detail**                                                                                                      |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Actor**       | Cancer patient                                                                                                  |
| **Trigger**     | Patient uploads a pathology report PDF                                                                          |
| **Process**     | The system parses the report, extracts diagnosis, stage, treatment plan, and clinical findings via automated analysis. Extracted entities are cross-referenced against the ingested literature corpus. The system generates a patient-friendly summary with citations to relevant research. |
| **Output**      | Structured summary covering diagnosis explanation, staging interpretation, treatment overview, and links to supporting literature. |

### Use Case 2: Treatment Query

| **Field**       | **Detail**                                                                                                      |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Actor**       | Cancer patient                                                                                                  |
| **Trigger**     | Patient submits a natural language query, e.g., "What are the side effects of doxorubicin for breast cancer?"   |
| **Process**     | Hybrid retrieval surfaces relevant chunks from oncology literature. RRF fusion and MMR re-ranking produce a diverse, relevant result set. The LLM generates a context-grounded response with citations and any relevant extracted images. |
| **Output**      | Cited, patient-friendly explanation of treatment side effects, with relevant figures rendered inline.            |

### Use Case 3: Dietary Guidance During Treatment

| **Field**       | **Detail**                                                                                                      |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Actor**       | Cancer patient undergoing chemotherapy                                                                          |
| **Trigger**     | Patient asks, "What should I eat during my second cycle of chemotherapy?"                                        |
| **Process**     | *Current system:* Retrieves dietary guidance from ingested literature. *Future (GraphRAG):* Routes to graph pathway, traverses drug–food interaction nodes, identifies contraindications, and generates a dietary plan grounded in graph-encoded nutritional guidelines. |
| **Output**      | Evidence-based dietary recommendations, with drug–food interaction warnings (GraphRAG) and citations.           |

### Use Case 4: Drug Interaction Inquiry (GraphRAG — Planned)

| **Field**       | **Detail**                                                                                                      |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Actor**       | Cancer patient or healthcare professional                                                                        |
| **Trigger**     | Query: "Does methotrexate interact with grapefruit?"                                                            |
| **Process**     | Query router directs to graph pathway. Cypher query traverses drug–food interaction edges. Results are augmented with literature evidence from the vector pathway if available. |
| **Output**      | Structured interaction report with severity classification, mechanism explanation, and supporting references.     |

### Use Case 5: Multi-Turn Conversational Support

| **Field**       | **Detail**                                                                                                      |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **Actor**       | Cancer patient                                                                                                  |
| **Trigger**     | Patient engages in a multi-turn conversation, progressively narrowing from general cancer information to specific treatment options. |
| **Process**     | Session-based chat history maintains conversational context. Each subsequent query benefits from accumulated context, enabling progressively more specific retrieval and response generation. |
| **Output**      | Coherent, contextually aware responses across multiple interaction turns.                                        |

---

## 10. Challenges & Solutions

| **Challenge**                                          | **Solution Implemented**                                                                                                               |
|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| **Heterogeneous PDF formats**                          | Dual-parser architecture (Docling + PyMuPDF fallback) with format-specific preprocessing pathways for scanned, vector, and mixed-layout documents. |
| **Medical text normalization**                         | Multi-stage cleaning pipeline tuned for medical text artifacts: ligature expansion, encoding normalization, and clinical notation preservation. |
| **Image noise in extracted figures**                   | Multi-layer filtering pipeline applying resolution thresholds, blank detection, and decorative element removal.                          |
| **Caption extraction inconsistency**                   | Primary extraction with spatial proximity analysis, supplemented by fallback strategies for non-standard caption formats.                |
| **Balancing precision and recall in retrieval**        | Hybrid dense+sparse retrieval with RRF fusion provides complementary coverage; MMR re-ranking ensures result diversity.                  |
| **LLM hallucination risk**                             | Strict context-grounding in the system prompt constrains the LLM to retrieved content. Source citations provide verifiability.           |
| **Insufficient internal knowledge coverage**           | Web search fallback with DuckDuckGo provides external context, with explicit safety disclaimers differentiating external from internal sources. |
| **Medical terminology accessibility**                  | System prompt directs the LLM to accompany medical terms with patient-friendly explanations.                                             |
| **Multimodal response coherence**                      | Image grounding via `[IMAGE]` tags ensures visual content is contextually integrated rather than randomly appended.                       |
| **Embedding model selection for medical domain**       | BAAI/bge-base-en-v1.5 selected for its strong general-domain performance and manageable resource footprint [3]; domain-specific fine-tuning identified as a future optimization. |

---

## 11. Performance Considerations

### 11.1 Ingestion Performance

| **Metric**                    | **Consideration**                                                                                                          |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **PDF Processing Throughput** | Docling provides higher accuracy but lower throughput than PyMuPDF. The fallback architecture ensures processing completes even for problematic documents, at the cost of reduced structural fidelity in fallback cases. |
| **Image Extraction Overhead** | Multi-layer noise filtering adds processing time per document but significantly reduces storage waste and downstream retrieval noise from irrelevant images. |
| **Embedding Generation**      | BAAI/bge-base-en-v1.5 provides a balance between embedding quality and inference speed. Batch embedding is used during ingestion to maximize GPU utilization. |

### 11.2 Retrieval Performance

| **Metric**                    | **Consideration**                                                                                                          |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Query Latency**             | Dense retrieval via Qdrant operates in sub-second latency for the current corpus scale. BM25 sparse retrieval adds marginal latency. RRF and MMR post-processing are computationally lightweight. |
| **Retrieval Quality**         | Hybrid retrieval consistently outperforms single-modality retrieval in internal evaluations, particularly for queries mixing specific medical terms with general-language descriptions. |
| **Scalability**               | Qdrant supports horizontal scaling via sharding. BM25 indices can be partitioned by cancer type for sub-linear scaling with corpus growth. |

### 11.3 Generation Performance

| **Metric**                    | **Consideration**                                                                                                          |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| **Inference Latency**         | Groq API provides low-latency LLaMA 3.3 inference via custom hardware (LPU). Typical response generation completes within 2–5 seconds. |
| **Context Window Utilization**| Context assembly manages chunk selection to maximize information density within the LLM's context window, prioritizing higher-relevance chunks. |
| **Response Quality**          | Context grounding and citation requirements in the system prompt trade marginal creativity for significant improvements in factual accuracy and verifiability. |

---

## 12. Scalability & Production Readiness

### 12.1 Current State Assessment

| **Dimension**              | **Current State**                                             | **Production Requirement**                                           |
|----------------------------|---------------------------------------------------------------|----------------------------------------------------------------------|
| **Deployment**             | Single-instance Streamlit application                         | Containerized deployment (Docker) behind a load balancer              |
| **Vector Database**        | Single Qdrant instance                                        | Qdrant cluster with replication and sharding                          |
| **LLM Access**             | Direct Groq API calls                                         | API gateway with rate limiting, retry logic, and fallback providers   |
| **Authentication**         | None (prototype)                                              | Role-based access control (RBAC)                                     |
| **Monitoring**             | Application-level logging                                     | Centralized logging, metrics dashboards, and alerting                 |
| **Data Privacy**           | Local processing                                              | HIPAA-compliant data handling, encryption at rest and in transit       |

### 12.2 Scaling Strategy

**Horizontal Scaling Path:**
1. **Frontend:** Migrate from Streamlit to a production web framework (FastAPI backend + React frontend) to support concurrent users.
2. **Retrieval:** Scale Qdrant horizontally via collection sharding, partitioned by cancer type or content category.
3. **Ingestion:** Parallelize PDF processing using task queues (e.g., Celery) to support batch ingestion of large document sets.
4. **Graph Database (future):** Neo4j supports causal clustering for read scaling and high availability [5].

### 12.3 Compliance Considerations

Given the healthcare domain, production deployment must address:
- **HIPAA Compliance:** Patient report data must be encrypted at rest and in transit, with audit logging for all access events.
- **Medical Disclaimer Framework:** Response generation must consistently include appropriate medical disclaimers, which the current system already implements.
- **Data Retention Policies:** Patient-uploaded reports must be subject to configurable retention and deletion policies.

---

## 13. Future Scope

| **Enhancement**                         | **Description**                                                                                                                       | **Priority** |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|--------------|
| **GraphRAG Integration**               | Deploy the Neo4j knowledge graph with query routing, Cypher-based reasoning, and drug–food interaction detection [1][2][5].           | High         |
| **Domain-Specific Embedding Fine-Tuning** | Fine-tune BAAI/bge-base-en-v1.5 on oncology-specific corpora to improve retrieval precision for medical terminology [3].              | High         |
| **Multi-Language Support**             | Extend ingestion and generation pipelines to support non-English medical literature and patient queries.                               | Medium       |
| **Voice Interface**                     | Add speech-to-text input and text-to-speech output for accessibility, particularly for patients with limited mobility.                 | Medium       |
| **Clinical Trial Matching**            | Integrate clinical trial databases to surface relevant active trials based on patient diagnosis and biomarker profiles.                 | Medium       |
| **Continuous Ingestion Pipeline**       | Implement automated monitoring of medical literature databases (e.g., PubMed) for new publications, with automatic ingestion.          | Medium       |
| **Feedback Loop & Model Improvement**   | Capture clinician and patient feedback on response quality to drive retrieval parameter tuning and prompt optimization.                 | Medium       |
| **Mobile Application**                  | Develop a mobile-native interface for improved accessibility.                                                                         | Low          |
| **EHR Integration**                     | Integrate with Electronic Health Record systems for automated report ingestion and context enrichment.                                 | Low          |

---

## 14. Conclusion

MedChat demonstrates a technically viable, multimodal RAG architecture engineered for the specific demands of oncology patient support. The current implementation establishes a robust foundation across all critical system layers:

- A resilient ingestion pipeline capable of handling the full spectrum of medical PDF formats with text, image, and metadata extraction.
- A hybrid retrieval system that materially outperforms single-modality approaches through the combination of dense and sparse retrieval with RRF fusion and MMR re-ranking.
- A generation layer that prioritizes factual grounding, source traceability, and patient accessibility.
- A functional multimodal pipeline that integrates visual medical content as a first-class output alongside text.

The planned GraphRAG extension will introduce structured relational reasoning—a capability absent from flat retrieval architectures—enabling the system to answer complex queries involving drug interactions, treatment protocol relationships, and biomarker-driven dietary recommendations through knowledge graph traversal.

This proof of concept validates the core technical approach and provides a clear path to production deployment, contingent on the scaling, compliance, and infrastructure enhancements outlined in Section 12. The system is positioned for iterative enhancement through the future scope items detailed in Section 13, with GraphRAG integration and domain-specific embedding fine-tuning identified as the highest-priority next steps.

---

## 15. References

[1] Internal Documentation, "GraphRAG: Cancer–Chemo–Nutrition Knowledge Graph using Neo4j and Vector Databases," Company Technical Resources, 2026.

[2] Internal Implementation Guide, "GraphRAG Integration Pipeline: Neo4j, Qdrant/FAISS, and LLM-based Retrieval," Company Engineering Documentation, 2026.

[3] J. Johnson *et al.*, "BGE: General Embedding Models for Text Retrieval," *arXiv preprint*, 2023.

[4] Qdrant Team, "Qdrant Vector Database Documentation," 2024. [Online]. Available: https://qdrant.tech/documentation/

[5] Neo4j Inc., "Neo4j Graph Database and Cypher Query Language Documentation," 2024. [Online]. Available: https://neo4j.com/docs/

---

*Document generated: March 17, 2026*
*Classification: Internal — For Technical Stakeholders & Leadership Review*
