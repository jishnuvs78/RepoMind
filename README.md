# 📘 AI Repository Assistant (timm / HuggingFace)

## 1. Project Title & Description

**AI Repository Assistant**  
An intelligent assistant that answers questions about GitHub repositories using hybrid search and LLMs.

> Built initially for the pytorch-image-models (timm) repository, but easily extensible to any repository.

---

## 2. Overview

### 🚩 Problem

Large GitHub repositories contain extensive documentation across many markdown files.  
Finding precise answers manually is time-consuming.

### 💡 Solution

This project builds an AI-powered assistant that:
- Downloads repository content dynamically
- Indexes documentation
- Enables semantic + keyword search
- Uses an LLM agent to answer questions with context

### ⚡ Why it’s useful / unique

- Hybrid search (keyword + vector)
- Tool-augmented LLM (pydantic-ai)
- Works on any repo (config-based)
- Streamlit UI + CLI support
- Context-aware answers with references

---

## 3. Installation

### 📋 Requirements

- Python 3.13+
- API Key (OpenAI or Groq)
- uv (recommended)

### ⚙️ Setup

```bash
uv sync
```

### ▶️ Run commands

```bash
uv run python main.py
uv run streamlit run app.py
```

### 🔐 Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
# or
GROQ_API_KEY=your_key_here
```

---

## 4. Usage

### ▶️ CLI

```bash
uv run python main.py
```

### 🌐 Streamlit

```bash
uv run streamlit run app.py
```

### ⚙️ Config

Modify:

```python
REPO_OWNER = "huggingface"
REPO_NAME = "pytorch-image-models"
```

---

## 5. Features

- Hybrid search (keyword + semantic)
- Automatic repo ingestion
- Markdown parsing + chunking
- LLM agent with tools
- CLI + Streamlit UI
- Logging system
- Easily extensible

---

## 6. Tests

Manual testing recommended:

- Ask questions about the repo
- Validate correctness of responses

---

## 7. Deployment

### 🌐 Options

- Streamlit Cloud
- Heroku
- Fly.io
- Render

### 📁 Logs

Logs are stored locally in:

```
logs/
```

Custom path:

```env
LOGS_DIRECTORY=your/custom/path
```

### ⚙️ Production Tips

- Add CI/CD (GitHub Actions)
- External logging storage
- Optimize indexing / vector DB

---

## 8. FAQ / Troubleshooting

### ❓ Missing dependencies

Run:

```bash
uv sync
```

Ensure Python 3.13+ is installed.

---

## 9. Credits

- Hugging Face
- Sentence Transformers
- minsearch
- pydantic-ai
- Streamlit

---

## 💬 Final Note

This project demonstrates how to build tool-augmented LLM systems over real-world repositories.
