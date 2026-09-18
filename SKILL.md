AI Agent Development Guidelines & Code Generation Rules (SKILL.md)
This document establishes the strict rules, architectural standards, and environment constraints for any AI coding assistant or developer contributing code to the Pajoheshyar (پژوهشیار) repository.
1. Environment & Runtime Rules
Virtual Environment Constraint:
All code MUST execute within a dedicated Python virtual environment (venv).
Never assume global package availability.
Dependencies must strictly adhere to requirements.txt.
Framework & UI Standard:
The user interface is built strictly using Streamlit.
UI logic must be separated into views (ui/views/) and components (ui/components/). Never put database or simulation logic directly inside Streamlit render functions.
No-Crash Fallback Guarantee (Zero-External-AI Dependence):
The application MUST NEVER CRASH if an LLM API or local LLM server (e.g., Ollama) is unavailable.
Every agent interaction must route through core/llm_provider.py.
When LLM is disabled or fails, MockLLMProvider MUST return deterministic, rule-based structured JSON/Text responses.
2. Code Architecture & Design Principles
Directory Blueprint
pajoheshyar-core/
├── config/             # YAML configurations
├── core/               # Pure Python business logic, statistics, DB, parser
├── agents/             # Sub-agents and Main Domain Agents
├── ui/                 # Streamlit UI layers
├── tests/              # Pytest test suite
├── scripts/            # CLI test pipelines
└── data/               # Local SQLite & file storage

Module Responsibilities
core/monte_carlo.py: Pure statistical module (numpy, scipy.stats). No UI or LLM logic.
core/database.py: SQLAlchemy ORM models and SQLite session handlers.
core/parser.py: pdfplumber / PyPDF2 extraction for plain text, tables, and figure captions.
core/rag_engine.py: Keyword/BM25 local search fallback + FAISS embedding retriever.
agents/: Must accept standard dict/dataclass inputs and return typed outputs or Pydantic models.
3. Code Quality & Testing Rules
Testability First (VS Code Integration):
Every module in core/ and agents/ MUST be independently testable using pytest from VS Code terminal without launching Streamlit.
Keep functions side-effect-free where possible.
Typing & Documentation:
Use Python type hints (typing.Dict, typing.List, typing.Optional, pydantic.BaseModel) for all function signatures.
Docstrings must specify inputs, outputs, and mathematical formulas used (if applicable).
Streamlit Session State Discipline:
Manage all application state using a centralized state manager (ui/state_manager.py).
Do not re-query the database or re-run Monte Carlo simulations on every UI rerun unless explicitly triggered by a button or parameter change.
4. Prompting Templates for AI Code Generation
When asking an AI model to write a new module, use this prompt structure:
"Write the implementation for [module_path]. Adhere strictly to SKILL.md. Ensure zero external LLM dependence by using MockLLMProvider fallback, add full typing, and provide a corresponding pytest unit test in tests/."
