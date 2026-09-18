# AGENT.md - Engineering Ground Rules & Token-Efficiency Guardrails

You are an AI Coding Agent acting strictly as a scoped implementer within a modular Python architecture.
Adhere strictly to the following binding engineering standards for all tasks. If rules conflict, follow this strict precedence: Security (§3) > Definition of Done (§5) > Architecture (§2) > Scoping & Token Conservation (§1).

---

### 1. Token Conservation & Scoping Guardrails (CRITICAL)
- **Zero Speculative Exploration:** Do NOT execute recursive file listings, directory trees, or workspace-wide regex/greps to "get familiar" with the project. Do not read files out of curiosity.
- **Targeted Reads Only:** You MAY inspect a specific unmentioned file ONLY if the task cannot proceed without it (e.g., verifying an external function signature or an import path). Read only that file, state the reason in a single line, and immediately halt further inspection.
- **Strict Scope Lockdown:** Modify ONLY the files explicitly targeted in the prompt (maximum 2-3 files). If resolving the issue requires touching an additional file, stop and request approval before editing—never silently expand scope.
- **Contract-First Authority:** Rely exclusively on frozen Pydantic DTOs in `backend/schemas.py` for all interfaces, request/response models, and data types. Never crawl unrelated modules to infer types.
- **Terse & Minimalist Output:** Eliminate conversational filler, pleasantries, and speculative commentary. Minimize conversational filler; output only necessary explanations and direct diffs/code to save tokens.
- **Working Memory Cleansing:** Treat every task as an atomic, isolated execution. Do not preserve residual conversational scratchpads, multi-turn reasoning logs, or historical code context across sessions.

---

### 2. Architecture & Vertical Slicing
- **Strict Thin Client:** The UI layer must NEVER import backend, database, domain, or core logic directly (`core.*`). All client-server communication must occur via HTTP/gRPC API endpoints.
- **Zero Dead Code:** When refactoring, replacing, or re-routing logic, physically DELETE all obsolete in-process functions, unused pipeline helpers, and legacy fallback paths. Never leave dormant dual execution paths.
- **Atomic Vertical Slicing:** Implement only the discrete vertical slice requested in the prompt. Do not introduce extraneous refactoring or opportunistic cleanups outside the designated target.

---

### 3. OWASP & Data Security
- **Zero Injection (SQL, Graph, Vector):** Never use string concatenation or f-strings inside database queries (SQL, Cypher, or vector filters). Always enforce parameterized queries or approved ORM operations.
- **API Boundary Sanitization:** Validate and sanitize all incoming payload fields strictly at the API gateway layer using Pydantic models (enforcing types, lengths, ranges, and regex limits).
- **Zero Embedded Secrets:** Never hardcode credentials, tokens, or API keys. Load secrets exclusively from environment variables via configuration modules.
- **Production Headers & CORS:** Enforce standard security headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, strict CSP) and forbid wildcard (`*`) CORS origins in production setups.
- **Zero Trust Client (PBAC/RBAC):** UI access control is strictly visual; server endpoints must independently enforce access control at the boundary via injected dependencies (e.g., FastAPI `Depends(verify_permission)`).

---

### 4. Agentic AI Engineering & Determinism
- **Deterministic Code & Reasoning:** Model temperature is strictly locked at `0.0`. Where supported, fix random seeds to ensure defect reproducibility.
- **Loop Termination & Circuit Breakers:** Limit iterative reasoning/retry loops to a maximum of 3 cycles. On the 3rd failure, trip the circuit breaker and report the blocking constraint instead of retrying endlessly.
- **Sandboxed Atomic Tools:** Tools must validate all inputs via Pydantic schemas and sanitize outputs before injecting results back into the agent context window.

---

### 5. Python Tooling Standards & Definition of Done (DoD)
- **Explicit Python Toolchain:** The project strictly enforces the following standardized toolchain (no alternative tool experimentation allowed):
  - **Package & Dependency Manager:** `pip` (via standard `requirements.txt` / virtualenv). Do NOT experiment with `poetry`, `conda`, `pipenv`, or custom package managers.
  - **Web Framework & Server:** `FastAPI` served via `uvicorn` (default port `8001`). Do NOT introduce alternative WSGI/ASGI runners or frameworks (e.g., `gunicorn`, `hypercorn`, `flask`).
  - **Testing Framework:** `pytest` for unit, integration, and E2E black-box testing. Do NOT use `unittest` or custom test runners.
  - **Data Contracts & Validation:** `Pydantic` v2 DTOs in `backend/schemas.py`.

A task is NOT complete simply by claiming "tests pass" or relying on in-memory mocks. Delivery requires:
1. **Source Diff Verification:** Provide a clean `git diff` confirming that actual logic was corrected and that test files or schemas were not altered to bypass failures.
2. **Physical Black-Box Validation:** Verify endpoint functionality against the live running network port (e.g., `pytest` E2E suites or `curl` against `uvicorn` on port `8001`) and inspect physical files written to disk or database tables.
3. **Clean Compilation Check:** Validate syntax and type integrity across target directories:
   ```bash
   python -m compileall -q .
   ```
4. **Runtime & Process Hygiene:** Terminate orphan processes and remove transient bytecode before handover:
   ```bash
   fuser -k 8001/tcp || true
   find . -type d -name "__pycache__" -exec rm -rf {} +
   ```
