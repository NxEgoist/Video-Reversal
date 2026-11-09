# Project structure and behavior specification

This document records the structure and externally visible behavior verified in the current repository. `PROJECT_SPEC.md` remains the contributor's investigation checklist; this file is the concise architecture and behavior reference. Update both when implementation changes invalidate their guidance.

## Purpose and current scope

The application is a local video analysis workbench. It stores source videos and task data locally, detects or edits shots, samples visual evidence, and optionally sends selected frames or audio to a configured model provider. Model output is a candidate analysis and prompt that requires human review. The React and FastAPI workbench is under migration; the Gradio interface remains available.

Currently supported workflows include shot analysis, independently analyzed clips, manual clip splitting/merging, evidence previews, prompt editing, audio transcript editing, confirmed cross-shot subject context, case snapshots, and ZIP export. Reconstruction planning produces a plan and does not generate video. Black-screen detection records review candidates and batch-skip state; it does not alter the source or timeline.

## Runtime map

```text
Gradio UI (app.py) ─────────────┐
                                ├── engine.py ── provider_settings.py
React UI (frontend/src) → backend/api.py ─┘       sampling.py
                                                   prompt_sampling.py
                                                   transitions.py
                                                   audio.py
                                                   cases.py
                                                   reconstruction.py

backend/api.py → clips.py → FFmpeg/FFprobe + local frame cache
                         └→ black_frames.py → OpenCV
```

- `engine.py` owns task creation, persistence, shot analysis orchestration, model calls, prompt generation, retry/checkpoint behavior, and export. It currently also contains provider request formatting and schema models.
- `backend/api.py` maps HTTP requests to domain operations and manages background operation state. It currently also owns a process-wide lock and executor, settings endpoints, compatibility projection from shot results to clips, and most route-specific validation.
- `clips.py` owns clip bounds/history, split/merge, frame timestamp indexing, extraction cache, and evidence selection. Time ranges use half-open bounds `[start_ms, end_ms)`.
- `sampling.py`, `prompt_sampling.py`, and `transitions.py` plan sample times; `audio.py`, `cases.py`, `clip_context.py`, `subjects.py`, and `reconstruction.py` own focused supporting workflows.
- `black_frames.py` identifies conservative all-dark shot candidates from sampled decoded frames. It records metadata only; user review and explicit batch-skip controls remain separate.
- `frontend/src` contains React UI surfaces. `App.tsx` coordinates application state and API interactions; `Analysis.tsx` and `Advanced.tsx` provide workflow panels.

## Data and compatibility rules

- Tasks are JSON documents under the local data directory; videos and derived evidence are stored alongside their task. `engine.save` is the persistence boundary. Existing tasks can omit newer optional fields, so readers should supply defaults without rewriting unrelated user data.
- A shot is the source timeline unit. A clip is an independently editable analysis interval within a shot. Model windows and generated-video segments are separate concepts and must not be represented as interchangeable ranges.
- A clip can have its own `analysis`. When its range exactly matches its source shot and it has no independent result, the API presents the shot result as a read-only compatibility projection (`analysis_source: shot`). Editing clip prompts materializes a separate copy; it must not mutate the shot result. Split/merge invalidates clip analysis and retains prior clip state in history.
- Frame evidence is tied to decoded source timestamps. Any change to video identity, clip/shot bounds, or evidence extraction rules must invalidate or verify affected caches and analysis checkpoints.
- User edits, confirmed context, model observations, inferred claims, and unknowns must remain distinguishable. Model self-reported coverage is not a quality score.
- API keys belong in local settings/environment only. They must not enter task JSON, logs, cases, exports, or test fixtures. Network transmission of frames/audio occurs only as part of an explicitly initiated analysis workflow.

## Design constraints and improvement priorities

1. Keep HTTP concerns in `backend/api.py` and move reusable operation/state coordination behind small service functions when API growth makes route behavior hard to follow. Avoid a broad rewrite while the two frontends share the current engine.
2. Keep `engine.py` as orchestration only over time: isolate provider transport/retry logic and persistence behind explicit modules before adding more providers or storage formats. Preserve checkpoint compatibility and scrub secrets from all error paths.
3. Replace duplicated route/UI enum validation with shared typed values or a single validation helper. Sampling strategies currently cross the UI, API, and `sampling.py` as display strings, which makes renames error-prone.
4. Make task mutation boundaries explicit. Functions that return public API data should avoid accidental mutation of loaded task objects; compatibility projections should remain copies.
5. Keep media access behind `clips.py` or a narrow media service so timestamp indexing, extraction, and cache invalidation have one owner.
6. Add feature-specific modules only when they own a coherent policy and have callers/tests at a stable boundary; avoid creating generic utility layers for one-off behavior.

## Verification guide

Run the relevant tests for changed behavior, and the full suite for cross-cutting changes:

```bash
python -m unittest discover -s tests
```

API tests require `requirements-api.txt`; media tests may require FFmpeg/FFprobe and OpenCV. Prefer synthetic media and mocked model calls. Passing local tests does not establish real provider compatibility, model accuracy, or successful external-service behavior.

## Document authority

Code and reproducible tests describe current behavior. `README.md` describes user workflows; `ALGORITHM.md` describes sampling algorithms; `FRONTEND_MIGRATION.md` describes migration scope. Planning documents are proposals, not proof of implementation. Resolve conflicts by tracing the current call path and tests, then update the appropriate documentation.
