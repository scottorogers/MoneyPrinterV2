"""
MoneyPrinterV2 — Pain Point Agent Web UI
Run from the project root:
    python src/web_ui.py
Then open http://localhost:5000 in your browser.
"""

import json
import os
import queue
import sys
import threading
import uuid

# Ensure src/ modules are importable
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from classes.PainPointAgent import PainPointAgent, _load_cache, _save_cache
from config import ROOT_DIR
import llm_provider

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["SECRET_KEY"] = "mpv2-pain-point-agent"

# In-memory job registry  {job_id: {"queue": Queue, "result": dict|None, "error": str|None}}
_jobs: dict = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------


@app.route("/api/research/start", methods=["POST"])
def research_start():
    """Kick off a background research job and return a job_id for SSE polling."""
    data = request.get_json(silent=True) or {}
    company_name = (data.get("company_name") or "").strip()
    website_url = (data.get("website_url") or "").strip()

    if not company_name:
        return jsonify({"error": "company_name is required"}), 400

    job_id = str(uuid.uuid4())
    q: queue.Queue = queue.Queue()

    with _jobs_lock:
        _jobs[job_id] = {"queue": q, "result": None, "error": None}

    def run():
        try:
            # Ensure a model is selected (pick first available if none set)
            try:
                if not llm_provider.get_active_model():
                    models = llm_provider.list_models()
                    if models:
                        llm_provider.select_model(models[0])
            except Exception:
                pass

            agent = PainPointAgent()

            def on_progress(message: str, stage: str = "info"):
                q.put({"type": "progress", "stage": stage, "message": message})

            result = agent.research_company(company_name, website_url, on_progress=on_progress)

            # Persist to cache
            cache = _load_cache()
            cache.append(result)
            _save_cache(cache)

            with _jobs_lock:
                _jobs[job_id]["result"] = result

            q.put({"type": "complete", "result": result})

        except Exception as exc:
            err = str(exc)
            with _jobs_lock:
                _jobs[job_id]["error"] = err
            q.put({"type": "error", "message": err})

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/research/stream/<job_id>")
def research_stream(job_id: str):
    """Server-Sent Events stream for a running research job."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"error": "job not found"}), 404

    q: queue.Queue = job["queue"]

    def generate():
        while True:
            try:
                event = q.get(timeout=60)
            except queue.Empty:
                # Keep-alive ping
                yield "data: {\"type\":\"ping\"}\n\n"
                continue

            yield f"data: {json.dumps(event)}\n\n"

            if event.get("type") in ("complete", "error"):
                break

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers=headers,
    )


@app.route("/api/results")
def list_results():
    """Return all cached research results."""
    return jsonify(_load_cache())


@app.route("/api/results/<int:index>", methods=["DELETE"])
def delete_result(index: int):
    """Delete a cached result by its list index."""
    cache = _load_cache()
    if index < 0 or index >= len(cache):
        return jsonify({"error": "index out of range"}), 404
    cache.pop(index)
    _save_cache(cache)
    return jsonify({"ok": True})


@app.route("/api/batch", methods=["POST"])
def batch_research():
    """
    Start batch research jobs for multiple companies at once.
    Expects JSON: {"companies": [{"company_name": "...", "website_url": "..."}, ...]}
    Returns a list of job_ids in the same order.
    """
    data = request.get_json(silent=True) or {}
    companies = data.get("companies", [])

    if not companies:
        return jsonify({"error": "companies list is required"}), 400

    job_ids = []
    for entry in companies:
        company_name = (entry.get("company_name") or "").strip()
        website_url = (entry.get("website_url") or "").strip()
        if not company_name:
            continue

        job_id = str(uuid.uuid4())
        q: queue.Queue = queue.Queue()

        with _jobs_lock:
            _jobs[job_id] = {"queue": q, "result": None, "error": None}

        def run(cname=company_name, wurl=website_url, jid=job_id, jq=q):
            try:
                try:
                    if not llm_provider.get_active_model():
                        models = llm_provider.list_models()
                        if models:
                            llm_provider.select_model(models[0])
                except Exception:
                    pass

                agent = PainPointAgent()

                def on_progress(message: str, stage: str = "info"):
                    jq.put({"type": "progress", "stage": stage, "message": message})

                result = agent.research_company(cname, wurl, on_progress=on_progress)

                cache = _load_cache()
                cache.append(result)
                _save_cache(cache)

                with _jobs_lock:
                    _jobs[jid]["result"] = result

                jq.put({"type": "complete", "result": result})

            except Exception as exc:
                err = str(exc)
                with _jobs_lock:
                    _jobs[jid]["error"] = err
                jq.put({"type": "error", "message": err})

        threading.Thread(target=run, daemon=True).start()
        job_ids.append(job_id)

    return jsonify({"job_ids": job_ids})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  MoneyPrinterV2 — Pain Point Agent Dashboard")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
