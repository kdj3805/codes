"""
cancer_app_v2_eval_patch.py
─────────────────────────────────────────────────────────────────────────────
INTEGRATION INSTRUCTIONS
─────────────────────────────────────────────────────────────────────────────
This file shows the minimal diff required to plug live_eval.py into
cancer_app_v2.py.  Do NOT copy this whole file — apply only the two
changes described below.

Change 1 — Add import at the top of cancer_app_v2.py
─────────────────────────────────────────────────────
Locate the line:
    from graphrag_integration import generate_answer_graphrag_stream as generate_answer_stream

Add immediately BELOW it:
    from live_eval import evaluate_live          # ← ADD THIS LINE

Change 2 — Call evaluate_live after every answer in the HANDLE NEW QUERY block
──────────────────────────────────────────────────────────────────────────────
Locate this block in cancer_app_v2.py (inside the "with tab_chat:" section):

    full_answer, sources, followups = render_streaming_answer(
        query=user_query,
        patient_report=patient_context,
        chat_history=history_for_llm,
        cancer_filter=cancer_filter,
    )

Add the following IMMEDIATELY AFTER that block (before the "Sources expander"):

    # ── LIVE EVAL (non-blocking, background thread) ──────────────────────
    _live_contexts = st.session_state.get("stream_sources", [])
    evaluate_live(
        question      = user_query,
        answer        = full_answer,
        contexts      = [
            s.get("label", str(s)) if isinstance(s, dict) else str(s)
            for s in _live_contexts
        ],
        cancer_filter = cancer_filter,
        metadata      = {
            "is_web_fallback": st.session_state.get("is_web_fallback", False),
            "n_history_turns": len(history_for_llm),
        },
    )
    # ─────────────────────────────────────────────────────────────────────

That is ALL.  Two changes, no architectural modifications.

─────────────────────────────────────────────────────────────────────────────
ENVIRONMENT VARIABLES (optional)
─────────────────────────────────────────────────────────────────────────────
Add to your .env file to control live eval behaviour:

    # Enable full RAGAS faithfulness on every live query (slower, costs API credits)
    LIVE_EVAL_RAGAS=0    # default off — set to 1 to enable

─────────────────────────────────────────────────────────────────────────────
READING THE LIVE LOG
─────────────────────────────────────────────────────────────────────────────
The live log is written to:  live_eval_logs.json  (NDJSON — one JSON per line)

To print a summary:
    python live_eval.py

To read programmatically:
    import json
    records = [json.loads(line) for line in open("live_eval_logs.json") if line.strip()]

─────────────────────────────────────────────────────────────────────────────
FULL PATCHED SNIPPET (copy-paste ready section of cancer_app_v2.py)
─────────────────────────────────────────────────────────────────────────────
Replace the entire "with st.chat_message("assistant"):" block (lines ~549-588)
with the version below.  Lines marked with  # ← ADDED  are the only new lines.
"""

# ══════════════════════════════════════════════════════════════════════════════
# TOP OF FILE — add this import (line ~10, after existing graphrag import)
# ══════════════════════════════════════════════════════════════════════════════
# from graphrag_integration import generate_answer_graphrag_stream as generate_answer_stream
# from live_eval import evaluate_live          # ← ADDED

# ══════════════════════════════════════════════════════════════════════════════
# HANDLE NEW QUERY block — patched version (inside "with tab_chat:")
# ══════════════════════════════════════════════════════════════════════════════

PATCHED_HANDLE_QUERY_BLOCK = '''
        # --- HANDLE NEW QUERY ---
        if user_query:

            with st.chat_message("user"):
                st.markdown(user_query)

            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                try:
                    history_for_llm = st.session_state.messages[:-1]

                    # Stream response (unchanged)
                    full_answer, sources, followups = render_streaming_answer(
                        query=user_query,
                        patient_report=patient_context,
                        chat_history=history_for_llm,
                        cancer_filter=cancer_filter,
                    )

                    # ── LIVE EVAL (non-blocking background thread) ────────────  ← ADDED
                    _live_contexts = st.session_state.get("stream_sources", [])     # ← ADDED
                    evaluate_live(                                                   # ← ADDED
                        question      = user_query,                                 # ← ADDED
                        answer        = full_answer,                                # ← ADDED
                        contexts      = [                                           # ← ADDED
                            s.get("label", str(s)) if isinstance(s, dict)          # ← ADDED
                            else str(s) for s in _live_contexts                     # ← ADDED
                        ],                                                          # ← ADDED
                        cancer_filter = cancer_filter,                              # ← ADDED
                        metadata      = {                                           # ← ADDED
                            "is_web_fallback": st.session_state.get(               # ← ADDED
                                "is_web_fallback", False),                          # ← ADDED
                            "n_history_turns": len(history_for_llm),               # ← ADDED
                        },                                                          # ← ADDED
                    )                                                               # ← ADDED
                    # ──────────────────────────────────────────────────────────  ← ADDED

                    # Sources expander (unchanged)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                if isinstance(s, dict) and s.get("url"):
                                    st.markdown(f"[{s[\'label\']}]({s[\'url\']})")
                                elif isinstance(s, dict):
                                    st.markdown(f"`{s.get(\'label\', str(s))}`")
                                else:
                                    st.markdown(f"`{s}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_answer,
                    })

                    new_turn_idx = len(st.session_state.messages) - 1
                    st.session_state["followups"][new_turn_idx] = followups

                    if len(st.session_state["followups"]) > 10:
                        oldest = min(st.session_state["followups"].keys())
                        del st.session_state["followups"][oldest]

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)
                    followups    = []
                    new_turn_idx = len(st.session_state.messages) - 1

            # Render follow-ups immediately for the new answer
            render_followup_buttons(
                st.session_state["followups"].get(new_turn_idx, []),
                turn_key=str(new_turn_idx),
            )

            # Rerun only if it was typed, so the input box clears and history aligns
            if typed_query:
                st.rerun()
'''

if __name__ == "__main__":
    print(__doc__)
    print("\nPatched query handling block:\n")
    print(PATCHED_HANDLE_QUERY_BLOCK)
