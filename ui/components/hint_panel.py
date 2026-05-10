"""
ui/components/hint_panel.py

Reusable Streamlit component for the Graduated Hints Panel (Screen 3).
This component can be imported into app.py for a modular UI architecture.
"""

import streamlit as st


def render_hint_panel(hints: list, correct_answer: str):
    """
    Renders the graduated hint reveal panel.

    Args:
        hints: A list of 3 hint strings (General, Specific, Most Specific).
        correct_answer: The correct answer string to reveal at the end.
    """
    if not hints:
        st.warning("No hints available. Submit a quiz first.")
        return

    used = st.session_state.get('hints_used', 0)
    hint_labels = [
        "🌐 Hint 1 — General Clue",
        "🎯 Hint 2 — Specific Clue",
        "🔍 Hint 3 — Most Specific Clue"
    ]
    hint_colors = ["#1a3a5c", "#0d4f3c", "#4a1942"]

    for i, (hint, label, color) in enumerate(zip(hints, hint_labels, hint_colors)):
        if i < used:
            st.markdown(
                f"""<div style='background:{color};padding:1rem 1.5rem;
                border-radius:10px;margin-bottom:0.75rem;'>
                <strong style='color:#aee;'>{label}</strong><br>
                <span style='color:#eee;font-size:1.05rem;'>{hint}</span>
                </div>""",
                unsafe_allow_html=True
            )
        elif i == used:
            if st.button(f"🔓 Reveal {label}", use_container_width=True):
                st.session_state.hints_used += 1
                st.rerun()

    st.markdown("---")

    if used >= len(hints):
        if not st.session_state.get('answer_revealed', False):
            if st.button("👁️ Reveal Answer (all hints used)", type="primary",
                         use_container_width=True):
                st.session_state.answer_revealed = True
                st.rerun()
        else:
            st.success(f"✅ The correct answer is: **{correct_answer}**")
    else:
        remaining = len(hints) - used
        st.info(f"Use all {len(hints)} hints to unlock the answer. "
                f"{remaining} hint(s) remaining.")
