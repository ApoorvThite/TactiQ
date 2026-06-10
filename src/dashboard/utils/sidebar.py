"""Shared sidebar for all TactiQ dashboard pages."""

import streamlit as st


def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown(
            """
            <div style='padding: 0.25rem 0 1rem;'>
                <div style='font-size:1.6rem; font-weight:900; letter-spacing:-0.03em;
                            background: linear-gradient(135deg, #00D4A0 0%, #7C6FE0 100%);
                            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                            background-clip: text;'>TactiQ</div>
                <div style='font-size:0.65rem; color:#3A4060; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.12em; margin-top:2px;'>
                    Tactical DNA Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tournament block
        st.markdown(
            """
            <div style='background:#0A0F1E; border:1px solid rgba(0,212,160,0.15);
                        border-radius:10px; padding:0.75rem 0.9rem; margin-bottom:0.75rem;'>
                <div style='font-size:0.62rem; color:#3A4060; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.5rem;'>
                    Tournament
                </div>
                <div style='font-size:0.82rem; color:#C0C8E8; margin-bottom:0.3rem;'>
                    🗓&nbsp; <span style='color:#6B7394;'>Starts</span>
                    &nbsp;<b style='color:#E8EDF8;'>June 11, 2026</b>
                </div>
                <div style='font-size:0.82rem; color:#C0C8E8;'>
                    🏆&nbsp; <span style='color:#6B7394;'>Champion</span>
                    &nbsp;<b style='color:#00D4A0;'>Spain</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Data stats block
        st.markdown(
            """
            <div style='background:#0A0F1E; border:1px solid rgba(255,255,255,0.05);
                        border-radius:10px; padding:0.75rem 0.9rem; margin-bottom:0.75rem;'>
                <div style='font-size:0.62rem; color:#3A4060; font-weight:700;
                            text-transform:uppercase; letter-spacing:0.12em; margin-bottom:0.6rem;'>
                    Data
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>StatsBomb Events</span>
                    <span class='stat-value'>843K</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>Tournaments</span>
                    <span class='stat-value'>4</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>Teams (StatsBomb)</span>
                    <span class='stat-value'>54</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>WC2026 Proxy Teams</span>
                    <span class='stat-value'>+14</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>Fixtures Predicted</span>
                    <span class='stat-value'>72</span>
                </div>
                <div class='stat-row'>
                    <span class='stat-label'>Monte Carlo Runs</span>
                    <span class='stat-value'>10,000</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style='font-size:0.62rem; color:#2A3050; text-align:center; padding-top:0.25rem;'>
                WC2026 &nbsp;·&nbsp; TactiQ
            </div>
            """,
            unsafe_allow_html=True,
        )
