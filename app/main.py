"""
main.py — Queens Vehicle Collision Dashboard
Run with: streamlit run app/main.py

Data loading and caching logic lives in utils.py.
Pages import from utils, not from main, to avoid circular re-execution.
"""

from pathlib import Path
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Queens Collision Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Navigation ────────────────────────────────────────────────────────────────

pages_dir = Path(__file__).parent / "pages"

pg = st.navigation(
    {
        "Analysis": [
            st.Page(
                str(pages_dir / "1_overview.py"),
                title="Overview",
                icon="📊",
                default=True,
            ),
            st.Page(str(pages_dir / "2_hotspot_map.py"), title="Hotspot Map", icon="🗺️"),
            st.Page(
                str(pages_dir / "3_contributing_factors.py"),
                title="Contributing Factors",
                icon="🔍",
            ),
            st.Page(str(pages_dir / "4_trends.py"), title="Trends", icon="📈"),
        ],
        "Model": [
            st.Page(
                str(pages_dir / "5_severity_predictor.py"),
                title="Severity Predictor",
                icon="🤖",
            ),
            st.Page(str(pages_dir / "6_pipeline.py"), title="Data Pipeline", icon="⚙️"),
        ],
    }
)

pg.run()