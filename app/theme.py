# ── Design tokens ─────────────────────────────────────────────────────────────

SEVERITY_ORDER = ["No Injury", "Minor Injury", "Major Injury", "Fatal"]
SEVERITY_COLORS = {
    "No Injury": "#4ade80",
    "Minor Injury": "#facc15",
    "Major Injury": "#f97316",
    "Fatal": "#ef4444",
}

BG = "#0d0d14"
SURFACE = "#13131f"
BORDER = "#1f1f30"
TEXT = "#e8e3f0"
SUBTEXT = "#6b6880"
ACCENT = "#ef4444"
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "#1f1f30"

HEATMAP_COLOR_RANGE = [
    [19, 19, 31],  # #13131f (background / low)
    [124, 58, 237],  # #7c3aed (mid density)
    [239, 68, 68],  # #ef4444 (high density)
]
