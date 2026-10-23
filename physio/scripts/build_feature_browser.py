"""
scripts/build_feature_browser.py

Generates docs/feature_browser.html to allow visual inspection of all
extracted feature tables directly in any browser.
"""

import json
from pathlib import Path
import pandas as pd

HTML_OUT = Path("docs/feature_browser.html")

TABLES = {
    "ECG (46 features)": "data/processed/features_csv/ecg_features.csv",
    "EDA (24 features)": "data/processed/features_csv/eda_features.csv",
    "Respiration (62 features)": "data/processed/features_csv/resp_features.csv",
    "Face (84 features)": "data/processed/features_csv/face_features.csv",
    "Voice (140 features)": "data/processed/features_csv/voice_features.csv",
    "Window Index (1,476 windows)": "data/processed/index/window_index.parquet",
}

def generate_table_html(path: str, max_rows: int = 25) -> str:
    p = Path(path)
    if not p.exists():
        return f"<p>File {path} not found</p>"
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    
    total_rows, total_cols = df.shape
    preview = df.head(max_rows).round(4)
    
    html = [
        f"<div class='info-badge'>Showing first {len(preview)} of {total_rows} rows | {total_cols} columns</div>",
        "<div class='table-container'>",
        "<table class='data-table'>",
        "<thead><tr>"
    ]
    for col in preview.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")
    
    for _, row in preview.iterrows():
        html.append("<tr>")
        for val in row:
            v_str = "" if pd.isna(val) else str(val)
            html.append(f"<td>{v_str}</td>")
        html.append("</tr>")
    
    html.append("</tbody></table></div>")
    return "\n".join(html)

def main():
    tab_buttons = []
    tab_contents = []
    
    for idx, (name, path) in enumerate(TABLES.items()):
        tab_id = f"tab_{idx}"
        active_cls = "active" if idx == 0 else ""
        tab_buttons.append(f"<button class='tab-btn {active_cls}' onclick=\"openTab(event, '{tab_id}')\">{name}</button>")
        
        display_style = "block" if idx == 0 else "none"
        content_html = generate_table_html(path)
        tab_contents.append(f"<div id='{tab_id}' class='tab-content' style='display: {display_style};'><h3>{name} — Table Preview</h3>{content_html}</div>")
        
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>StressRes Extracted Features Browser</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 25px; background: #0f172a; color: #f8fafc; }}
h1 {{ color: #38bdf8; margin-top: 0; }}
.subtitle {{ color: #94a3b8; margin-bottom: 25px; }}
.tab-bar {{ display: flex; gap: 8px; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.tab-btn {{ background: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 14px; transition: all 0.2s ease; }}
.tab-btn:hover {{ background: #334155; color: white; }}
.tab-btn.active {{ background: #0284c7; color: white; border-color: #38bdf8; }}
.table-container {{ overflow-x: auto; background: #1e293b; border-radius: 8px; border: 1px solid #334155; max-height: 700px; }}
.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; font-family: Consolas, monospace; white-space: nowrap; }}
.data-table th {{ background: #0f172a; color: #38bdf8; position: sticky; top: 0; padding: 10px 14px; text-align: left; border-bottom: 2px solid #334155; font-weight: 600; }}
.data-table td {{ padding: 8px 14px; border-bottom: 1px solid #293548; color: #e2e8f0; }}
.data-table tr:hover td {{ background: #273549; }}
.info-badge {{ display: inline-block; background: #0369a1; color: #e0f2fe; padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: 500; margin-bottom: 12px; }}
.links-bar {{ margin-top: 25px; padding: 15px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; }}
.links-bar a {{ color: #38bdf8; text-decoration: none; margin-right: 18px; font-weight: 500; }}
.links-bar a:hover {{ text-decoration: underline; }}
</style>
<script>
function openTab(evt, tabId) {{
    var contents = document.getElementsByClassName("tab-content");
    for (var i = 0; i < contents.length; i++) {{
        contents[i].style.display = "none";
    }}
    var btns = document.getElementsByClassName("tab-btn");
    for (var i = 0; i < btns.length; i++) {{
        btns[i].className = btns[i].className.replace(" active", "");
    }}
    document.getElementById(tabId).style.display = "block";
    evt.currentTarget.className += " active";
}}
</script>
</head>
<body>
<h1>StressRes Feature Table Browser</h1>
<div class="subtitle">Inspect extracted features, window ground truth, and calibrated matrices across all modalities.</div>

<div class="tab-bar">
{''.join(tab_buttons)}
</div>

{''.join(tab_contents)}

<div class="links-bar">
<strong>Raw CSV Files on Disk:</strong>
<a href="../data/processed/features_csv/ecg_features.csv">ecg_features.csv</a>
<a href="../data/processed/features_csv/eda_features.csv">eda_features.csv</a>
<a href="../data/processed/features_csv/resp_features.csv">resp_features.csv</a>
<a href="../data/processed/features_csv/face_features.csv">face_features.csv</a>
<a href="../data/processed/features_csv/voice_features.csv">voice_features.csv</a>
</div>
</body>
</html>
"""
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Generated interactive feature browser: {HTML_OUT}")

if __name__ == "__main__":
    main()
