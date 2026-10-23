"""
scripts/export_eda_html.py

Exports notebooks/01_stressid_eda.ipynb to a self-contained HTML report:
docs/01_stressid_eda_report.html
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/01_stressid_eda.ipynb")
HTML_PATH = Path("docs/01_stressid_eda_report.html")


def main():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>StressID Exploratory Data Analysis Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 1050px; margin: 30px auto; padding: 0 20px; color: #1e293b; background: #f8fafc; }",
        ".card { background: white; padding: 25px; margin-bottom: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e2e8f0; }",
        "h1 { color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; margin-top: 0; }",
        "h2 { color: #1e40af; margin-top: 0; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; }",
        ".output-text { background: #f1f5f9; color: #334155; padding: 12px; border-radius: 6px; font-family: Consolas, monospace; white-space: pre-wrap; font-size: 13px; border-left: 4px solid #3b82f6; margin: 12px 0; }",
        "img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #cbd5e1; margin: 15px 0; display: block; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }",
        ".conclusion { background: #ecfdf5; border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 4px; margin-top: 15px; font-weight: 500; color: #065f46; }",
        "ul { padding-left: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class='card'>"
    ]

    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            if text.startswith("**Conclusion"):
                html.append(f"<div class='conclusion'>{text}</div>")
            elif text.startswith("# "):
                html.append(f"<h1>{text[2:]}</h1>")
            elif text.startswith("## "):
                html.append(f"</div><div class='card'><h2>{text[3:]}</h2>")
            else:
                paragraphs = text.split("\n\n")
                for p in paragraphs:
                    if p.strip().startswith("- "):
                        items = [f"<li>{item[2:].strip()}</li>" for item in p.strip().split("\n") if item.startswith("- ")]
                        html.append(f"<ul>{''.join(items)}</ul>")
                    elif p.strip().startswith("1. "):
                        items = [f"<li>{item[3:].strip()}</li>" for item in p.strip().split("\n") if len(item) > 3 and item[0].isdigit()]
                        html.append(f"<ol>{''.join(items)}</ol>")
                    else:
                        html.append(f"<p>{p.strip()}</p>")
        elif cell["cell_type"] == "code":
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    txt = "".join(out.get("text", []))
                    html.append(f"<div class='output-text'>{txt}</div>")
                elif "data" in out and "image/png" in out["data"]:
                    b64 = out["data"]["image/png"]
                    html.append(f'<img src="data:image/png;base64,{b64}" />')

    html.append("</div></body></html>")

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(html))

    print(f"Generated standalone HTML report: {HTML_PATH}")


if __name__ == "__main__":
    main()
