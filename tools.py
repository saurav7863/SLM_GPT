import subprocess
import webbrowser
import requests
import re
import PyPDF2
from PyPDF2 import generic

# Safari

def open_safari():
    subprocess.Popen(["open","-a","Safari"])

# App

def open_app(name: str):
    subprocess.Popen(["open","-a",name])

# URL

def open_url(url: str):
    webbrowser.open(url if url.startswith("http") else f"http://{url}")

# Fetch data

def fetch_data(url: str) -> str:
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.text[:1000]
    except Exception as e:
        return f"❌ Error: {e}"

# Fill PDF

def fill_pdf(command: str, pdf_path: str=None) -> str:
    if not pdf_path:
        return "❌ No PDF uploaded."
    m = re.match(r"fill pdf.*with\s*(.+)", command, re.IGNORECASE)
    if not m:
        return "❌ Use: fill pdf with field1=val1,field2=val2"
    data = dict(item.split('=') for item in m.group(1).split(','))
    reader = PyPDF2.PdfReader(pdf_path)
    writer = PyPDF2.PdfWriter()
    page = reader.pages[0]
    if '/Annots' in page:
        for annot in page['/Annots']:
            key = annot.get_object().get('/T')
            if key and key in data:
                annot.get_object().update({generic.NameObject('/V'): generic.createStringObject(data[key])})
    writer.add_page(page)
    out = pdf_path.replace('.pdf','_filled.pdf')
    with open(out,'wb') as f:
        writer.write(f)
    return f"✅ Filled PDF saved to {out}"

# Schedule task (stub)

def schedule_task(command: str) -> str:
    return f"✅ Scheduled: {command}"