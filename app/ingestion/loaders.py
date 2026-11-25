from pathlib import Path
from bs4 import BeautifulSoup
import fitz

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown"}

def load_document(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        text = load_pdf(path)
    elif extension in {".html", ".htm"}:
        text = load_html(path)
    else:
        text = path.read_text(encoding="utf-8")

    return {
        "text": text.strip(),
        "source": str(path.resolve()),
        "filename": path.name,
        "file_type": extension
    }


def load_pdf(path: Path) -> str:
    pages = []

    with fitz.open(path) as pdf:
        for page in pdf:
            text = page.get_text()
            if text.strip():
                pages.append(text)

    return "\n\n".join(pages)


def load_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8")

    soup = BeautifulSoup(html, "html.parser")

    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()

    return soup.get_text(separator="\n", strip=True)
