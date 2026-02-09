"""
read_pdf.py — Simple tool to extract text and info from PDF datasheets.

Usage:
    python read_pdf.py <pdf_path> [page_numbers]

Examples:
    python read_pdf.py ../docs/cn0566.pdf              # all pages
    python read_pdf.py ../docs/ADAR1000.pdf 1-5         # pages 1 through 5
    python read_pdf.py ../docs/ADAR1000.pdf 1,3,7       # specific pages
    python read_pdf.py ../docs/cn0566.pdf 12             # single page
"""

import sys
import io
import fitz  # PyMuPDF

# Fix Windows console encoding for Unicode characters in datasheets
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def parse_page_spec(spec, max_page):
    """Parse page specification string into a list of 0-based page numbers.

    Supports: "5" (single), "1-10" (range), "1,3,7" (list), "1-3,7,10-12" (mixed).
    Input is 1-based, output is 0-based.
    """
    pages = set()
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            start = max(1, int(start))
            end = min(max_page, int(end))
            pages.update(range(start - 1, end))
        else:
            p = int(part) - 1
            if 0 <= p < max_page:
                pages.add(p)
    return sorted(pages)


def read_pdf(path, page_spec=None):
    """Read a PDF and print its text content."""
    doc = fitz.open(path)
    total = len(doc)
    print(f"=== {path} ===")
    print(f"Pages: {total}, Title: {doc.metadata.get('title', 'N/A')}")
    print(f"Author: {doc.metadata.get('author', 'N/A')}")
    print()

    if page_spec:
        pages = parse_page_spec(page_spec, total)
    else:
        pages = range(total)

    for pnum in pages:
        page = doc[pnum]
        text = page.get_text()
        print(f"--- Page {pnum + 1}/{total} ---")
        if text.strip():
            print(text)
        else:
            # Page might be scanned/image-only
            images = page.get_images()
            print(f"[No extractable text — page contains {len(images)} image(s)]")
        print()

    doc.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_spec = sys.argv[2] if len(sys.argv) > 2 else None
    read_pdf(pdf_path, page_spec)
