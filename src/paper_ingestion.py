"""Persistent PDF ingestion, source hashes and page-preserving paragraphs."""
from __future__ import annotations
import hashlib
import io
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


class PaperLibrary:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def rows(self):
        return [json.loads(path.read_text(encoding='utf-8')) for path in sorted(self.root.glob('pdf-*.json'))]

    def import_pdf(self, content: bytes, filename: str):
        with self._lock:
            return self._import_pdf(content, filename)

    def _import_pdf(self, content: bytes, filename: str):
        if not content.startswith(b'%PDF-'):
            raise ValueError('Only PDF files are accepted')
        if len(content) > 10 * 1024 * 1024:
            raise ValueError('PDF exceeds the 10 MiB upload limit')
        digest = hashlib.sha256(content).hexdigest()
        paper_id = 'pdf-' + digest
        metadata_path = self.root / (paper_id + '.json')
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding='utf-8')), False
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ValueError('Encrypted PDFs must be decrypted before import')
            if len(reader.pages) > 200:
                raise ValueError('PDF exceeds the 200-page import limit')
            sections, paragraphs = [], []
            chars = 0
            for number, page in enumerate(reader.pages, 1):
                text = (page.extract_text() or '').strip()
                chars += len(text)
                if chars > 1000000:
                    raise ValueError('Extracted text exceeds the one-million-character limit')
                # Paragraph breaks are retained where the parser exposes them;
                # page boundaries always remain stable, including empty pages.
                blocks = [re.sub(r'\s+', ' ', part).strip() for part in re.split(r'\n\s*\n', text) if part.strip()]
                sections.append(f'Page {number}')
                paragraphs.append(blocks)
            if sum(len(p) for ps in paragraphs for p in ps) < 40:
                raise ValueError('No usable text layer found; OCR is required for this PDF')
            title = str((reader.metadata or {}).get('/Title') or Path(filename).stem).strip()[:500]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('PDF parsing failed: ' + type(exc).__name__) from exc
        row = {'id': paper_id, 'title': title or 'Imported paper', 'abstract': '',
               'full_text': {'section_name': sections, 'paragraphs': paragraphs},
               'source': {'kind': 'uploaded_pdf', 'sha256': digest, 'filename': Path(filename).name[:255],
                          'pages': len(sections), 'imported_at': datetime.now(timezone.utc).isoformat(),
                          'parser': 'pypdf', 'layout_review_required': True}}
        # The original file is content-addressed; a user filename never becomes
        # a writable path. A manifest is published only after the PDF is stored.
        (self.root / (paper_id + '.pdf')).write_bytes(content)
        temporary = metadata_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(metadata_path)
        return row, True

    def pdf_path(self, paper_id: str) -> Path:
        if not re.fullmatch(r'pdf-[0-9a-f]{64}', paper_id):
            raise KeyError(paper_id)
        path = self.root / (paper_id + '.pdf')
        if not path.is_file():
            raise KeyError(paper_id)
        return path
