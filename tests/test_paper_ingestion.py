import io
import tempfile
import unittest
from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from fastapi.testclient import TestClient
from src.research_api import build_app
from src.research_workbench import QasperEvidenceWorkbench
from src.paper_ingestion import PaperLibrary


def pdf_bytes():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 50 720 Td (The treatment reduced the measured outcome after twelve weeks.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.add_metadata({'/Title':'Imported treatment study'})
    out = io.BytesIO(); writer.write(out)
    return out.getvalue()


class PaperIngestionTests(unittest.TestCase):
    def test_real_pdf_import_source_roundtrip_and_restart(self):
        seed = {'id':'seed', 'title':'Seed paper', 'full_text':{'section_name':['Intro'], 'paragraphs':[['Original seed corpus.']]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def app():
                return build_app(QasperEvidenceWorkbench([seed]), history_db_path=root/'history.sqlite3', library_path=root/'library')
            with TestClient(app()) as client:
                imported = client.post('/v1/library/pdf?filename=study.pdf', content=pdf_bytes(), headers={'Content-Type':'application/pdf'}).raise_for_status().json()
                paper_id = imported['paper']['paper_id']
                self.assertTrue(imported['created'])
                duplicate = client.post('/v1/library/pdf', content=pdf_bytes()).raise_for_status().json()
                self.assertFalse(duplicate['created'])
                source = client.get(f'/v1/papers/{paper_id}/paragraphs/s1:p1').raise_for_status().json()
                self.assertEqual(source['section_name'], 'Page 1')
                self.assertIn('twelve weeks', source['text'])
                self.assertEqual(client.get(f'/v1/library/{paper_id}/pdf').content, pdf_bytes())
                question = client.post('/v1/questions', json={'paper_id':paper_id, 'question':'What outcome did treatment reduce?'}).raise_for_status().json()
                self.assertFalse(question['result']['abstained'])
                self.assertEqual(client.post('/v1/library/pdf', content=b'not a PDF').status_code, 422)
            with TestClient(app()) as client:
                self.assertEqual(client.get(f'/v1/papers/{paper_id}').json()['title'], 'Imported treatment study')
            library = PaperLibrary(root/'library')
            with self.assertRaises(KeyError): library.pdf_path('../../outside')


if __name__ == '__main__': unittest.main()
