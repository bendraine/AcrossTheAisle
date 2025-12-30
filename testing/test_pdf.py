import unittest
from document_store import add_pdf_from_docs, get_store_stats, _document_store

class TestPDFLoading(unittest.TestCase):
    def test_pdf_loading_and_passage_creation(self):
        pdf_filename = "ben testing doc.pdf"
        # Try to add the PDF
        result = add_pdf_from_docs(pdf_filename)
        self.assertTrue(result, f"PDF '{pdf_filename}' should load successfully.")

        # Check document stats
        stats = get_store_stats()
        self.assertIn(pdf_filename, stats["document_registry"], "PDF should be in document registry.")
        passage_count = stats["document_registry"][pdf_filename]["passages_count"]
        self.assertGreater(passage_count, 0, "PDF should have at least one passage created.")

        print(f"Loaded '{pdf_filename}' with {passage_count} passages.")
                # Print passages for the PDF

        # Get all passages for this document
        registry_entry = stats["document_registry"][pdf_filename]
        file_path = registry_entry["file_path"]
        passages = _document_store.create_passages(
            _document_store.extract_text_from_pdf(file_path, pdf_filename),
            pdf_filename,
            registry_entry["hash"]
        )

        print("\nPassages from '{}':".format(pdf_filename))
        for i, passage in enumerate(passages, 1):
            print(f"\n--- Passage {i} ---\n{passage.text}\n")

if __name__ == "__main__":
    unittest.main()