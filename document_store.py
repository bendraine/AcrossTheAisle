import pdfplumber
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
import logging
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import hashlib
import json
from datetime import datetime

load_dotenv()

@dataclass
class DocumentPassage:
    """Enhanced document passage with better metadata and source tracking"""
    text: str
    page_number: int
    document_source: str
    passage_index: int
    semantic_type: str = "general"
    topic: str = "general" # <-- ADDED: Explicit topic field
    word_count: int = 0
    created_at: str = ""
    document_hash: str = ""
    
    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.created_at = datetime.now().isoformat()

@dataclass
class SearchResult:
    """Container for search results with source attribution"""
    text: str
    source: str
    page_number: int
    relevance_score: float
    document_hash: str
    passage_index: int

class DocumentStore:
    """Enhanced document store with better chunking, error handling, and source tracking"""

    def __init__(self, collection_name: str = "documents",
                 min_passage_length: int = 100,
                 max_passage_length: int = 800):
        self.min_passage_length = min_passage_length
        self.max_passage_length = max_passage_length
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        try:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.openai_client.models.list()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        try:
            self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
            self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.openai_ef
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
        
        self.document_registry = {}
        self._load_document_registry()

    def _load_document_registry(self):
        registry_path = "./chroma_store/document_registry.json"
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.document_registry = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load document registry: {e}")

    def _save_document_registry(self):
        registry_path = "./chroma_store/document_registry.json"
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        try:
            with open(registry_path, 'w') as f:
                json.dump(self.document_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save document registry: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Could not hash file {file_path}: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str, document_source: str = None) -> List[str]:
        # ... (This function remains unchanged) ...
        if document_source is None:
            document_source = Path(pdf_path).name
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        pages_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = self._clean_text(text)
                            if len(cleaned_text) > 50:
                                pages_text.append(cleaned_text)
                    except Exception as e:
                        self.logger.error(f"Error processing page {page_num} of {document_source}: {e}")
                        continue
        except Exception as e:
            self.logger.error(f"Error opening PDF {pdf_path}: {e}")
            raise
        if not pages_text:
            raise ValueError(f"No extractable text found in {document_source}")
        self.logger.info(f"Successfully extracted text from {len(pages_text)} pages of {document_source}")
        return pages_text

    def _clean_text(self, text: str) -> str:
        # ... (This function remains unchanged) ...
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\s+', ' ', text)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if not re.match(r'^\d+$', line.strip()) and len(line.strip()) >= 10]
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'[^\w\s.,;:!?()\[\]"\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _semantic_chunk_text(self, text: str, document_source: str) -> List[str]:
        # ... (This function remains unchanged) ...
        paragraphs = re.split(r'\n\s*\n', text)
        chunks, current_chunk = [], ""
        for p in paragraphs:
            if not p.strip(): continue
            if current_chunk and len(current_chunk) + len(p) + 2 > self.max_passage_length:
                if len(current_chunk) >= self.min_passage_length: chunks.append(current_chunk)
                current_chunk = p
            else:
                current_chunk = f"{current_chunk}\n\n{p}" if current_chunk else p
        if current_chunk and len(current_chunk) >= self.min_passage_length: chunks.append(current_chunk)
        return chunks

    def create_passages(self, pages_text: List[str], document_source: str, document_hash: str, topic: str) -> List[DocumentPassage]:
        """Enhanced passage creation to include the topic"""
        passages = []
        passage_index = 0
        for page_num, page_text in enumerate(pages_text, 1):
            chunks = self._semantic_chunk_text(page_text, document_source)
            for chunk in chunks:
                semantic_type = self._classify_chunk_type(chunk)
                passage = DocumentPassage(
                    text=chunk,
                    page_number=page_num,
                    document_source=document_source,
                    passage_index=passage_index,
                    semantic_type=semantic_type,
                    document_hash=document_hash,
                    topic=topic  # <-- ADDED: Pass the topic to the passage object
                )
                passages.append(passage)
                passage_index += 1
        self.logger.info(f"Created {len(passages)} passages from {document_source}")
        return passages

    def _classify_chunk_type(self, text: str) -> str:
        # ... (This function remains unchanged) ...
        system_prompt = """You are a text classification assistant. Classify the passage into one of four categories: 'argument', 'evidence', 'policy', or 'general'. Respond with ONLY the category name."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text[:1000]}],
                temperature=0, max_tokens=10
            )
            classification = response.choices[0].message.content.strip().lower()
            return classification if classification in ["argument", "evidence", "policy", "general"] else "general"
        except Exception as e:
            self.logger.warning(f"AI classification failed: {e}. Defaulting to 'general'.")
            return "general"

    def add_document(self, pdf_path: str, document_source: str = None, force_reprocess: bool = False):
        """Document addition now infers and stores a topic for each passage"""
        document_source = document_source or Path(pdf_path).name
        file_hash = self._get_file_hash(pdf_path)
        
        if not force_reprocess and self.document_registry.get(document_source, {}).get('hash') == file_hash:
            self.logger.info(f"Document {document_source} already processed (unchanged)")
            return
        
        try:
            self.logger.info(f"Processing document: {document_source}")
            
            # --- CHANGE START ---
            # Infer topic from filename (e.g., "income_tax_pro.pdf" -> "income_tax")
            topic = document_source.split('_')[0] if '_' in document_source else 'general'
            self.logger.info(f"Inferred topic '{topic}' for document '{document_source}'")
            # --- CHANGE END ---

            pages_text = self.extract_text_from_pdf(pdf_path, document_source)
            new_passages = self.create_passages(pages_text, document_source, file_hash, topic)
            
            if not new_passages:
                raise ValueError(f"No valid passages created from {document_source}")
            
            documents, ids, metadatas = [], [], []
            for passage in new_passages:
                documents.append(passage.text)
                ids.append(f"{document_source}_{passage.document_hash[:8]}_{passage.passage_index}")
                metadatas.append({
                    "document_source": passage.document_source,
                    "page_number": passage.page_number,
                    "passage_index": passage.passage_index,
                    "semantic_type": passage.semantic_type,
                    "word_count": passage.word_count,
                    "document_hash": passage.document_hash,
                    "topic": passage.topic, # <-- ADDED: Store topic in metadata
                    "created_at": passage.created_at
                })
            
            self.collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            
            self.document_registry[document_source] = {
                "hash": file_hash, "passages_count": len(new_passages),
                "processed_at": datetime.now().isoformat(), "file_path": pdf_path
            }
            self._save_document_registry()
            self.logger.info(f"Successfully added {len(new_passages)} passages from {document_source}")
        except Exception as e:
            self.logger.error(f"Failed to add document {document_source}: {e}", exc_info=True)
            raise

    def retrieve_passages(self, query: str, top_k: int = 5, semantic_types: List[str] = None, topic: str = None) -> List[SearchResult]:
        """Retrieve passages, using the supported '$eq' operator for topic filtering."""
        if not query.strip(): return []
        try:
            # --- CHANGE START ---
            conditions = []
            if semantic_types:
                conditions.append({"semantic_type": {"$in": semantic_types}})
            if topic:
                # Use the supported '$eq' operator on the new 'topic' metadata field
                conditions.append({"topic": {"$eq": topic}})

            where_clause = None
            if len(conditions) > 1:
                where_clause = {"$and": conditions}
            elif len(conditions) == 1:
                where_clause = conditions[0]
            # --- CHANGE END ---

            results = self.collection.query(
                query_texts=[query], n_results=min(top_k, 20),
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            if not results["documents"] or not results["documents"][0]: return []

            search_results = []
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                search_results.append(SearchResult(
                    text=doc, source=meta.get("document_source", "Unknown"),
                    page_number=meta.get("page_number", 0),
                    relevance_score=max(0, 1 - dist),
                    document_hash=meta.get("document_hash", ""),
                    passage_index=meta.get("passage_index", 0)
                ))
            
            self.logger.info(f"Retrieved {len(search_results)} passages for query with topic '{topic}'")
            return search_results
        except Exception as e:
            self.logger.error(f"Error retrieving passages: {e}", exc_info=True)
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        # ... (This function remains unchanged) ...
        try:
            total_count = self.collection.count()
            sample = self.collection.peek(limit=min(100, total_count))
            sources, semantic_types = {}, {}
            if sample and sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    source = metadata.get("document_source", "Unknown")
                    sources[source] = sources.get(source, 0) + 1
                    sem_type = metadata.get("semantic_type", "general")
                    semantic_types[sem_type] = semantic_types.get(sem_type, 0) + 1
            return {
                "total_passages": total_count, "unique_documents": len(self.document_registry),
                "document_sources": sources, "semantic_type_distribution": semantic_types,
                "document_registry": self.document_registry
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

# Global instance and helper functions
_document_store = DocumentStore()
def add_pdf_from_docs(filename: str, docs_folder: str = "./data/docs") -> bool:
    try:
        pdf_path = os.path.join(docs_folder, filename)
        if not os.path.exists(pdf_path):
            logging.error(f"PDF not found: {pdf_path}")
            return False
        _document_store.add_document(pdf_path, filename)
        return True
    except Exception as e:
        logging.error(f"Error adding PDF {filename}: {e}")
        return False

def retrieve_passages(query: str, top_k: int = 5, semantic_types: List[str] = None, topic: str = None) -> List[SearchResult]:
    return _document_store.retrieve_passages(query, top_k, semantic_types, topic)

def get_store_stats() -> Dict[str, Any]:
    return _document_store.get_document_stats()