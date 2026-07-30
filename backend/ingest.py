#!/usr/bin/env python3
"""Standalone PDF Ingestion Script for Kenyan Legal Assistant
Usage:
    python ingest.py                    # Ingest all PDFs (sequential)
    python ingest.py --parallel         # Ingest with parallel processing (faster)
    python ingest.py --workers 4        # Use 4 workers
    python ingest.py --clear            # Clear existing index before ingestion
    python ingest.py --reload-model     # Clear embeddings cache and reload model
"""

import sys
import re
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, str(Path(__file__).parent))

from logic.embeddings import embeddings_service


class StandaloneIngestor:
    def __init__(self, parallel=False, max_workers=2):
        self.parallel = parallel
        self.max_workers = max_workers
        self.collection = None
        self.client = None
        self.stats = {"processed": 0, "chunks": 0, "failed": 0}
        self._lock = threading.Lock()
    
    def initialize_database(self):
        """Initialize ChromaDB connection (no model loading)"""
        print("\n[1/3] Connecting to ChromaDB...")
        
        import chromadb
        
        chroma_path = Path(__file__).parent / "data" / "chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_path))
        
        try:
            self.collection = self.client.get_collection("kenyan_law")
            print(f"✓ Connected to existing database with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection("kenyan_law")
            print("✓ Created new database")
        
        return True
    
    def load_embeddings_model(self):
        """Load the shared embeddings model"""
        print("\n[2/3] Loading Embeddings Model...")
        
        success = embeddings_service.load()
        
        if success:
            print("✓ Embeddings model loaded")
        else:
            print("✗ Failed to load embeddings model")
        
        return success
    
    def _filename_to_title(self, filename):
        """Convert filename to readable title"""
        name = filename.replace('.pdf', '')
        name = name.replace('_', ' ')
        name = name.title()
        name = name.replace(' Of ', ' of ').replace(' The ', ' the ')
        name = name.replace(' And ', ' and ').replace(' For ', ' for ')
        return name
    
    def _remove_toc_lines(self, text):
        """Remove TOC lines and page numbers"""
        text = re.sub(r'\n.+\s*\.{4,}\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*\.{4,}\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page\s+\d+\s+of\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
        return text
    
    def _chunk_legal_text(self, text, doc_name):
        """Split text into meaningful chunks"""
        patterns = [
            r'(?:ARTICLE|Article)\s+\d+[\.\:\—\-]',
            r'(?:PART|Part)\s+[IVX]+[\.\:\—\-]',
            r'(?:CHAPTER|Chapter)\s+[IVX\d]+[\.\:\—\-]',
            r'(?:SECTION|Section)\s+\d+[A-Z]?[\.\:\—\-]',
        ]
        
        chunks = []
        current_chunk = ""
        current_title = doc_name
        
        lines = text.split('\n')
        for line in lines:
            is_header = any(re.match(p, line) for p in patterns)
            
            if is_header and len(current_chunk) > 200:
                if current_chunk.strip():
                    chunks.append({'text': current_chunk.strip(), 'title': current_title})
                current_title = f"{doc_name} — {line[:80]}"
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
                if len(current_chunk) > 1500:
                    if current_chunk.strip():
                        chunks.append({'text': current_chunk.strip(), 'title': current_title})
                    current_chunk = ""
                    current_title = doc_name
        
        if current_chunk.strip() and len(current_chunk.strip()) > 100:
            chunks.append({'text': current_chunk.strip(), 'title': current_title})
        
        return chunks if chunks else [{'text': text[:2000], 'title': doc_name}]
    
    def _filter_quality_chunks(self, chunks):
        """Filter out low-quality chunks"""
        filtered = []
        for chunk in chunks:
            text = chunk.get('text', '')
            if len(text) < 100:
                continue
            if re.search(r'\.{4,}\s*\d+', text):
                continue
            filtered.append(chunk)
        return filtered
    
    def _clean_text(self, text):
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('-\n', '')
        text = text.replace('\n', ' ')
        text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\.{3,}\s*\d+', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 not installed. Run: pip install PyPDF2")
            return None
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                total_pages = len(reader.pages)
                
                # Skip TOC pages (first 8-12 pages for most Kenyan law PDFs)
                toc_pages = min(12, total_pages // 3)
                
                for page_num, page in enumerate(reader.pages):
                    if page_num < toc_pages:
                        continue
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    
                    if total_pages > 20 and (page_num + 1) % 20 == 0:
                        print(f"    Page {page_num + 1}/{total_pages}")
                
                if not full_text.strip():
                    print(f"  ✗ No text extracted (after skipping TOC)")
                    return None
                
                return full_text
                
        except Exception as e:
            print(f"  ✗ Error extracting text: {e}")
            return None
    
    def ingest_single_pdf(self, pdf_path, tracking_file=None):
        """Ingest a single PDF using shared embeddings"""
        try:
            print(f"\n📄 Processing: {pdf_path.name}...")
            
            # Extract text
            full_text = self._extract_text_from_pdf(pdf_path)
            if not full_text:
                return False, 0
            
            # Get document name
            doc_name = self._filename_to_title(pdf_path.name)
            
            # Clean and chunk
            cleaned_text = self._remove_toc_lines(full_text)
            chunks = self._chunk_legal_text(cleaned_text, doc_name)
            
            # Filter quality chunks
            quality_chunks = self._filter_quality_chunks(chunks)
            
            if not quality_chunks:
                print(f"  ✗ No quality chunks found")
                return False, 0
            
            print(f"    Created {len(chunks)} chunks, {len(quality_chunks)} quality chunks")
            
            # Prepare texts for embedding
            chunk_texts = [chunk['text'] for chunk in quality_chunks]
            
            # Generate embeddings using shared service
            print(f"    Generating embeddings...")
            embeddings = embeddings_service.encode(chunk_texts)
            
            # Add to ChromaDB
            indexed = 0
            for i, (chunk, embedding) in enumerate(zip(quality_chunks, embeddings)):
                doc_id = hashlib.md5(f"{pdf_path.stem}_{i}".encode()).hexdigest()[:20]
                text = self._clean_text(chunk['text'])
                
                if len(text) < 100:
                    continue
                
                try:
                    self.collection.add(
                        embeddings=[embedding],
                        documents=[text],
                        metadatas=[{
                            "source": doc_name,
                            "title": chunk.get('title', doc_name),
                            "category": "kenyan_law_pdf",
                            "chunk_index": i,
                            "chunk_total": len(quality_chunks),
                            "filename": pdf_path.name,
                            "ingested_at": datetime.now().isoformat()
                        }],
                        ids=[doc_id]
                    )
                    indexed += 1
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        print(f"    Warning: Could not index chunk {i}: {e}")
            
            if indexed > 0:
                # Track processed file
                if tracking_file:
                    with open(tracking_file, 'a') as f:
                        f.write(f"{pdf_path.name}\n")
                
                print(f"  ✓ Success: {indexed} chunks indexed")
                return True, indexed
            
            print(f"  ✗ Failed: No chunks indexed")
            return False, 0
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return False, 0
    
    def load_processed_files(self, tracking_file):
        """Load list of already processed files"""
        if tracking_file.exists():
            with open(tracking_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def ingest_folder(self, folder_path, clear_first=False):
        """Main ingestion function"""
        pdf_dir = Path(folder_path)
        if not pdf_dir.exists():
            print(f"✗ PDF folder not found: {folder_path}")
            return 0
        
        # Tracking file for idempotency
        tracking_file = pdf_dir / ".ingested.txt"
        
        # Clear existing index if requested
        if clear_first:
            print("\n[0/3] Clearing existing PDF index...")
            self._clear_pdf_index()
            if tracking_file.exists():
                tracking_file.unlink()
                print("  ✓ Cleared tracking file")
        
        # Get PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"✗ No PDFs found in {folder_path}")
            return 0
        
        print(f"\n📁 Found {len(pdf_files)} PDF(s)")
        
        # Load already processed files
        processed = self.load_processed_files(tracking_file)
        pending = [f for f in pdf_files if f.name not in processed]
        
        if pending:
            print(f"   {len(processed)} already processed, {len(pending)} pending")
        else:
            print(f"   All {len(pdf_files)} files already processed!")
            return 0
        
        # Initialize database and embeddings
        if not self.initialize_database():
            return 0
        
        if not self.load_embeddings_model():
            return 0
        
        print(f"\n[3/3] Ingesting {len(pending)} PDF(s)...")
        
        if self.parallel and len(pending) > 1:
            print(f"    Using parallel processing with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self.ingest_single_pdf, pdf, tracking_file): pdf for pdf in pending}
                for future in as_completed(futures):
                    success, chunks = future.result()
                    if success:
                        self.stats["processed"] += 1
                        self.stats["chunks"] += chunks
                    else:
                        self.stats["failed"] += 1
        else:
            print(f"    Using sequential processing")
            for pdf in pending:
                success, chunks = self.ingest_single_pdf(pdf, tracking_file)
                if success:
                    self.stats["processed"] += 1
                    self.stats["chunks"] += chunks
                else:
                    self.stats["failed"] += 1
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"  Successfully ingested: {self.stats['processed']} files")
        print(f"  Total chunks indexed: {self.stats['chunks']}")
        print(f"  Failed: {self.stats['failed']}")
        
        if self.collection:
            print(f"  Total documents in database: {self.collection.count()}")
        
        return self.stats["processed"]
    
    def _clear_pdf_index(self):
        """Remove all PDF-indexed documents from collection"""
        if not self.collection:
            print("  No database connection")
            return
        
        try:
            results = self.collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(results.get('metadatas', [])):
                if metadata and metadata.get('category') == 'kenyan_law_pdf':
                    ids_to_delete.append(results['ids'][i])
            
            if ids_to_delete:
                # Delete in batches of 100
                for i in range(0, len(ids_to_delete), 100):
                    batch = ids_to_delete[i:i+100]
                    self.collection.delete(ids=batch)
                print(f"  ✓ Removed {len(ids_to_delete)} PDF-indexed documents")
            else:
                print("  No PDF-indexed documents to remove")
        except Exception as e:
            print(f"  Error clearing index: {e}")


def main():
    parser = argparse.ArgumentParser(description='Ingest PDFs into RAG database')
    parser.add_argument('--parallel', '-p', action='store_true', help='Use parallel processing')
    parser.add_argument('--workers', '-w', type=int, default=2, help='Number of workers (default: 2)')
    parser.add_argument('--clear', '-c', action='store_true', help='Clear existing index before ingestion')
    parser.add_argument('--reload-model', action='store_true', help='Clear embeddings cache and reload model')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KENYAN LEGAL ASSISTANT - PDF INGESTION TOOL")
    print("="*60)
    
    # Force reload model if requested
    if args.reload_model:
        print("\n🔄 Clearing embeddings cache...")
        embeddings_service.clear_cache()
    
    ingestor = StandaloneIngestor(parallel=args.parallel, max_workers=args.workers)
    
    pdf_dir = Path(__file__).parent / "data" / "pdfs"
    
    # Create PDF directory if it doesn't exist
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True)
        print(f"\n📁 Created PDF directory: {pdf_dir}")
        print("   Place your PDF files here and run again.")
        return
    
    count = ingestor.ingest_folder(pdf_dir, clear_first=args.clear)
    
    if count > 0:
        print(f"\n✅ Ingestion complete! {count} new PDF(s) processed.")
    elif count == 0:
        print("\n✅ No new PDFs to ingest.")
    else:
        print("\n❌ Ingestion failed.")


if __name__ == "__main__":
    main()