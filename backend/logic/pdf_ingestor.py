"""PDF Ingestion — Load Kenyan legal PDFs into the RAG vector database"""

import os
import re
import hashlib
import time
from pathlib import Path


class PDFIngestor:
    """Ingests PDFs and indexes them into ChromaDB"""

    # Generic fallback citation format, e.g. "[2020] eKLR" — used only if
    # neither of the two more specific patterns below find anything.
    _CITATION_PATTERN = re.compile(r'\[\d{4}\]\s*eKLR', re.IGNORECASE)

    # Kenya Law's own downloaded filenames embed the neutral citation as a
    # fused run of digits+court-code+number immediately before "_KLR", e.g.
    # "...2018KEHC6157_KLR..." (year=2018, court=KEHC, number=6157) or
    # "...2013KECA541_KLR..." (year=2013, court=KECA, number=541). This is
    # the most reliable source since it's Kenya Law's own generated
    # filename, not free text.
    _FILENAME_CITATION_PATTERN = re.compile(r'(\d{4})(KE[A-Z]{2,5})(\d+)_KLR', re.IGNORECASE)

    # Some judgments print their own neutral citation directly on the first
    # page as "Neutral citation: [2013] KECA 541 (KLR)" — when present,
    # this is authoritative and preferred over both patterns above.
    _NEUTRAL_CITATION_TEXT_PATTERN = re.compile(
        r'Neutral citation:\s*(\[\d{4}\]\s*[A-Z]{2,6}\s*\d+\s*\(?KLR\)?)', re.IGNORECASE
    )

    # A loose "X v Y" / "X vs Y" / "X versus Y" case-name-shaped line, used
    # as a fallback when the filename alone doesn't give us a usable case
    # name — see _extract_case_metadata.
    _CASE_NAME_LINE_PATTERN = re.compile(
        r'^[A-Z][A-Za-z0-9 ,.\'()&-]{2,150}\s+(?:v\.?|vs\.?|versus)\s+[A-Za-z0-9 ,.\'()&-]{2,150}$',
        re.IGNORECASE
    )

    # Extracts the section/article number from a structural header produced
    # by the section/article split in _chunk_legal_text, e.g. "Section
    # 296." -> "296", "Article 49." -> "49". Used to tag chunks with
    # section_number metadata and to decide whether subsection splitting
    # is even applicable (only Section/Article bodies have numbered
    # subsections in this corpus; Part/Chapter/Schedule headers don't).
    _SECTION_HEADER_NUMBER_PATTERN = re.compile(
        r'(?:SECTION|Section|ARTICLE|Article)\s+(\d+[A-Za-z]?)', re.IGNORECASE
    )

    # Splits a section/article's body text at genuine subsection
    # boundaries — "(1)", "(2)", "(3A)", etc. Deliberately requires the
    # marker to be preceded by a period or newline (i.e. right after the
    # previous subsection's closing sentence, or right after the section
    # number itself), NOT just any "(N)" anywhere in the text. This is
    # what keeps it from misfiring on an in-sentence cross-reference like
    # "...as described in subsection (3) of this section..." — that "(3)"
    # sits mid-sentence with no preceding period/newline, so it's left
    # alone rather than treated as a new subsection boundary.
    _SUBSECTION_SPLIT_PATTERN = re.compile(r'(?<=[.\n])\s*(?=\(\d+[A-Za-z]?\)\s)')
    _SUBSECTION_NUMBER_PATTERN = re.compile(r'^\((\d+[A-Za-z]?)\)')

    def __init__(self, rag_engine):
        self.rag = rag_engine
    
    def wait_for_rag(self, timeout=30):
        """Wait for RAG engine to be ready"""
        print("Waiting for RAG engine to load...")
        start = time.time()
        while not self.rag.is_ready() and (time.time() - start) < timeout:
            time.sleep(0.5)
            print(".", end="", flush=True)
        
        if self.rag.is_ready():
            print(f"\nRAG ready after {time.time() - start:.1f}s")
            return True
        else:
            print(f"\nRAG not ready after {timeout}s")
            return False
    
    def ingest_folder(self, folder_path):
        """Ingest all statute/constitution PDFs from a folder"""
        # Wait for RAG to be ready first
        if not self.wait_for_rag():
            print("RAG not loaded after waiting. PDF ingestion skipped.")
            return 0
        
        pdf_dir = Path(folder_path)
        if not pdf_dir.exists():
            print(f"PDF folder not found: {folder_path}")
            return 0
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDFs found in {folder_path}")
            return 0
        
        print(f"\nFound {len(pdf_files)} PDF(s) to ingest")
        
        count = 0
        for pdf_path in pdf_files:
            try:
                success = self._ingest_pdf(pdf_path)
                if success:
                    count += 1
                    print(f"✓ Successfully ingested: {pdf_path.name}")
                else:
                    print(f"✗ Failed to ingest: {pdf_path.name}")
            except Exception as e:
                print(f"✗ Error ingesting {pdf_path.name}: {e}")
        
        print(f"\nPDF ingestion complete: {count}/{len(pdf_files)} files processed")
        if self.rag.collection:
            print(f"Total documents in RAG: {self.rag.collection.count()}")
        
        return count

    def ingest_case_law_folder(self, folder_path):
        """Ingest all case-law (judgment) PDFs downloaded from Kenya Law
        into the same collection, tagged so the RAG engine's
        retrieve_examples() can find them separately from statute text.
        Point this at a folder of Kenya Law case downloads, e.g. real
        "robbery with violence" conviction judgments."""
        if not self.wait_for_rag():
            print("RAG not loaded after waiting. Case law ingestion skipped.")
            return 0

        case_dir = Path(folder_path)
        if not case_dir.exists():
            print(f"Case law folder not found: {folder_path}")
            return 0

        pdf_files = list(case_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDFs found in {folder_path}")
            return 0

        print(f"\nFound {len(pdf_files)} case law PDF(s) to ingest")

        count = 0
        for pdf_path in pdf_files:
            try:
                success = self._ingest_case_law_pdf(pdf_path)
                if success:
                    count += 1
                    print(f"✓ Successfully ingested case: {pdf_path.name}")
                else:
                    print(f"✗ Failed to ingest case: {pdf_path.name}")
            except Exception as e:
                print(f"✗ Error ingesting case {pdf_path.name}: {e}")

        print(f"\nCase law ingestion complete: {count}/{len(pdf_files)} files processed")
        if self.rag.collection:
            print(f"Total documents in RAG: {self.rag.collection.count()}")

        return count
    
    def _ingest_pdf(self, pdf_path):
        """Extract text from PDF, chunk it, and index into ChromaDB"""
        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 not installed. Run: pip install PyPDF2")
            return False
        
        print(f"\nProcessing: {pdf_path.name}...")
        
        # Extract text
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                full_text = ""
                total_pages = len(reader.pages)
                
                # Estimate TOC pages (first 8-12 pages for most Kenyan law PDFs)
                toc_pages = min(12, total_pages // 3)
                
                for page_num, page in enumerate(reader.pages):
                    # Skip table of contents pages
                    if page_num < toc_pages:
                        continue
                    
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
                    
                    # Progress indicator for large PDFs
                    if total_pages > 20 and (page_num + 1) % 20 == 0:
                        print(f"  Page {page_num + 1}/{total_pages}")
        except Exception as e:
            print(f"  Error extracting text: {e}")
            return False
        
        if not full_text.strip():
            print(f"  No text extracted from {pdf_path.name} (after skipping TOC)")
            return False
        
        print(f"  Extracted {len(full_text)} characters from {total_pages} pages")
        
        # Get a readable name from the filename
        doc_name = self._filename_to_title(pdf_path.name)
        
        # Clean TOC lines before chunking
        full_text = self._remove_toc_lines(full_text)
        
        # Chunk into sections
        chunks = self._chunk_legal_text(full_text, doc_name)
        print(f"  Created {len(chunks)} chunks")
        
        # Index each chunk
        indexed = 0
        skipped = 0
        
        for i, chunk in enumerate(chunks):
            text = chunk['text'].strip()
            
            # Skip very short chunks
            if len(text) < 100:
                skipped += 1
                continue
            
            # Clean text
            text = self._clean_text(text)
            
            if len(text) < 100:
                skipped += 1
                continue
            
            doc_id = hashlib.md5(f"{pdf_path.stem}_{i}".encode()).hexdigest()[:20]
            
            try:
                self.rag.collection.add(
                    documents=[text],
                    metadatas=[{
                        "source": doc_name,
                        "title": chunk.get('title', doc_name),
                        "category": "kenyan_law_pdf",
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "filename": pdf_path.name,
                        # NEW: lets rag_engine's grounding checks verify a
                        # cited section/subsection number directly against
                        # metadata instead of fragile text substring
                        # matching. granularity is "section" (whole
                        # provision, subsection_number=None) or
                        # "subsection" (one numbered subsection only) —
                        # see _chunk_legal_text / _split_section_into_subsections.
                        "granularity": chunk.get('granularity', 'section'),
                        "section_number": chunk.get('section_number') or "",
                        "subsection_number": chunk.get('subsection_number') or "",
                        # NEW: see _split_section_into_subsections' return
                        # value docstring — lets rag_engine.py exclude
                        # inherited (quoted-from-a-different-subsection)
                        # text from precise subsection-attribution checks.
                        "inherited_context_snippet": chunk.get('inherited_context_snippet') or "",
                    }],
                    ids=[doc_id]
                )
                indexed += 1
                if (i + 1) % 50 == 0:
                    print(f"  Indexed {indexed} chunks...")
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    skipped += 1
                else:
                    print(f"  Error indexing chunk {i}: {e}")
                    skipped += 1
        
        print(f"  Indexed: {indexed} chunks, Skipped: {skipped}")
        return indexed > 0

    def _ingest_case_law_pdf(self, pdf_path):
        """Extract text from a case-law PDF, chunk it, and index into
        ChromaDB tagged as doc_type='case_law' with case_name/citation
        metadata so the RAG engine's retrieve_examples() can find it.

        Unlike _ingest_pdf, this does NOT skip a table-of-contents-sized
        block of leading pages — judgments don't have one, and case names
        commonly appear on page 1."""
        try:
            import PyPDF2
        except ImportError:
            print("PyPDF2 not installed. Run: pip install PyPDF2")
            return False

        print(f"\nProcessing case law: {pdf_path.name}...")

        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                full_text = ""

                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

                    if total_pages > 20 and (page_num + 1) % 20 == 0:
                        print(f"  Page {page_num + 1}/{total_pages}")
        except Exception as e:
            print(f"  Error extracting text: {e}")
            return False

        if not full_text.strip():
            print(f"  No text extracted from {pdf_path.name}")
            return False

        print(f"  Extracted {len(full_text)} characters from {total_pages} pages")

        case_name, citation = self._extract_case_metadata(pdf_path, full_text)
        print(f"  Case: {case_name} {citation}".strip())

        full_text = self._remove_toc_lines(full_text)

        # Reuses the same chunker as statute PDFs. Judgment text won't
        # match the ARTICLE/SECTION/PART/CHAPTER/SCHEDULE header patterns
        # it looks for first, so it naturally falls through to the
        # paragraph-merge fallback branch — which works fine for narrative
        # judgment text without needing separate chunking logic.
        chunks = self._chunk_legal_text(full_text, case_name)
        print(f"  Created {len(chunks)} chunks")

        indexed = 0
        skipped = 0

        # Read the field/value names from the RAG engine itself rather
        # than hardcoding "doc_type"/"case_law" here, so ingestion always
        # stays in sync if those constants ever change on the engine side.
        doc_type_field = self.rag.DOC_TYPE_FIELD
        case_law_type = self.rag.CASE_LAW_TYPE

        for i, chunk in enumerate(chunks):
            text = chunk['text'].strip()

            if len(text) < 100:
                skipped += 1
                continue

            text = self._clean_text(text)

            if len(text) < 100:
                skipped += 1
                continue

            # Prefixed "case_" so IDs can never collide with statute chunk
            # IDs even if a statute and a case-law PDF happened to share
            # the same filename stem.
            doc_id = "case_" + hashlib.md5(f"{pdf_path.stem}_{i}".encode()).hexdigest()[:20]

            try:
                self.rag.collection.add(
                    documents=[text],
                    metadatas=[{
                        "source": case_name,
                        "title": chunk.get('title', case_name),
                        "category": "case_law_pdf",
                        doc_type_field: case_law_type,
                        "case_name": case_name,
                        "citation": citation,
                        "chunk_index": i,
                        "chunk_total": len(chunks),
                        "filename": pdf_path.name
                    }],
                    ids=[doc_id]
                )
                indexed += 1
                if (i + 1) % 50 == 0:
                    print(f"  Indexed {indexed} chunks...")
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower() or "duplicate" in error_msg.lower():
                    skipped += 1
                else:
                    print(f"  Error indexing chunk {i}: {e}")
                    skipped += 1

        print(f"  Indexed: {indexed} chunks, Skipped: {skipped}")
        return indexed > 0

    def _extract_case_metadata(self, pdf_path, full_text):
        """Best-effort extraction of case name + neutral citation, tuned
        against real Kenya Law download filenames, e.g.:

            James_Kariuki_Wagana_v_Republic_2018KEHC6157_KLR_.pdf
            Kittiny_v_Republic__Criminal_Appeal_56of2013__2018KECA851_KLR___8February2018___Judgment_.pdf

        Priority order for the citation:
        1. An explicit "Neutral citation: [...]" line printed on the
           judgment's own first page, if present — this is Kenya Law's own
           authoritative statement of the citation, not something we're
           parsing out of a filename.
        2. The neutral citation fused into the filename itself
           (see _FILENAME_CITATION_PATTERN) — Kenya Law's own generated
           filename, so reliable even when the PDF text doesn't print it.
        3. A generic "[YYYY] eKLR" pattern anywhere on the first page, as a
           last resort.

        The case name is taken from the filename text preceding the
        embedded citation blob, with underscores turned back into spaces.
        This is heuristic, not guaranteed correct — if your downloads use a
        different naming convention than the three files this was tuned
        against, check a few _extract_case_metadata results against the
        actual PDFs and adjust.
        """
        stem = pdf_path.stem

        citation = ""
        neutral_text_match = self._NEUTRAL_CITATION_TEXT_PATTERN.search(full_text[:2000])
        if neutral_text_match:
            citation = neutral_text_match.group(1).strip()

        filename_citation_match = self._FILENAME_CITATION_PATTERN.search(stem)
        if not citation and filename_citation_match:
            year, court, number = filename_citation_match.groups()
            citation = f"[{year}] {court.upper()} {number} (KLR)"

        if not citation:
            eklr_match = self._CITATION_PATTERN.search(full_text[:3000])
            if eklr_match:
                citation = eklr_match.group(0)

        # Case name = everything in the filename before the embedded
        # citation blob (or the whole stem if no citation blob was found),
        # with underscores collapsed back into spaces.
        name_part = stem[:filename_citation_match.start()] if filename_citation_match else stem
        case_name = re.sub(r'_+', ' ', name_part)
        case_name = re.sub(r'\s{2,}', ' ', case_name).strip(' -')

        # Two small readability fixes matching Kenya Law's own display
        # conventions: "2 others" -> "& 2 others" (multi-party appeals are
        # conventionally joined with "&"), and "56of2013" -> "56 of 2013"
        # (appeal number formatting).
        case_name = re.sub(r'\b(\d+)\s+others\b', r'& \1 others', case_name, flags=re.IGNORECASE)
        case_name = re.sub(r'(\d+)of(\d{4})', r'\1 of \2', case_name)

        # Filename didn't give us anything "X v Y"-shaped at all — scan the
        # first page of text for a case-name-shaped line instead.
        if len(case_name) < 5 or not re.search(r'\bv\.?\b|\bvs\.?\b|\bversus\b', case_name, re.IGNORECASE):
            for line in full_text[:3000].split('\n'):
                line = line.strip()
                if self._CASE_NAME_LINE_PATTERN.match(line):
                    case_name = line
                    break

        if not case_name:
            case_name = pdf_path.stem

        return case_name, citation
    
    def _filename_to_title(self, filename):
        """Convert filename like constitution_of_kenya_2010.pdf to Constitution of Kenya 2010"""
        name = filename.replace('.pdf', '')
        name = name.replace('_', ' ')
        name = name.title()
        name = name.replace(' Of ', ' of ').replace(' The ', ' the ').replace(' And ', ' and ')
        name = name.replace(' For ', ' for ').replace(' To ', ' to ').replace(' In ', ' in ')
        name = re.sub(r'\bCap\b', 'Cap', name)
        return name
    
    def _remove_toc_lines(self, text):
        """Remove table of contents lines and page numbers before chunking"""
        # Remove TOC dot-leader lines: "64. Arrest to prevent offences ......... 18"
        text = re.sub(r'\n.+\s*\.{4,}\s*\d+\s*\n', '\n', text)
        
        # Remove lines that are mostly dots
        text = re.sub(r'\n\s*\.{4,}\s*\n', '\n', text)
        
        # Remove "Page X of Y" patterns
        text = re.sub(r'\n\s*Page\s+\d+\s+of\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Remove standalone page numbers on their own line
        text = re.sub(r'\n\s*\d{1,4}\s*\n', '\n', text)
        
        # Remove headers/footers that repeat
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that are just CAPITALS and short (likely headers)
            if line.isupper() and len(line.strip()) < 60:
                continue
            # Skip lines with just "CHAPTER X" without content
            if re.match(r'^\s*CHAPTER\s+[IVX]+\s*$', line, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    # A deliberately LOOSER "bare numbered heading" pattern than anything
    # in the primary `patterns` list in _chunk_legal_text — e.g. matches
    # "296. Punishment of robbery" with no "Section"/"Article" word at
    # all, and doesn't require the primary patterns' stricter boundary
    # conditions (like a literal preceding newline). This is deliberately
    # NOT used as a primary split candidate — used directly against raw
    # full-document text it would false-positive constantly (numbered
    # list items, dates, cross-references, etc.). It's only safe to use
    # as a RECOVERY check inside a chunk that's already been provisionally
    # assigned a single section number: false positives there are rare
    # (real prose containing "NN. Capitalized-word" mid-sentence is
    # uncommon), and the cost of silently missing a real embedded heading
    # (see _recover_embedded_sections below) is high — a full statute
    # section mislabeled under a different section's number entirely.
    _INTERNAL_HEADING_RECOVERY_PATTERN = re.compile(
        r'(?<![\d.\-])\b(\d{1,3}[A-Za-z]?)\.\s+[A-Z][a-z]'
    )

    # A real Kenyan Penal Code section is a paragraph or two. A raw
    # chunk this large has almost certainly swallowed multiple sections
    # due to a missed header boundary (see _chunk_legal_text's docstring)
    # — used as a trigger to run the recovery scan even when the
    # recovery pattern alone doesn't obviously fire, and as a last-resort
    # flag (needs_review) when recovery finds no matches either.
    MAX_TRUSTED_SECTION_CHUNK_SIZE = 3000

    def _recover_embedded_sections(self, text, fallback_section_num):
        """Given text that was provisionally assigned ONE section number
        (fallback_section_num), scans for additional numbered-heading-
        looking positions embedded within it — evidence that the primary
        header-detection pass in _chunk_legal_text missed one or more
        real boundaries and silently merged multiple sections into one
        mislabeled blob (a real, observed failure mode: PDF text
        extraction frequently loses or shifts the newline the primary
        fallback pattern requires immediately before a header, often
        around inline amendment/footnote annotations).

        Returns a list of {'text': ..., 'section_num': ...} pieces. If no
        embedded headers are found, returns a single piece unchanged
        (same text, same fallback_section_num) — this is the common,
        expected case for a genuinely well-formed single-section chunk.

        Each recovered piece is labeled with the section number found AT
        THAT boundary, not the parent's fallback number — so if this
        text actually contains §205 followed by §206, §296, etc. merged
        together, each piece gets its own correct number instead of all
        of them inheriting whichever section was last correctly detected
        before things broke."""
        matches = [
            m for m in self._INTERNAL_HEADING_RECOVERY_PATTERN.finditer(text)
            if m.start() > 5  # skip a match at the very start — that's
                               # just this chunk's own declared heading
        ]

        if not matches:
            return [{'text': text, 'section_num': fallback_section_num}]

        pieces = []
        first_start = matches[0].start()
        leading_text = text[:first_start].strip()
        if leading_text:
            pieces.append({'text': leading_text, 'section_num': fallback_section_num})

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            piece_text = text[start:end].strip()
            if piece_text:
                pieces.append({'text': piece_text, 'section_num': m.group(1)})

        return pieces

    def _extract_section_number(self, header_text):
        """Extract a section/article number from raw header text, trying
        two shapes in order:
        1. Explicitly worded: "Section 296.", "Article 49." (what
           _SECTION_HEADER_NUMBER_PATTERN alone recognized before this
           fix).
        2. Bare numbered, no preceding word at all: "296. Punishment of
           robbery" — this turns out to be how MOST real Penal Code
           section headers are actually formatted in practice. Relying
           only on the worded pattern meant bare-numbered sections never
           got ANY section_number metadata, completely independent of
           whether header-BOUNDARY detection itself succeeded — i.e. this
           was blocking correct tagging even in cases where chunking
           worked perfectly.
        Returns None if neither shape matches."""
        m = self._SECTION_HEADER_NUMBER_PATTERN.search(header_text)
        if m:
            return m.group(1)
        # Titles are always built as f"{doc_name} — {header_text}" (see
        # _chunk_legal_text), so the bare number never actually starts
        # the string — it comes right after the "— " separator. Using
        # re.match here (anchored to true string start) would silently
        # never match any bare-numbered title at all. Searching for the
        # number specifically after "— " (or at true start, for callers
        # that pass a bare header fragment directly rather than a full
        # title) handles both cases without risking a false match on a
        # stray number elsewhere in doc_name itself.
        m2 = re.search(r'(?:—\s*|^)(\d{1,3}[A-Za-z]?)\.\s', header_text)
        if m2:
            return m2.group(1)
        return None

    def _split_section_into_subsections(self, section_text, doc_name, section_num):
        """Split one Section/Article's full body text into per-subsection
        chunks — e.g. Section 296's "(1) Any person who steals... liable
        to imprisonment for seven years. (2) If the offender is armed..."
        becomes two separate chunks, one for (1) and one for (2), instead
        of one chunk containing both offenses' penalties concatenated.

        This is the direct fix for a real observed failure: with the old
        single-chunk-per-section behavior, a query about "robbery with
        violence" (section 296(2), death penalty) would retrieve a chunk
        that ALSO contained 296(1)'s unrelated "seven years" penalty for
        simple robbery sitting right next to it — and the model attached
        the wrong subsection's penalty to its answer. That's not a
        citation-formatting bug; the phrase really was in the retrieved
        text, just attached to the wrong subsection. Splitting subsections
        into separate chunks means a query about (2) can no longer
        retrieve (1)'s text as an undifferentiated blob in the first
        place — fixing this at retrieval instead of trying to detect it
        after generation.

        Returns [] if fewer than 2 subsections are found (nothing to
        split — most Parts/Chapters/Schedules and many simple Sections
        have no numbered subsections at all), signaling the caller should
        keep the original single whole-section chunk as-is.

        Each subsection chunk keeps a short inherited-context prefix
        quoting the start of subsection (1) — subsection (2) alone
        ("If the offender is armed...") doesn't make grammatical or legal
        sense without knowing what offense subsection (1) just defined,
        so a bare split would trade one grounding problem for a
        readability/embedding-quality one. Quoting only the first ~150
        characters keeps this cheap and avoids re-creating the original
        problem (subsection (1)'s specific penalty number reappearing
        inside subsection (2)'s chunk)."""
        parts = self._SUBSECTION_SPLIT_PATTERN.split(section_text)

        subsections = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = self._SUBSECTION_NUMBER_PATTERN.match(part)
            if m:
                subsections.append((m.group(1), part))

        if len(subsections) < 2:
            return []

        first_sub_num, first_sub_text = subsections[0]
        # Truncate on a word boundary so the inherited-context snippet
        # doesn't end mid-word.
        context_snippet = first_sub_text[:150].rsplit(' ', 1)[0].strip()

        chunks = []
        for sub_num, sub_text in subsections:
            label = f"{doc_name} — Section {section_num}({sub_num})"
            if sub_num == first_sub_num:
                enriched_text = f"Section {section_num}({sub_num}): {sub_text}"
                inherited_snippet = None
            else:
                enriched_text = (
                    f"Section {section_num}({sub_num}) of {doc_name} "
                    f"(continuing from subsection ({first_sub_num}): "
                    f"\"{context_snippet}...\"): {sub_text}"
                )
                inherited_snippet = context_snippet
            chunks.append({
                'text': enriched_text,
                'title': label,
                'granularity': 'subsection',
                'section_number': section_num,
                'subsection_number': sub_num,
                # NEW: the exact inherited-context quote embedded in this
                # chunk's text (None for the first subsection, which
                # carries no inherited prefix). Downstream grounding
                # checks (rag_engine.py's _validate_subsection_attribution)
                # need this to avoid a real precision gap: without it, a
                # phrase from subsection (1) that's ONLY present here
                # because it was quoted as inherited context — not because
                # THIS subsection actually states it — would incorrectly
                # validate as "grounded in this subsection." Storing the
                # exact snippet lets that check strip it out before
                # matching, so only this subsection's genuine own content
                # counts as evidence.
                'inherited_context_snippet': inherited_snippet,
            })

        return chunks

    def _chunk_legal_text(self, text, doc_name):
        """Split legal text into meaningful chunks — each containing complete
        provisions at the finest granularity the text's structure actually
        supports.

        Two-level approach:
        1. Split by Section/Article/Part/Chapter/Schedule headers, same as
           before — this identifies each provision's full text block.
        2. For any Section/Article block that contains multiple numbered
           subsections — "(1)", "(2)", etc. — ALSO emit one chunk per
           subsection (via _split_section_into_subsections), alongside the
           original whole-section chunk. Both get indexed: the whole-
           section chunk still serves broad queries ("what is robbery"),
           while the new subsection chunks let a specific query ("robbery
           WITH VIOLENCE") retrieve just the relevant subsection instead
           of a blob containing every subsection's penalty mixed together.

        IMPORTANT CHANGE from the previous version: this no longer merges
        different structural units (different sections, different
        articles) together just to hit a target character count. The old
        "merge adjacent chunks to reach target size (800-2000 chars)" step
        was optimizing purely for chunk size with no awareness that it
        could be fusing two unrelated legal provisions into one chunk —
        which is the same class of problem subsection-splitting fixes,
        just one level up (across whole sections rather than within one).
        Legal provisions should be chunked on legal structure, not a
        target byte count. A short section chunk is still a short, but
        complete and unambiguous, legal unit — the existing 100-character
        minimum-length filter in _ingest_pdf already discards genuinely
        tiny/junk fragments, so there's no need to force-merge short
        legitimate chunks just to make them bigger.

        The size-based merge is still used, unchanged, in the paragraph
        fallback branch below — that path only runs for text with no
        detectable Section/Article/Part/Chapter/Schedule structure at all
        (e.g. this is also what case-law judgment text falls through to,
        via _ingest_case_law_pdf), where there's no legal structure to
        preserve in the first place, so optimizing purely for size is a
        reasonable, low-risk default there."""

        # Patterns for Kenyan legal document structure
        patterns = [
            r'(?:ARTICLE|Article)\s+\d+[\.\:\—\-]',
            r'(?:PART|Part)\s+[IVX]+[\.\:\—\-]',
            r'(?:CHAPTER|Chapter)\s+[IVX\d]+[\.\:\—\-]',
            r'(?:SECTION|Section)\s+\d+[A-Z]?[\.\:\—\-]',
            r'\n\d+[\.\)]\s+[A-Z]',
            r'(?:SCHEDULE|Schedule)\s+[A-Z\d]',
        ]
        
        # Try splitting by section/article boundaries
        best_splits = []
        for pattern in patterns:
            test_splits = re.split(f'({pattern})', text)
            if len(test_splits) > 5:
                best_splits = test_splits
                break
        
        if not best_splits or len(best_splits) < 3:
            # Fallback: split by double newlines and merge. No reliable
            # Section/Article structure was found (typical for case-law
            # judgment text, or a statute PDF whose headers don't match
            # any of the patterns above) — target-size merging is a
            # reasonable default here since there's no legal structure to
            # accidentally destroy.
            paragraphs = re.split(r'\n\s*\n', text)
            chunks = []
            current_chunk = ""
            current_title = doc_name
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If this paragraph looks like a header
                if len(para) < 120 and (para.isupper() or re.match(r'^[\dIVX]+[\.\)]', para)):
                    if len(current_chunk) > 200:
                        chunks.append({'text': current_chunk.strip(), 'title': current_title})
                    current_title = f"{doc_name} — {para[:80]}"
                    current_chunk = para + "\n"
                else:
                    current_chunk += para + "\n"
                    
                    if len(current_chunk) > 1500:
                        chunks.append({'text': current_chunk.strip(), 'title': current_title})
                        current_chunk = ""
                        current_title = doc_name
            
            if len(current_chunk.strip()) > 100:
                chunks.append({'text': current_chunk.strip(), 'title': current_title})
            
            return chunks if chunks else [{'text': text[:2000], 'title': doc_name}]
        
        # Build one raw chunk per structural unit (Section/Article/Part/
        # Chapter/Schedule) — same as before, just no longer merged
        # together afterward.
        raw_chunks = []
        current_title = doc_name
        current_text = ""
        
        for part in best_splits:
            if not part:
                continue
            
            is_header = any(re.match(p, part) for p in patterns)
            
            if is_header:
                # FIX: this used to require > 200 characters before
                # keeping the PREVIOUS section's accumulated text, which
                # was meant to skip empty/near-empty leading preamble
                # before the very first header — but it also silently
                # discarded genuinely short, complete provisions (e.g. a
                # one-sentence section like "Any person who steals
                # anything is guilty of the felony termed theft.", well
                # under 200 chars) every time a new header came along.
                # That's real data loss: whole short sections never made
                # it into the corpus at all. A low threshold (20 chars)
                # still skips true noise/whitespace-only preamble while
                # keeping every real provision, however short.
                if len(current_text.strip()) > 20:
                    raw_chunks.append({
                        'text': current_text.strip(),
                        'title': current_title
                    })
                    current_text = ""
                current_title = f"{doc_name} — {part.strip()[:80]}"
                current_text = part
            else:
                current_text += part
        
        if len(current_text.strip()) > 20:
            raw_chunks.append({
                'text': current_text.strip(),
                'title': current_title
            })

        # For each structural unit, attempt subsection-level splitting.
        # A unit that isn't a numbered Section/Article (e.g. a Part,
        # Chapter, or Schedule header), or one with fewer than 2 detected
        # subsections, is kept as a single whole-unit chunk — same
        # granularity as before. A Section/Article with 2+ subsections
        # gets BOTH the whole-unit chunk (for broad queries) AND one chunk
        # per subsection (for precise queries) — see
        # _split_section_into_subsections' docstring for why both are
        # kept rather than replacing one with the other.
        #
        # BEFORE any of that: run _recover_embedded_sections on every
        # numbered chunk first. The primary header-detection pass above
        # picks ONE pattern for the whole document and never recovers
        # from a missed match (see _recover_embedded_sections' docstring
        # for why real PDF extraction makes this a real, observed
        # failure — not just a theoretical edge case). Without this
        # step, a chunk that silently absorbed several unrelated
        # sections' worth of text would get subsection-split under a
        # single wrong section number, and the real sections buried
        # inside it would never get their own section_number tag at all
        # — exactly what happened before this fix: querying for a real,
        # correctly-ingested section returned zero chunks because its
        # content was sitting, unlabeled, inside a different section's
        # mega-chunk.
        final_chunks = []
        for chunk in raw_chunks:
            declared_section_num = self._extract_section_number(chunk['title'])

            # FIX: recovery used to only run when declared_section_num was
            # already successfully extracted from the chunk's OWN title —
            # meaning a chunk whose own heading was itself unparseable (or
            # one that never matched ANY header pattern at all, e.g. a
            # huge Table-of-Contents-shaped block whose original tabular
            # layout has no real newlines once PDF text extraction
            # flattens it) got skipped entirely: no section_number tag,
            # AND no recovery scan of its body, even when that body
            # contains perfectly recoverable real section text (a real,
            # observed case: a 99,650-character first chunk whose title
            # never advanced past the bare doc_name, containing a ToC
            # preamble followed by real section text — including a
            # complete, substantive "25. Sentence of death" section —
            # that never got tagged because recovery was never even
            # attempted). Running recovery on EVERY chunk, using None as
            # a valid (rather than skip-triggering) fallback value, closes
            # this gap.
            recovered_pieces = self._recover_embedded_sections(
                chunk['text'], declared_section_num
            )

            if len(recovered_pieces) == 1 and len(chunk['text']) <= self.MAX_TRUSTED_SECTION_CHUNK_SIZE:
                if declared_section_num is None:
                    # No number at all, and nothing recoverable inside —
                    # genuinely non-numbered content (Part/Chapter/Schedule
                    # header, or a chunk with no detectable structure).
                    # Same as the original pre-recovery behavior.
                    final_chunks.append({
                        'text': chunk['text'],
                        'title': chunk['title'],
                        'granularity': 'section',
                        'section_number': None,
                        'subsection_number': None,
                    })
                else:
                    # Common case: genuinely a single, well-formed section.
                    section_num = declared_section_num
                    subsection_chunks = self._split_section_into_subsections(
                        chunk['text'], doc_name, section_num
                    )
                    final_chunks.append({
                        'text': chunk['text'],
                        'title': chunk['title'],
                        'granularity': 'section',
                        'section_number': section_num,
                        'subsection_number': None,
                    })
                    final_chunks.extend(subsection_chunks)

            elif len(recovered_pieces) > 1:
                # Recovery found real embedded headers — this chunk had
                # silently absorbed multiple sections (or, when
                # declared_section_num was already None, had absorbed
                # them from the very start with no declared number at
                # all). Re-split, each piece correctly labeled and
                # independently run through subsection-splitting under
                # ITS OWN section number.
                print(
                    f"CHUNK_RECOVERY: '{chunk['title']}' contained "
                    f"{len(recovered_pieces)} embedded section(s) "
                    f"(declared as {declared_section_num!r}) — "
                    f"re-split with corrected labels."
                )
                for piece in recovered_pieces:
                    piece_section_num = piece['section_num']
                    piece_text = piece['text']

                    if piece_section_num is None:
                        # Leading text before the first recoverable
                        # heading, with no declared number to attribute
                        # it to either (e.g. genuine preamble/ToC noise
                        # before real content starts) — kept as a
                        # searchable but untagged fragment rather than
                        # discarded outright, since it may still contain
                        # useful prose; just not subsection-split, since
                        # there's no section number to split it under.
                        final_chunks.append({
                            'text': piece_text,
                            'title': chunk['title'],
                            'granularity': 'section',
                            'section_number': None,
                            'subsection_number': None,
                        })
                        continue

                    final_chunks.append({
                        'text': piece_text,
                        'title': f"{doc_name} — Section {piece_section_num}",
                        'granularity': 'section',
                        'section_number': piece_section_num,
                        'subsection_number': None,
                    })
                    final_chunks.extend(
                        self._split_section_into_subsections(
                            piece_text, doc_name, piece_section_num
                        )
                    )

            else:
                # Oversized (> MAX_TRUSTED_SECTION_CHUNK_SIZE) but recovery
                # found no recognizable embedded headers either — likely
                # still corrupted (real Penal Code sections are a
                # paragraph or two), just in a shape this recovery
                # pattern doesn't catch. Only meaningful to flag when
                # there WAS a declared number to be suspicious of in the
                # first place; an already-untagged oversized chunk has
                # nothing more specific to warn about than "no structure
                # detected," which is already implied by section_number
                # being None.
                final_chunks.append({
                    'text': chunk['text'],
                    'title': chunk['title'],
                    'granularity': 'section_needs_review' if declared_section_num else 'section',
                    'section_number': declared_section_num,
                    'subsection_number': None,
                })

        return final_chunks if final_chunks else [{'text': text[:2000], 'title': doc_name}]
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = text.replace('-\n', '')
        text = text.replace('\n', ' ')
        
        # Remove page numbers
        text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[\d{4}\]\s*', '', text)
        
        # Remove TOC remnants
        text = re.sub(r'\.{3,}\s*\d+', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    def clear_pdf_index(self):
        """Remove all statute PDF-indexed documents from the collection.

        Returns the number of documents actually removed, so callers (see
        reingest_statutes.py) can verify the clear genuinely succeeded
        before re-ingesting on top of it — silently trusting this to have
        worked and then re-ingesting anyway is exactly how old, badly-
        chunked entries end up coexisting with new ones."""
        if not self.rag.is_ready() or not self.rag.collection:
            print("RAG not loaded, cannot clear index")
            return None

        try:
            results = self.rag.collection.get()
            ids_to_delete = []

            for i, metadata in enumerate(results.get('metadatas', [])):
                if metadata and metadata.get('category') == 'kenyan_law_pdf':
                    ids_to_delete.append(results['ids'][i])

            if ids_to_delete:
                self.rag.collection.delete(ids=ids_to_delete)
                print(f"Removed {len(ids_to_delete)} PDF-indexed documents")
            else:
                print("No PDF-indexed documents to remove")

            return len(ids_to_delete)
        except Exception as e:
            # FIX: was print(f"Error clearing PDF index: {e}") only — the
            # same silent-failure pattern fixed elsewhere in this project.
            # A failure here is especially dangerous because the caller
            # would otherwise have no way to know re-ingestion is about to
            # run on top of un-cleared, badly-chunked old data.
            print(f"Error clearing PDF index: {e}")
            return None

    def clear_case_law_index(self):
        """Remove all case-law-indexed documents from the collection.
        Returns the number removed, or None on failure — see
        clear_pdf_index's docstring for why this matters."""
        if not self.rag.is_ready() or not self.rag.collection:
            print("RAG not loaded, cannot clear index")
            return None

        try:
            results = self.rag.collection.get()
            ids_to_delete = []

            for i, metadata in enumerate(results.get('metadatas', [])):
                if metadata and metadata.get('category') == 'case_law_pdf':
                    ids_to_delete.append(results['ids'][i])

            if ids_to_delete:
                self.rag.collection.delete(ids=ids_to_delete)
                print(f"Removed {len(ids_to_delete)} case-law-indexed documents")
            else:
                print("No case-law-indexed documents to remove")

            return len(ids_to_delete)
        except Exception as e:
            # FIX: same silent-failure issue as clear_pdf_index above.
            print(f"Error clearing case law index: {e}")
            return None
