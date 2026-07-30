"""RAG Engine — grounded answers only, real citations, no silent fallbacks.

Key fixes vs the previous version:
1. A distance/similarity threshold on retrieval, so weakly-matching or
   irrelevant chunks are never handed to the model as "context" (this was
   a major source of confident-sounding wrong answers: the model will
   answer fluently from whatever text it's given, even if that text is
   only loosely related to the question).
2. The prompt requires Llama to cite which DOCUMENT NUMBER it used for
   each claim, and the response is parsed to attach the real source
   metadata (filename/title) for those documents back into the final
   answer. If Llama doesn't cite anything, or cites a document number
   that wasn't actually retrieved, we treat that as a failed answer
   rather than returning ungrounded text.
3. No more raw-text dump fallback (_format_raw is gone). Any failure path
   (DB not loaded, no results above threshold, Ollama down, Ollama
   returned something uncitable) now returns one clear, generic
   "unavailable" message to the user and a real, detailed log entry —
   so failures are debuggable from logs without ever leaking internals
   to the end user.
4. All previously-bare `except:` clauses now log the real exception.
5. Real-world case-law EXAMPLES (e.g. actual "robbery with violence"
   convictions pulled from Kenya Law downloads) can now be surfaced
   alongside the statute text. This requires those case files to be
   ingested into the same Chroma collection with metadata
   `doc_type: "case_law"` (plus ideally `case_name` and `citation`) so
   they can be retrieved separately from statute chunks. Examples are
   purely additive — if no case-law docs are loaded/tagged, retrieval
   returns an empty example list and answers behave exactly as before.
   Any case citation the model produces is still validated against what
   was actually retrieved, same as document citations, since inventing a
   case name/citation is a well-known LLM failure mode and is treated as
   a hard grounding failure, not a soft warning.
6. NEW: Hard gate on fabricated PENALTY/SENTENCE claims. A citation
   validator that only checks "did the model point at a real section
   number" does NOT guarantee the specific fact attached to that section
   (e.g. "seven years" vs "death") is actually what the source says. This
   was a real observed failure: the model correctly cited Section 296(2)
   of the Penal Code, but stated the penalty as "seven years" when the
   retrieved text actually specifies a mandatory death sentence (subject
   to the Muruatetu Supreme Court ruling on sentencing discretion) —
   fabricating a materially different, and materially less severe,
   outcome for a serious felony. Since a wrong sentence/penalty number is
   exactly the kind of fact a person might act on, it's treated with the
   same severity as a hallucinated case citation: any specific
   sentence/penalty phrase the model states must appear verbatim in the
   retrieved source text, or the whole answer is rejected outright (see
   _validate_penalty_claims). The prompt now also instructs the model to
   quote penalty language verbatim rather than paraphrase it, which both
   reduces the chance of the model inventing a number and makes this new
   substring check far more reliable.
7. NEW: Generalized the penalty-specific verbatim-quote check (#6) into
   _validate_quoted_claims, which checks EVERY quoted span in an answer,
   not just penalty-shaped ones. The prompt now requires quotes for any
   specific right/rule/threshold/procedure, not just sentences. This
   exists because #6 only catches ONE category of fabrication — real
   observed failures also included an entire unrelated statute
   (forging documents) getting pulled into an answer about inherited
   land, and one Act's content being attributed to a different Act's
   citation. Writing a new hand-crafted regex for each newly observed
   hallucination shape doesn't scale; requiring quotes for anything
   specific and checking ALL of them against the exact document they
   were cited under does. Also: generation temperature dropped from 0.3
   to 0.0. At temp>0 the same question could come back correctly
   grounded on one run and hallucinated on the next purely from
   sampling variance — that's not a content bug this file can fix with
   more validators, it's a determinism bug fixed by not sampling at all
   for factual legal Q&A.
8. NEW: Two general (not query-specific) fixes, aimed at whatever the
   *next* failing query turns out to be rather than the two that were
   already observed:
   - _filter_relevant / _filter_relevant_examples now apply a relative
     RELEVANCE_MARGIN on top of the absolute MAX_RELEVANT_DISTANCE
     ceiling, so relevance self-adjusts to how strong the best match
     for THIS query actually is, instead of relying on a single
     constant hand-tuned from one past query and then silently wrong
     for the next differently-shaped one (observed failure: an
     inheritance query where every retrieved distance was mediocre
     still passed the old absolute-only threshold in full, pulling an
     unrelated Penal Code forgery/debt chunk into the answer).
   - _reconcile_section_numbers repairs a stated "Section N(sub)" using
     the section_number metadata pdf_ingestor.py already records for
     every chunk at ingestion, rather than only flagging a mismatch and
     hoping a retry fixes it. This targets a specific, repeat failure
     mode: llama3.2 kept writing "Section 296(2)" (the textbook-known
     section for robbery with violence) even when the chunk actually
     retrieved and cited was headed "Section 205(2)" — the model
     substituting its own pretrained prior for the document in front of
     it, which a same-model retry can't fix since the prior doesn't
     change between attempts. Because this is driven by stored
     metadata covering the whole corpus, it generalizes to any section
     number mismatch, not just this one offense.
9. NEW: _verify_fact_pattern adds a second, narrow Ollama pass that
   checks whether the answer's CONCLUSION matches the facts the user's
   question actually described — a different failure class than
   anything above. Real observed case: the model correctly quoted BOTH
   the sole-name and joint-name clauses of the Matrimonial Property
   Act, verbatim, correctly cited under real [DOCUMENT N] tags — then
   applied the joint-name clause to a user who explicitly said the
   land was in his name alone. Nothing was ungrounded; every check
   above would have passed this answer. The error was in which
   correctly-quoted rule got applied to the user's stated facts, which
   is a reading-comprehension failure, not a text-fidelity one, so no
   amount of quote/citation/section-number checking can catch it.
   Deliberately scoped narrow (CORRECT/INCORRECT + one sentence, 60
   tokens) rather than asked to rewrite the answer — a 3B model is far
   more reliable at a tight binary check than at open-ended legal
   synthesis, which is the very task that's failing. Gated behind
   _CONDITIONAL_MARKER_PATTERN (2+ conditional markers) so this second
   Ollama call only runs on answers where a fact-pattern mismatch is
   even structurally possible, not on every query.
10. NEW: Two fixes to _verify_fact_pattern after it went to production and
    was observed causing a 100% failure rate on one real query across 5
    attempts (~45 min) — the verifier itself was misreading the question
    or the answer, inconsistently, across separate calls, not just
    rendering a bad verdict on a correct read:
    - The prompt now forces explicit "QUESTION FACT: ... / ANSWER FACT:
      ... / VERDICT: ..." steps instead of asking directly for a verdict,
      so the verifier's read of EACH side is visible and logged, not just
      an opaque judgment — and separating extraction from judgment is
      itself a known way to reduce comparison errors in small models.
    - Architecturally: a fact-pattern mismatch that survives the
      correction retry no longer discards the answer the way a grounding
      issue does. Grounding issues (ungrounded quotes, wrong citations)
      are deterministic — a quote either is or isn't a verbatim substring
      — so blocking on them is warranted. A fact-pattern verdict is a
      judgment call from the same small model, which has already shown it
      can misread either side of the comparison; treating it with equal
      blocking weight took a fully-grounded, plausible answer to zero
      availability. It now degrades to a visible caveat prepended to the
      answer instead, so the (lawyer) user sees the real answer plus an
      explicit "verify this applies to your exact facts" flag, rather
      than nothing at all.
"""

import re
import random
import difflib
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from logic.embeddings import embeddings_service

logger = logging.getLogger("haki_ai.rag_engine")

UNAVAILABLE_MESSAGE = "Haki AI is currently unavailable. Please try again shortly."


class RAGEngine:
    """Database-first RAG engine. RAG retrieves, Llama answers, only from cited sources."""

    # Chroma cosine distance threshold. Lower distance = more similar.
    # Anything retrieved with distance above this is treated as not
    # relevant enough to use as context. This value needs tuning against
    # your actual embedding model/collection — start here and adjust based
    # on real query logs (see _log_retrieval below).
    # Calibrated from real collection data: "rights of arrested persons"
    # and "what to do when arrested by police" both returned genuinely
    # relevant results out to ~1.07-1.08. Set just above that so good
    # embedding matches aren't rejected. Article/section-number exact
    # lookups (see _exact_number_lookup) bypass this threshold entirely
    # via distance=0.0, since they don't rely on embedding similarity.
    MAX_RELEVANT_DISTANCE = 1.1

    # Relative margin, applied ON TOP of the absolute ceiling above. A
    # single static cutoff has to be re-tuned by hand every time a new
    # query phrasing reveals it was wrong — e.g. the "inherited land"
    # query returned distances [0.965, 0.975, 0.998, 1.006, 1.023, ...]:
    # every one of those is under 1.1, so ALL of them passed as
    # "relevant" even though, compared to each other, only the first
    # couple are actually close — the rest are weak matches that just
    # happened to be the least-bad options retrieval had. A margin
    # self-corrects for this per query: when the best match is strong
    # (e.g. 0.50, as "robbery with violence" gets), only genuine
    # near-matches are kept; when the best match is itself only
    # mediocre (e.g. 0.96), nothing further away gets pulled in purely
    # for being under the absolute ceiling. This is what should have
    # kept the unrelated Penal Code forgery/debt chunks out of the
    # inheritance answer, without needing a query-specific fix.
    RELEVANCE_MARGIN = 0.15

    # Minimum number of documents that must pass the distance threshold
    # before we even bother asking the model. If retrieval is this weak,
    # the honest answer is "we don't have this," not a guess from thin context.
    MIN_USABLE_DOCS = 1

    # Separate, independently-tunable threshold for case-law EXAMPLES.
    # Kept as its own constant (rather than reusing MAX_RELEVANT_DISTANCE)
    # because case-law chunks are typically longer and noisier than
    # statute text, so their distance distribution may not match — tune
    # this against your own case-law retrieval logs once you have some.
    MAX_RELEVANT_EXAMPLE_DISTANCE = 1.1

    # Examples are enrichment, not a requirement — an answer is still
    # valid with zero examples attached, so there's no MIN_USABLE_EXAMPLES.
    MAX_EXAMPLES = 2

    # Easy off-switch for the ONE remaining check that costs a full extra
    # Ollama call (the correction retry, the quote/citation/section-number
    # checks are all free -- pure Python against text already generated).
    # Flip to False to skip _verify_fact_pattern entirely if hardware
    # capacity is the binding constraint rather than answer quality --
    # this trades away the fact-pattern safety net, not the (free)
    # grounding checks, which stay active either way.
    ENABLE_FACT_PATTERN_CHECK = True

    # Metadata field/value your ingestion pipeline should set on case-law
    # chunks so they can be retrieved separately from statute text. If
    # your ingestion script already uses different field names, change
    # these two constants to match rather than renaming your data.
    # FIX: these previously said DOC_TYPE_FIELD = "doc_type" and
    # CASE_LAW_TYPE = "case_law" — but pdf_ingestor.py actually writes the
    # metadata field "category" with values "kenyan_law_pdf" / 
    # "case_law_pdf". Chroma's `where` filter on a nonexistent field/value
    # simply matches nothing, so retrieve_examples() has been silently
    # returning zero results this whole time — meaning [CASE N] citations
    # (and their dedicated, stricter validation in
    # _validate_case_citations) never actually fired. Every case
    # reference seen in practice was instead coming through the general,
    # UNFILTERED statute search (doc_type=None), sitting in
    # relevant_results and getting cited as an ordinary [DOCUMENT N] —
    # indistinguishable from real statute text. This is what produced
    # answers attributing statutory text to a court judgment (e.g. "According
    # to [case name], this section states...") instead of the Penal Code
    # itself. These constants now match what's actually stored.
    DOC_TYPE_FIELD = "category"
    CASE_LAW_TYPE = "case_law_pdf"
    STATUTE_TYPE = "kenyan_law_pdf"

    def __init__(self):
        self.collection = None
        self.client = None
        self._loaded = False
        self._loading = False
        self.last_exchange = None
        self.conversation_history = []
        self.ollama_model = "llama3.2:latest"

        self._load_database()

    # ============================================
    # RETRIEVAL
    # ============================================

    # Matches "article 49", "Article 49", "section 36", "sec 36A", etc.
    _ARTICLE_QUERY_PATTERN = re.compile(r'\b(?:article|section|sec)\s+(\d+[A-Za-z]?)\b', re.IGNORECASE)

    def search_database(
        self, query: str, top_k: int = 10,
        doc_type: Optional[str] = None, exclude_doc_type: Optional[str] = None,
    ) -> List[Dict]:
        """
        doc_type=None, exclude_doc_type=None (default) searches
        everything, matching the original behavior. Pass
        doc_type=self.CASE_LAW_TYPE to search ONLY case-law chunks — used
        by retrieve_examples() below. Pass exclude_doc_type=self.CASE_LAW_TYPE
        to search everything EXCEPT case-law chunks — used by
        generate_answer's primary statute search, so a case-law chunk can
        never end up cited as an ordinary [DOCUMENT N] (indistinguishable
        from real statute text) instead of going through the dedicated,
        separately-validated [CASE N] pathway. The two are mutually
        exclusive; doc_type takes priority if both are somehow passed.

        The article/section exact-number lookup only makes sense for
        statute text, so it's skipped whenever a doc_type filter is given,
        and — since _exact_number_lookup scans the whole collection
        directly rather than going through Chroma's query() where-clause —
        its own hits are separately filtered against exclude_doc_type too,
        so that path can't reintroduce the same case-law leakage this
        method is otherwise closing off.
        """
        if not self._loaded or not self.collection:
            logger.error("SEARCH_SKIPPED: collection not loaded")
            return []
        if not embeddings_service.is_ready():
            logger.error("SEARCH_SKIPPED: embeddings_service not ready")
            return []

        # Exact-match fallback: if the query names a specific article/section
        # number, try a direct text match first. Embedding similarity is
        # known to miss these (a query containing "article" may share no
        # vocabulary with the source heading, which often omits "Article"
        # entirely), so this catches the case deterministically rather than
        # hoping the embedding closes the gap. Skipped for filtered
        # (e.g. case-law-only) searches — see docstring above.
        exact_hits = self._exact_number_lookup(query) if doc_type is None else []
        if exclude_doc_type:
            exact_hits = [
                h for h in exact_hits
                if h.get("metadata", {}).get(self.DOC_TYPE_FIELD) != exclude_doc_type
            ]

        try:
            embedding = embeddings_service.encode_single(query)
        except Exception as e:
            logger.error(f"EMBEDDING_FAILED: query='{query}' err={e}")
            return exact_hits

        if doc_type:
            where_clause = {self.DOC_TYPE_FIELD: doc_type}
        elif exclude_doc_type:
            where_clause = {self.DOC_TYPE_FIELD: {"$ne": exclude_doc_type}}
        else:
            where_clause = None

        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_clause,
            )
        except Exception as e:
            logger.error(
                f"CHROMA_QUERY_FAILED: query='{query}' doc_type={doc_type} "
                f"exclude_doc_type={exclude_doc_type} err={e}"
            )
            return exact_hits

        output = []
        try:
            if results and results.get('documents'):
                for i, doc in enumerate(results['documents'][0]):
                    output.append({
                        "document": doc,
                        "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None,
                    })
        except Exception as e:
            logger.error(f"RESULT_PARSE_FAILED: query='{query}' err={e}")
            return exact_hits

        # Merge: exact hits first (treated as distance 0.0 so they always
        # pass the relevance filter), then embedding results, deduped by
        # document text so we don't double-count the same chunk.
        merged = list(exact_hits)
        seen_docs = {r["document"] for r in merged}
        for r in output:
            if r["document"] not in seen_docs:
                merged.append(r)
                seen_docs.add(r["document"])

        self._log_retrieval(query, merged, doc_type=doc_type)
        return merged[:top_k]

    def _exact_number_lookup(self, query: str) -> List[Dict]:
        """
        If the query references a specific article/section number, search
        the raw document text directly for that number immediately
        followed by a clause marker like "(1)" — the pattern Kenyan legal
        text consistently uses to open a numbered article/section. This
        bypasses embedding similarity entirely for this case.
        """
        match = self._ARTICLE_QUERY_PATTERN.search(query)
        if not match:
            return []

        number = match.group(1)
        # Look for "49. (1)" allowing for the heading text right before it,
        # which is how these documents are actually structured (see
        # rag_engine diagnostics: "Rights of arrested persons. 49. (1)...").
        number_pattern = re.compile(
            r'\b' + re.escape(number) + r'\.\s*\(1\)', re.IGNORECASE
        )

        try:
            all_docs = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.error(f"EXACT_LOOKUP_FAILED: query='{query}' number={number} err={e}")
            return []

        hits = []
        for doc, meta in zip(all_docs.get("documents", []), all_docs.get("metadatas", [])):
            if number_pattern.search(doc):
                hits.append({
                    "document": doc,
                    "metadata": meta or {},
                    "distance": 0.0,  # exact match — always passes the relevance filter
                })

        if hits:
            logger.info(f"EXACT_NUMBER_MATCH: query='{query}' number={number} hits={len(hits)}")
        else:
            logger.info(f"EXACT_NUMBER_MATCH: query='{query}' number={number} hits=0 (no exact match found)")

        return hits

    def _log_retrieval(self, query: str, results: List[Dict], doc_type: Optional[str] = None):
        """Log distances so you can tune MAX_RELEVANT_DISTANCE /
        MAX_RELEVANT_EXAMPLE_DISTANCE from real data."""
        distances = [r.get("distance") for r in results]
        label = doc_type or "all"
        logger.info(f"RETRIEVAL: query='{query}' doc_type={label} n={len(results)} distances={distances}")

    def _filter_relevant(self, results: List[Dict]) -> List[Dict]:
        """Drop documents that are too dissimilar to be trustworthy
        context. Two stages: first the absolute ceiling (a hard backstop
        so a query with no good matches at all doesn't pull in results
        just because they're clustered together), then a relative
        margin off the best match actually returned for THIS query —
        see RELEVANCE_MARGIN's docstring for why the margin is what
        actually generalizes across query types."""
        within_ceiling = [
            r for r in results
            if r.get("distance") is not None and r["distance"] <= self.MAX_RELEVANT_DISTANCE
        ]
        if not within_ceiling:
            return []
        best = min(r["distance"] for r in within_ceiling)
        return [r for r in within_ceiling if r["distance"] <= best + self.RELEVANCE_MARGIN]

    def _filter_relevant_examples(self, results: List[Dict]) -> List[Dict]:
        """Same two-stage idea as _filter_relevant (absolute ceiling,
        then relative margin off the best match), but against the
        example-specific ceiling.

        Selection among the qualifying pool is RANDOM, not "always the
        closest," when there are more relevant examples than MAX_EXAMPLES
        can show. This is deliberately a data-selection choice, not a
        generation one: it exists so the same query doesn't always cite
        the identical case example every single time, without touching
        Ollama's temperature at all. Randomizing generation itself was
        the original source of this session's very first bug (the same
        question coming back correctly grounded on one run and
        hallucinated on the next) — temperature is staying at 0.0.
        Randomizing WHICH already-vetted, already-relevant document gets
        shown is a completely different axis: every candidate in
        `relevant` has already passed the same ceiling + margin filter
        as before, so variety only happens among options that were
        already good matches — this can't surface a weak or irrelevant
        example, only vary which of several comparably strong ones is
        used.

        The final selection is still sorted by distance for citation
        order, so which examples get PICKED varies across calls, but how
        they're PRESENTED within a single answer stays relevance-ordered."""
        within_ceiling = [
            r for r in results
            if r.get("distance") is not None and r["distance"] <= self.MAX_RELEVANT_EXAMPLE_DISTANCE
        ]
        if not within_ceiling:
            return []
        best = min(r["distance"] for r in within_ceiling)
        relevant = [r for r in within_ceiling if r["distance"] <= best + self.RELEVANCE_MARGIN]

        if len(relevant) <= self.MAX_EXAMPLES:
            return sorted(relevant, key=lambda r: r.get("distance", 999))

        selected = random.sample(relevant, self.MAX_EXAMPLES)
        return sorted(selected, key=lambda r: r.get("distance", 999))

    def retrieve_context(self, query: str, top_k: int = 10) -> List[Dict]:
        return self.search_database(query, top_k)

    def retrieve_examples(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve illustrative real-world case-law examples (e.g. actual
        "robbery with violence" convictions) tagged doc_type='case_law' at
        ingestion time. Returns [] gracefully — including if no case-law
        documents have been ingested/tagged at all yet — since examples
        are an enrichment, never a requirement for a valid answer."""
        results = self.search_database(query, top_k=top_k, doc_type=self.CASE_LAW_TYPE)
        return self._filter_relevant_examples(results)

    # ============================================
    # GENERATION
    # ============================================

    def check_ollama_available(self, timeout: int = 5) -> bool:
        """
        Fast health check — confirms the Ollama server is reachable and the
        configured model is actually loaded/known, BEFORE committing to a
        full generation request. This exists because a slow or overloaded
        Ollama (or a model that got unloaded from memory after its idle
        keep-alive window) will otherwise silently eat the full 120s
        generation timeout on every single query before failing, with no
        earlier signal that something's wrong.

        This does not guarantee a subsequent generate call won't time out
        (the model could still be slow to respond, or get unloaded again
        between this check and the real call), but it catches the common
        case fast: Ollama not running at all, wrong port, or model not
        pulled.
        """
        import requests

        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        except Exception as e:
            logger.error(f"OLLAMA_HEALTH_CHECK_FAILED: cannot reach server err={e}")
            return False

        if not r.ok:
            logger.error(f"OLLAMA_HEALTH_CHECK_FAILED: bad status={r.status_code}")
            return False

        try:
            models = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            logger.error(f"OLLAMA_HEALTH_CHECK_FAILED: bad response body err={e}")
            return False

        if not any(self.ollama_model in m for m in models):
            logger.error(
                f"OLLAMA_HEALTH_CHECK_FAILED: model '{self.ollama_model}' not found. "
                f"Available models: {models}. Run: ollama pull {self.ollama_model}"
            )
            return False

        return True

    def _ask_ollama(
        self, context: str, query: str, doc_count: int,
        example_count: int = 0, correction_notes: Optional[str] = None,
    ) -> Optional[str]:
        import requests

        if example_count:
            example_rules = f"""
- Below the numbered documents you'll also find {example_count} REAL-WORLD
  CASE EXAMPLE(S), labeled [CASE 1], [CASE 2], etc. These are real,
  previously decided cases provided for illustration only — they are NOT
  the primary legal basis for your answer.
- If (and only if) one of the provided case examples is clearly relevant
  to the question, you may briefly mention it to illustrate the answer
  with a concrete real-world example (e.g. "For example, in [CASE 1], a
  person was convicted of..."). Cite it exactly as [CASE N].
- Do NOT invent, name, or reference any case that is not one of the
  provided [CASE N] examples. If none of the provided examples are
  actually relevant, simply don't mention a case at all — do not force one in.
"""
        else:
            example_rules = """
- No real-world case examples were provided for this question. Do NOT
  invent or reference any case name, citation, or court decision — answer
  from the legal text above only.
"""

        # Only populated on the bounded self-correction retry (see
        # generate_answer) — tells the model exactly which numbers/phrases
        # from its PREVIOUS answer didn't check out against the retrieved
        # text, so it can fix or drop just those instead of the whole
        # answer being thrown away for one bad detail.
        correction_block = ""
        if correction_notes:
            correction_block = f"""
IMPORTANT — CORRECTION NEEDED: A previous attempt at this answer had the
following problem(s), which must be fixed:
{correction_notes}
If a problem is about a claim that could not be verified against the
legal text above, do not repeat it unless you can find it verbatim in
the text above — remove it rather than guessing at a replacement. If a
problem is about the conclusion not matching the facts the user
described, re-read the user's question carefully and make sure your
final answer applies the correct scenario to the facts they actually
gave, not a different one discussed in the same source text.
"""

        prompt = f"""You are a Kenyan legal assistant helping someone understand the law in plain language.

LEGAL TEXT FROM DATABASE:
{context}

USER QUESTION: {query}

Explain the answer the way a knowledgeable person would talk to someone
who isn't a lawyer — in your own words, in full sentences, with a brief
lead-in before any list of rights or steps. You can quote short exact
phrases (in quotation marks) when the precise wording matters, but don't
just paste the statute as a wall of clauses with no explanation around it.
If there's a list (e.g. several rights or sub-clauses), it's fine to
present it as a short list, but introduce it first and keep the wording
natural rather than copying the legal numbering verbatim for every item.

Rules you must still follow:
- Base your answer ONLY on the legal text above. Do not add outside
  knowledge, and do not guess or fill gaps.
- Cite the document number you used for each main claim, in the format
  [DOCUMENT N] — for example [DOCUMENT 1]. Put the citation at the end of
  the sentence it supports, not mixed into the middle of a quote.
- There are exactly {doc_count} documents provided (numbered 1 to {doc_count}).
  If you reference a document outside that range, say so plainly rather
  than presenting it as if it were available.
{example_rules}- If the provided text doesn't actually answer the question, say exactly:
  "The retrieved legal text does not address this question."
- Mention the specific article/section number when the text gives one,
  but don't just be the article number — explain what it actually means.
- Whenever you state a specific PENALTY, SENTENCE LENGTH, TIME LIMIT,
  MONETARY AMOUNT, or a specific RULE/RIGHT/CONDITION that determines the
  outcome of the person's situation (e.g. "seven years", "within 14 days",
  "rebuttable presumption that the property is held in trust", "sentenced
  to death"), you MUST quote the exact words from the source text in
  quotation marks, copied verbatim character-for-character — not
  paraphrased, rounded, averaged, or restated with different wording or a
  different number than what the source text literally says. Every
  quoted span you write will be checked against the specific document you
  just cited it under, so only quote text you can see in that document
  above. If the source text does not state something specific, do not
  invent it — say plainly that the text doesn't specify that detail.
- Finish your answer properly; do not stop mid-sentence or mid-list.
- Keep your full answer under 150 words. Be concise — cover the main
  point and cite it, rather than listing every sub-clause. If there are
  multiple relevant provisions, mention the most important one or two
  rather than all of them.
- State only what the cited text directly says. Do NOT draw a legal
  conclusion, outcome, or remedy that the text doesn't explicitly state —
  for example, do not say a right being violated "means you can ask for
  a case dismissal" or "entitles you to compensation" unless the text
  itself says that. If the person's real question (e.g. what happens
  next, what remedy is available) goes beyond what the text states,
  say plainly that the text doesn't cover that specific consequence, and
  that they should ask a licensed advocate about it.
{correction_block}
YOUR ANSWER:"""

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Deterministic generation. This is factual legal
                        # Q&A, not creative writing — sampling at temp>0 is
                        # exactly why the same question could come back
                        # correctly grounded on one run and hallucinated on
                        # the next (see the "robbery with violence" and
                        # "inherited land" transcripts). temp=0 costs
                        # nothing extra on the same hardware/model.
                        "temperature": 0.0,
                        "num_predict": 250,
                        "num_ctx": 4096,
                        "stop": ["---", "Legal information"]
                    }
                },
                timeout=240
            )
        except Exception as e:
            logger.error(f"OLLAMA_REQUEST_FAILED: query='{query}' err={e}")
            return None

        if not r.ok:
            logger.error(f"OLLAMA_BAD_STATUS: query='{query}' status={r.status_code} body={r.text[:300]}")
            return None

        try:
            response_json = r.json()
            text = response_json.get("response", "")
            done_reason = response_json.get("done_reason", "")
        except Exception as e:
            logger.error(f"OLLAMA_JSON_PARSE_FAILED: query='{query}' err={e}")
            return None

        if not text or len(text.strip()) < 20:
            logger.warning(f"OLLAMA_EMPTY_RESPONSE: query='{query}'")
            return None

        text = text.strip()

        if done_reason == "length":
            # Generation hit num_predict before finishing naturally. Trim
            # back to the last complete sentence so the user sees a clean
            # ending rather than a mid-word/mid-sentence cutoff (e.g. "...or
            # to dis"). We keep the citations that already landed rather
            # than discarding the whole answer, since a truncated-but-cited
            # answer is still useful — see _validate_citations, which
            # already accepts this as long as a real citation appears
            # before the cut.
            logger.warning(f"OLLAMA_TRUNCATED: query='{query}' raw_len={len(text)}")
            text = self._trim_to_last_complete_sentence(text)

        return text

    def _verify_fact_pattern(self, query: str, answer: str) -> Optional[str]:
        """Second, narrow pass on the SAME small model — not "write a
        correct legal answer" (the open-ended synthesis task that's
        actually the unreliable one) but "does this already-written
        answer's conclusion match the fact pattern the user described."
        That's a much easier, closer-to-binary-classification task, and
        a 3B model is far more likely to get a yes/no comprehension
        check right than open-ended legal reasoning.

        Targets a failure class the grounding validators structurally
        can't see: a real observed case had the model correctly quote
        BOTH the sole-name clause and the joint-name clause of the
        Matrimonial Property Act, verbatim, correctly cited — then
        apply the joint-name clause to a user who explicitly said the
        land was in his name alone. Every quote was real; the citation
        was real; the conclusion was still wrong. No text-fidelity
        check catches that, because there's no ungrounded text — the
        error is in which real, correctly-quoted rule got applied to
        the user's stated facts, not in the text itself.

        Only called when the ANSWER contains 2+ conditional markers
        (branching language a conclusion could be misapplied within) AND
        the QUESTION itself contains a personal-scenario indicator (a
        pronoun/possessive) — both gates have to pass. The answer-side
        gate alone isn't enough: real law often explains a topic with
        "if armed... if not..." structure even when the user's question
        is a bare topic lookup with no personal facts in it at all, and
        checking "does the answer match the question's facts" against a
        question that has no facts produces near-arbitrary mismatches
        (observed in production on "robbery with violence" — a topic
        query with zero personal scenario — flagged INCORRECT twice, for
        two different, unrelated reasons each time). This doesn't add a
        second Ollama call to every query, only the subset where a
        fact-pattern mismatch is even a coherent thing to check for.

        Fails open: if the verifier call itself errors, times out, or
        returns something unparseable, this returns None (no issue
        raised) rather than blocking an otherwise-fine answer on a
        flaky verifier response — the point is to catch an extra class
        of error, not to add a new way for the pipeline to give up."""
        if not self.ENABLE_FACT_PATTERN_CHECK:
            return None
        if len(self._CONDITIONAL_MARKER_PATTERN.findall(answer)) < 2:
            return None
        if not self._SCENARIO_INDICATOR_PATTERN.search(query):
            return None

        import requests

        prompt = f"""USER QUESTION: {query}

ANSWER GIVEN: {answer}

The answer above may describe more than one condition or scenario (for
example: "if acquired alone... but if acquired jointly..."). Follow
these steps IN ORDER, copying only what is actually stated, not what
you assume or infer:

STEP 1: In a few words, what specific circumstance does the USER'S
QUESTION describe? (e.g. "land in one spouse's name alone")
STEP 2: In a few words, what specific circumstance does the ANSWER's
final conclusion apply to?
STEP 3: If STEP 1 and STEP 2 describe the SAME circumstance, the
verdict is CORRECT. If they describe DIFFERENT circumstances, the
verdict is INCORRECT.

Reply in EXACTLY this format, nothing else:
QUESTION FACT: ...
ANSWER FACT: ...
VERDICT: CORRECT or INCORRECT"""

        try:
            r = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Still a narrow check, not a second full answer —
                        # the extra room over the original 60 is for the
                        # two short extracted-fact lines (STEP 1/STEP 2),
                        # not for open-ended reasoning. Forcing those two
                        # lines out explicitly, instead of asking directly
                        # for a verdict, is a real fix, not padding: it
                        # makes the verifier's read of EACH side visible
                        # and loggable, so a wrong verdict can be traced to
                        # "misread the question" vs "misread the answer"
                        # instead of being one opaque judgment call — and
                        # forcing the extraction step before the verdict
                        # is itself a known way to reduce comprehension
                        # errors in small models on comparison tasks.
                        "temperature": 0.0,
                        "num_predict": 100,
                        "num_ctx": 2048,
                    },
                },
                timeout=120,
            )
        except Exception as e:
            logger.error(f"FACT_PATTERN_CHECK_FAILED: query='{query}' err={e}")
            return None

        if not r.ok:
            logger.error(f"FACT_PATTERN_CHECK_BAD_STATUS: query='{query}' status={r.status_code}")
            return None

        try:
            verdict_text = r.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"FACT_PATTERN_CHECK_PARSE_FAILED: query='{query}' err={e}")
            return None

        if not verdict_text:
            return None

        match = re.search(r'\b(INCORRECT|CORRECT)\b', verdict_text, re.IGNORECASE)
        if not match:
            logger.warning(f"FACT_PATTERN_CHECK_UNPARSEABLE: query='{query}' raw='{verdict_text[:150]}'")
            return None

        q_fact_match = re.search(r'QUESTION FACT:\s*(.+)', verdict_text, re.IGNORECASE)
        a_fact_match = re.search(r'ANSWER FACT:\s*(.+)', verdict_text, re.IGNORECASE)
        q_fact = q_fact_match.group(1).strip() if q_fact_match else "?"
        a_fact = a_fact_match.group(1).strip() if a_fact_match else "?"
        # Trim trailing content that spilled from the next line, in case
        # the model didn't cleanly newline-separate the three fields.
        a_fact = re.split(r'VERDICT:', a_fact, flags=re.IGNORECASE)[0].strip()

        if match.group(1).upper() == "CORRECT":
            logger.info(f"FACT_PATTERN_OK: query='{query}' question_fact='{q_fact}' answer_fact='{a_fact}'")
            return None

        reason = f"the question describes \"{q_fact}\" but the answer's conclusion applies to \"{a_fact}\""
        logger.warning(f"FACT_PATTERN_MISMATCH: query='{query}' question_fact='{q_fact}' answer_fact='{a_fact}'")
        return f"- The answer's conclusion may not match the specific facts in the question: {reason}"

    @staticmethod
    def _trim_to_last_complete_sentence(text: str) -> str:
        """Cut trailing incomplete text back to the last sentence-ending
        punctuation, so a truncated answer still reads as a finished
        thought rather than stopping mid-word."""
        last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_end == -1 or last_end < len(text) * 0.5:
            # No good sentence boundary found, or it would cut away more
            # than half the answer — better to keep the original text than
            # risk gutting a short, otherwise-fine response.
            return text
        return text[:last_end + 1].strip()

    # ============================================
    # CITATION VALIDATION
    # ============================================

    def _reconcile_section_numbers(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> str:
        """Auto-correct a stated "Section N(sub)" citation to wherever its
        quoted claim actually lives in ingestion metadata, instead of only
        flagging a mismatch and hoping a retry fixes it.

        Observed failure this targets: llama3.2 kept writing "Section
        296(2)" — the textbook-known section for robbery with violence —
        even when the chunk that actually contains "sentenced to death"
        is headed "Section 205(2)" in this corpus, and did so
        consistently across many retries. That's the model substituting
        its own pretrained prior for the document actually in front of
        it; a same-model retry can't fix that, since the prior doesn't
        change between attempts. pdf_ingestor.py already records the real
        section_number/subsection_number for every chunk, so the number
        the model TYPES doesn't need to be trusted at all — only where
        its quoted claim (see _validate_quoted_claims / the prompt's
        verbatim-quote requirement) is actually found.

        Deliberately mirrors _validate_subsection_attribution's approach
        rather than tracking [DOCUMENT N] citation order: association is
        by "most recently stated NUM(SUB) citation" to "next quoted span",
        the same reading-order pattern real prose uses (a citation, then
        the quote it's introducing) — NOT by which [DOCUMENT N] happens
        to appear nearby, since that citation can trail the section
        number in the same sentence (e.g. "Section 296(2) of the Penal
        Code [DOCUMENT 1], ...") and a forward-only DOCUMENT-tracking
        approach would miss it entirely.

        Only corrects when the quote is found verbatim in exactly one
        OTHER statute chunk with a different, non-empty section_number —
        never invents a number, never touches an already-correct
        citation, and skips case-law-tagged chunks (a case's sentencing
        outcome correcting a statute citation would be its own, different
        kind of error, already caught by _validate_subsection_attribution
        rather than silently rewritten here)."""
        all_chunks = list(results) + list(examples or [])
        if not all_chunks:
            return answer

        events = []
        for m in self._SECTION_SUBSECTION_PATTERN.finditer(answer):
            events.append((m.start(), "citation", m))
        for m in self._QUOTED_SPAN_PATTERN.finditer(answer):
            events.append((m.start(), "quote", m.group(1)))
        events.sort(key=lambda e: e[0])

        current_citation = None  # the regex Match for the most recent "NUM(SUB)"
        replacements = []  # (start, end, new_text, old_text), keyed to the citation span

        for _, kind, payload in events:
            if kind == "citation":
                current_citation = payload
                continue

            # kind == "quote"
            if current_citation is None:
                continue

            quote = re.sub(r'\s+', ' ', payload).strip().lower()
            if len(quote) < 4:
                continue

            claimed_section, claimed_subsection = current_citation.group(1), current_citation.group(2)

            found_in_claimed = False
            found_elsewhere = None
            for chunk in all_chunks:
                meta = chunk.get("metadata", {}) or {}
                if meta.get(self.DOC_TYPE_FIELD) == self.CASE_LAW_TYPE:
                    continue
                doc_text = chunk.get("document", "").lower()
                if not self._fuzzy_quote_match(quote, doc_text):
                    continue

                chunk_section = str(meta.get("section_number") or "")
                chunk_subsection = str(meta.get("subsection_number") or "")
                if not chunk_section:
                    continue  # whole-section merged chunk — not precise enough to correct off of

                if chunk_section == claimed_section and (not chunk_subsection or chunk_subsection == claimed_subsection):
                    found_in_claimed = True
                    break

                found_elsewhere = (chunk_section, chunk_subsection or claimed_subsection)

            if found_in_claimed or not found_elsewhere:
                continue

            new_section, new_subsection = found_elsewhere
            start, end = current_citation.start(), current_citation.end()
            replacements.append((
                start, end, f"{new_section}({new_subsection})",
                f"{claimed_section}({claimed_subsection})",
            ))
            # Don't correct the same citation span twice off a second quote.
            current_citation = None

        if not replacements:
            return answer

        corrected = answer
        for start, end, new_text, old_text in sorted(replacements, key=lambda r: -r[0]):
            corrected = corrected[:start] + new_text + corrected[end:]
            logger.info(f"SECTION_NUMBER_AUTOCORRECTED: {old_text} -> {new_text} (from ingestion metadata)")

        return corrected

    # Matches the prompted format [DOCUMENT N], plus the looser variants
    # Llama actually produces in practice (e.g. "DOCUMENT 1 [SOURCE NAME]",
    # "Document 1:", "(Document 1)"). We validate on the document NUMBER
    # being present and in range, not on exact bracket placement.
    _CITATION_PATTERN = re.compile(r"document\s+(\d+)", re.IGNORECASE)

    # Matches the prompted format for real-world case examples, [CASE N],
    # with the same loose-variant tolerance as _CITATION_PATTERN above.
    _CASE_CITATION_PATTERN = re.compile(r"case\s+(\d+)", re.IGNORECASE)

    # Matches a real article/clause-style citation like "49(1)(a)(i)" or
    # "Article 49(1)" — if Llama cites the actual legal provision number
    # instead of (or alongside) a document index, that's also evidence of
    # grounding and shouldn't be rejected just because it skipped the
    # "DOCUMENT N" phrasing.
    _CLAUSE_CITATION_PATTERN = re.compile(r"\b\d{1,3}\s*\(\d+\)")

    # Same shape as _CLAUSE_CITATION_PATTERN above but WITH capturing
    # groups for the base section number and the subsection number
    # separately — e.g. "296(2)" -> ("296", "2"). Used by
    # _validate_subsection_attribution to check a claim against the
    # SPECIFIC subsection cited, not just the base section. Kept as a
    # separate pattern rather than adding groups to
    # _CLAUSE_CITATION_PATTERN itself, since that one is used purely as
    # an existence check (re.search) all over this file and mixing groups
    # in wouldn't change behavior there, but keeping them as two clearly-
    # named patterns makes each call site's intent unambiguous.
    _SECTION_SUBSECTION_PATTERN = re.compile(r'\b(\d{1,3}[A-Za-z]?)\s*\((\d+[A-Za-z]?)\)')

    # Rough sentence splitter used only to scope "which subsection number
    # is this specific penalty claim near" — doesn't need to be perfect,
    # just needs to keep a citation and the claim it supports in the same
    # segment in the common case (citation at the end of the sentence it
    # backs, matching what the prompt asks for).
    _SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')

    # Civil Procedure Rules (and other subsidiary legislation) cite
    # provisions as "Order 25, rule 1" / "Order 25 rule 1(2)" rather than
    # the "49(1)" style the Constitution uses. A correct answer citing
    # this format was previously rejected purely because it didn't match
    # either pattern — see the "steps for withdrawing case" truncation
    # case, where "Order 25, rule 1" was a genuine, correct citation.
    _ORDER_RULE_CITATION_PATTERN = re.compile(
        r"\bOrder\s+\d+[A-Za-z]?\s*,?\s*rule\s+\d+", re.IGNORECASE
    )

    # Matches specific sentence/penalty claims the model tends to
    # fabricate or mix up between similar-but-distinct provisions (e.g.
    # borrowing the 14-year term from "simple robbery" while citing the
    # "robbery with violence" section, or inventing "seven years" out of
    # thin air). Covers both spelled-out and numeral year counts, plus the
    # "death sentence" / "sentenced to death" phrasing.
    _PENALTY_CLAIM_PATTERN = re.compile(
        r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|'
        r'thirty|forty|fifty|\d{1,3})\s+years?\b'
        r'|\bdeath\s+sentence\b|\bsentenced\s+to\s+death\b',
        re.IGNORECASE,
    )

    # Any span the model puts in quotation marks is, by definition, a claim
    # of exact wording — the prompt now requires this for ANY specific
    # right/rule/threshold/procedure, not just penalties (see _ask_ollama).
    # This backs _validate_quoted_claims, the general-purpose replacement
    # for writing a new category-specific regex (penalty, subsection, ...)
    # every time a new hallucination shape shows up in production. Length
    # floor of 4 chars filters stray punctuation artifacts without
    # excluding short but real quoted terms.
    _QUOTED_SPAN_PATTERN = re.compile(r'"([^"]{4,300})"')

    # Cheap, deterministic pre-filter for _verify_fact_pattern below — no
    # model call, just a regex count. See that method's docstring for why
    # 2+ conditional markers, not just 1, gates whether the (paid)
    # verifier pass is even worth running.
    _CONDITIONAL_MARKER_PATTERN = re.compile(
        r'\bif\b|\bunless\b|\bwhereas\b|\bon the other hand\b|\bhowever\b|'
        r'\bprovided that\b|\bin contrast\b',
        re.IGNORECASE,
    )

    # Second gate for _verify_fact_pattern, alongside the conditional-
    # marker one above. A bare topic lookup ("robbery with violence",
    # "adverse possession") has no personal scenario in it at all — there
    # is nothing for the answer's conclusion to be checked against, so
    # asking "does the answer match the question's facts" is an ill-posed
    # comparison that produces near-arbitrary mismatches depending on how
    # the verifier happens to paraphrase that call (observed in
    # production: two separate MISMATCH verdicts on the same "robbery
    # with violence" query, each citing a different, unrelated paraphrase
    # of the answer as the supposed conflicting "fact"). Personal
    # pronouns/possessives are a cheap, general signal that a question
    # actually describes a scenario with a determinable fact pattern,
    # rather than being a keyword/topic search.
    _SCENARIO_INDICATOR_PATTERN = re.compile(
        r'\b(i|my|me|mine|we|our|us|ours|he|she|him|her|his|hers|they|them|'
        r'their|theirs|you|your|yours)\b',
        re.IGNORECASE,
    )

    def _extract_cited_doc_numbers(self, answer: str) -> List[int]:
        return [int(n) for n in self._CITATION_PATTERN.findall(answer)]

    def _extract_cited_case_numbers(self, answer: str) -> List[int]:
        return [int(n) for n in self._CASE_CITATION_PATTERN.findall(answer)]

    def _validate_citations(self, answer: str, doc_count: int) -> bool:
        """
        An answer is considered grounded if any of these hold:
        - it explicitly says the text doesn't address the question, OR
        - it contains AT LEAST ONE "document N" reference with N in range
          (accepts "[DOCUMENT 1]", "DOCUMENT 1 [SOURCE]", "Document 1:",
          etc.), OR
        - it contains at least one clause-style citation like "49(1)(a)"
          (evidence the model is quoting specific provision numbers from
          the source text, not generic outside knowledge), OR
        - it contains an "Order N, rule N" style citation (Civil Procedure
          Rules and similar subsidiary legislation use this format instead
          of the Constitution's "49(1)" style).

        Note: we deliberately do NOT require every cited number to be in
        range. Llama sometimes references a document number that wasn't
        retrieved while honestly flagging it (e.g. "[DOCUMENT 6] is not
        provided, but according to [DOCUMENT 1], ...") and still answers
        correctly from a valid document. That kind of self-correction is
        evidence of good grounding, not a reason to reject — what matters
        is whether at least one real, in-range citation backs the answer.

        Case citations ([CASE N]) are deliberately NOT part of this check
        — they're validated separately and strictly in generate_answer,
        since a hallucinated case example is a much higher-risk failure
        mode than a hallucinated document reference (it can fabricate an
        entire real-sounding precedent), so it's rejected outright rather
        than treated as one signal among several. Penalty/sentence claims
        are likewise validated separately and strictly — see
        _validate_penalty_claims — for the same reason: a wrong number
        there is a fact someone could act on, not just a formatting slip.
        """
        if "does not address this question" in answer.lower():
            return True

        cited = self._extract_cited_doc_numbers(answer)
        if any(1 <= n <= doc_count for n in cited):
            return True

        if self._CLAUSE_CITATION_PATTERN.search(answer):
            return True

        if self._ORDER_RULE_CITATION_PATTERN.search(answer):
            return True

        return False

    def _validate_case_citations(self, answer: str, example_count: int) -> bool:
        """Strict, separate gate for [CASE N] references: every cited case
        number must fall within the examples actually retrieved and handed
        to the model. Unlike document citations, there's no tolerance for
        a self-corrected out-of-range reference here — inventing a case
        name/citation is exactly the kind of confident-sounding fabrication
        this whole engine exists to prevent, so any out-of-range or
        unexpected (example_count == 0 but a case is cited anyway)
        reference fails the answer rather than being logged as a soft
        warning."""
        cited = self._extract_cited_case_numbers(answer)
        if not cited:
            return True
        if example_count == 0:
            return False
        return all(1 <= n <= example_count for n in cited)

    def _validate_section_citations(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """Hard-gate check: returns the list of explicit "Article N" /
        "Section N" / "Sec. N" numbers the model cited that do NOT appear
        (that exact number, on a word boundary) anywhere in the text that
        was actually placed in context — an empty list means the answer
        is fully grounded on this check.

        This catches a specific, higher-risk failure mode than what
        _validate_citations checks: a model can name a real-*looking*,
        in-range-*feeling* section number that was simply never in the
        retrieved text at all — including naming TWO DIFFERENT numbers
        for what's presented as the same offense in the same answer
        (e.g. "section 297(2)" in one sentence, "section 296(2)" two
        sentences later). _validate_citations only confirms *a* citation
        pattern exists somewhere in the answer; it can't tell a correct
        number from a fabricated one.

        IMPORTANT: checks against BOTH `results` (statute chunks) AND
        `examples` (case-law chunks) — the model is explicitly permitted
        to cite a section number while discussing a case example, and
        that number often only appears in the case text, not the statute
        chunk. Checking `results` alone was a bug: it would falsely flag
        a real, correctly-grounded citation just because it happened to
        come from the case-law side of the context rather than the
        statute side.

        Matching is on the bare number only (e.g. "296"), not the full
        "Section 296" phrase or "296(2)" clause, and stays lenient about
        surrounding formatting (source documents may render the heading
        as "296. (1)" with no word "Section" at all) so it doesn't reject
        valid answers over formatting differences — it only flags when
        the number itself is nowhere in the retrieved text (statute or
        case-law) that was shown to the model.

        Returning the mismatched numbers (rather than just True/False)
        lets the caller attempt a targeted, bounded self-correction pass
        instead of discarding a possibly-otherwise-correct answer outright
        — see generate_answer, which retries once with these numbers
        called out explicitly before giving up."""
        named_numbers = set(self._NAMED_REFERENCE_PATTERN.findall(answer))
        if not named_numbers:
            return []

        examples = examples or []
        combined_source_text = " ".join(r.get("document", "") for r in results)
        combined_source_text += " " + " ".join(r.get("document", "") for r in examples)

        mismatched = []
        for n in named_numbers:
            if not re.search(r'\b' + re.escape(n) + r'\b', combined_source_text, re.IGNORECASE):
                mismatched.append(n)

        if mismatched:
            logger.warning(f"UNGROUNDED_SECTION_CITATION: numbers={sorted(mismatched)}")

        return mismatched


    def _validate_penalty_claims(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """Returns the list of penalty/sentence phrases (e.g. "seven
        years", "sentenced to death") the answer states that do NOT
        appear, near-verbatim, anywhere in the text actually placed in
        context — an empty list means the answer is fully grounded on
        this check.

        Citing a real, in-range section number does NOT guarantee the
        specific penalty attached to it in the answer is what the source
        text actually says — the model can correctly point at Section
        296(2) while still inventing or mixing in a penalty number from a
        different provision entirely (observed failure: "seven years"
        stated for an offence whose retrieved text specifies a death
        sentence).

        Checks against BOTH `results` (statute chunks) and `examples`
        (case-law chunks), since a penalty phrase like "sentenced to
        fifteen years" can legitimately come from a case's sentencing
        discussion rather than the statute text itself.

        No claim at all (the answer doesn't state a specific sentence
        length) returns [] trivially — there's nothing to check, and
        plenty of valid answers won't mention a penalty at all.

        Returning the mismatched phrases (rather than True/False) lets
        the caller attempt a targeted, bounded self-correction pass
        before discarding a possibly-otherwise-correct answer — see
        generate_answer."""
        claims = self._PENALTY_CLAIM_PATTERN.findall(answer)
        if not claims:
            return []

        examples = examples or []
        combined_source_text = re.sub(r'\s+', ' ', " ".join(
            r.get("document", "") for r in results
        ) + " " + " ".join(
            r.get("document", "") for r in examples
        )).lower()

        mismatched = []
        for match in self._PENALTY_CLAIM_PATTERN.finditer(answer):
            phrase = re.sub(r'\s+', ' ', match.group(0)).strip().lower()
            if phrase not in combined_source_text:
                mismatched.append(phrase)

        if mismatched:
            logger.warning(f"UNGROUNDED_PENALTY_CLAIM: phrases={mismatched}")

        return mismatched

    def _fuzzy_quote_match(self, quote: str, text: str, threshold: float = 0.88, min_fuzzy_length: int = 40) -> bool:
        """True if `quote` appears in `text` verbatim, OR — ONLY for
        quotes at least `min_fuzzy_length` characters long — as a
        near-exact match tolerant of the kind of small copy drift a 3B
        local model actually produces when "quoting": a dropped/added
        comma, an article ("a"/"the"), a swapped connective word.

        This exists because a hard exact-substring check turned out to
        be too strict for what llama3.2 can reliably reproduce
        character-for-character: real observed case — a phrase that had
        appeared verbatim, confirmed correct, across multiple earlier
        sessions — was rejected purely over a comma the model added
        when repeating it.

        Scoring approach: find the single best-aligned position in
        `text` for `quote` via difflib's longest-match anchor, THEN score
        the ratio against a window at that anchor sized to the quote
        itself (not a padded window). This two-step anchor-then-score
        was a deliberate fix for a real false rejection: an earlier
        padded-window version compared the quote against windows LONGER
        than the quote, so whenever the source sentence legitimately
        continued past where the quote stopped (e.g. quote ends
        "...held in trust for the other spouse." but the real clause
        continues "...spouse for the duration of the marriage unless...",
        with no period there at all), that extra trailing real text — 
        which the quote was never claiming to reproduce — inflated the
        comparison length and dragged the ratio down (0.876 in the
        observed case, just under the 0.88 threshold) even though the
        quote was accurate for everything it actually claimed. Scoring
        against a same-length window anchored at the real alignment
        point removes that penalty for stopping where the quote stops,
        while a materially different claim still fails, since a wrong
        number or wrong right changes characters WITHIN the quote's own
        length, which this approach is just as sensitive to as before.

        The length gate is NOT cosmetic — it's the difference between
        "this fixes drift/truncation" and "this lets a wrong penalty
        through": a short quote like "seven years" scores a dangerously
        high similarity ratio against "eleven years" (they share most of
        their characters), because a similarity SCORE can't distinguish
        "same fact, drifted ending" from "different fact, similar
        spelling" once the string is short. Below the length gate, only
        an exact match counts, same as before this fix.

        Only worth the anchor search for reasonably short TEXT spans, so
        this stays cheap — no model call, just string comparison."""
        if not quote or not text:
            return False
        if quote in text:
            return True
        if len(quote) < min_fuzzy_length:
            return False

        # Step 1: find where quote best aligns in text at all, via the
        # single longest contiguous matching block. This just locates a
        # good anchor point -- it does not itself decide pass/fail.
        matcher = difflib.SequenceMatcher(None, quote, text, autojunk=False)
        match = matcher.find_longest_match(0, len(quote), 0, len(text))
        if match.size == 0:
            return False

        # Step 2: score against a window in `text` anchored at that match,
        # sized to the quote itself (not padded longer) -- so a source
        # sentence that simply continues past the quote's own length
        # doesn't get compared against text the quote never claimed to
        # reproduce. anchor_start is where the matching block's start in
        # `text` implies the quote as a whole should begin.
        anchor_start = max(0, match.b - match.a)
        segment = text[anchor_start:anchor_start + len(quote)]
        return difflib.SequenceMatcher(None, quote, segment, autojunk=False).ratio() >= threshold

    def _validate_quoted_claims(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """General-purpose successor to _validate_penalty_claims: instead
        of only checking phrases that match a pre-written "looks like a
        penalty" regex, this checks EVERY span the model put in quotation
        marks — which the prompt now requires for any specific right,
        rule, threshold, or procedure, not just sentences/penalties.

        This is the fix for the whack-a-mole problem: previously, every
        newly observed hallucination shape (a fabricated property-law
        rule, a forging-related tangent, a misattributed procedural step)
        needed its own hand-written pattern before it could be caught.
        Because the model is now required to quote anything specific, any
        wrong "fact in quotes" is caught the same way regardless of what
        category it falls into — including ones never seen in production
        yet.

        Like _validate_subsection_attribution, this tracks the most
        recently mentioned citation ([DOCUMENT N], [CASE N]) in reading
        order and checks each quote against THAT specific chunk first —
        catching a quote that's real but attributed to the wrong source
        (e.g. text that only appears in the Matrimonial Property Act
        chunk, quoted right after a "[DOCUMENT N]" citation that actually
        points at the Penal Code chunk). If no citation has appeared yet
        before a given quote, falls back to checking it against the whole
        combined context, so an answer that quotes before citing isn't
        unfairly rejected.

        Returns the list of unverifiable quotes (empty = fully grounded),
        following the same "return mismatches, not bool" convention as
        the other validators so generate_answer can surface them in a
        single targeted self-correction retry."""
        quotes = self._QUOTED_SPAN_PATTERN.findall(answer)
        if not quotes:
            return []

        examples = examples or []
        combined_text = re.sub(r'\s+', ' ', " ".join(
            r.get("document", "") for r in results
        ) + " " + " ".join(
            r.get("document", "") for r in examples
        )).lower()

        events = []
        for m in self._CITATION_PATTERN.finditer(answer):
            events.append((m.start(), "doc", int(m.group(1))))
        for m in self._CASE_CITATION_PATTERN.finditer(answer):
            events.append((m.start(), "case", int(m.group(1))))
        for m in self._QUOTED_SPAN_PATTERN.finditer(answer):
            events.append((m.start(), "quote", m.group(1)))
        events.sort(key=lambda e: e[0])

        mismatches = []
        current_source_text = None  # text of the most recently cited chunk, or None

        for _, kind, payload in events:
            if kind == "doc":
                idx = payload - 1
                current_source_text = (
                    results[idx].get("document", "").lower()
                    if 0 <= idx < len(results) else None
                )
                continue
            if kind == "case":
                idx = payload - 1
                current_source_text = (
                    examples[idx].get("document", "").lower()
                    if 0 <= idx < len(examples) else None
                )
                continue

            # kind == "quote"
            quote = re.sub(r'\s+', ' ', payload).strip().lower()
            if len(quote) < 4:
                continue

            if current_source_text and self._fuzzy_quote_match(quote, current_source_text):
                continue
            if self._fuzzy_quote_match(quote, combined_text):
                continue
            mismatches.append(quote[:120])

        if mismatches:
            logger.warning(f"UNGROUNDED_QUOTED_CLAIM: phrases={mismatches}")

        return mismatches

    def _validate_subsection_attribution(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """Returns descriptions of penalty/sentence claims that ARE
        grounded somewhere in the retrieved text (so _validate_penalty_claims
        alone would pass them) but are attributed to the WRONG place —
        either the wrong subsection, the wrong section entirely, or a
        genuine STATUTE claim ("Section X states...") that's actually only
        backed by CASE-LAW text (a specific court's sentencing outcome in
        one case, misrepresented as the general statutory rule).

        Compares against ALL retrieved chunks (statute AND case-law), not
        just same-section statute chunks, so this catches both:
        - citing an entirely different section number for the right
          content (e.g. "Section 205(2)" for what's actually 296(2)'s
          death-penalty text — 205 is a real, unrelated section that
          happened to also be in context, so the base-number check in
          _validate_section_citations passed it: "205" genuinely exists
          in the retrieved text, just not attached to this claim);
        - stating "Section 296(2) states [X years]" when [X years] is
          really a specific court's sentencing decision found only in a
          case-law chunk, not the statute itself.

        For each penalty/sentence phrase in the answer, associates it with
        the MOST RECENTLY MENTIONED "NUM(SUBNUM)" citation that appears
        before it in reading order — NOT restricted to the same sentence.
        This matters because real answers routinely split a citation from
        its elaboration across sentence boundaries (e.g. "...as defined
        under Section 205(2) of the Penal Code. According to this
        section, if an offender is armed... he shall be sentenced to
        death." — the citation and the penalty phrase it's introducing are
        two separate sentences, but obviously refer to each other). An
        earlier, same-sentence-only version of this check completely
        missed exactly this pattern — the citation and the phrase it was
        supposed to catch were one sentence apart, so it silently skipped
        it. Tracking the "current citation in force" across sentence
        boundaries (reset only when a new citation appears) is how a
        human reader would parse this prose, and is what the check needs
        to match.

        For each phrase, searches every chunk for it:
        - A chunk whose section_number+subsection_number metadata exactly
          matches the citation AND isn't case-law-tagged -> grounded, no
          issue.
        - Otherwise, if found in a DIFFERENT statute section/subsection ->
          flag the actual location.
        - Otherwise, if found ONLY in case-law-tagged chunk(s) -> flag as
          a statute claim that's really only backed by case-specific text
          (a court's sentencing decision in one case, not the general
          statutory rule).
        - A whole-section ('section'-granularity, no subsection_number)
          chunk is never used as contradicting evidence either way — it
          can't disambiguate any more precisely than the claim itself,
          so asserting a mismatch off of it would risk false positives.
        This means the check only fires when precise, unambiguous
        contradicting evidence exists, not merely on absence of a match
        (absence alone is already _validate_penalty_claims' job)."""
        all_chunks = list(results) + list(examples or [])

        # Collect every citation match and every penalty-phrase match
        # across the WHOLE answer, each tagged with its start position,
        # then walk them in reading order so we always know which
        # citation was most recently announced when we hit a phrase.
        events = []
        for m in self._SECTION_SUBSECTION_PATTERN.finditer(answer):
            events.append((m.start(), "citation", m.groups()))
        for m in self._PENALTY_CLAIM_PATTERN.finditer(answer):
            phrase = re.sub(r'\s+', ' ', m.group(0)).strip().lower()
            events.append((m.start(), "phrase", phrase))
        events.sort(key=lambda e: e[0])

        mismatches = []
        current_citation = None  # (section, subsection) or None

        for _, kind, payload in events:
            if kind == "citation":
                current_citation = payload
                continue

            # kind == "phrase"
            if current_citation is None:
                # No citation has appeared yet in the answer at all —
                # nothing to check this phrase's attribution against.
                continue

            claimed_section, claimed_subsection = current_citation
            phrase = payload

            found_in_claimed = False
            found_in_other_statute = None  # (section, subsection)
            found_only_in_case_law = False

            for chunk in all_chunks:
                meta = chunk.get("metadata", {}) or {}
                doc_text = chunk.get("document", "")
                inherited_snippet = meta.get("inherited_context_snippet") or ""
                if inherited_snippet:
                    # Exclude quoted-from-a-different-subsection text
                    # before matching — otherwise a phrase that's ONLY
                    # present here because it was quoted as inherited
                    # context (see pdf_ingestor.py's
                    # _split_section_into_subsections) would incorrectly
                    # validate as grounded in THIS subsection, when really
                    # it only proves the OTHER subsection (whichever one
                    # the snippet was quoted from) says it.
                    doc_text = doc_text.replace(inherited_snippet, "")
                doc_text = doc_text.lower()

                if phrase not in doc_text:
                    continue

                is_case_law = meta.get(self.DOC_TYPE_FIELD) == self.CASE_LAW_TYPE
                chunk_section = str(meta.get("section_number") or "")
                chunk_subsection = str(meta.get("subsection_number") or "")

                if is_case_law:
                    found_only_in_case_law = True
                    continue

                if not chunk_section or not chunk_subsection:
                    # Whole-section merged chunk — ambiguous, not usable
                    # as contradicting evidence either way.
                    continue

                if chunk_section == claimed_section and chunk_subsection == claimed_subsection:
                    found_in_claimed = True
                    break

                found_in_other_statute = (chunk_section, chunk_subsection)

            if found_in_claimed:
                continue

            if found_in_other_statute:
                other_sec, other_sub = found_in_other_statute
                mismatches.append(
                    f"\"{phrase}\" attributed to Section {claimed_section}"
                    f"({claimed_subsection}), but only found in Section "
                    f"{other_sec}({other_sub})"
                )
            elif found_only_in_case_law:
                mismatches.append(
                    f"\"{phrase}\" presented as what Section {claimed_section}"
                    f"({claimed_subsection}) states, but this phrase only "
                    f"appears in case-law text — likely a specific court's "
                    f"sentencing decision in one case, not the general statute"
                )

        if mismatches:
            logger.warning(f"SUBSECTION_ATTRIBUTION_MISMATCH: {mismatches}")

        return mismatches


    def _attach_sources(self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None) -> str:
        """Replace [DOCUMENT N] markers with the real, human-readable source
        name pulled from metadata, and [CASE N] markers with the real case
        name/citation, then append source/example lists at the end.

        If the answer instead grounds itself via a clause-style citation
        ("49(1)(a)") or an Order/rule citation ("Order 25, rule 1") rather
        than a "[DOCUMENT N]" marker, we can't map that citation to a
        specific result by index — the model didn't tell us which one it
        used. In that case we fall back to listing every source that was
        actually placed in context, since all of them were genuinely
        available to ground the answer, rather than silently showing no
        source list at all."""
        examples = examples or []
        sources_used = set()
        examples_used = set()

        def replace_doc(match):
            n = int(match.group(1))
            idx = n - 1
            if 0 <= idx < len(results):
                source = results[idx].get('metadata', {}).get('source', 'Unknown source')
                sources_used.add(source)
                return f"[{source}]"
            return match.group(0)

        def replace_case(match):
            n = int(match.group(1))
            idx = n - 1
            if 0 <= idx < len(examples):
                meta = examples[idx].get('metadata', {})
                case_name = meta.get('case_name') or meta.get('source', 'Unknown case')
                citation = meta.get('citation', '')
                label = f"{case_name} {citation}".strip()
                examples_used.add(label)
                return f"[{label}]"
            return match.group(0)

        rewritten = self._CITATION_PATTERN.sub(replace_doc, answer)
        rewritten = self._CASE_CITATION_PATTERN.sub(replace_case, rewritten)

        if not sources_used and (
            self._CLAUSE_CITATION_PATTERN.search(answer)
            or self._ORDER_RULE_CITATION_PATTERN.search(answer)
        ):
            for r in results:
                source = r.get('metadata', {}).get('source')
                if source:
                    sources_used.add(source)

        if sources_used:
            source_list = "\n".join(f"- {s}" for s in sorted(sources_used))
            rewritten += f"\n\nSources cited:\n{source_list}"

        if examples_used:
            example_list = "\n".join(f"- {s}" for s in sorted(examples_used))
            rewritten += f"\n\nCase examples cited:\n{example_list}"

        return rewritten

    # ============================================
    # MAIN ENTRY POINT
    # ============================================

    def generate_answer(self, query: str, chat_messages: List[Dict] = None) -> str:
        if not self._loaded:
            self.wait_for_loading(30)
        if not self._loaded:
            logger.error("RAG_DB_NOT_LOADED: collection failed to initialize in time")
            return UNAVAILABLE_MESSAGE

        # Fail fast (≈5s) if Ollama is unreachable or the model isn't
        # loaded, rather than discovering this after a 120s generation
        # timeout. This is a best-effort check, not a guarantee — see
        # check_ollama_available's docstring.
        if not self.check_ollama_available():
            logger.error(f"OLLAMA_UNAVAILABLE: query='{query}' skipping generation attempt")
            return UNAVAILABLE_MESSAGE

        if chat_messages:
            try:
                self._sync_history(chat_messages)
            except Exception as e:
                logger.error(f"SYNC_HISTORY_FAILED: err={e}")

        # Case law is deliberately excluded from this primary search now
        # that DOC_TYPE_FIELD/CASE_LAW_TYPE actually match what's stored
        # (see the constants' docstring above). Before that fix, this
        # call searched everything unfiltered, so a case judgment could
        # out-rank the real statute text for a given query and get cited
        # as an ordinary [DOCUMENT N] — indistinguishable from the Penal
        # Code itself, and exactly how a case ended up credited with
        # "this section states..." in place of the actual statute.
        # Case-law relevance is still fully available — just only through
        # retrieve_examples() below, as [CASE N], with its own dedicated
        # (and stricter) validation.
        raw_results = self.search_database(query, top_k=8, exclude_doc_type=self.CASE_LAW_TYPE)
        if not raw_results:
            logger.warning(f"NO_RAW_RESULTS: query='{query}'")
            return UNAVAILABLE_MESSAGE

        relevant_results = self._filter_relevant(raw_results)
        if len(relevant_results) < self.MIN_USABLE_DOCS:
            logger.warning(
                f"NO_RELEVANT_RESULTS: query='{query}' "
                f"best_distance={min((r.get('distance') for r in raw_results if r.get('distance') is not None), default=None)}"
            )
            return UNAVAILABLE_MESSAGE

        # Cap context to the 5 most relevant after filtering, to control
        # prompt size and keep the model focused on the strongest matches.
        relevant_results = sorted(relevant_results, key=lambda r: r.get("distance", 999))[:5]

        # Real-world case-law examples, retrieved separately from the
        # statute search above via a doc_type filter — see
        # retrieve_examples(). This is a distinct, filtered query rather
        # than just reusing raw_results so an example doesn't have to
        # out-rank the statute text in the same unfiltered search to be
        # surfaced. Purely additive: [] here just means no examples
        # section gets built, the rest of the flow is unaffected.
        example_results = self.retrieve_examples(query, top_k=8)

        context = self._build_context(relevant_results, example_results)
        answer = self._ask_ollama(
            context, query,
            doc_count=len(relevant_results),
            example_count=len(example_results),
        )

        if not answer:
            logger.error(f"OLLAMA_NO_ANSWER: query='{query}'")
            return UNAVAILABLE_MESSAGE

        # Repair any section number the model retyped incorrectly using
        # the ground truth already captured at ingestion, BEFORE running
        # grounding validation — a mismatch fixed here is one fewer
        # answer that needs a retry or gets discarded. See
        # _reconcile_section_numbers for why this generalizes beyond any
        # single offense/section.
        answer = self._reconcile_section_numbers(answer, relevant_results)

        grounding_issues = self._collect_grounding_issues(
            answer, relevant_results, example_results
        )

        # Tracked separately from grounding_issues on purpose: this checks
        # whether the answer's CONCLUSION matches the facts the user
        # actually described, not whether its text is real — a different
        # failure class the grounding checks structurally can't see (see
        # _verify_fact_pattern's docstring). It's also a fundamentally
        # LOWER-CONFIDENCE signal than grounding_issues: a quote either is
        # or isn't a verbatim substring of the source (deterministic), but
        # "does this small model's second call correctly compare two fact
        # patterns" is itself a judgment call that can be wrong — observed
        # in production: the verifier misread which side of the
        # comparison said what, differently, across separate attempts on
        # an unchanged question. Treating that judgment with the same
        # blocking weight as a deterministic grounding failure is what
        # took this exact query to a 100% failure rate across 5 real
        # attempts and ~45 minutes before this fix. See below: a
        # persistent grounding issue still blocks the answer; a
        # persistent fact-pattern issue degrades to a visible caveat
        # instead, so an uncertain-but-plausible answer still reaches the
        # user rather than being silently replaced with nothing.
        fact_pattern_issue = self._verify_fact_pattern(query, answer)

        if fact_pattern_issue and not grounding_issues:
            # A fact-pattern-only mismatch's outcome doesn't depend on
            # retrying: whether the retry "fixes" it or not, the answer
            # still ends up returned with the same caveat either way (see
            # the persistent-grounding-issues / had_fact_pattern_issue
            # logic below). So spending a full extra Ollama call here
            # buys nothing — it can only ever arrive at the same place
            # this shortcut reaches immediately. On hardware where even
            # ONE generation call is not reliably completing inside its
            # timeout, a needless second one is not a marginal cost:
            # requiring two independent calls to both succeed multiplies
            # the failure probability rather than adding to it (observed
            # directly — the primary generation succeeded, and it was
            # SPECIFICALLY the retry, triggered only by this check, that
            # then timed out and lost the whole answer).
            logger.warning(
                f"FACT_PATTERN_ISSUE_SKIP_RETRY (attaching caveat directly): query='{query}'"
            )
            return (
                "⚠️ Please double-check this answer applies to your exact facts "
                "(an automated check could not confirm the scenario below matches "
                "your question precisely) — verify against the cited section directly "
                "or with a licensed advocate.\n\n" + answer
            )

        if grounding_issues:
            had_fact_pattern_issue = bool(fact_pattern_issue)
            # Give the model ONE bounded chance to fix exactly the flagged
            # numbers/phrases/fact-pattern mismatch before deciding what to
            # do next. This exists because a hard, single-shot reject is
            # blunt: source chunking can legitimately split a heading from
            # its body across two retrieved chunks, so a real, correct
            # number can occasionally look "unverified" purely from where
            # the text got cut, even though the model saw it. A correction
            # retry costs one extra Ollama call, and only on the failure
            # path (which should be the minority of queries), not on every
            # query — so it doesn't meaningfully change typical latency.
            all_issues = grounding_issues + ([fact_pattern_issue] if fact_pattern_issue else [])
            logger.warning(
                f"GROUNDING_ISSUES_FOUND (attempt 1, retrying): query='{query}' "
                f"issues={all_issues} raw_answer='{answer[:200]}'"
            )
            correction_notes = "\n".join(all_issues)
            answer = self._ask_ollama(
                context, query,
                doc_count=len(relevant_results),
                example_count=len(example_results),
                correction_notes=correction_notes,
            )
            if not answer:
                logger.error(f"OLLAMA_NO_ANSWER_ON_RETRY: query='{query}'")
                return UNAVAILABLE_MESSAGE

            answer = self._reconcile_section_numbers(answer, relevant_results)

            grounding_issues = self._collect_grounding_issues(
                answer, relevant_results, example_results
            )
            # Deliberately NOT re-running _verify_fact_pattern here. Worst
            # case, a single query was costing up to 4 sequential Ollama
            # calls (generate, fact-check, retry-generate, retry-fact-
            # check), each potentially minutes long on this hardware —
            # observed directly compounding into Ollama becoming fully
            # unresponsive (a 3s health-check itself timing out). Reusing
            # whether attempt 1 flagged a fact-pattern issue, rather than
            # spending a second verifier call to re-confirm, cuts that to
            # 3. The tradeoff is deliberately conservative in the safe
            # direction: if attempt 1 flagged it, the caveat gets applied
            # to the retry's answer even if the retry happened to fix the
            # mismatch — an occasional unnecessary caveat costs nothing
            # but a sentence of extra caution; a fourth Ollama call costs
            # minutes of latency this system doesn't reliably have.

            if grounding_issues:
                # A deterministic text-fidelity problem (ungrounded quote,
                # wrong citation) survived the correction attempt — this
                # is where we actually give up, same as before: these are
                # high-confidence, verifiable-as-wrong problems, not a
                # judgment call, so blocking the answer is the right call.
                logger.warning(
                    f"GROUNDING_ISSUES_PERSIST (attempt 2, giving up): query='{query}' "
                    f"issues={grounding_issues} raw_answer='{answer[:200]}'"
                )
                return UNAVAILABLE_MESSAGE

            if had_fact_pattern_issue:
                # Everything text-fidelity related checks out; the only
                # open question is the lower-confidence fact-pattern
                # judgment from attempt 1, which we're not re-spending a
                # call to re-verify (see above). Don't discard a
                # plausible, fully-grounded answer over that — surface
                # the uncertainty to the (lawyer) user instead of hiding
                # the answer entirely.
                logger.warning(
                    f"FACT_PATTERN_ISSUE_CARRIED_FORWARD (attempt 2, returning with caveat): "
                    f"query='{query}'"
                )
                answer = (
                    "⚠️ Please double-check this answer applies to your exact facts "
                    "(an automated check could not confirm the scenario below matches "
                    "your question precisely) — verify against the cited section directly "
                    "or with a licensed advocate.\n\n" + answer
                )

        if "does not address this question" in answer.lower():
            logger.info(f"HONEST_NO_ANSWER: query='{query}'")
            return ("The legal documents available don't address this question directly. "
                    "Please consult a licensed Kenyan advocate for guidance.")

        # Soft, log-only check for language that asserts a legal
        # conclusion/remedy beyond what the cited text states (e.g. "you
        # can ask for a case dismissal"). Article/Section number
        # grounding used to also be checked here, log-only — it's now a
        # hard gate above (_collect_grounding_issues), so this call only
        # covers the conclusion-language heuristic.
        self._check_number_grounding(query, answer, relevant_results)

        # Same log-only treatment, same reasoning: surfaces unquoted
        # claims with weak textual support against their cited chunk for
        # monitoring, without costing an Ollama call or risking a false
        # positive blocking a fine answer. See _audit_unquoted_claims.
        self._audit_unquoted_claims(answer, relevant_results, example_results)

        final_answer = self._attach_sources(answer, relevant_results, example_results)
        self._save_exchange(query, final_answer)
        return final_answer + "\n\n---\nLegal information, not legal advice."

    _AUDIT_STOPWORDS = frozenset({
        'the', 'a', 'an', 'and', 'or', 'but', 'of', 'to', 'in', 'on', 'at', 'by', 'for',
        'with', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'its',
        'that', 'this', 'these', 'those', 'which', 'who', 'whom', 'their', 'they', 'he',
        'she', 'his', 'her', 'if', 'than', 'then', 'so', 'not', 'no', 'from', 'into',
        'under', 'over', 'shall', 'may', 'must', 'can', 'will', 'would', 'about', 'also',
        'such', 'any', 'all', 'has', 'have', 'had', 'when', 'where', 'while', 'one',
        'according', 'however', 'therefore', 'thus', 'because', 'since', 'upon', 'within',
        'between', 'sources', 'cited', 'legal', 'information', 'advice',
    })

    def _audit_unquoted_claims(
        self, answer: str, results: List[Dict], examples: Optional[List[Dict]] = None
    ) -> List[str]:
        """LOG-ONLY diagnostic. Deliberately NOT wired into
        _collect_grounding_issues or the retry/blocking pipeline —
        returns findings purely for logging/monitoring, never feeds a
        retry, a caveat, or UNAVAILABLE_MESSAGE.

        Why log-only rather than another blocking check: this whole
        session has been about the gap it targets — a real, verified
        example (the Mwaura "duplex charge" claim) was accurate but had
        ZERO quoted spans, so _validate_quoted_claims had nothing to
        check and the claim's correctness was never actually verified by
        this pipeline, only by us reading the source PDF by hand
        afterward. But the person who asked for this fix ALSO just
        identified, correctly, that guardrails costing an extra Ollama
        call are directly hurting availability on hardware that's
        already failing most single calls outright. A blocking version
        of this check would add exactly that cost, on exactly the
        answers already burning the most retries — the opposite of what
        was asked for. This version costs nothing beyond string
        comparison: it exists so the pattern that took manual PDF
        cross-referencing to catch here can instead be *seen* in the
        logs going forward, at zero latency cost, so it can be promoted
        to a real (Ollama-costing) check later if capacity allows and if
        watching real findings shows it's not too noisy to trust.

        Approach: for each sentence attributed to a [DOCUMENT N] or
        [CASE N] citation (same reading-order "most recent citation"
        tracking as the other validators) that contains NO quoted span
        (already-quoted content is _validate_quoted_claims's job), scores
        what fraction of the sentence's significant words literally
        appear in that citation's own chunk text. Low coverage doesn't
        prove a hallucination — legitimate synthesis across multiple
        chunks, or a fair paraphrase, can also score low — which is
        exactly why this stays log-only rather than being trusted to
        block anything on its own."""
        examples = examples or []
        split_matches = list(self._SENTENCE_SPLIT_PATTERN.finditer(answer))
        starts = [0] + [m.end() for m in split_matches]
        ends = [m.start() for m in split_matches] + [len(answer)]
        sentences = [(s, e, answer[s:e]) for s, e in zip(starts, ends) if e > s]

        events = []
        for m in self._CITATION_PATTERN.finditer(answer):
            events.append((m.start(), "doc", int(m.group(1))))
        for m in self._CASE_CITATION_PATTERN.finditer(answer):
            events.append((m.start(), "case", int(m.group(1))))
        events.sort(key=lambda e: e[0])

        findings = []
        event_idx = 0
        current_kind, current_idx = None, None

        for s_start, s_end, sentence_text in sentences:
            while event_idx < len(events) and events[event_idx][0] <= s_end:
                _, current_kind, current_idx = events[event_idx]
                event_idx += 1

            if current_kind is None or self._QUOTED_SPAN_PATTERN.search(sentence_text):
                continue

            words = re.findall(r"[a-zA-Z']{3,}", sentence_text.lower())
            significant = [w for w in words if w not in self._AUDIT_STOPWORDS]
            if len(significant) < 4:
                continue

            source_list = results if current_kind == "doc" else examples
            idx = current_idx - 1
            if not (0 <= idx < len(source_list)):
                continue
            chunk_words = set(re.findall(r"[a-zA-Z']{3,}", source_list[idx].get("document", "").lower()))

            covered = sum(1 for w in significant if w in chunk_words)
            coverage = covered / len(significant)
            if coverage < 0.5:
                findings.append(
                    f"coverage={coverage:.2f} {current_kind}{current_idx} sentence='{sentence_text.strip()[:150]}'"
                )

        if findings:
            logger.info(f"UNQUOTED_CLAIM_AUDIT: {len(findings)} low-overlap unquoted sentence(s): {findings}")

        return findings


    def _collect_grounding_issues(
        self, answer: str, relevant_results: List[Dict], example_results: List[Dict]
    ) -> List[str]:
        """Run every grounding check and return a list of human-readable
        issue descriptions (empty list = fully grounded). Centralizing
        this lets generate_answer run the exact same validation twice —
        once on the original answer, once on the corrected retry — without
        duplicating the check logic, and lets the retry prompt be built
        from a single combined list covering every problem found at once
        (rather than surfacing issues one gate at a time)."""
        # FIX: an honest "the retrieved text doesn't cover this" refusal
        # must bypass EVERY check below, not just _validate_citations
        # (which already had this exact bypass, from the very first
        # version of this pipeline). The newer checks added since then —
        # _validate_section_citations, _validate_penalty_claims,
        # _validate_subsection_attribution — were bolted on without
        # inheriting it, so an honest refusal that happens to restate the
        # user's own requested-but-unavailable section number (e.g. "I
        # couldn't find Section 297; the text only covers up to 205") was
        # getting flagged as an "ungrounded citation" and rejected twice
        # in a row — punishing the model for correctly admitting it
        # doesn't have the answer, and forcing a pointless retry loop
        # that just repeats the same correct refusal. This is the exact
        # kind of self-correction _validate_citations already recognized
        # as good behavior; the other checks just weren't told about it.
        if "does not address this question" in answer.lower():
            return []

        issues = []

        if not self._validate_citations(answer, doc_count=len(relevant_results)):
            issues.append(
                "- No valid, in-range document or section/clause citation "
                "was found backing the answer at all."
            )

        if not self._validate_case_citations(answer, example_count=len(example_results)):
            issues.append(
                "- A [CASE N] reference was used that doesn't correspond to "
                "any of the real case examples actually provided."
            )

        bad_sections = self._validate_section_citations(
            answer, relevant_results, example_results
        )
        if bad_sections:
            issues.append(
                "- These cited section/article numbers could not be verified "
                f"in the provided legal text: {', '.join(sorted(bad_sections))}."
            )

        bad_penalties = self._validate_penalty_claims(
            answer, relevant_results, example_results
        )
        if bad_penalties:
            issues.append(
                "- These stated penalties/sentences could not be found "
                f"verbatim in the provided legal text: {'; '.join(bad_penalties)}."
            )

        bad_quotes = self._validate_quoted_claims(
            answer, relevant_results, example_results
        )
        if bad_quotes:
            issues.append(
                "- These quoted claims could not be verified verbatim against "
                f"the source they were attributed to: {'; '.join(bad_quotes)}."
            )

        subsection_mismatches = self._validate_subsection_attribution(
            answer, relevant_results, example_results
        )
        if subsection_mismatches:
            issues.append(
                "- These claims cite the wrong subsection for the penalty "
                f"described: {'; '.join(subsection_mismatches)}."
            )

        return issues


    _NAMED_REFERENCE_PATTERN = re.compile(
        r'\b(?:Article|Section|Sec\.)\s+(\d+[A-Za-z]?)\b', re.IGNORECASE
    )

    # Phrases that signal the model is asserting a legal CONCLUSION,
    # OUTCOME, or REMEDY (e.g. "you can ask for a case dismissal") rather
    # than stating what the source text says. These aren't inherently
    # wrong, but they're the highest-risk place for an inferential leap
    # beyond what's actually grounded — see the real example from testing:
    # "you may be able to request a dismissal of the case" was derived
    # from Article 48 (access to justice) and 49(b) (right to silence),
    # neither of which actually establishes a dismissal remedy. This is a
    # cheap, pattern-based flag, NOT a real verification that the
    # conclusion is wrong — confirming that would need either a second
    # LLM pass or a human reviewer. On hardware already running near its
    # generation-time ceiling, a second Ollama call per query would
    # roughly double worst-case latency, which isn't worth it for a
    # heuristic flag. This catches the common phrasing pattern for free
    # and logs it for review, rather than verifying correctness.
    _CONCLUSION_PHRASES = [
        "you may be able to", "you can ask for", "you are entitled to",
        "this means you can", "you have grounds to", "you can request",
        "this entitles you to", "you can sue for", "you can claim",
        "this means you are entitled", "you can demand",
    ]

    def _check_number_grounding(self, query: str, answer: str, results: List[Dict]):
        """Log-only: flag language that asserts a legal conclusion/remedy
        that goes beyond what the cited text states (e.g. "you can ask
        for a case dismissal"). Doesn't block the answer — it exists
        purely for review visibility, since confirming a conclusion is
        actually wrong (rather than just risky-sounding) would need
        either a second model call or a human reviewer.

        Article/Section reference-number grounding used to also be
        checked here, but only as a soft NUMBER_MISMATCH log line — real
        fabricated/self-contradicting statute numbers (like "section
        297(2)" vs "section 296(2)" for the same offense) shipped to
        users anyway. That check is now a hard reject via
        _validate_section_citations, called earlier in generate_answer,
        so it's been removed from here to avoid dead, misleading code.
        Penalty/sentence numbers are similarly hard-gated separately by
        _validate_penalty_claims."""
        answer_lower = answer.lower()
        matched_phrases = [p for p in self._CONCLUSION_PHRASES if p in answer_lower]
        if matched_phrases:
            logger.warning(
                f"POSSIBLE_OVERREACH: query='{query}' "
                f"conclusion_phrases={matched_phrases} "
                f"model={self.ollama_model} "
                f"-- review whether the cited source text actually states "
                f"this outcome/remedy, or whether the model inferred it."
            )

    # ============================================
    # CONTEXT BUILDING
    # ============================================

    def _build_context(self, results: List[Dict], examples: Optional[List[Dict]] = None) -> str:
        parts = []
        for i, r in enumerate(results):
            doc = re.sub(r'\s+', ' ', r.get('document', '')).strip()
            source = r.get('metadata', {}).get('source', 'Unknown')
            parts.append(f"[DOCUMENT {i + 1}: {source}]\n{doc}")

        if examples:
            parts.append("--- REAL-WORLD CASE EXAMPLES (illustration only) ---")
            for i, r in enumerate(examples):
                doc = re.sub(r'\s+', ' ', r.get('document', '')).strip()
                meta = r.get('metadata', {})
                case_name = meta.get('case_name') or meta.get('source', 'Unknown case')
                citation = meta.get('citation', '')
                label = f"{case_name} {citation}".strip()
                parts.append(f"[CASE {i + 1}: {label}]\n{doc}")

        return "\n\n".join(parts)

    # ============================================
    # CONVERSATION HISTORY (unchanged behavior)
    # ============================================

    def _sync_history(self, messages: List[Dict]):
        if not messages:
            return
        exchanges = []
        pair = {}
        for msg in messages:
            if msg.get('role') == 'user':
                if pair.get('question') and pair.get('answer'):
                    exchanges.append(pair)
                    pair = {}
                pair['question'] = msg.get('content', '')
            elif msg.get('role') == 'assistant' and pair.get('question'):
                pair['answer'] = msg.get('content', '')
                exchanges.append(pair)
                pair = {}
        if exchanges:
            self.conversation_history = exchanges[-10:]
            self.last_exchange = exchanges[-1]

    def _save_exchange(self, query: str, answer: str):
        if not answer or len(answer) < 20:
            return
        self.last_exchange = {
            "question": query[:500],
            "answer": answer[:500],
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(self.last_exchange)
        self.conversation_history = self.conversation_history[-10:]

    # ============================================
    # LIFECYCLE
    # ============================================

    def wait_for_loading(self, timeout: int = 30):
        start = time.time()
        while self._loading and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self._loaded

    def is_ready(self) -> bool:
        return self._loaded

    def _load_database(self):
        def load():
            self._loading = True
            try:
                import chromadb
                path = Path(__file__).parent.parent / "data" / "chroma_db"
                path.mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=str(path))
                try:
                    self.collection = self.client.get_collection("kenyan_law")
                except Exception:
                    self.collection = self.client.create_collection("kenyan_law")
                self._loaded = True
                logger.info(f"RAG_DB_LOADED: count={self.collection.count()}")
            except Exception as e:
                logger.error(f"RAG_DB_LOAD_FAILED: err={e}")
            finally:
                self._loading = False

        threading.Thread(target=load, daemon=True).start()
