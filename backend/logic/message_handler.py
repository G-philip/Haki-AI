"""Message processing logic — routes by explicit session mode (qa | document)"""

import re
import logging
from datetime import datetime
from logic.document_generator import DocumentGenerator

logger = logging.getLogger(__name__)


class MessageHandler:
    """
    Processes user messages.

    Mode is explicit and sticky for the session (set once via set_mode(),
    chosen by the user up front — e.g. "Ask a question" vs "Draft a
    document"). There is no more inference from intent classification or
    from rag.last_exchange. Once a mode is set, every message in that
    session is handled by that mode's flow until reset() is called.
    """

    MODE_PROMPT = (
        "Welcome to Kenyan Legal Assistant. Would you like to:\n\n"
        "1. Ask a legal question\n"
        "2. Draft a legal document (affidavit)\n\n"
        "Reply with 'question' or 'document' to get started."
    )

    def __init__(self, chat_manager, interview_engine, data_manager, rag_engine=None, search_tool=None):
        self.cm = chat_manager
        self.engine = interview_engine
        self.dm = data_manager
        self.rag = rag_engine
        self.search_tool = search_tool
        self._errors = []

        # If no shared instances provided, create new ones (fallback)
        if self.rag is None:
            from logic.rag_engine import RAGEngine
            self.rag = RAGEngine()
        if self.search_tool is None:
            from logic.search_tool import LegalSearchTool
            self.search_tool = LegalSearchTool()

    def _set_field(self, key, value):
        """
        Correctly persist a single collected field.

        IMPORTANT: ChatManager.collected_fields is a property whose getter
        returns self._collected_fields.copy() (by design, for thread
        safety). Code like `self.cm.collected_fields[key] = value` or
        `self.cm.collected_fields.update(...)` mutates that throwaway copy
        and silently discards it — the real internal dict never changes.
        This was a real, serious bug found in this exact file: every field
        "collected" during an interview was being lost, so the interview
        could never actually complete. Always go through this helper (or
        ChatManager.update_field directly) instead of touching
        self.cm.collected_fields[...] as if it were mutable in place.
        """
        self.cm.update_field(key, value)

    def _set_fields(self, fields: dict):
        """Bulk version of _set_field, for dict.update()-style call sites."""
        for k, v in fields.items():
            self.cm.update_field(k, v)

    # ============================================
    # TOP-LEVEL ROUTER
    # ============================================

    def process(self, user_input):
        """
        Main entry point. Routes purely on self.cm.mode.

        user_type (set once at login, see ChatManager.user_type) gates
        which modes are even reachable:
        - 'normal' users have exactly one allowed mode ('qa'), so there's
          no real choice to make — mode is auto-assigned on the first
          message rather than prompting them to pick from one option.
        - 'lawyer' (and 'admin', currently unrestricted as a placeholder)
          can choose freely, and the choice is still validated against
          allowed_modes() — a user could bypass a hidden/disabled
          frontend control by POSTing directly to the API, so this check
          is the actual enforcement point, not the frontend dropdown.

        The mode control in the UI is now always visible for accounts
        with more than one allowed mode (not just before the first
        choice), so an exact, bare 'question' or 'document' input is
        treated as an explicit mode-switch command REGARDLESS of whether
        a mode is already set — this is what the dropdown actually sends
        (see ChatInput.jsx), as opposed to a free-text message that
        happens to contain those words.
        """
        if not user_input or not user_input.strip():
            return None

        allowed = self.cm.allowed_modes()
        stripped = user_input.strip().lower()

        if stripped in ('question', 'document') and self.cm.has_mode():
            # Explicit mode switch mid-conversation via the dropdown.
            return self._handle_mode_selection(user_input, allowed)

        if not self.cm.has_mode():
            if allowed == {'qa'}:
                # Only one mode is reachable for this user_type — no
                # choice to prompt for. Assign it directly and treat
                # this message as the first real Q&A turn, not as the
                # mode-selection reply.
                self.cm.set_mode('qa')
                return self.process_qa(user_input)
            return self._handle_mode_selection(user_input, allowed)

        mode = self.cm.mode

        if mode not in allowed:
            # Defensive: mode was set before user_type was known, or
            # user_type changed somehow. Force back to an allowed mode
            # rather than continue serving a mode this session shouldn't
            # have access to.
            logger.warning(
                f"Mode '{mode}' not allowed for user_type "
                f"'{self.cm.user_type}' on session {self.cm.session_id}; resetting"
            )
            self.cm.mode = None
            return self.process(user_input)

        if mode == 'qa':
            return self.process_qa(user_input)

        if mode == 'document':
            return self.process_document(user_input)

        # Should be unreachable given VALID_MODES, but fail safely.
        logger.warning(f"Unknown mode '{mode}' on session {self.cm.session_id}; resetting mode")
        self.cm.mode = None
        return self.MODE_PROMPT

    # Mode-selection confirmation/rejection texts are transient UI
    # feedback, not real conversation content — see main.py's chat()
    # route, which checks the response against these exact strings and
    # skips persisting them to chat history (same pattern as
    # UNAVAILABLE_MESSAGE). Defined as constants here rather than
    # inline so the two files can't drift out of sync.
    QA_MODE_CONFIRMATION = "You're in legal Q&A mode. Ask me anything about Kenyan law."
    DOCUMENT_MODE_CONFIRMATION = (
        "You're in document drafting mode. "
        "Describe your situation (e.g. 'I lost my ID') and I'll start the affidavit."
    )
    QA_MODE_DENIED = "Your account doesn't have access to Q&A mode. Please contact support if this seems wrong."
    DOCUMENT_MODE_DENIED = "Document drafting isn't available on your account. This feature is currently limited to lawyer accounts."

    # All transient (non-persisted) mode-related responses, for a single
    # set membership check in main.py.
    TRANSIENT_MODE_RESPONSES = {
        QA_MODE_CONFIRMATION,
        DOCUMENT_MODE_CONFIRMATION,
        QA_MODE_DENIED,
        DOCUMENT_MODE_DENIED,
        MODE_PROMPT,
    }

    def _handle_mode_selection(self, user_input, allowed=None):
        """Capture the user's explicit mode choice before anything else runs."""
        if allowed is None:
            allowed = self.cm.allowed_modes()

        choice = user_input.strip().lower()

        qa_words = {'question', 'q', '1', 'ask', 'ask a question', 'legal question'}
        doc_words = {'document', 'd', '2', 'draft', 'draft a document', 'affidavit'}

        if choice in qa_words or 'question' in choice:
            if 'qa' not in allowed:
                return self.QA_MODE_DENIED
            self.cm.set_mode('qa')
            return self.QA_MODE_CONFIRMATION

        if choice in doc_words or 'document' in choice or 'affidavit' in choice:
            if 'document' not in allowed:
                return self.DOCUMENT_MODE_DENIED
            self.cm.set_mode('document')
            return self.DOCUMENT_MODE_CONFIRMATION

        return self.MODE_PROMPT

    # ============================================
    # MODE: QA  (RAG + Ollama only — no interview state involved)
    # ============================================

    def process_qa(self, user_input):
        """
        Pure legal Q&A. Every message in this mode goes to the RAG engine.
        No InterviewEngine, no DocumentGenerator, no stage machine.
        """
        if user_input.strip().lower() in ('exit', 'switch', 'change mode', 'menu'):
            self.cm.mode = None
            return self.MODE_PROMPT

        return self.rag.generate_answer(user_input, self.cm.messages)

    # ============================================
    # MODE: DOCUMENT (interview / confirm / generate)
    # ============================================

    def process_document(self, user_input):
        """Document drafting flow. No RAG/legal-question branches here —
        see the TODO below for the still-open decision on mid-interview
        legal questions."""
        stage = self.cm.stage

        if user_input.strip().lower() in ('exit', 'switch', 'change mode', 'menu'):
            self.cm.mode = None
            return self.MODE_PROMPT

        # TODO(decide): if we want a scoped "quick legal question" escape
        # hatch during interviewing/confirming, detect it here explicitly
        # (e.g. a leading "?" or a dedicated command like "ask: ...") and
        # call self.rag.generate_answer(...) WITHOUT changing stage, then
        # re-prompt the same outstanding question afterwards. Until that's
        # decided, document mode does not consult RAG at all, so a vague
        # question typed here will just be treated as interview input by
        # InterviewEngine's field extraction (which is fine — worst case
        # it's treated as free text for whatever field is currently being
        # collected).

        intent_data = self.engine.classify_intent(user_input)
        intent = intent_data['intent']

        if intent in ('help', 'cost'):
            return self.engine.get_intent_response(intent_data)

        if stage == 'greeting':
            if intent == 'document':
                doc_type = intent_data.get('doc_type')
                if not doc_type:
                    return "Tell me what you need — describe your situation for an affidavit."
                self.cm.doc_type = doc_type
                fields = self.engine.extract_all_fields(user_input, doc_type)
                self._set_fields(fields)
                if self.engine.is_complete(doc_type, self.cm.collected_fields):
                    self.cm.stage = 'confirming'
                    return self._confirm()
                self.cm.stage = 'interviewing'
                return self._extract_response()

            # Fall back to trying to detect a document type directly,
            # since not every "describe your situation" message will be
            # classified as intent == 'document' by classify_intent().
            doc_type = self.engine.detect_document_type(user_input)
            if doc_type:
                self.cm.doc_type = doc_type
                fields = self.engine.extract_all_fields(user_input, doc_type)
                self._set_fields(fields)
                if self.engine.is_complete(doc_type, self.cm.collected_fields):
                    self.cm.stage = 'confirming'
                    return self._confirm()
                self.cm.stage = 'interviewing'
                return self._extract_response()

            return ("Tell me what kind of document you need — for example "
                    "'I lost my ID' or 'I need an affidavit of service'.")

        elif stage == 'interviewing':
            return self._handle_interviewing(user_input)

        elif stage == 'confirming':
            if any(w in user_input.lower() for w in ['yes', 'generate', 'ok', 'proceed', 'create']):
                return self._generate()
            elif user_input.lower().strip() in ['no', 'change', 'edit', 'wrong', 'what to change', 'i want to change']:
                self.cm.stage = 'interviewing'
                return "Sure, what would you like to change? You can update your name, ID number, postal address, city, or the facts."
            else:
                return self._handle_confirming_edits(user_input)

        elif stage == 'done':
            return ("Your document has already been generated. "
                    "Type 'switch' to change mode, or describe a new situation to start another document.")

        return "How can I help? Describe your situation for an affidavit."

    def _handle_confirming_edits(self, user_input):
        doc_type = self.cm.doc_type
        updated_name = None
        updated_id = None
        updated_box = None
        updated_city = None
        any_change = False
        self._errors = []

        def extract_meta(text):
            nonlocal updated_name, updated_id, updated_box, updated_city
            nm = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:name|deponent|na)\s*(?:to|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,3})", text)
            im = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:id|identification|id number)\s*(?:to|:)?\s*(\d{6,8})", text)
            bm = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:p\.?o\.?\s*box|box|postal)\s*(?:to|:)?\s*(\d{1,6})", text)
            cc = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:city|town|location)", text, re.IGNORECASE)
            cm = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:city|town|location)\s*(?:to|:)?\s*([A-Za-z]+)", text)
            if nm: updated_name = nm.group(1).strip().title()
            if im: updated_id = im.group(1).strip()
            if bm: updated_box = bm.group(1).strip()
            if cc and cm: updated_city = cm.group(1).strip().title()

        commands = re.split(r'\.\s*(?=(?:replace|change|edit|update|add)\s)', user_input)

        for command in commands:
            command = command.strip()
            if not command:
                continue

            fact_num_replace = re.search(r"(?:change|replace|edit|update)\s*(?:fact|paragraph|point|number)\s*(\d+)\s*(?:to|with)\s+(.+)", command, re.IGNORECASE)
            fact_num_reversed = re.search(r"(?:change|replace|edit|update)\s*(?:fact|paragraph|point|number)\s*(?:to|with)\s*(\d+)\s+(.+)", command, re.IGNORECASE)

            if fact_num_replace:
                fact_num = int(fact_num_replace.group(1))
                new_fact_text = fact_num_replace.group(2).strip()
            elif fact_num_reversed:
                fact_num = int(fact_num_reversed.group(1))
                new_fact_text = fact_num_reversed.group(2).strip()
                self._errors.append(f"I noticed you wrote 'replace fact to {fact_num}'. The usual format is 'change fact {fact_num} to [new text]'. I've made the change anyway, just so you know for next time.")
            else:
                fact_num = None

            if fact_num:
                new_fact_text = re.sub(r'\.\s+', '.\n', new_fact_text)
                existing_facts = self.cm.collected_fields.get('facts', '')
                fact_lines = existing_facts.split('\n') if existing_facts else []

                if 1 <= fact_num <= len(fact_lines):
                    fact_lines[fact_num - 1] = new_fact_text
                    self._set_field('facts', '\n'.join(fact_lines))
                    any_change = True
                else:
                    self._errors.append(f"Fact number {fact_num} doesn't exist. You have {len(fact_lines)} facts, numbered 1 to {len(fact_lines)}. Try something like 'change fact 2 to [new text]'.")
                continue

            simple_replace = re.search(r"(?:replace|edit)\s+(?:to|with)\s+(.+)", command, re.IGNORECASE)
            if simple_replace:
                new_text = simple_replace.group(1).strip()
                add_part = None
                add_match_in_replace = re.search(r'\.\s*add\s+(.+)', new_text, re.IGNORECASE)
                if add_match_in_replace:
                    add_part = add_match_in_replace.group(1).strip()
                    new_text = new_text[:add_match_in_replace.start()].strip()

                if len(new_text) > 3:
                    new_text = re.sub(r'\.\s+', '.\n', new_text)
                    self._set_field('facts', new_text)
                    if add_part:
                        add_part = re.sub(r'\.\s+', '.\n', add_part)
                        self._set_field('facts', new_text + '\n' + add_part)
                    any_change = True
                continue

            add_match = re.search(r"(?:add|include|also)\s+(.+)", command, re.IGNORECASE)
            if add_match:
                existing = self.cm.collected_fields.get('facts', '')
                new_fact = add_match.group(1).strip()
                new_fact = re.sub(r'\.\s+', '.\n', new_fact)
                self._set_field('facts', existing + '\n' + new_fact if existing else new_fact)
                any_change = True
                continue

        extract_meta(user_input)
        self._apply_meta(updated_name, updated_id, updated_box, updated_city)
        if updated_name or updated_id or updated_box or updated_city:
            any_change = True

        if any_change or self._errors:
            items = [f"• {f.replace('_', ' ').title()}: {v}" for f, v in self.cm.collected_fields.items() if v]
            msg = f"Here's what I have now:\n\n{chr(10).join(items)}"
            if self._errors:
                msg += f"\n\nJust a note: {chr(10).join(self._errors)}"
                self._errors = []
            msg += "\n\nWould you like to change anything else, or should I generate the document?"
            return msg

        new_fields = self.engine.extract_all_fields(user_input, doc_type)
        if new_fields and any(v for v in new_fields.values()):
            for k, v in new_fields.items():
                if v:
                    if k == 'city_town':
                        city_cmd = re.search(r"(?:change|update|set|edit)\s*(?:the\s+)?(?:city|town|location)", user_input, re.IGNORECASE)
                        if city_cmd: self._set_field(k, v.title())
                    elif k == 'deponent_name': self._set_field(k, v.title())
                    elif k == 'facts':
                        existing = self.cm.collected_fields.get('facts', '')
                        v = re.sub(r'\.\s+', '.\n', v)
                        if existing and v not in existing: self._set_field('facts', existing + '\n' + v)
                        elif not existing: self._set_field('facts', v)
                    else: self._set_field(k, v)
            items = [f"• {f.replace('_', ' ').title()}: {v}" for f, v in self.cm.collected_fields.items() if v]
            return f"Here's what I have now:\n\n{chr(10).join(items)}\n\nWould you like to change anything else, or should I generate the document?"

        return "I didn't quite catch that. Here are some things you can do:\n\n• Update your name: change name to John Doe\n• Change your ID: update ID to 12345678\n• Add more details: add I reported to the police\n• Replace all facts: replace to new text\n• Edit a specific point: change fact 2 to new text\n\nOr just type generate when you're ready."

    def _apply_meta(self, name, id_num, box, city):
        if name:
            self._set_field('deponent_name', name)
        if id_num:
            self._set_field('id_number', id_num)
        if box:
            self._set_field('postal_address', box)
        if city:
            self._set_field('city_town', city)

    def _handle_interviewing(self, user_input):
        doc_type = self.cm.doc_type
        new_fields = self.engine.extract_all_fields(user_input, doc_type)
        missing = self.engine.get_missing_fields(doc_type, self.cm.collected_fields)

        if missing:
            cf = missing[0]['field']
            if cf == 'facts':
                if user_input.lower().strip() in ['done', 'finished', 'complete', 'next']:
                    nq = self.engine.get_next_question(doc_type, self.cm.collected_fields)
                    return f"Got it. {nq}" if nq else self._confirm()
                sentences = re.split(r'(?<=[.!?])\s+', user_input.strip())
                existing = self.cm.collected_fields.get('facts', '')
                new_facts = (existing + '\n' + '\n'.join(sentences)) if existing else '\n'.join(sentences)
                self._set_field('facts', new_facts)
                total = len(new_facts.split('\n'))
                return "Got it. Continue describing, or type done when you're finished." if total < 3 else f"I've noted {total} points. You can add more or type done to continue."
            else:
                if not new_fields or all(v is None for v in new_fields.values()):
                    self._set_field(cf, user_input.strip().title() if cf == 'deponent_name' else user_input.strip())
                else:
                    for k, v in new_fields.items():
                        if v:
                            self._set_field(k, v.title() if k == 'deponent_name' else v)
                nq = self.engine.get_next_question(doc_type, self.cm.collected_fields)
                return f"Got it. {nq}" if nq else self._confirm()

        if self.engine.is_complete(doc_type, self.cm.collected_fields):
            self.cm.stage = 'confirming'
            return self._confirm()
        return f"Got it. {self.engine.get_next_question(doc_type, self.cm.collected_fields)}"

    def _extract_response(self):
        fields = self.cm.collected_fields
        found = [f"• {f.replace('_', ' ').title()}: {v}" for f, v in fields.items() if v]
        summary = "\n".join(found) if found else "I'm collecting your information."
        nq = self.engine.get_next_question(self.cm.doc_type, fields)
        return f"{summary}\n\n{nq}"

    def _confirm(self):
        fields = self.cm.collected_fields
        items = [f"• {f.replace('_', ' ').title()}: {v}" for f, v in fields.items() if v]
        return f"Here's a summary of your {self.cm.doc_type}:\n\n{chr(10).join(items)}\n\nDoes everything look right? Reply yes to generate the document, or tell me what you'd like to change."

    def _generate(self):
        doc_type = self.cm.doc_type
        fields = self.cm.collected_fields

        generator = DocumentGenerator()
        document = generator.generate(doc_type, fields)

        self.cm.generated_document = document
        self.cm.stage = 'done'

        chats = self.dm.load_chats()
        dn = fields.get('deponent_name', '')
        title = f"{doc_type} — {dn}" if dn else doc_type

        for cid in list(chats.keys()):
            if chats[cid].get('title') == title:
                del chats[cid]

        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        chats[chat_id] = {
            'title': title,
            'deponent_name': dn,
            'doc_type': doc_type,
            'date': datetime.now().isoformat(),
            'messages': [m for m in self.cm.messages],
            'collected_fields': dict(fields),
            'has_document': True
        }

        self.dm.save_chats(chats)

        return f"Your {doc_type} is ready. It's {len(document.split())} words and formatted according to the Oaths and Statutory Declarations Act, Cap. 15.\n\nYou can download it below as a DOCX or PDF file. Remember to have it reviewed by a licensed Kenyan advocate before swearing."
