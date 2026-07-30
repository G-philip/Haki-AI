"""
Session State Manager - Production Ready
Thread-safe chat session management with serialization
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class ChatManager:
    """
    Manages state for a single chat session.
    Thread-safe for concurrent access.
    """

    VALID_STAGES = {'greeting', 'interviewing', 'confirming', 'done'}
    # NEW: explicit session mode. None = not yet chosen by the user.
    # 'qa'       -> pure RAG + Ollama legal Q&A, no interview/document state used
    # 'document' -> affidavit drafting interview/confirm/generate flow
    VALID_MODES = {None, 'qa', 'document'}

    # User type, set once at (mock) login and never changed mid-session.
    # This gates which modes are even available — see
    # MessageHandler.process()'s enforcement of ALLOWED_MODES_BY_TYPE.
    # 'admin'  -> access TBD; currently unrestricted as a placeholder.
    # 'lawyer' -> may freely choose 'qa' or 'document'.
    # 'normal' -> locked to 'qa' only; 'document' is not offered or
    #             accepted even if explicitly requested.
    VALID_USER_TYPES = {None, 'admin', 'lawyer', 'normal'}

    MAX_MESSAGES = 200
    MAX_FIELD_LENGTH = 5000

    def __init__(self, session_id: str = None, user_type: Optional[str] = None):
        self.session_id = session_id
        self._lock = Lock()
        self._messages: List[Dict] = []
        self._collected_fields: Dict[str, Any] = {}
        self._doc_type: Optional[str] = None
        self._generated_document: Optional[str] = None
        self._stage: str = 'greeting'
        self._mode: Optional[str] = None  # NEW
        self._user_type: Optional[str] = (
            user_type if user_type in self.VALID_USER_TYPES else None
        )
        self._created_at = datetime.now().isoformat()
        self._updated_at = datetime.now().isoformat()
        self._version: int = 1

    # ============ PROPERTIES ============

    @property
    def messages(self) -> List[Dict]:
        with self._lock:
            return self._messages.copy()

    @messages.setter
    def messages(self, value: List[Dict]):
        with self._lock:
            self._messages = value[-self.MAX_MESSAGES:] if len(value) > self.MAX_MESSAGES else value
            self._touch()

    @property
    def collected_fields(self) -> Dict[str, Any]:
        with self._lock:
            return self._collected_fields.copy()

    @collected_fields.setter
    def collected_fields(self, value: Dict[str, Any]):
        with self._lock:
            # Truncate long field values
            self._collected_fields = {
                k: v[:self.MAX_FIELD_LENGTH] if isinstance(v, str) and len(v) > self.MAX_FIELD_LENGTH else v
                for k, v in value.items()
            }
            self._touch()

    @property
    def doc_type(self) -> Optional[str]:
        with self._lock:
            return self._doc_type

    @doc_type.setter
    def doc_type(self, value: Optional[str]):
        with self._lock:
            self._doc_type = value
            self._touch()

    @property
    def generated_document(self) -> Optional[str]:
        with self._lock:
            return self._generated_document

    @generated_document.setter
    def generated_document(self, value: Optional[str]):
        with self._lock:
            self._generated_document = value
            self._touch()

    @property
    def stage(self) -> str:
        with self._lock:
            return self._stage

    @stage.setter
    def stage(self, value: str):
        with self._lock:
            if value in self.VALID_STAGES:
                self._stage = value
                self._touch()
            else:
                logger.warning(f"Invalid stage '{value}', keeping '{self._stage}'")

    @property
    def mode(self) -> Optional[str]:
        """
        Session mode, chosen explicitly by the user once per session.
        None until the user picks 'qa' or 'document'.
        """
        with self._lock:
            return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        with self._lock:
            if value in self.VALID_MODES:
                self._mode = value
                self._touch()
            else:
                logger.warning(f"Invalid mode '{value}', keeping '{self._mode}'")

    def set_mode(self, value: str) -> bool:
        """
        Explicit setter with a boolean result, for callers (e.g. the route
        handler) that want to confirm the choice was accepted rather than
        silently keeping the old value.
        """
        if value not in self.VALID_MODES or value is None:
            logger.warning(f"Rejected mode selection '{value}'")
            return False
        self.mode = value
        return True

    def has_mode(self) -> bool:
        with self._lock:
            return self._mode is not None

    @property
    def user_type(self) -> Optional[str]:
        with self._lock:
            return self._user_type

    # user_type is intentionally NOT settable after construction — it's
    # established once at (mock) login via ChatManager(user_type=...) and
    # treated as fixed identity for the session, not a value that
    # changes mid-conversation the way `mode` does.

    def allowed_modes(self) -> set:
        """
        Which modes this session's user_type may use. This is the actual
        enforcement point — MessageHandler must check this before honoring
        a mode selection, since a frontend-only restriction (e.g. just
        hiding the dropdown option) can't be trusted; a user could still
        POST {"message": "document"} directly to the API.
        """
        if self._user_type == 'normal':
            return {'qa'}
        if self._user_type == 'lawyer':
            return {'qa', 'document'}
        if self._user_type == 'admin':
            # TBD — unrestricted for now as an explicit placeholder, not
            # a considered decision about what admins should actually see.
            return {'qa', 'document'}
        # No user_type set (e.g. legacy session, or login was skipped) —
        # default to the safest option rather than silently allowing
        # document drafting to anyone unidentified.
        return {'qa'}

    # ============ METHODS ============

    def add_message(self, role: str, content: str, metadata: Dict = None) -> Dict:
        """Add a message to the conversation"""
        with self._lock:
            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self._messages.append(message)

            # Trim if too many
            if len(self._messages) > self.MAX_MESSAGES:
                self._messages = self._messages[-self.MAX_MESSAGES:]

            self._touch()
            return message

    def truncate_and_replace(self, index: int, role: str, content: str, metadata: Dict = None) -> Dict:
        """
        Used when the user edits a previously-sent message. Drops every
        message from `index` onward (the old reply, and anything after
        it, is stale once the question itself changes) and appends a
        fresh message with the revised content in its place.

        If `index` is out of range, falls back to a plain add_message so
        an edit never silently fails to persist.
        """
        with self._lock:
            if 0 <= index < len(self._messages):
                self._messages = self._messages[:index]

            message = {
                'role': role,
                'content': content,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {}
            }
            self._messages.append(message)

            if len(self._messages) > self.MAX_MESSAGES:
                self._messages = self._messages[-self.MAX_MESSAGES:]

            self._touch()
            return message

    def remove_last_message(self, role: str = None) -> bool:
        """
        Removes the most recent message, optionally only if it matches
        `role`. Used when a message turns out to be transient UI noise
        rather than real conversation content — e.g. the literal
        'question'/'document' a user sends via the mode-switch dropdown
        (see main.py's chat() route) — and was added to history before
        process() revealed what kind of turn it actually was. Returns
        True if a message was removed, False otherwise (e.g. role
        mismatch, or empty history) so the caller can tell whether the
        removal actually happened.
        """
        with self._lock:
            if not self._messages:
                return False
            if role is not None and self._messages[-1].get('role') != role:
                return False
            self._messages.pop()
            self._touch()
            return True

    def get_last_message(self, role: str = None) -> Optional[Dict]:
        """Get the last message, optionally filtered by role"""
        with self._lock:
            if not self._messages:
                return None

            if role:
                for msg in reversed(self._messages):
                    if msg['role'] == role:
                        return msg
                return None

            return self._messages[-1]

    def get_message_count(self) -> int:
        """Get total message count"""
        with self._lock:
            return len(self._messages)

    def update_field(self, key: str, value: Any):
        """Update a single collected field"""
        with self._lock:
            if isinstance(value, str) and len(value) > self.MAX_FIELD_LENGTH:
                value = value[:self.MAX_FIELD_LENGTH]
            self._collected_fields[key] = value
            self._touch()

    def get_field(self, key: str, default: Any = None) -> Any:
        """Get a single collected field"""
        with self._lock:
            return self._collected_fields.get(key, default)

    def clear_fields(self):
        """Clear all collected fields"""
        with self._lock:
            self._collected_fields.clear()
            self._touch()

    def reset(self):
        """Reset session to initial state"""
        with self._lock:
            self._messages = []
            self._collected_fields = {}
            self._doc_type = None
            self._generated_document = None
            self._stage = 'greeting'
            self._mode = None  # NEW: force the user to choose a mode again
            self._version += 1
            self._touch()

    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage"""
        with self._lock:
            return {
                'session_id': self.session_id,
                'messages': self._messages,
                'collected_fields': self._collected_fields,
                'doc_type': self._doc_type,
                'generated_document': self._generated_document,
                'stage': self._stage,
                'mode': self._mode,  # NEW
                'user_type': self._user_type,  # NEW
                'created_at': self._created_at,
                'updated_at': self._updated_at,
                'version': self._version,
            }

    def load_from_dict(self, data: Dict):
        """Load state from dictionary"""
        with self._lock:
            self.session_id = data.get('session_id', self.session_id)
            self._messages = data.get('messages', [])[-self.MAX_MESSAGES:]
            self._collected_fields = data.get('collected_fields', {})
            self._doc_type = data.get('doc_type')
            self._generated_document = data.get('generated_document')
            self._stage = data.get('stage', 'greeting')
            self._mode = data.get('mode')  # NEW
            self._user_type = data.get('user_type')  # NEW
            self._created_at = data.get('created_at', self._created_at)
            self._updated_at = data.get('updated_at', datetime.now().isoformat())
            self._version = data.get('version', 1)

            if self._stage not in self.VALID_STAGES:
                self._stage = 'greeting'
            if self._mode not in self.VALID_MODES:
                self._mode = None
            if self._user_type not in self.VALID_USER_TYPES:
                self._user_type = None

    def get_summary(self) -> Dict:
        """Get session summary without full messages"""
        with self._lock:
            return {
                'session_id': self.session_id,
                'message_count': len(self._messages),
                'doc_type': self._doc_type,
                'stage': self._stage,
                'mode': self._mode,  # NEW
                'user_type': self._user_type,  # NEW
                'has_document': bool(self._generated_document),
                'created_at': self._created_at,
                'updated_at': self._updated_at,
                'version': self._version,
                'fields_collected': list(self._collected_fields.keys()),
            }

    def _touch(self):
        """Update the last-modified timestamp"""
        self._updated_at = datetime.now().isoformat()
