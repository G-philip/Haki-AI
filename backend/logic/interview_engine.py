"""
Smart Interview Engine
Handles ALL user inputs gracefully — questions ALWAYS take priority
"""

import re


class InterviewEngine:
    """Smart interview engine for Kenyan legal documents"""
    
    GREETINGS = [
        'hello', 'hi', 'hey', 'jambo', 'habari', 'good morning',
        'good afternoon', 'good evening', 'howdy', 'greetings',
        'yo', 'sup', 'whats up', "what's up", 'how are you'
    ]
    
    HELP_WORDS = [
        'help', 'what can you do', 'how does this work',
        'what do you do', 'menu', 'options', 'commands'
    ]
    
    QUESTION_WORDS = [
        'what is', 'what are', 'how do', 'explain',
        'meaning', 'define', 'difference between', 'when',
        'can i', 'can my', 'can you', 'can they', 'can an',
        'is it', 'are there', 'do i', 'does a', 'does an',
        'what happens', 'what can', 'what should',
        'how long', 'how much', 'how many', 'how can',
        'am i', 'should i', 'will i', 'must i',
        'where can', 'where do', 'who can', 'who is',
        'why is', 'why do', 'why does', 'why can',
        'what to do', 'what can i do', 'how to', 'how can i',
        'what happens if', 'what happens when', 'how do you',
        'what is the process', 'what are the steps',
        'what are my rights', 'is there a way',
        'how does one', 'where do i', 'when can i',
    ]
    
    COST_WORDS = ['cost', 'price', 'fee', 'charge', 'how much', 'payment', 'pay']
    
    CANNOT_DO = [
        'represent me', 'be my lawyer', 'represent me in court',
        'file for me', 'go to court for me', 'appear for me',
        'sue on my behalf', 'draft a plaint for me',
    ]
    
    FEEDBACK_WORDS = [
        'bad', 'terrible', 'useless', 'stupid', 'not working',
        'wrong', 'error', 'mistake', 'bug', 'broken'
    ]
    
    GENERAL_SIGNALS = [
        "lost", "id card", "identity", "passport", "certificate",
        "name change", "residence", "proof", "declare", "declaration",
        "general", "lost my", "missing", "misplaced", "gone",
        "affidavit", "swear", "sworn", "oath"
    ]
    
    SERVICE_SIGNALS = [
        "service", "served", "process server", "delivered",
        "documents to", "summons", "plaint", "defendant",
        "case", "court case", "civil suit", "pleadings"
    ]
    
    SUPPORT_SIGNALS = [
        "support", "application", "motion", "injunction",
        "stay", "appeal", "setting aside", "review",
        "order", "ruling", "judgment", "dismiss"
    ]
    
    FIELD_QUESTIONS = {
        "General Affidavit": {
            "deponent_name": {
                "question": "What is the full name of the person swearing this affidavit?",
                "patterns": [
                    r"(?:change|update|set|edit)\s*(?:the\s+)?(?:name|deponent|na)\s*(?:to|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,3})",
                    r"(?:my name is|I am|i am|name:)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
                    r"(?:deponent|applicant)\s*(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
                    r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
                ],
                "extract": lambda m: m.group(1).strip()
            },
            "id_number": {
                "question": "What is the National ID number?",
                "patterns": [
                    r"(?:change|update|set|edit)\s*(?:the\s+)?(?:id|identification|id number)\s*(?:to|:)?\s*(\d{6,8})",
                    r"(?:ID|id)\s*(?:number|no\.?|#)?\s*(?:is|:)?\s*(\d{6,8})",
                    r"(\d{6,8})\s*(?:is my|is the)?\s*(?:ID|id)",
                ],
                "extract": lambda m: m.group(1).strip()
            },
            "postal_address": {
                "question": "What is the P.O. Box number?",
                "patterns": [
                    r"(?:change|update|set|edit)\s*(?:the\s+)?(?:p\.?o\.?\s*box|box|postal)\s*(?:to|:)?\s*(\d{1,6})",
                    r"(?:P\.?O\.?\s*Box|box)\s*(?:number|no\.?)?\s*(?:is|:)?\s*(\d{1,6})",
                    r"Box\s*(\d{1,6})",
                ],
                "extract": lambda m: m.group(1).strip()
            },
            "city_town": {
                "question": "Which city or town?",
                "patterns": [
                    r"(?:change|update|set|edit)\s*(?:the\s+)?(?:city|town|location)\s*(?:to|:)?\s*([A-Za-z]+)",
                    r"(?:in|at|from|of)\s+(Nairobi|Mombasa|Kisumu|Nakuru|Eldoret|Thika|Nyeri|Meru|Machakos|Kakamega|Kisii|Bungoma|Garissa|Malindi|Kericho|Kitale|Embu)",
                    r"(?:city|town)\s*(?:is|:)?\s*([A-Z][a-z]+)",
                ],
                "extract": lambda m: m.group(1).strip()
            },
            "place_sworn": {
                "question": "Where will this affidavit be sworn?",
                "patterns": [r"(?:swear|sworn)\s*(?:in|at)\s*([A-Z][a-z]+)"],
                "extract": lambda m: m.group(1).strip()
            },
            "facts": {
                "question": "Please describe what happened. What facts need to be sworn?",
                "patterns": [
                    r"(?:add|include|also)\s+(.+)",
                    r"(?:I lost|lost my|need to|happened|that\s+I)(.*)",
                ],
                "extract": lambda m: m.group(1).strip() if m.lastindex and m.group(1) else (m.group(0).strip() if m else None)
            }
        },
        "Affidavit of Service": {
            "court_name": {
                "question": "Which court? (e.g., High Court of Kenya)",
                "patterns": [r"(High Court|Court of Appeal|Supreme Court|Magistrate'?s Court|ELRC|ELC|Kadhi'?s Court)"],
                "extract": lambda m: m.group(1).strip()
            },
            "court_location": {
                "question": "Where is the court located?",
                "patterns": [r"(?:at|in)\s+(Milimani|Nairobi|Mombasa|Kisumu|Nakuru|Eldoret)[,\s]"],
                "extract": lambda m: m.group(1).strip()
            },
            "case_number": {
                "question": "What is the case number?",
                "patterns": [r"(?:case|suit|application)\s*(?:no\.?|number)?\s*([A-Z]?\d+\s*(?:of\s*\d{4})?)", r"([A-Z]\d+\s*of\s*\d{4})"],
                "extract": lambda m: m.group(1).strip()
            },
            "plaintiff_name": {
                "question": "Who is the plaintiff/applicant?",
                "patterns": [r"(?:plaintiff|applicant)\s*(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip()
            },
            "defendant_name": {
                "question": "Who is the defendant/respondent?",
                "patterns": [r"(?:defendant|respondent)\s*(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip()
            },
            "deponent_name": {
                "question": "What is the process server's full name?",
                "patterns": [r"(?:I am|my name is|server is|served by)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip()
            },
            "deponent_role": {
                "question": "What is the process server's role?",
                "patterns": [r"(?:I am a|as a|process server|advocate'?s clerk)"],
                "extract": lambda m: m.group(0).strip() if m else None
            },
            "id_number": {
                "question": "What is the process server's ID number?",
                "patterns": [r"(?:ID|id)\s*(?:number|no\.?)?\s*(?:is|:)?\s*(\d{6,8})"],
                "extract": lambda m: m.group(1).strip()
            },
            "postal_address": {
                "question": "What is the P.O. Box number?",
                "patterns": [r"P\.?O\.?\s*Box\s*(\d{1,6})"],
                "extract": lambda m: m.group(1).strip()
            },
            "city_town": {
                "question": "Which city/town?",
                "patterns": [r"(?:in|at|from)\s+(Nairobi|Mombasa|Kisumu|Nakuru|Eldoret)"],
                "extract": lambda m: m.group(1).strip()
            },
            "date_of_service": {
                "question": "When was service effected?",
                "patterns": [r"(?:on|served on|service on)\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s*\d{4})"],
                "extract": lambda m: m.group(1).strip()
            },
            "service_address": {
                "question": "Where were the documents served?",
                "patterns": [r"(?:at|address|served at)\s+([A-Za-z0-9\s,]+(?:Estate|Road|Street|Avenue|Lane|Building|House)[A-Za-z0-9\s,]*)"],
                "extract": lambda m: m.group(1).strip() if m else None
            },
            "document_1": {
                "question": "What document was served?",
                "patterns": [r"(?:served|documents?)\s+(?:the\s+)?(Plaint|Summons|Notice|Petition|Application|Order|Decree|Writ)"],
                "extract": lambda m: m.group(1).strip() if m else None
            },
            "person_served": {
                "question": "Who received the documents?",
                "patterns": [r"(?:served upon|received by|given to|accepted by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip() if m else None
            },
            "relationship_to_recipient": {
                "question": "Relationship to the defendant?",
                "patterns": [r"(wife|husband|secretary|clerk|agent|employee|relative|spouse|son|daughter|brother|sister)"],
                "extract": lambda m: m.group(0).strip() if m else None
            },
            "reason_for_not_signing": {
                "question": "Why did they not sign?",
                "patterns": [r"(?:declined|refused|did not sign|could not sign)\s*(?:because|as|citing)?\s*(.*?)(?:\.|$)"],
                "extract": lambda m: m.group(1).strip()[:100] if m else "they declined to sign"
            },
            "place_sworn": {
                "question": "Where will this be sworn?",
                "patterns": [r"(?:swear|sworn)\s*(?:in|at)\s*([A-Z][a-z]+)"],
                "extract": lambda m: m.group(1).strip() if m else "Nairobi"
            }
        },
        "Affidavit in Support": {
            "court_name": {
                "question": "Which court?",
                "patterns": [r"(High Court|Court of Appeal|Supreme Court|Magistrate'?s Court)"],
                "extract": lambda m: m.group(1).strip()
            },
            "court_location": {
                "question": "Where is the court located?",
                "patterns": [r"(?:at|in)\s+(Milimani|Nairobi|Mombasa|Kisumu|Nakuru|Eldoret)"],
                "extract": lambda m: m.group(1).strip()
            },
            "case_number": {
                "question": "What is the case number?",
                "patterns": [r"([A-Z]?\d+\s*of\s*\d{4})"],
                "extract": lambda m: m.group(1).strip()
            },
            "subject_matter": {
                "question": "What is the application about?",
                "patterns": [r"(?:application for|seeking|praying for)\s+(.*?)(?:\.|$)"],
                "extract": lambda m: m.group(1).strip()[:200] if m else None
            },
            "applicant_name": {
                "question": "Who is the applicant?",
                "patterns": [r"(?:applicant is|on behalf of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip()
            },
            "deponent_name": {
                "question": "Who is swearing this affidavit?",
                "patterns": [r"(?:change|update|set|edit)\s*(?:the\s+)?(?:name|deponent)\s*(?:to|:)?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})", r"(?:my name is|I am|i am)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"],
                "extract": lambda m: m.group(1).strip()
            },
            "deponent_relationship": {
                "question": "What is your relationship to the applicant?",
                "patterns": [r"(?:I am the|as the)\s*(applicant|advocate|director|manager|agent|representative|employee)"],
                "extract": lambda m: m.group(1).strip() if m else None
            },
            "id_number": {
                "question": "What is the ID number?",
                "patterns": [r"(?:change|update|set|edit)\s*(?:the\s+)?(?:id|identification|id number)\s*(?:to|:)?\s*(\d{6,8})", r"(?:ID|id)\s*(?:number|no\.?)?\s*(?:is|:)?\s*(\d{6,8})"],
                "extract": lambda m: m.group(1).strip()
            },
            "postal_address": {
                "question": "What is the P.O. Box?",
                "patterns": [r"(?:change|update|set|edit)\s*(?:the\s+)?(?:p\.?o\.?\s*box|box|postal)\s*(?:to|:)?\s*(\d{1,6})", r"P\.?O\.?\s*Box\s*(\d{1,6})"],
                "extract": lambda m: m.group(1).strip()
            },
            "city_town": {
                "question": "Which city?",
                "patterns": [r"(?:change|update|set|edit)\s*(?:the\s+)?(?:city|town|location)\s*(?:to|:)?\s*([A-Z][a-z]+)", r"(?:in|at|from)\s+(Nairobi|Mombasa|Kisumu|Nakuru|Eldoret)"],
                "extract": lambda m: m.group(1).strip()
            },
            "facts": {
                "question": "What are the grounds in support?",
                "patterns": [r"(?:add|include|also)\s+(.+)", r"(?:grounds|reasons|because|that\s+the)\s*(.*)"],
                "extract": lambda m: m.group(1).strip() if m.lastindex and m.group(1) else (m.group(0).strip() if m else None)
            },
            "place_sworn": {
                "question": "Where will this be sworn?",
                "patterns": [r"(?:swear|sworn)\s*(?:in|at)\s*([A-Z][a-z]+)"],
                "extract": lambda m: m.group(1).strip() if m else "Nairobi"
            }
        }
    }
    
    # ============================================
    # INTENT CLASSIFICATION
    # ============================================
    
    def classify_intent(self, user_input):
        """Classify what the user wants — questions ALWAYS take priority"""
        user_lower = user_input.lower().strip()
        
        # 1. GREETINGS
        if any(g == user_lower or user_lower.startswith(g) for g in self.GREETINGS):
            return {'intent': 'greeting', 'doc_type': None}
        
        # 2. QUESTIONS — ALWAYS check first
        if self._is_clearly_question(user_input):
            return {'intent': 'question', 'doc_type': None}
        if any(q in user_lower for q in self.QUESTION_WORDS):
            return {'intent': 'question', 'doc_type': None}
        
        # 3. HELP
        if any(h in user_lower for h in self.HELP_WORDS):
            return {'intent': 'help', 'doc_type': None}
        
        # 4. COST
        if any(c in user_lower for c in self.COST_WORDS):
            return {'intent': 'cost', 'doc_type': None}
        
        # 5. DOCUMENT — only if not a question
        doc_type = self.detect_document_type(user_input)
        if doc_type:
            return {'intent': 'document', 'doc_type': doc_type}
        
        # 6. ANSWER
        if self._looks_like_answer(user_input):
            return {'intent': 'answer', 'doc_type': None}
        
        # 7. CANNOT DO
        if any(c in user_lower for c in self.CANNOT_DO):
            return {'intent': 'cannot_do', 'doc_type': None}
        
        # 8. FEEDBACK
        if any(f in user_lower for f in self.FEEDBACK_WORDS):
            return {'intent': 'feedback', 'doc_type': None}
        
        # 9. UNKNOWN
        return {'intent': 'unknown', 'doc_type': None}
    
    def _is_clearly_question(self, user_input):
        """Check if input is clearly a question"""
        user_lower = user_input.lower()
        
        question_starters = ['what', 'how', 'can', 'could', 'would', 'is', 'are', 
                            'do', 'does', 'why', 'when', 'where', 'who', 'should',
                            'will', 'did', 'has', 'have', 'am', 'was', 'were']
        
        words = user_lower.split()
        first_word = words[0] if words else ''
        first_two = ' '.join(words[:2]) if len(words) >= 2 else ''
        first_three = ' '.join(words[:3]) if len(words) >= 3 else ''
        
        all_starters = question_starters + [
            'i want to', 'i need to', 'i wish to', 'i would like to',
            'tell me', 'explain', 'help me', 'advise me',
        ]
        
        if first_word in all_starters or first_two in all_starters or first_three in all_starters:
            return True
        
        if '?' in user_lower:
            return True
        
        question_phrases = [
            'tell me', 'explain', 'process', 'procedure', 'how do', 'how can',
            'what is the', 'what are the', 'steps to', 'steps for', 'guide me',
            'i want to know', 'i need to know', 'what happens', 'how long',
            'how much', 'what to do', 'where to', 'who to', 'when to',
            'what can i', 'how do i', 'what are my', 'is there a',
            'how to', 'what do i', 'i wish to', 'i would like',
        ]
        if any(phrase in user_lower for phrase in question_phrases):
            return True
        
        if user_lower.startswith(('i want', 'i need', 'i wish', 'i would like', 'i am looking')):
            return True
        
        return False
    
    def _looks_like_answer(self, user_input):
        """Check if input looks like an answer to a question"""
        user_stripped = user_input.strip()
        
        if len(user_stripped.split()) == 1 and user_stripped[0].isupper():
            return True
        if user_stripped.replace(' ', '').isdigit():
            return True
        if len(user_stripped.split()) <= 3 and any(c.isdigit() for c in user_stripped):
            return True
        
        months = ['january','february','march','april','may','june','july',
                  'august','september','october','november','december',
                  'jan','feb','mar','apr','jun','jul','aug','sep','oct','nov','dec']
        if any(m in user_stripped.lower() for m in months):
            return True
        
        confirm_words = ['yes', 'no', 'correct', 'wrong', 'change', 'edit', 
                        'ok', 'okay', 'generate', 'proceed', 'create']
        if any(w == user_stripped.lower() for w in confirm_words):
            return True
        
        if len(user_stripped.split()) <= 3:
            command_words = ['help', 'cost', 'what', 'how', 'why', 'who', 'where', 'when']
            if not any(w in user_stripped.lower() for w in command_words):
                return True
        
        return False
    
    def get_intent_response(self, intent_data):
        """Get response for non-document intents"""
        intent = intent_data['intent']
        
        if intent == 'greeting':
            return "Jambo, welcome to Kenyan Legal Assistant. Describe your situation for an affidavit, or ask a legal question."
        elif intent == 'help':
            return "I can help you draft affidavits or answer legal questions. Try typing 'lost id' for an affidavit, or ask any legal question."
        elif intent == 'cost':
            return "This service is free. The only cost is the Commissioner for Oaths fee when swearing your affidavit (usually Ksh 200-500)."
        elif intent == 'cannot_do':
            return "I can help with affidavits and legal information, but I can't provide legal representation. Contact a licensed Kenyan advocate for court representation."
        elif intent == 'feedback':
            return "I'm sorry about that. Try rephrasing or ask a different question."
        elif intent == 'unknown':
            return "I'm not sure what you need. You can describe your situation for an affidavit, or ask a legal question."
        
        return "How can I help you?"
    
    # ============================================
    # DOCUMENT DETECTION
    # ============================================
    
    def detect_document_type(self, user_input):
        """Detect which document type the user needs"""
        user_lower = user_input.lower().strip()
        
        if user_lower in ['id', 'lost id', 'lost', 'identity', 'general', 'affidavit']:
            return "General Affidavit"
        if user_lower in ['service', 'served', 'serve']:
            return "Affidavit of Service"
        if user_lower in ['support', 'application', 'motion']:
            return "Affidavit in Support"
        
        if any(w in user_lower for w in ['lost', 'id', 'passport', 'certificate', 'name change', 'residence']):
            return "General Affidavit"
        if any(w in user_lower for w in ['service', 'served', 'process server', 'summons', 'plaint']):
            return "Affidavit of Service"
        if any(w in user_lower for w in ['support', 'application', 'motion', 'injunction', 'stay', 'appeal']):
            return "Affidavit in Support"
        
        gs = sum(1 for s in self.GENERAL_SIGNALS if s in user_lower)
        ss = sum(1 for s in self.SERVICE_SIGNALS if s in user_lower)
        ps = sum(1 for s in self.SUPPORT_SIGNALS if s in user_lower)
        
        if ss > gs and ss > ps: return "Affidavit of Service"
        if ps > gs and ps > ss: return "Affidavit in Support"
        if gs > 0 or ss > 0 or ps > 0: return "General Affidavit"
        
        return None
    
    # ============================================
    # FIELD EXTRACTION
    # ============================================
    
    def extract_all_fields(self, user_input, doc_type):
        """Extract all possible fields from user input"""
        fields = {}
        for fn, fc in self.FIELD_QUESTIONS.get(doc_type, {}).items():
            for p in fc.get("patterns", []):
                m = re.search(p, user_input, re.IGNORECASE)
                if m:
                    try:
                        v = fc["extract"](m)
                        if v:
                            fields[fn] = v
                            break
                    except:
                        pass
        return fields
    
    def get_missing_fields(self, doc_type, collected_fields):
        """Get fields still needed"""
        af = self.FIELD_QUESTIONS.get(doc_type, {})
        return [{"field": fn, "question": fc["question"]} 
                for fn, fc in af.items() 
                if fn not in collected_fields or not collected_fields[fn]]
    
    def get_next_question(self, doc_type, collected_fields):
        """Get the next question to ask"""
        m = self.get_missing_fields(doc_type, collected_fields)
        return m[0]["question"] if m else None
    
    def is_complete(self, doc_type, collected_fields):
        """Check if essential fields are collected"""
        m = self.get_missing_fields(doc_type, collected_fields)
        essential = ["deponent_name", "facts", "id_number"]
        for x in m:
            if x["field"] in essential:
                return False
        return len(m) <= 2
    
    def generate_summary(self, doc_type, collected_fields):
        """Generate human-readable summary"""
        lines = [f"{doc_type}\n"]
        for f, v in collected_fields.items():
            if v:
                lines.append(f"• {f.replace('_', ' ').title()}: {v}")
        if not lines[1:]:
            lines.append("No information extracted yet")
        return "\n".join(lines)