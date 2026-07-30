"""
Legal Q&A Search Tool - Production Ready
Self-learning knowledge base with AI fallback
"""

import json
import re
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class LegalSearchTool:
    """
    Searches Kenyan legal knowledge base.
    Self-learns from AI responses for future instant answers.
    """
    
    MAX_KB_SIZE = 1000
    MAX_UNKNOWN_SIZE = 5000
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.kb_file = self.data_dir / "legal_knowledge.json"
        self.unknown_file = self.data_dir / "unknown_questions.jsonl"
        
        self.knowledge_base = self._load_knowledge_base()
        
        # Load API key
        self.gemini_api_key = self._load_api_key()
        self.gemini_available = bool(self.gemini_api_key)
        
        logger.info(
            f"Search tool ready. Topics: {len(self.knowledge_base)}. "
            f"Gemini: {'Configured' if self.gemini_available else 'Not configured'}"
        )
    
    def _load_api_key(self) -> str:
        """Load Gemini API key from environment or .env file"""
        # Check environment first
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            return api_key
        
        # Check .env file
        try:
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip()
        except:
            pass
        
        return ""
    
    def _load_knowledge_base(self) -> Dict:
        """Load knowledge base from file"""
        if self.kb_file.exists():
            try:
                with open(self.kb_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                logger.warning("Knowledge base corrupted, starting fresh")
        return {}
    
    def _save_knowledge_base(self):
        """Save knowledge base atomically"""
        temp_file = self.kb_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.kb_file)
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def _stem_word(self, word: str) -> str:
        """Simple word stemming"""
        word = word.lower()
        if word.endswith('ies'): return word[:-3] + 'y'
        elif word.endswith('es'): return word[:-2]
        elif word.endswith('s') and not word.endswith('ss'): return word[:-1]
        if word.endswith('ing') and len(word) > 5: return word[:-3]
        if word.endswith('ed') and len(word) > 4: return word[:-2]
        return word
    
    def search(self, query: str, min_score: int = 5) -> List[Dict]:
        """
        Search knowledge base for matching answers.
        
        Args:
            query: User's question
            min_score: Minimum relevance score to include
            
        Returns:
            List of matching results with scores
        """
        query_lower = query.lower()
        
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are',
            'the', 'a', 'an', 'i', 'my', 'me', 'do', 'does', 'can',
            'if', 'his', 'her', 'their', 'one', 'happens', 'happen',
            'get', 'got', 'need', 'want', 'has', 'have', 'had',
            'was', 'were', 'be', 'been', 'being', 'will', 'would',
            'could', 'should', 'may', 'might', 'shall', 'to', 'of',
            'in', 'on', 'at', 'for', 'with', 'by', 'from', 'up',
        }
        
        query_terms = {
            self._stem_word(w) 
            for w in query_lower.split() 
            if w not in stop_words and len(w) > 1
        }
        
        if not query_terms:
            return []
        
        results = []
        
        for topic_id, topic_data in self.knowledge_base.items():
            keywords = topic_data.get("keywords", [])
            question = topic_data.get("question", "").lower()
            
            # Keyword matching
            all_keyword_stems = set()
            for kw in keywords:
                for word in kw.split():
                    stem = self._stem_word(word)
                    if len(stem) > 1:
                        all_keyword_stems.add(stem)
            
            matching_terms = query_terms & all_keyword_stems
            
            # Phrase matching (higher weight)
            phrase_matches = sum(10 for kw in keywords if kw in query_lower)
            
            # Question similarity
            question_stems = {
                self._stem_word(w) 
                for w in question.split() 
                if w not in stop_words
            }
            question_match = len(query_terms & question_stems)
            
            # Calculate total score
            total_score = (
                len(matching_terms) * 5 +
                phrase_matches +
                question_match * 2
            )
            
            if total_score >= min_score:
                results.append({
                    "topic_id": topic_id,
                    "question": topic_data.get("question", ""),
                    "answer": topic_data.get("answer", ""),
                    "source": topic_data.get("source", ""),
                    "score": total_score,
                    "keywords_matched": list(matching_terms),
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def answer_question(self, query: str) -> str:
        """
        Answer question with local knowledge → Gemini AI fallback.
        """
        # Step 1: Check local knowledge
        results = self.search(query)
        
        if results:
            best = results[0]
            logger.info(f"Local answer found (score: {best['score']})")
            
            # Include related answer if relevant
            if len(results) > 1 and results[1]["score"] >= best["score"] * 0.8:
                return (
                    f"{best['answer']}\n\n"
                    f"📌 Related: {results[1]['question']}\n"
                    f"{results[1]['answer']}"
                )
            
            return best['answer']
        
        # Step 2: Try Gemini AI
        if self.gemini_available:
            ai_result = self._ask_gemini(query)
            if ai_result:
                self._learn_from_source(query, ai_result, "Google Gemini AI")
                return ai_result
        
        # Step 3: Save as unknown
        self._save_unknown_question(query)
        
        return self._get_fallback_response(query)
    
    def _ask_gemini(self, query: str) -> Optional[str]:
        """Ask Google Gemini AI for answer"""
        if not self.gemini_api_key:
            return None
        
        try:
            import requests
            
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/gemini-2.0-flash:generateContent?key={self.gemini_api_key}"
            )
            
            prompt = f"""You are a Kenyan legal expert. Answer this question accurately.

Question: {query}

Guidelines:
- Provide specific information about Kenyan law
- Cite relevant acts, sections, or regulations
- Use plain language without markdown
- Include practical steps if relevant
- End with: "This is legal information, not legal advice. Consult a licensed Kenyan advocate."
- Keep under 500 words"""
            
            response = requests.post(
                url,
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }]
                },
                timeout=20
            )
            
            if response.ok:
                data = response.json()
                candidates = data.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        answer = parts[0].get('text', '')
                        if answer and len(answer) > 50:
                            logger.info(f"Gemini answered: {query[:50]}...")
                            return answer
            
            logger.warning(f"Gemini API error: {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            return None
    
    def _learn_from_source(self, query: str, answer: str, source: str):
        """Save answer to knowledge base for future instant responses"""
        # Generate topic ID
        topic_id = re.sub(r'[^a-z0-9]', '_', query.lower())[:50]
        
        # Don't overwrite existing
        if topic_id in self.knowledge_base:
            return
        
        # Don't exceed max size
        if len(self.knowledge_base) >= self.MAX_KB_SIZE:
            # Remove oldest entry
            oldest = min(
                self.knowledge_base.keys(),
                key=lambda k: self.knowledge_base[k].get('learned_at', '')
            )
            del self.knowledge_base[oldest]
        
        # Extract keywords
        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are',
            'the', 'a', 'an', 'i', 'my', 'me', 'do', 'does', 'can',
            'if', 'has', 'have', 'to', 'of', 'in', 'on', 'at',
        }
        keywords = [
            w.lower() for w in query.split()
            if w.lower() not in stop_words and len(w) > 2
        ]
        
        question = query.strip()
        if not question.endswith('?'):
            question += '?'
        
        self.knowledge_base[topic_id] = {
            "keywords": keywords,
            "source": source,
            "question": question,
            "answer": answer,
            "learned_at": datetime.now().isoformat(),
            "access_count": 0,
        }
        
        self._save_knowledge_base()
        logger.info(f"Learned new topic: {question[:60]}...")
    
    def _save_unknown_question(self, query: str):
        """Save question that couldn't be answered"""
        try:
            # Check file size
            if self.unknown_file.exists():
                size = self.unknown_file.stat().st_size
                if size > self.MAX_UNKNOWN_SIZE * 200:  # ~1MB
                    # Rotate file
                    old = self.unknown_file.with_suffix('.old.jsonl')
                    self.unknown_file.rename(old)
            
            entry = {
                "question": query,
                "timestamp": datetime.now().isoformat(),
            }
            
            with open(self.unknown_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save unknown question: {e}")
    
    def _get_fallback_response(self, query: str) -> str:
        """Get helpful fallback when no answer found"""
        query_lower = query.lower()
        
        # Find related topics
        topics_mentioned = []
        for topic_id, data in self.knowledge_base.items():
            if any(kw in query_lower for kw in data.get("keywords", [])):
                topics_mentioned.append(data.get("question", ""))
        
        if topics_mentioned:
            return (
                "I have some related information. Try asking:\n\n" +
                "\n".join([f"• {q}" for q in topics_mentioned[:3]])
            )
        
        return (
            "I don't have information on that yet.\n\n"
            "To get AI-powered answers, add a free Google Gemini API key "
            "to the backend/.env file.\n"
            "Get one at: https://aistudio.google.com\n\n"
            "In the meantime, I can help with:\n"
            "• Drafting affidavits for lost documents or court cases\n"
            "• Questions about Kenyan legal procedures\n"
            "• Court document requirements\n\n"
            "What would you like to know?"
        )
    
    def get_stats(self) -> Dict:
        """Get search tool statistics"""
        return {
            "known_topics": len(self.knowledge_base),
            "gemini_available": self.gemini_available,
            "unknown_questions": (
                sum(1 for _ in open(self.unknown_file)) 
                if self.unknown_file.exists() else 0
            ),
            "top_categories": self._get_top_categories(),
        }
    
    def _get_top_categories(self, limit: int = 5) -> List[str]:
        """Get most common knowledge categories"""
        sources = {}
        for data in self.knowledge_base.values():
            source = data.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
        
        return sorted(sources, key=sources.get, reverse=True)[:limit]