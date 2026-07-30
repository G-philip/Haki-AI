# backend/training_data/seed_knowledge.py

import json
from pathlib import Path

# Common Kenyan legal questions
seed_questions = [
    # Employment
    "Can my employer fire me without notice in Kenya?",
    "What is unfair dismissal in Kenya?",
    "How much notice should I get before termination?",
    "What are my rights if I'm made redundant?",
    "How do I file a complaint against my employer?",
    "What is the minimum wage in Kenya?",
    "Can I be fired while on sick leave?",
    "What is constructive dismissal?",
    
    # Land & Property
    "How do I transfer land in Kenya?",
    "What is a title deed search?",
    "How do I resolve a land dispute?",
    "What is adverse possession in Kenya?",
    "How do I register a lease?",
    "What are land rates and how do I pay them?",
    
    # Family Law
    "How do I file for divorce in Kenya?",
    "What are the grounds for divorce?",
    "How is child custody decided?",
    "How do I change my child's name?",
    "What is a prenuptial agreement in Kenya?",
    "How do I adopt a child in Kenya?",
    
    # Criminal
    "What are my rights after arrest?",
    "How do I get bail in Kenya?",
    "What is a police abstract?",
    "How do I report a crime?",
    "What happens at a plea hearing?",
    
    # Business
    "How do I register a company in Kenya?",
    "What is a sole proprietorship?",
    "How do I get a business permit?",
    "What taxes does a small business pay?",
    "How do I register a trademark?",
    
    # More categories...
]

print(f"Add these {len(seed_questions)} questions and let Ollama answer them all")