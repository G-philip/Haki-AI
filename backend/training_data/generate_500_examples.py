"""Generate 500+ training examples for Kenyan legal AI — proper JSON escaping"""

import json
import random
from pathlib import Path

# Load knowledge base
kb_file = Path(__file__).parent.parent / "data" / "legal_knowledge.json"
with open(kb_file, 'r', encoding='utf-8') as f:
    knowledge = json.load(f)

training_data = []

# ============================================
# 1. Q&A PAIRS (150 examples)
# ============================================
question_variations = [
    "{}",
    "can you tell me {}",
    "I need to know {}",
    "help me understand {}",
    "explain {}",
    "tell me about {}",
    "what do you know about {}",
]

for topic_id, topic in knowledge.items():
    base_question = topic["question"].lower().rstrip('?')
    answer = topic["answer"]
    
    for var in question_variations:
        training_data.append({
            "instruction": var.format(base_question),
            "input": "",
            "output": answer.replace('\n', ' ').strip()
        })

# ============================================
# 2. AFFIDAVIT GENERATION (200 examples)
# ============================================
names = ["John Mwangi", "Jane Wanjiku", "Peter Njoroge", "Alice Akinyi",
         "David Otieno", "Mary Wambui", "James Kiprotich", "Grace Akoth",
         "Tom Odhiambo", "Sarah Chebet"]

ids = [str(random.randint(1000000, 9999999)) for _ in range(30)]
cities = ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Thika", "Nyeri", "Juja"]
boxes = [str(random.randint(100, 9999)) for _ in range(30)]

scenarios = [
    "lost my National ID card",
    "lost my passport",
    "need proof of residence",
    "changed my name",
    "lost my driving license",
    "lost my birth certificate",
    "need to prove my identity",
    "lost my academic certificates",
]

for _ in range(200):
    name = random.choice(names)
    scenario = random.choice(scenarios)
    city = random.choice(cities)
    id_num = random.choice(ids)
    box = random.choice(boxes)
    
    questions = [
        f"Generate an affidavit for {name} who {scenario} in {city}",
        f"I need an affidavit because I {scenario}",
        f"Draft a general affidavit for {name}",
        f"Help me swear an affidavit. I {scenario} in {city}",
        f"I {scenario}. I need an affidavit",
    ]
    
    response = f"I'll help you draft a General Affidavit for {name}. The deponent details are: {name}, ID {id_num}, P.O. Box {box}, {city}. The facts are: {scenario}. The affidavit will follow the Oaths and Statutory Declarations Act, Cap. 15. Would you like me to generate it?"
    
    training_data.append({
        "instruction": random.choice(questions),
        "input": f"Name: {name}, ID: {id_num}, Box: {box}, City: {city}",
        "output": response
    })

# ============================================
# 3. LEGAL PROCEDURES (100 examples)
# ============================================
procedure_qa = [
    {
        "q": "how do I file a case in Kenya",
        "a": "To file a civil case in Kenya: 1. Draft a Plaint stating your claim. 2. Prepare a Verifying Affidavit sworn before a Commissioner for Oaths. 3. File at the court registry and pay filing fees. 4. The court issues Summons to the defendant who has 15 days to respond. 5. The case proceeds to hearing. This is legal information, not legal advice."
    },
    {
        "q": "how do I get a police abstract in Kenya",
        "a": "To get a Police Abstract: 1. Visit the nearest police station. 2. Report the incident at the OB desk. 3. Provide details of what happened. 4. You'll receive an OB number. 5. Request a Police Abstract which costs Ksh 100-200. The abstract is needed for replacing lost documents like ID cards and passports."
    },
    {
        "q": "how to register a business in Kenya",
        "a": "To register a business in Kenya: 1. Conduct a name search on eCitizen (Ksh 150). 2. Reserve your business name. 3. For companies, prepare Memorandum and Articles of Association. 4. Register on the BRS portal. 5. Pay registration fees (Ksh 10,000-50,000). 6. Get a KRA PIN. 7. Obtain county permits. This is legal information, not legal advice."
    },
    {
        "q": "what is the limitation period for civil cases in Kenya",
        "a": "Under the Limitation of Actions Act: Contract claims have 6 years, tort claims (negligence, defamation) have 3 years, land recovery has 12 years, and personal injury claims have 3 years from the date of injury. Judgments are enforceable for 12 years. Claims against the government require 30 days notice."
    },
    {
        "q": "how to apply for a work permit in Kenya",
        "a": "To apply for a work permit: 1. The employer must show no Kenyan can fill the position. 2. Apply through the eFNS portal with Form 25. 3. Submit passport copy, certificates, company registration, and tax compliance certificate. 4. Pay fees (Ksh 10,000-100,000 per year). 5. Processing takes 2-3 months."
    },
    {
        "q": "how to get a certificate of good conduct in Kenya",
        "a": "To get a Certificate of Good Conduct: 1. Register on eCitizen. 2. Go to the DCI section. 3. Fill the application form and upload your ID and photo. 4. Pay Ksh 1,050 via M-Pesa. 5. Visit DCI headquarters or regional office for fingerprinting. 6. Processing takes 5-10 working days."
    },
    {
        "q": "how to apply for a birth certificate in Kenya",
        "a": "For newborns within 6 months: Get a birth notification from the hospital, visit Civil Registration Services or Huduma Centre with parents' IDs, fill the form. Registration is free. After 6 months: Additional documents like clinic card or baptismal card may be needed plus a late fee of Ksh 200-500."
    },
    {
        "q": "how to change my name legally in Kenya",
        "a": "To legally change your name: 1. Swear an affidavit before a Commissioner for Oaths stating current name, new name, and reasons. 2. Register the affidavit at the Registrar of Documents. 3. Apply for Kenya Gazette publication (Ksh 2,000-5,000). 4. Take the gazette notice to update your ID, passport, and other documents."
    },
    {
        "q": "how to transfer land in Kenya",
        "a": "To transfer land: 1. Conduct an official land search. 2. Seller provides title deed, ID, KRA PIN, spousal consent, and rates clearance. 3. Both parties sign the transfer form. 4. Pay stamp duty (2% urban, 4% rural). 5. Submit to Land Registry. 6. New title deed issued in 2-4 weeks. Always use a licensed advocate."
    },
    {
        "q": "how to get married legally in Kenya",
        "a": "To marry legally in Kenya: 1. Give 21 days notice to the Registrar of Marriages. 2. Provide IDs, photos, and if previously married, divorce decree or death certificate. 3. Attend the ceremony with two witnesses. 4. Pay fees (Ksh 3,000-10,000). 5. Receive marriage certificate same day. Religious and customary marriages are also recognized."
    },
]

for item in procedure_qa:
    for _ in range(10):
        variations = [
            item["q"],
            f"can you tell me {item['q']}",
            f"I need to know {item['q']}",
            f"help with {item['q']}",
            f"what is the process for {item['q']}",
            f"how can {item['q']}",
            f"steps for {item['q']}",
            f"what are the requirements for {item['q']}",
            f"explain {item['q']}",
            f"guide me on {item['q']}",
        ]
        training_data.append({
            "instruction": random.choice(variations),
            "input": "",
            "output": item["a"]
        })

# ============================================
# 4. DOCUMENT EDITING (50 examples)
# ============================================
for _ in range(50):
    name = random.choice(names)
    id_num = random.choice(ids)
    city = random.choice(cities)
    
    edit_commands = [
        f"change name to {name}",
        f"update ID to {id_num}",
        f"change city to {city}",
        "add more details to the facts",
        "replace the facts with new information",
    ]
    
    cmd = random.choice(edit_commands)
    response = f"I've updated the information. Would you like to change anything else, or should I generate the document?"
    
    training_data.append({
        "instruction": cmd,
        "input": "",
        "output": response
    })

# ============================================
# SAVE — one JSON object per line, properly escaped
# ============================================
output_file = Path(__file__).parent / "kenyan_legal_training.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for item in training_data:
        # json.dumps handles all escaping properly
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Generated {len(training_data)} training examples")
print(f"Saved to: {output_file}")