"""
Document Generator - Production Ready
Generates properly formatted Kenyan legal documents
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DocumentGenerator:
    """Generates Kenyan legal affidavits with proper formatting"""
    
    PLACEHOLDERS = {
        "deponent_name": "[FULL NAME]",
        "id_number": "[ID NUMBER]",
        "postal_address": "[P.O. BOX]",
        "city_town": "[CITY]",
        "place_sworn": "[PLACE]",
        "facts": "[FACTS]",
        "court_name": "[COURT]",
        "court_location": "[LOCATION]",
        "case_number": "[CASE NO]",
        "plaintiff_name": "[PLAINTIFF]",
        "defendant_name": "[DEFENDANT]",
        "subject_matter": "[SUBJECT]",
        "applicant_name": "[APPLICANT]",
        "deponent_relationship": "[RELATIONSHIP]",
        "deponent_role": "[ROLE]",
        "date_of_service": "[DATE]",
        "service_address": "[ADDRESS]",
        "document_1": "[DOCUMENT]",
        "person_served": "[PERSON]",
        "relationship_to_recipient": "[RELATIONSHIP]",
        "reason_for_not_signing": "they declined to sign",
    }
    
    ALL_FIELDS = {
        "General Affidavit": [
            "deponent_name", "id_number", "postal_address",
            "city_town", "facts", "place_sworn"
        ],
        "Affidavit of Service": [
            "court_name", "court_location", "case_number",
            "plaintiff_name", "defendant_name", "deponent_name",
            "deponent_role", "id_number", "postal_address",
            "city_town", "date_of_service", "service_address",
            "document_1", "person_served", "relationship_to_recipient",
            "reason_for_not_signing", "place_sworn"
        ],
        "Affidavit in Support": [
            "court_name", "court_location", "case_number",
            "subject_matter", "applicant_name", "deponent_name",
            "deponent_relationship", "id_number", "postal_address",
            "city_town", "facts", "place_sworn"
        ],
    }
    
    # Validation rules
    VALIDATION = {
        "id_number": (r'^\d{6,8}$', "ID number must be 6-8 digits"),
        "postal_address": (r'^\d{1,6}$', "P.O. Box must be 1-6 digits"),
        "deponent_name": (r'^[A-Za-z\s\'-]{3,100}$', "Name must be 3-100 characters"),
        "case_number": (r'^[A-Za-z0-9\s/]+$', "Invalid case number format"),
    }
    
    def __init__(self):
        self._generated_count = 0
        self._error_count = 0
    
    def fill_missing(self, doc_type: str, fields: Dict) -> Dict:
        """Fill missing fields with placeholders"""
        filled = dict(fields)
        for field_name in self.ALL_FIELDS.get(doc_type, []):
            if field_name not in filled or not filled[field_name]:
                filled[field_name] = self.PLACEHOLDERS.get(
                    field_name, f"[{field_name.upper()}]"
                )
        return filled
    
    def validate_fields(self, doc_type: str, fields: Dict) -> list:
        """Validate field values"""
        errors = []
        
        for field_name, (pattern, message) in self.VALIDATION.items():
            if field_name in fields and fields[field_name]:
                value = str(fields[field_name])
                if not re.match(pattern, value):
                    errors.append(f"{field_name}: {message}")
        
        return errors
    
    def generate(self, doc_type: str, fields: Dict) -> str:
        """
        Generate document text.
        
        Args:
            doc_type: Type of affidavit
            fields: Dictionary of field values
            
        Returns:
            Formatted document text
            
        Raises:
            ValueError: If doc_type is invalid
        """
        if doc_type not in self.ALL_FIELDS:
            raise ValueError(f"Unknown document type: {doc_type}")
        
        try:
            # Validate fields
            errors = self.validate_fields(doc_type, fields)
            if errors:
                logger.warning(f"Validation warnings: {errors}")
            
            # Fill missing fields
            filled_fields = self.fill_missing(doc_type, fields)
            
            # Generate based on type
            if "Service" in doc_type:
                document = self._generate_service_affidavit(filled_fields)
            elif "Support" in doc_type:
                document = self._generate_supporting_affidavit(filled_fields)
            else:
                document = self._generate_general_affidavit(filled_fields)
            
            self._generated_count += 1
            return document
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Document generation failed: {e}", exc_info=True)
            raise
    
    def _generate_general_affidavit(self, fields: Dict) -> str:
        """Generate General Affidavit"""
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        month_name = now.strftime("%B")
        day = now.strftime("%d")
        year = now.strftime("%Y")
        
        # Format facts as numbered paragraphs
        facts_text = self._format_facts(fields.get('facts', ''))
        
        document = f"""REPUBLIC OF KENYA
IN THE MATTER OF OATHS AND STATUTORY DECLARATIONS ACT
(CAP. 15 LAWS OF KENYA)

GENERAL AFFIDAVIT

I, {fields.get('deponent_name', '[FULL NAME]')}, 
of P.O. Box {fields.get('postal_address', '[P.O. BOX]')}, 
{fields.get('city_town', '[CITY]')}, 
in the Republic of Kenya, do hereby make oath and state as follows:-

{facts_text}

What is deponed to hereinabove is true to the best of my knowledge, 
information and belief.

SWORN at {fields.get('place_sworn', '[PLACE]')} by the said
{fields.get('deponent_name', '[FULL NAME]')}
This {day} day of {month_name}, {year}

.........................................
DEPONENT

Before me:

.........................................
COMMISSIONER FOR OATHS / MAGISTRATE

---
Document generated on {date_str}
Oaths and Statutory Declarations Act, Cap. 15 Laws of Kenya
"""
        return document
    
    def _generate_service_affidavit(self, fields: Dict) -> str:
        """Generate Affidavit of Service"""
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        month_name = now.strftime("%B")
        day = now.strftime("%d")
        year = now.strftime("%Y")
        
        document = f"""REPUBLIC OF KENYA
IN THE {fields.get('court_name', '[COURT]')} AT {fields.get('court_location', '[LOCATION]')}
CIVIL CASE NO. {fields.get('case_number', '[CASE NO]')}

BETWEEN

{fields.get('plaintiff_name', '[PLAINTIFF]')}
.....................................................PLAINTIFF

AND

{fields.get('defendant_name', '[DEFENDANT]')}
.....................................................DEFENDANT

AFFIDAVIT OF SERVICE

I, {fields.get('deponent_name', '[FULL NAME]')}, 
of P.O. Box {fields.get('postal_address', '[P.O. BOX]')}, 
{fields.get('city_town', '[CITY]')}, 
in the Republic of Kenya, do hereby make oath and state as follows:-

1. THAT I am the {fields.get('deponent_role', '[ROLE]')} herein duly 
   authorized and competent to swear this affidavit.

2. THAT on the {fields.get('date_of_service', '[DATE]')}, I served 
   the following documents upon {fields.get('person_served', '[PERSON]')} 
   at {fields.get('service_address', '[ADDRESS]')}:

   a) {fields.get('document_1', '[DOCUMENT]')}

3. THAT the said {fields.get('person_served', '[PERSON]')} is the 
   {fields.get('relationship_to_recipient', '[RELATIONSHIP]')} herein.

4. THAT I requested the said {fields.get('person_served', '[PERSON]')} 
   to sign on the principal copy but they declined to sign because 
   {fields.get('reason_for_not_signing', 'they declined to sign')}.

What is deponed to hereinabove is true to the best of my knowledge, 
information and belief.

SWORN at {fields.get('place_sworn', '[PLACE]')} by the said
{fields.get('deponent_name', '[FULL NAME]')}
This {day} day of {month_name}, {year}

.........................................
DEPONENT

Before me:

.........................................
COMMISSIONER FOR OATHS / MAGISTRATE

---
Document generated on {date_str}
"""
        return document
    
    def _generate_supporting_affidavit(self, fields: Dict) -> str:
        """Generate Affidavit in Support"""
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        month_name = now.strftime("%B")
        day = now.strftime("%d")
        year = now.strftime("%Y")
        
        facts_text = self._format_facts(fields.get('facts', ''))
        
        document = f"""REPUBLIC OF KENYA
IN THE {fields.get('court_name', '[COURT]')} AT {fields.get('court_location', '[LOCATION]')}
CIVIL CASE NO. {fields.get('case_number', '[CASE NO]')}

IN THE MATTER OF: {fields.get('subject_matter', '[SUBJECT]')}

AND

IN THE MATTER OF: AN APPLICATION BY {fields.get('applicant_name', '[APPLICANT]')}

AFFIDAVIT IN SUPPORT

I, {fields.get('deponent_name', '[FULL NAME]')}, 
of P.O. Box {fields.get('postal_address', '[P.O. BOX]')}, 
{fields.get('city_town', '[CITY]')}, 
in the Republic of Kenya, do hereby make oath and state as follows:-

1. THAT I am the {fields.get('deponent_relationship', '[RELATIONSHIP]')} 
   to the {fields.get('applicant_name', '[APPLICANT]')} herein and I am 
   duly authorized and competent to swear this affidavit.

{facts_text}

What is deponed to hereinabove is true to the best of my knowledge, 
information and belief.

SWORN at {fields.get('place_sworn', '[PLACE]')} by the said
{fields.get('deponent_name', '[FULL NAME]')}
This {day} day of {month_name}, {year}

.........................................
DEPONENT

Before me:

.........................................
COMMISSIONER FOR OATHS / MAGISTRATE

---
Document generated on {date_str}
"""
        return document
    
    def _format_facts(self, facts: str) -> str:
        """Format facts as numbered paragraphs"""
        if not facts or facts.strip() == '[FACTS]':
            return "1. [FACTS TO BE SWORN]\n"
        
        # Split by newlines or sentences
        lines = []
        for part in re.split(r'\n+', facts.strip()):
            part = part.strip()
            if not part:
                continue
            
            # Split long paragraphs into sentences
            if len(part) > 200:
                sentences = re.split(r'(?<=[.!?])\s+', part)
                for sent in sentences:
                    if sent.strip():
                        lines.append(sent.strip())
            else:
                lines.append(part)
        
        # Number the paragraphs (start from 2 for support affidavits)
        formatted = []
        for i, line in enumerate(lines, 1):
            if line:
                formatted.append(f"{i}. {line}")
        
        return '\n\n'.join(formatted) if formatted else "1. [FACTS TO BE SWORN]\n"
    
    def get_stats(self) -> Dict:
        """Get generator statistics"""
        return {
            'generated_count': self._generated_count,
            'error_count': self._error_count,
            'success_rate': (
                self._generated_count / max(self._generated_count + self._error_count, 1)
            ) * 100
        }









# from templates import (
#     generate_general_affidavit,
#     generate_service_affidavit,
#     generate_supporting_affidavit
# )


# class DocumentGenerator:
#     PLACEHOLDERS = {
#         "deponent_name": "[FULL NAME]", "id_number": "[ID NUMBER]",
#         "postal_address": "[P.O. BOX]", "city_town": "[CITY]",
#         "place_sworn": "[PLACE]", "facts": "[FACTS]",
#         "court_name": "[COURT]", "court_location": "[LOCATION]",
#         "case_number": "[CASE NO]", "plaintiff_name": "[PLAINTIFF]",
#         "defendant_name": "[DEFENDANT]", "subject_matter": "[SUBJECT]",
#         "applicant_name": "[APPLICANT]", "deponent_relationship": "[RELATIONSHIP]",
#         "deponent_role": "[ROLE]", "date_of_service": "[DATE]",
#         "service_address": "[ADDRESS]", "document_1": "[DOCUMENT]",
#         "person_served": "[PERSON]", "relationship_to_recipient": "[RELATIONSHIP]",
#         "reason_for_not_signing": "they declined to sign",
#     }
    
#     ALL_FIELDS = {
#         "General Affidavit": ["deponent_name","id_number","postal_address","city_town","facts","place_sworn"],
#         "Affidavit of Service": ["court_name","court_location","case_number","plaintiff_name","defendant_name","deponent_name","deponent_role","id_number","postal_address","city_town","date_of_service","service_address","document_1","person_served","relationship_to_recipient","reason_for_not_signing","place_sworn"],
#         "Affidavit in Support": ["court_name","court_location","case_number","subject_matter","applicant_name","deponent_name","deponent_relationship","id_number","postal_address","city_town","facts","place_sworn"],
#     }
    
#     def fill_missing(self, doc_type, fields):
#         for f in self.ALL_FIELDS.get(doc_type, []):
#             if f not in fields or not fields[f]:
#                 fields[f] = self.PLACEHOLDERS.get(f, f"[{f.upper()}]")
#         return fields
    
#     def generate(self, doc_type, fields):
#         fields = self.fill_missing(doc_type, dict(fields))
#         if "Service" in doc_type:
#             return generate_service_affidavit(fields)
#         elif "Support" in doc_type:
#             return generate_supporting_affidavit(fields)
#         else:
#             return generate_general_affidavit(fields)