from templates.base_template import REPUBLIC_HEADER, COMMISSIONER_BLOCK, DRAWN_BY
from datetime import datetime

def generate_general_affidavit(data):
    facts_text = data.get('facts', '[FACTS TO BE SWORN]')
    facts_list = [f.strip() for f in facts_text.split('\n') if f.strip()]
    
    fact_paragraphs = ""
    fact_num = 2
    for fact in facts_list:
        if fact and not fact.endswith('.'):
            fact += '.'
        fact_paragraphs += f"     {fact_num}. THAT {fact}\n\n"
        fact_num += 1
    
    verification_num = fact_num
    
    document = f"""                                REPUBLIC OF KENYA

            IN THE MATTER OF THE OATHS AND STATUTORY DECLARATIONS ACT, CAP. 15 LAWS OF KENYA                             
                                          AND
            IN THE MATTER OF AN APPLICATION BY {data.get('deponent_name', '[FULL NAME]').upper()}
                                    AFFIDAVIT

I, {data.get('deponent_name', '[FULL NAME]').upper()}, of P.O. Box {data.get('postal_address', '[P.O. BOX]')} – {data.get('city_town', '[CITY]').upper()}, in the Republic of Kenya, holder of National Identity Card Number {data.get('id_number', '[ID NUMBER]')}, do hereby make oath and state as follows:—

     1. THAT I am the Deponent herein and therefore competent to swear this Affidavit.

{fact_paragraphs}     {verification_num}. THAT what is deponed to hereinabove is true to the best of my knowledge, information and belief.

SWORN at {data.get('place_sworn', '[PLACE]').upper()} by the said
{data.get('deponent_name', '[FULL NAME]').upper()}
this _____ day of _______________, {datetime.now().year}

…………………………………
DEPONENT

{COMMISSIONER_BLOCK}

{DRAWN_BY}
"""
    return document