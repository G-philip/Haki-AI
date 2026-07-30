from templates.base_template import REPUBLIC_HEADER, COMMISSIONER_BLOCK, DRAWN_BY
from datetime import datetime

def generate_supporting_affidavit(data):
    facts_text = data.get('facts', '[GROUNDS IN SUPPORT]')
    facts_list = [f.strip() for f in facts_text.split('\n') if f.strip()]
    
    fact_paragraphs = ""
    fact_num = 2
    for fact in facts_list:
        if fact and not fact.endswith('.'):
            fact += '.'
        fact_paragraphs += f"     {fact_num}. THAT {fact}\n\n"
        fact_num += 1
    
    app_num = fact_num
    ver_num = fact_num + 1
    
    document = f"""                                REPUBLIC OF KENYA

                     IN THE {data.get('court_name', '[COURT]').upper()}
                         AT {data.get('court_location', '[LOCATION]').upper()}

                    APPLICATION NO. {data.get('case_number', '[CASE NUMBER]')}

                       IN THE MATTER OF {data.get('subject_matter', '[SUBJECT MATTER]').upper()}

                                          AND

                 IN THE MATTER OF AN APPLICATION BY {data.get('applicant_name', '[APPLICANT]').upper()}


                               AFFIDAVIT IN SUPPORT

I, {data.get('deponent_name', '[FULL NAME]').upper()}, of P.O. Box {data.get('postal_address', '[P.O. BOX]')} – {data.get('city_town', '[CITY]').upper()}, holder of National Identity Card Number {data.get('id_number', '[ID NUMBER]')}, do hereby make oath and state as follows:—

     1. THAT I am the {data.get('deponent_relationship', '[RELATIONSHIP]').upper()} herein and therefore competent and duly authorized to swear this Affidavit on behalf of the Applicant.

{fact_paragraphs}     {app_num}. THAT I swear this Affidavit in support of the Application dated {data.get('application_date', '[DATE]')} seeking the orders set out therein.

     {ver_num}. THAT what is deponed to hereinabove is true to the best of my knowledge, information and belief.

SWORN at {data.get('place_sworn', '[PLACE]').upper()} by the said
{data.get('deponent_name', '[FULL NAME]').upper()}
this _____ day of _______________, {datetime.now().year}

…………………………………
DEPONENT

{COMMISSIONER_BLOCK}

{DRAWN_BY}
"""
    return document