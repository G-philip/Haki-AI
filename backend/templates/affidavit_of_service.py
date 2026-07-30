from templates.base_template import REPUBLIC_HEADER, COMMISSIONER_BLOCK, DRAWN_BY
from datetime import datetime

def generate_service_affidavit(data):
    document = f"""                                REPUBLIC OF KENYA

                     IN THE {data.get('court_name', '[COURT]').upper()}
                         AT {data.get('court_location', '[LOCATION]').upper()}

                          CASE NO. {data.get('case_number', '[CASE NUMBER]')}

BETWEEN

{data.get('plaintiff_name', '[PLAINTIFF]').upper()}
                                                              PLAINTIFF/APPLICANT

AND

{data.get('defendant_name', '[DEFENDANT]').upper()}
                                                              DEFENDANT/RESPONDENT


                            AFFIDAVIT OF SERVICE

I, {data.get('deponent_name', '[FULL NAME]').upper()}, of P.O. Box {data.get('postal_address', '[P.O. BOX]')} – {data.get('city_town', '[CITY]').upper()}, holder of National Identity Card Number {data.get('id_number', '[ID NUMBER]')}, do hereby make oath and state as follows:—

     1. THAT I am a {data.get('deponent_role', 'Process Server').upper()} in this matter and therefore competent to swear this Affidavit.

     2. THAT on the {data.get('date_of_service', '[DATE]')}, I received from {data.get('source_of_documents', '[SOURCE]').upper()} the following documents for service upon the {data.get('recipient_role', '[RECIPIENT]').upper()}:
        (a) {data.get('document_1', '[DOCUMENT]')}
        (b) {data.get('document_2', '') if data.get('document_2') else ''}
        (c) {data.get('document_3', '') if data.get('document_3') else ''}

     3. THAT on the same date, at approximately {data.get('time_of_service', '[TIME]')}, I proceeded to the {data.get('recipient_role', '[RECIPIENT]').upper()}'s known address at {data.get('service_address', '[ADDRESS]').upper()} and effected service upon {data.get('person_served', '[PERSON]').upper()}, {data.get('relationship_to_recipient', '[RELATIONSHIP]').upper()} of the {data.get('recipient_role', '[RECIPIENT]').upper()}.

     4. THAT the said {data.get('person_served', '[PERSON]').upper()} accepted service but declined to sign on the copies retained by me citing that {data.get('reason_for_not_signing', 'they were not authorized to sign')}.

     5. THAT I verily believe that the {data.get('recipient_role', '[RECIPIENT]').upper()} was duly served within the prescribed time and this Honourable Court now has jurisdiction to proceed with this matter.

     6. THAT what is deponed to hereinabove is true to the best of my knowledge, information and belief.

SWORN at {data.get('place_sworn', '[PLACE]').upper()} by the said
{data.get('deponent_name', '[FULL NAME]').upper()}
this _____ day of _______________, {datetime.now().year}

…………………………………
DEPONENT

{COMMISSIONER_BLOCK}

{DRAWN_BY}
"""
    return document