from datetime import datetime

def format_paragraphs(facts_text, start_number=2):
    facts_list = [f.strip() for f in facts_text.split('\n') if f.strip()]
    if not facts_list:
        return "", start_number
    
    paragraphs = ""
    for i, fact in enumerate(facts_list, start=start_number):
        fact = fact.strip()
        if fact and not fact.endswith(('.', '!', '?')):
            fact += '.'
        paragraphs += f"{i}. THAT {fact}\n\n"
    
    return paragraphs, len(facts_list) + start_number


def format_date_legal(d):
    if isinstance(d, str):
        d = datetime.strptime(d, '%Y-%m-%d').date()
    day = d.strftime('%d').lstrip('0')
    if day.endswith('1') and day != '11': suffix = 'st'
    elif day.endswith('2') and day != '12': suffix = 'nd'
    elif day.endswith('3') and day != '13': suffix = 'rd'
    else: suffix = 'th'
    return f"{day}{suffix} day of {d.strftime('%B, %Y')}"