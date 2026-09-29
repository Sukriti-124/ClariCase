"""Stable import location for preprocessing persisted in fitted vectorizers."""
import html, re, unicodedata

def clean_text(text):
    text=html.unescape(unicodedata.normalize('NFKC',text)).lower().replace('’',"'")
    for a,b in [("can't","can not"),("won't","will not"),("shan't","shall not")]: text=text.replace(a,b)
    text=re.sub(r"n\'t\b",' not',text)
    text=re.sub(r'https?://\S+|www\.\S+|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b',' ',text)
    text=re.sub(r'\bx{2,}\b',' ',text)
    text=re.sub(r'\b\d{6,}\b',' number ',text)
    return ' '.join(text.split())
