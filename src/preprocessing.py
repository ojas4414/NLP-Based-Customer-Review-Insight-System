import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords",quiet=True)
nltk.download("wordnet",quiet=True)
nltk.download("omw-1.4", quiet=True)
_stopwords=set(stopwords.words("english"))
_lemmatizers=WordNetLemmatizer()

def clean_text(text:str)-> str:
    if not isinstance(text,str):
        return ""
    
    text=text.lower()

    text=re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens=text.split()

    tokens=[_lemmatizers.lemmatize(word) for word in tokens if word not in _stopwords ]

    return " ".join(tokens)