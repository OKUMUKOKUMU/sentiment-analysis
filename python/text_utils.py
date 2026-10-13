"""Text processing helpers for the Maziwa Bora sentiment analysis.

Everything here is hand-built (no NLTK / spaCy downloads), so the logic is visible and auditable:
  * clean_text()        lower-casing, brand masking, emoji -> tokens, slang & (optionally) Kiswahili/Sheng normalisation
  * tokenize()          simple regex tokeniser
  * lexicon_text()      text prepared for the VADER lexicon (brand & product names masked, Swahili/Sheng translated)
  * extended_analyzer() VADER + a dairy / Kenyan-English domain lexicon
  * detect_aspects()    keyword / regex aspect detection, clause by clause
"""
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------------- brand & catalogue
BRAND_RE = re.compile(r"@?maziwa\s?bora\b|@maziwabora\b|#maziwabora\b|\bmaziwabora\b", re.I)
CATALOGUE = ["baobab honey yoghurt", "greek-style yoghurt", "strawberry yoghurt", "vanilla yoghurt", "plain yoghurt",
             "mango yoghurt", "500ml milk packet", "1l long-life milk", "fresh milk", "vanilla ice cream",
             "choc-chip ice cream", "2l ice cream tub", "ice cream cones", "cheese slices", "brown bread", "milk buns",
             "queen cakes", "beef sausages", "chicken sausages", "ham slices", "smokies", "cheddar", "mozzarella",
             "mala", "lala", "yoghurt", "ice cream", "cheese", "bread", "milk"]
CATALOGUE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in CATALOGUE) + r")\b", re.I)

# ----------------------------------------------------------------------------- emojis
EMOJI_MAP = {"😍": "emj_love", "👌": "emj_ok", "🔥": "emj_fire", "😋": "emj_yum", "👍": "emj_thumbsup", "❤": "emj_heart",
             "🙌": "emj_praise", "💯": "emj_100", "😡": "emj_angry", "😠": "emj_angry", "🤮": "emj_vomit",
             "👎": "emj_thumbsdown", "😤": "emj_huff", "😞": "emj_sad", "💔": "emj_brokenheart", "🤔": "emj_thinking",
             "🙏": "emj_pray", "👀": "emj_eyes", "🙄": "emj_eyeroll", "🙃": "emj_upsidedown", "😒": "emj_unamused",
             "👏": "emj_clap", "😅": "emj_sweatsmile"}
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

# ----------------------------------------------------------------------------- slang / typos (English side)
SLANG = {"u": "you", "gud": "good", "pls": "please", "coz": "because", "tho": "though", "rly": "really", "wth": "with",
         "4": "for", "gr8": "great", "thx": "thanks", "ppl": "people", "b4": "before", "smh": "shaking my head"}

# ----------------------------------------------------------------------------- Kiswahili / Sheng -> English
# ORDER MATTERS: multi-word expressions first, because single words change meaning in context
# ("tamu mbaya" = extremely tasty, but "mbaya" alone = bad; "sio mbaya" = not bad).
SW_PHRASES = [
    (r"\btamu mbaya\b", "extremely delicious"), (r"\bsio mbaya\b", "not bad"),
    (r"\bbila kuharibika\b", "without spoiling"), (r"\bsio kama zamani\b", "worse than before"),
    (r"\bhaina ladha\b", "tasteless"), (r"\bina ladha mbaya\b", "tastes bad"), (r"\bmaji maji\b", "watery"),
    (r"\bni maji tu\b", "is just watery"), (r"\bimekuwa mbaya\b", "has become bad"),
    (r"\bbei iko juu\b", "expensive"), (r"\bbei (ya \w+(?: \w+)? )?imepanda\b", "price hike"),
    (r"\bbei iko (poa|sawa)\b", "fair price"), (r"\bbei ni nafuu\b", "affordable price"),
    (r"\bhawashiki simu\b", "unresponsive"), (r"\bhaikuja\b", "failed to come"), (r"\bilikuja nusu\b", "incomplete"),
    (r"\bwatoto wanaipenda\b", "kids love it"), (r"\bsitanunua tena\b", "boycott"), (r"\bnimechoka na\b", "fedup with"),
    (r"\brahisi kufungua\b", "easy to open"), (r"\bkwa urahisi\b", "easily"), (r"\btumekosa stock\b", "unavailable"),
    (r"\bimeisha\b", "unavailable"), (r"\bhaipatikani\b", "unavailable"), (r"\binapatikana\b", "available"),
    (r"\bimeharibika\b", "spoiled"), (r"\bkuharibika\b", "spoiling"), (r"\bimeexpire\b", "expired"),
    (r"\bimeganda\b", "curdled"), (r"\binanuka\b", "smells bad"), (r"\bukungu\b", "mould"), (r"\binavuja\b", "leaking"),
    (r"\bilipasuka\b", "burst"), (r"\bimechelewa\b", "late"), (r"\bwalinisaidia\b", "helpful"),
    (r"\buchachu\b", "sourness"), (r"\bmpole\b", "polite"), (r"\bmapema\b", "early"), (r"\bharaka\b", "quickly"),
    (r"\bladha\b", "taste"), (r"\bnafuu\b", "affordable"), (r"\bghali\b", "expensive"),
]
SW_WORDS = {"poa": "good", "safi": "great", "noma": "awesome", "inabamba": "awesome", "fiti": "fine", "tamu": "delicious",
            "mbaya": "bad", "kabisa": "totally", "sio": "not", "hapana": "no", "asante": "thanks", "maziwa": "milk",
            "mkate": "bread", "bei": "price", "kifuniko": "lid", "mpya": "new", "lakini": "but", "kila": "every",
            "duka": "shop", "maduka": "shops", "wiki": "week", "siku": "days", "hizi": "these", "tena": "again",
            "iko": "is", "ni": "is", "hii": "this", "na": "and"}
# English multi-word expressions -> single domain tokens the lexicon can score
EN_PHRASES = [
    (r"\b(out of stock|sold out|no stock|empty shelves)\b", "unavailable"),
    (r"\b(gone bad|went bad|go bad|gone off|had gone bad)\b", "spoiled"),
    (r"\bprice (went up|has gone up|increase)\b", "price hike"), (r"\bon time\b", "punctual"),
    (r"\brip-off\b", "ripoff"), (r"\bvalue for money\b", "good value"), (r"\bon hold for ages\b", "kept waiting"),
]


def _sana(text):
    """Kiswahili puts the intensifier AFTER the word ('poa sana'); VADER expects it BEFORE ('very good')."""
    return re.sub(r"\b(\w+) sana\b", r"sana \1", text)


def normalise_swahili(text):
    for pat, rep in SW_PHRASES:
        text = re.sub(pat, rep, text)
    text = _sana(text).replace("sana", "very")
    return " ".join(SW_WORDS.get(w, w) for w in text.split(" "))


def clean_text(text, normalise_sw=False):
    """Clean one raw feedback string for the ML models."""
    t = str(text)
    t = BRAND_RE.sub(" brandtoken ", t)
    for e, tok in EMOJI_MAP.items():
        t = t.replace(e, f" {tok} ")
    t = EMOJI_RE.sub(" emj_other ", t).replace("️", "")
    t = t.lower()
    t = re.sub(r"https?://\S+", " url ", t)
    t = re.sub(r"@\w+", " mention ", t)
    t = re.sub(r"#(\w+)", r" \1 ", t)
    t = re.sub(r"\bmb\d{4,}\b", " batchno ", t)              # batch numbers
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)                      # sooooo -> soo
    t = re.sub(r"[^a-z0-9_'\s-]", " ", t)
    t = " ".join(SLANG.get(w, w) for w in t.split())
    if normalise_sw:
        t = normalise_swahili(t)
    return re.sub(r"\s+", " ", t).strip()


TOKEN_RE = re.compile(r"[a-z_][a-z_'-]*|\d+")


def tokenize(text):
    return TOKEN_RE.findall(text)


EN_STOP = set("""a an the and or of to in on at for from by with is are was were be been being am i me my we our you your
it its this that these those he she they them their his her as so than then there here just also very really too
do does did have has had will would can could should may might shall what which who whom when where why how all any
some such into out up down over again once only own same other more most about after before because while if
s t don don't doesn't didn't can't won't im i'm it's thats that's get got go going went one two three week day days""".split())
NEGATIONS = {"not", "no", "never", "nobody", "nothing", "none", "cannot", "can't", "don't", "doesn't", "didn't", "isn't",
             "wasn't", "won't", "without", "sio", "hapana"}
SW_STOP = set("""na ya wa kwa ni iko hii hiyo kama tu pia za la cha vya kuhusu sasa hizi siku wiki mimi wewe yeye sisi
nasema habari tena""".split())
INTENSIFIERS = set("""totally super extremely really so very much too again now always these lately anymore though although
however but also even ever""".split())
STOPWORDS = (EN_STOP | SW_STOP | {"brandtoken", "mention", "url", "batchno", "emj_other"}) - NEGATIONS

# ----------------------------------------------------------------------------- lexicon
DOMAIN_LEXICON = {
    # freshness / expiry
    "expired": -2.3, "expire": -1.5, "expires": -1.5, "spoiled": -2.3, "spoiling": -2.0, "spoil": -1.8,
    "stale": -2.0, "mouldy": -2.5, "mould": -2.0, "slimy": -2.0, "curdled": -2.5, "sour": -1.5, "sourness": -1.5,
    "fresh": 1.6,
    # taste / quality
    "watery": -1.6, "bland": -1.5, "chalky": -1.5, "tasteless": -2.2, "aftertaste": -1.0, "lumps": -1.0,
    "creamy": 1.8, "tasty": 2.2, "smooth": 1.2, "rich": 1.2,
    # price / value
    "expensive": -1.6, "overpriced": -2.2, "pricey": -1.2, "costly": -1.5, "ripoff": -2.5, "hike": -1.6,
    "affordable": 1.8, "cheaper": 0.8,
    # packaging
    "leaking": -2.0, "leaks": -2.0, "leaked": -2.0, "leak": -1.8, "burst": -1.5, "flimsy": -1.5, "cracked": -1.5,
    "spilling": -1.0, "resealable": 1.0, "premium": 1.2, "handy": 1.2,
    # availability / service
    "unavailable": -1.8, "punctual": 1.6, "polite": 1.6, "unresponsive": -1.8, "incomplete": -1.5, "late": -1.5,
    "delayed": -1.5, "boycott": -2.0, "fedup": -2.2,
}


def lexicon_text(text):
    """Text prepared for the extended lexicon: brand & product names masked, slang and Swahili/Sheng translated.
    Emojis are left in place because VADER scores them itself."""
    t = BRAND_RE.sub(" ", str(text))
    t = CATALOGUE_RE.sub(" product ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    low = t.lower()
    low = " ".join(SLANG.get(w, w) for w in low.split())
    for pat, rep in EN_PHRASES:
        low = re.sub(pat, rep, low)
    low = normalise_swahili(low)
    # keep VADER's capitalisation emphasis for words that were ALL CAPS in the original
    caps = {w.lower() for w in re.findall(r"\b[A-Z]{3,}\b", t)}
    return " ".join(w.upper() if w in caps else w for w in low.split())


def extended_analyzer():
    a = SentimentIntensityAnalyzer()
    a.lexicon.update(DOMAIN_LEXICON)
    return a


def to_label(compound, pos=0.05, neg=-0.05):
    return "positive" if compound >= pos else ("negative" if compound <= neg else "neutral")


# ----------------------------------------------------------------------------- boilerplate (for topic modelling)
# Channel greetings and sign-offs are frequent but say nothing about WHAT the problem is; left in, they
# dominate topic models ("hello customer care", "shop owner says"). Removed for topic discovery only.
BOILERPLATE_RE = re.compile(
    r"^\s*(hello|hi|good morning|habari|admin|honestly|mimi nasema|manze|aki|wueh|hey|review)\b[ ,:]*(customer care)?[ ,:]*|"
    r"\b(customer reports|caller says|outlet feedback|merchandiser note|from the duka owner|retailer comment|"
    r"shop owner in [\w' ]+? says)\s*:?|"
    r"\b(tried the [\w -]+?|bought the [\w -]+? last week|been buying [\w ]+? for years)\.|"
    r"\b(please sort this out|fix this|never again|very disappointed|kindly call back|please follow up|kindly advise|"
    r"please assist|asante sana|asante|keep it up|would buy again|highly recommend|thanks|smh)\b[.!]*|"
    r"batch no\.? ?mb\d+\.?|#\w+", re.I)


def strip_boilerplate(text):
    t = BRAND_RE.sub(" ", str(text))
    for _ in range(2):
        t = BOILERPLATE_RE.sub(" ", t.strip())
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------- aspects
ASPECT_PATTERNS = {
    "taste_quality": r"tast|flavou?r|creamy|delicious|yumm|smooth|texture|quality|watery|bland|chalky|aftertaste|"
                     r"too sweet|lumps|plastic|kids love|family loves|ladha|tamu|noma|inabamba|fiti|maji maji|maji tu|"
                     r"uchachu|ever had|excellent|top notch|disappointing|sugar|sukari|nothing beats|wanaipenda|fat content|whole milk|real (milk|product)|farm (milk|product)|thick|best thing",
    "freshness_expiry": r"fresh|expir|spoil|gone bad|went bad|go bad|gone off|sour|smel|mou?ld|stale|curdl|shelf life|"
                        r"slimy|separated|melted|refrozen|ice crystals|harib|inanuka|ukungu|imeganda|\blasts?\b(?! week)|"
                        r"refrigerat|siku ngapi",
    "price_value": r"pric|expensive|afford|cheap|cost|value|shilling|\bkes\b|rip-off|\bbei\b|ghali|nafuu|promotion|kidney",
    "packaging": r"packag|\blid\b|seal|\bcups?\b|packet|\bpack\b|straw|leak|burst|spill|crack|kifuniko|vuja|pasuka|"
                 r"recyclable|design|size",
    "availability": r"(?<!expired )stock|have none|sold out|shelves|\bfind\b|availab|in bulk|buy the|hide and seek|patikan|imeisha|naweza pata|"
                    r"\bempty\b|anywhere",
    "delivery_service": r"deliver|driver|\bvan\b|\borders?\b|customer care|call centre|call(ed)? (me )?back|on hold|merchandiser|"
                        r"service|replaced|reply|picked up|answers|\bgari\b|simu|msee|mnipigie|chelewa|haikuja",
}
ASPECT_RE = {a: re.compile(p, re.I) for a, p in ASPECT_PATTERNS.items()}
CLAUSE_SPLIT = re.compile(r"[.!?;]+|,|\s(?:but|however|lakini|although|though|also)\s|\.\.\.", re.I)


GREETING_RE = re.compile(r"^\s*(hello|hi|good morning|habari)\s*(customer care)?\s*,", re.I)


def split_clauses(text):
    t = GREETING_RE.sub(" ", BRAND_RE.sub(" ", str(text)))
    return [c.strip() for c in CLAUSE_SPLIT.split(t) if c and len(c.strip()) > 2]


def detect_aspects(text):
    """Return {aspect: [clauses that mention it]} using keyword rules on product-masked text."""
    out = {}
    for c in split_clauses(text):
        masked = CATALOGUE_RE.sub(" product ", c)
        for a, rx in ASPECT_RE.items():
            if rx.search(masked):
                out.setdefault(a, []).append(c)
    return out
