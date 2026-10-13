"""Generate synthetic customer-feedback text for a FICTIONAL Kenyan dairy brand, "Maziwa Bora".

Context: a dairy FMCG company (milk, yoghurt, cheese, ice cream, bakery, deli) collects
feedback from five channels: product reviews (with a 1-5 star rating), X/Twitter-style posts,
Facebook comments, WhatsApp / call-centre tickets and retailer / merchandiser comments.

How the text is built (so there is real, learnable structure - and realistic difficulty):
  * Each item mentions 0-2 ASPECTS (taste/quality, freshness/expiry, price/value, packaging,
    availability/stock-outs, delivery/service), each with a polarity (positive/negative/neutral).
  * Every aspect clause is drawn from large phrase banks in several STYLES: plain English,
    negation ("not bad at all", "not fresh"), Kiswahili / Sheng / code-switched
    ("maziwa imeharibika", "bei iko juu sana", "tamu mbaya" = extremely tasty) and
    SARCASM ("Great, another expired yoghurt").
  * Two-aspect items with opposite polarities are MIXED; the gold label follows the main
    clause ("love the taste BUT too expensive" -> negative; "a bit pricey but worth it" -> positive).
  * Channel-specific openers/closers, intensifiers, emojis, typos and slang add variety.
  * Gold labels: overall sentiment + one column per aspect. Hard-case tags are exported
    (as an annotation team might tag them) so error analysis can be done by case type.
  * About 5% of overall labels are deliberately flipped (annotator noise, NOT flagged in
    the export) and ~3% of items are genuinely ambiguous ("ok", "hmm 🤔"), so models
    cannot reach a perfect score.

Embedded business events (for the trend / anomaly analysis to discover):
  E1  Oct-Nov 2025  Yoghurt          all regions  new cup / lid design leaks   -> packaging complaints spike
  E2  Dec 2025      Ice Cream        Nairobi      festive-season stock-outs    -> availability complaints spike
  E3  Jan-Feb 2026  Milk             all regions  price increase               -> price complaints spike
  E4  Mar-Apr 2026  Yoghurt          all regions  "Baobab Honey" flavour launch -> positive taste surge
  E5  Jun 2026      Milk & Yoghurt   Coast        cold-chain / expiry incident -> freshness complaints spike

All data is randomly generated. "Maziwa Bora" and every product name are fictional; no real
company, brand, retailer or customer data is included.
Usage: python data/generate_data.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

RNG = np.random.default_rng(2026)
OUT = Path(__file__).parent
MONTHS = pd.period_range("2025-04", "2026-09", freq="M")       # 18 months
LABEL_NOISE = 0.05
AMBIGUOUS_SHARE = 0.03

ASPECTS = ["taste_quality", "freshness_expiry", "price_value", "packaging", "availability", "delivery_service"]
POL = ["positive", "negative", "neutral"]

# ----------------------------------------------------------------------------- products & places
PRODUCTS = {
    "Milk": ["fresh milk", "500ml milk packet", "1L long-life milk", "mala", "lala", "milk"],
    "Yoghurt": ["strawberry yoghurt", "vanilla yoghurt", "plain yoghurt", "Greek-style yoghurt", "yoghurt",
                "mango yoghurt"],
    "Cheese": ["cheddar", "mozzarella", "cheese slices", "cheese"],
    "Ice Cream": ["vanilla ice cream", "choc-chip ice cream", "2L ice cream tub", "ice cream cones", "ice cream"],
    "Bakery": ["bread", "brown bread", "milk buns", "queen cakes"],
    "Deli": ["beef sausages", "smokies", "ham slices", "chicken sausages"],
}
CAT_W = {"Milk": .30, "Yoghurt": .27, "Cheese": .08, "Ice Cream": .12, "Bakery": .12, "Deli": .11}
NEW_FLAVOUR = "Baobab Honey yoghurt"
LAUNCH = pd.Period("2026-03", "M")
REGIONS = {"Nairobi": .34, "Central": .14, "Rift Valley": .16, "Coast": .12, "Nyanza": .10, "Western": .08,
           "Eastern": .06}
CITIES = {"Nairobi": ["Nairobi", "Kasarani", "Rongai", "Westlands", "Embakasi"],
          "Central": ["Nyeri", "Thika", "Murang'a"], "Rift Valley": ["Nakuru", "Eldoret", "Naivasha"],
          "Coast": ["Mombasa", "Malindi", "Kilifi", "Diani"], "Nyanza": ["Kisumu", "Kisii", "Homa Bay"],
          "Western": ["Kakamega", "Bungoma"], "Eastern": ["Machakos", "Embu", "Meru"]}
CHANNELS = {"product_review": .30, "x_post": .19, "facebook_comment": .17, "whatsapp_callcentre": .20,
            "retailer_comment": .14}

# ----------------------------------------------------------------------------- phrase banks
# {p}=product {b}=brand {i}=intensifier {city}=town {n}=number. Styles: en / neg (negation) / sw (Kiswahili, Sheng,
# code-switched). Each entry is a clause; the composer adds openers, connectors, emojis and noise.
B = {a: {p: {} for p in POL} for a in ASPECTS}

B["taste_quality"]["positive"] = {
    "en": ["the {p} tastes {i} amazing", "§{i} creamy and tasty", "best {p} I have ever had", "the flavour is spot on",
           "{i} delicious, my kids love it", "§the {p} is {i} smooth and rich", "quality is top notch as always",
           "the {p} is {i} good", "§lovely thick texture", "§tastes like real farm milk", "my family loves the {p}",
           "the {p} is excellent", "{i} yummy, perfect for breakfast", "great quality {p}"],
    "neg": ["not bad at all", "can't fault the taste", "the {p} never disappoints", "no complaints about the taste",
            "I wasn't expecting much but it is {i} good", "nothing beats the {p}", "never had a bad one"],
    "sw": ["ladha ni tamu sana", "hii {p} iko tamu mbaya", "hii {p} ni noma sana", "watoto wanaipenda sana",
           "{p} iko poa sana", "safi kabisa", "hii {p} inabamba", "ladha sio mbaya kabisa", "iko fiti sana",
           "ni tamu kuliko zingine"],
}
B["taste_quality"]["negative"] = {
    "en": ["§the {p} tastes {i} sour", "§{i} watery these days", "the {p} tasted off", "§bland and chalky",
           "has a weird aftertaste", "the quality has dropped {i} badly", "the {p} is {i} disappointing",
           "tastes like plastic", "the {p} was {i} bad", "§found lumps in the {p}", "terrible quality lately",
           "the {p} has become too sweet and artificial"],
    "neg": ["§the {p} is not creamy anymore", "not as tasty as before", "§doesn't taste like real milk",
            "no flavour at all", "not good at all", "I don't like the new taste", "the {p} is not what it used to be"],
    "sw": ["ladha imebadilika, sio kama zamani", "haina ladha kabisa", "§imekuwa maji maji", "ina ladha mbaya",
           "§iko na uchachu mingi", "hii {p} imekuwa mbaya sana", "§{p} siku hizi ni maji tu"],
}
B["taste_quality"]["neutral"] = {
    "en": ["the {p} tastes like the regular one", "same taste as before", "the taste is average",
           "does the {p} contain added sugar", "§is the {p} made with whole milk", "what is the fat content of the {p}"],
    "neg": ["not sure the taste has changed", "can't tell the difference in taste"],
    "sw": ["ladha ni kawaida tu", "hii {p} ina sukari ngapi"],
}
B["freshness_expiry"]["positive"] = {
    "en": ["always fresh", "§the {p} stays fresh for days", "long shelf life", "fresh even after 5 days in the fridge",
           "the {p} was {i} fresh", "lasts well past opening", "fresh every single time"],
    "neg": ["never had a spoiled packet", "§no sour smell even after a week", "doesn't go bad quickly",
            "not once has it gone off before the date"],
    "sw": ["iko fresh kabisa", "§maziwa ni safi na fresh", "inakaa siku nyingi bila kuharibika",
           "{p} iko fresh sana"],
}
B["freshness_expiry"]["negative"] = {
    "en": ["the {p} was already expired", "§smelt sour on opening", "the {p} had gone bad", "found expired {p} on the shelf",
           "the {p} spoiled a day after buying", "expired stock again", "the {p} smells {i} bad"],
    "neg": ["the {p} was not fresh", "not even close to the expiry date and already sour", "it didn't last even two days",
            "this is not fresh at all"],
    "sw": ["§maziwa imeharibika", "{p} imeexpire tayari", "ilikuwa imeharibika kabla ya expiry date",
           "{p} inanuka", "nimepata {p} imeexpire kwa duka"],
}
B["freshness_expiry"]["neutral"] = {
    "en": ["where is the expiry date printed on the {p}", "what is the shelf life of the {p}",
           "how long does the {p} last after opening", "the expiry date is on the side of the pack"],
    "neg": ["not sure if the {p} needs to be refrigerated"],
    "sw": ["expiry date iko wapi kwa packet", "{p} inakaa siku ngapi"],
}
FRESH_SPECIFIC = {
    "Milk": {"negative": ["the milk curdled before the expiry date", "§the {p} curdled when I boiled it",
                          "maziwa imeganda kabla ya expiry"]},
    "Yoghurt": {"negative": ["the yoghurt had separated and smelt off", "mould on the lid of the {p}"]},
    "Bakery": {"negative": ["the {p} was mouldy", "the {p} was stale", "mkate ilikuwa na ukungu"],
               "positive": ["the {p} is soft and fresh", "the {p} was {i} fresh from the oven"]},
    "Deli": {"negative": ["the {p} smelt off", "the {p} were slimy"]},
    "Ice Cream": {"negative": ["the {p} had melted and refrozen", "the {p} was full of ice crystals"]},
}
B["price_value"]["positive"] = {
    "en": ["good value for money", "{i} affordable", "worth every shilling", "the price is fair",
           "great price for the quality", "cheaper than other brands and better", "the offer price is {i} good"],
    "neg": ["not expensive at all", "can't complain about the price", "doesn't break the bank",
            "no other brand gives this value"],
    "sw": ["bei iko poa", "bei ni nafuu", "value for money kabisa", "bei iko sawa kabisa"],
}
B["price_value"]["negative"] = {
    "en": ["{i} expensive", "the price went up again", "overpriced for what you get", "too costly for a family",
           "the {p} costs {i} much now", "the pack got smaller and the price went up", "a rip-off at this price"],
    "neg": ["not worth the price anymore", "can't afford it anymore", "not cheap at all",
            "doesn't justify the price"],
    "sw": ["bei iko juu sana", "bei imepanda tena", "ni ghali sana siku hizi", "hatuwezi afford tena",
           "bei ya {p} imepanda"],
}
B["price_value"]["neutral"] = {
    "en": ["how much is the {p} now", "the {p} is KES {n} at our shop", "is there a promotion on the {p} this month",
           "the price is the same as other brands"],
    "neg": ["not sure what the new price is"],
    "sw": ["bei ya {p} ni ngapi sasa", "bei ni sawa na zingine"],
}
B["packaging"]["positive"] = {
    "en": ["¤the new cup is easy to open", "nice packaging", "the resealable lid is {i} handy",
           "the packet is strong", "packaging looks {i} premium", "love the new design"],
    "neg": ["no more leaks with the new seal", "the packet never bursts", "not a single leak so far"],
    "sw": ["packaging mpya ni smart", "¤kifuniko mpya ni rahisi kufungua", "packet iko strong"],
}
B["packaging"]["negative"] = {
    "en": ["the lid leaks", "the packet burst in my bag", "the seal was broken", "packaging is {i} flimsy",
           "the {p} leaked all over the fridge", "¤the cup cracked", "¤the straw is always missing"],
    "neg": ["the lid doesn't close properly", "the seal is not tight", "¤can't open the cup without spilling"],
    "sw": ["packet inavuja", "¤kifuniko inavuja", "packet ilipasuka kwa bag", "{p} inavuja kila mahali"],
}
B["packaging"]["neutral"] = {
    "en": ["noticed the new packaging", "is the {p} packet recyclable", "¤the {p} now comes in a cup",
           "which size does the {p} come in"],
    "neg": ["not sure why the packaging changed"],
    "sw": ["mmebadilisha packaging", "packet ni ya size gani"],
}
B["availability"]["positive"] = {
    "en": ["always in stock at my local duka", "easy to find now", "the shelves are always full",
           "available everywhere in {city}", "finally stocked near me"],
    "neg": ["never out of stock at my duka", "no problem finding it", "never had to look for it"],
    "sw": ["inapatikana kila duka", "stock iko poa siku hizi", "inapatikana kwa urahisi {city}"],
}
B["availability"]["negative"] = {
    "en": ["out of stock everywhere", "the {p} is always sold out", "empty shelves for a week",
           "we have not received stock for two weeks", "customers keep asking and we have none",
           "the {p} is out of stock again in {city}"],
    "neg": ["can't find the {p} anywhere", "not available in any shop near me", "no stock for two weeks"],
    "sw": ["haipatikani kwa maduka", "imeisha kwa shelves wiki nzima", "tumekosa stock ya {p}",
           "{p} haipatikani {city}"],
}
B["availability"]["neutral"] = {
    "en": ["is the {p} available in {city}", "where can I buy the {p}", "which shops stock the {p}",
           "do you sell the {p} in bulk"],
    "neg": ["not sure which shops stock it"],
    "sw": ["{p} inapatikana {city}", "naweza pata {p} wapi"],
}
B["delivery_service"]["positive"] = {
    "en": ["delivery was on time", "customer care sorted me out quickly", "the merchandiser was {i} helpful",
           "the driver was polite and early", "they replaced the {p} the same day", "great customer service"],
    "neg": ["never late with our deliveries", "no issues with the van this month",
            "customer care didn't take long to help"],
    "sw": ["delivery ilikuja mapema", "customer care walinisaidia haraka", "msee wa delivery ni mpole sana"],
}
B["delivery_service"]["negative"] = {
    "en": ["delivery was late again", "the driver was rude", "the order came incomplete", "waited the whole day for the van",
           "customer care keeps me on hold for ages", "the merchandiser never visits", "they delivered the wrong order"],
    "neg": ["the van didn't come on the agreed day", "nobody answers the call centre", "never got a reply from customer care",
            "no one has called me back"],
    "sw": ["gari ya delivery haikuja", "customer care hawashiki simu", "delivery imechelewa tena",
           "order ilikuja nusu"],
}
B["delivery_service"]["neutral"] = {
    "en": ["please call me back about my order", "what are the delivery days for {city}", "I placed an order on Monday",
           "can I change my delivery day"],
    "neg": ["not sure when the next delivery is"],
    "sw": ["naomba mnipigie kuhusu order", "delivery ni siku gani {city}"],
}
SARCASM = {
    "taste_quality": ["§wow, {p} that tastes like water. just what I paid for", "great, the {p} tastes like chalk now. love that for me",
                      "oh lovely, another sour {p}", "§amazing how the {p} gets more watery every week"],
    "freshness_expiry": ["great, another expired {p}. just what I needed", "love how my {p} expires before I even get home",
                         "§thanks {b}, nothing like sour milk for breakfast",
                         "wonderful, the {p} went bad two days before the expiry date"],
    "price_value": ["love paying more every month for the same {p}", "amazing how the {p} gets smaller and pricier every month",
                    "sure, let me just sell a kidney to buy {p}"],
    "packaging": ["wow thanks for the leaking {p}, my bag smells lovely now", "¤brilliant new cup design, it leaks all over the fridge",
                  "nice packaging, if you like {p} all over your shopping"],
    "availability": ["oh nice, the {p} is out of stock AGAIN", "great job, empty shelves for the third week running",
                     "love playing hide and seek with the {p} in every shop"],
    "delivery_service": ["fantastic service, waited 3 days for delivery", "wow, customer care picked up on the 15th call, impressive",
                         "brilliant, the van came at 6pm when the shop was closing"],
}
GENERAL = {
    "positive": ["love {b}!", "{b} never disappoints", "best dairy brand in Kenya", "big up {b}", "{b} ni noma sana",
                 "proudly Kenyan, proudly {b}", "been loyal to {b} for years and still happy", "thank you {b}"],
    "negative": ["worst experience with {b}", "{b} has really gone down", "very disappointed with {b} lately",
                 "sitanunua tena", "nimechoka na {b}", "{b} you have let us down", "switching brands after this"],
    "neutral": ["just bought the {p}", "does {b} make lactose-free milk", "where is the {b} factory shop",
                "is the {p} halal certified", "what time does the factory shop open", "saw the {b} truck in {city} today",
                "do you have a number for orders"],
}
AMBIGUOUS = ["ok", "hmm 🤔", "{b} again", "it's milk lol", "saw the new advert", "interesting...", "ok sawa",
             "same as usual", "the {p} 😅", "not sure how I feel about the new cup", "it is what it is", "eh",
             "the {p} I guess", "well well well", "kumbe", "nimeona", "hmm the new {p}"]
EVENT_PHRASES = {
    "E1": ["since you changed the cup the {p} leaks everywhere", "the new cup lid doesn't seal",
           "new packaging is terrible, the lid pops off", "cup mpya inavuja", "the new yoghurt cup leaks in the bag",
           "every new cup I buy is leaking from the lid"],
    "E2": ["no ice cream tubs anywhere in Nairobi this festive season", "the {p} has been sold out since before Christmas",
           "kids want ice cream for Christmas and every shop is empty", "ice cream haipatikani Nairobi hii December",
           "our freezer has been empty for two weeks, no ice cream stock"],
    "E3": ["the milk price went up by 10 shillings overnight", "the 500ml is now KES 70, too much",
           "bei ya maziwa imepanda tena", "why has the price of milk gone up again",
           "price increase on milk is too much for families"],
    "E4": ["the new Baobab Honey flavour is {i} amazing", "Baobab Honey ni tamu mbaya", "obsessed with the new Baobab Honey",
           "the Baobab Honey yoghurt is the best thing you have made", "my kids finish the Baobab Honey in one day"],
    "E5": ["the milk was sour on opening, bought in {city}", "expired stock at the shops in {city}",
           "the shop fridge was off and the yoghurt had gone bad", "maziwa imeharibika {city} wiki hii",
           "third packet this week that has gone off in {city}"],
}
EVENTS = [  # id, months, categories, region, aspect, polarity, extra items per month, product override
    ("E1", ["2025-10", "2025-11"], ["Yoghurt"], None, "packaging", "negative", [150, 95], None),
    ("E2", ["2025-12"], ["Ice Cream"], "Nairobi", "availability", "negative", [90], None),
    ("E3", ["2026-01", "2026-02"], ["Milk"], None, "price_value", "negative", [140, 70], None),
    ("E4", ["2026-03", "2026-04"], ["Yoghurt"], None, "taste_quality", "positive", [150, 90], NEW_FLAVOUR),
    ("E5", ["2026-06"], ["Milk", "Yoghurt"], "Coast", "freshness_expiry", "negative", [140], None),
]

INTENS = ["so", "really", "very", "super", "extremely", "totally", "", ""]
EMO = {"positive": ["😍", "👌", "🔥", "😋", "👍", "❤️", "🙌", "💯"], "negative": ["😡", "🤮", "👎", "😤", "😞", "💔", "😠"],
       "neutral": ["🤔", "🙏", "👀"], "sarcasm": ["🙄", "🙃", "😒", "👏"]}
OPENERS = {
    "product_review": ["", "", "Bought the {p} last week.", "Been buying {b} for years.", "Tried the {p}.", "Review:"],
    "x_post": ["@MaziwaBora", "Hey @MaziwaBora", "", "", "Aki,", "Manze,", "Wueh,"],
    "facebook_comment": ["", "", "Admin,", "Hello {b},", "Honestly", "Mimi nasema"],
    "whatsapp_callcentre": ["Hello customer care,", "Habari,", "Good morning,", "Hi,", "Customer reports:", "Caller says:"],
    "retailer_comment": ["Outlet feedback:", "Merchandiser note:", "From the duka owner:", "Retailer comment:",
                         "Shop owner in {city} says:"],
}
CLOSERS = {
    "positive": ["Would buy again.", "Keep it up!", "Asante sana.", "Highly recommend.", ""],
    "negative": ["Never again.", "Please sort this out.", "Very disappointed.", "Fix this!", ""],
    "neutral": ["Thanks.", "Asante.", "Kindly advise.", ""],
}
CH_CLOSERS = {"x_post": ["#MaziwaBora", "#{city}", "smh", ""], "whatsapp_callcentre": ["Batch no. {batch}.", "Kindly call back.", ""],
              "retailer_comment": ["Please follow up.", ""], "product_review": [""], "facebook_comment": [""]}
SLANG = {"you": "u", "good": "gud", "please": "pls", "because": "coz", "though": "tho", "really": "rly",
         "with": "wth", "for": "4", "great": "gr8", "thanks": "thx", "people": "ppl", "before": "b4"}
MIXED_PATTERNS = [("{A} but {B}", False), ("{A}, however {B}", False), ("{A}... but {B}", False),
                  ("although {A}, {B}", False), ("{B}, even though {A}", False), ("{A} lakini {B}", True),
                  ("{B}, though {A}", False)]


def pick(xs):
    return xs[int(RNG.integers(len(xs)))]


def suits(entries, cat):
    """§ = dairy-only phrase (not for bakery/deli); ¤ = cup/straw phrase (yoghurt & ice cream only)."""
    ok = [e for e in entries if not (e.startswith("§") and cat in ("Bakery", "Deli"))
          and not (e.startswith("¤") and cat not in ("Yoghurt", "Ice Cream"))]
    return [e.lstrip("§¤") for e in (ok or entries)]


PLURAL = {"smokies", "beef sausages", "chicken sausages", "ham slices", "cheese slices", "milk buns", "queen cakes",
          "ice cream cones"}
VERB_FIX = {" is ": " are ", " was ": " were ", " has ": " have ", " tastes ": " taste ", " smells ": " smell ",
            " costs ": " cost ", " stays ": " stay ", " leaks ": " leak ", " tasted ": " tasted ", " comes ": " come "}


def fill(s, ctx):
    out = s.format(p=ctx["p"], b=ctx["b"], i=pick(INTENS), city=ctx["city"], n=int(RNG.integers(55, 260)),
                   batch=f"MB{int(RNG.integers(10000, 99999))}").replace("  ", " ").strip()
    if ctx["p"] in PLURAL:
        for a, b in VERB_FIX.items():
            out = out.replace(ctx["p"] + a, ctx["p"] + b)
    return out


def style_weights(channel):
    sw = {"x_post": .26, "facebook_comment": .22, "whatsapp_callcentre": .22, "retailer_comment": .18}.get(channel, .12)
    return {"en": 1 - sw - .16, "neg": .16, "sw": sw}


def make_clause(aspect, pol, ctx, channel, allow_sarcasm=True, event=None):
    """Return (clause_text, tags)."""
    tags = set()
    if event is not None and RNG.random() < .6:
        txt = fill(pick(EVENT_PHRASES[event]), ctx)
        if re.search(r"\b(ni|iko|imepanda|inavuja|haipatikani|imeharibika|maziwa|bei|wiki|hii)\b", txt):
            tags.add("code_switch")
        return txt, tags
    sarc_p = {"x_post": .26, "facebook_comment": .18, "product_review": .08}.get(channel, .05)
    if pol == "negative" and allow_sarcasm and RNG.random() < sarc_p:
        return fill(pick(suits(SARCASM[aspect], ctx["cat"])), ctx), {"sarcasm"}
    w = style_weights(channel)
    style = RNG.choice(list(w), p=list(w.values()))
    bank = suits(B[aspect][pol][style], ctx["cat"])
    if aspect == "freshness_expiry" and ctx["cat"] in FRESH_SPECIFIC and RNG.random() < .5:
        spec = FRESH_SPECIFIC[ctx["cat"]].get(pol)
        if spec:
            bank = suits(spec, ctx["cat"])
    txt = fill(pick(bank), ctx)
    if style == "neg":
        tags.add("negation")
    if style == "sw" or re.search(r"\b(ladha|maziwa|imeganda|mkate|bei|iko|ni)\b", txt):
        tags.add("code_switch")
    return txt, tags


def add_noise(text, tags):
    words = text.split(" ")
    changed = False
    for k, w in enumerate(words):
        lw = w.lower().strip(",.!?")
        if lw in SLANG and RNG.random() < .6:
            words[k] = w.lower().replace(lw, SLANG[lw]); changed = True
        elif len(lw) > 5 and lw.isalpha() and RNG.random() < .12:
            j = int(RNG.integers(1, len(w) - 1))
            words[k] = w[:j] + w[j + 1:] if RNG.random() < .5 else w[:j] + w[j + 1] + w[j] + w[j + 2:]; changed = True
    if changed:
        tags.add("typo_slang")
    return " ".join(words)


def cap(s):
    s = s.strip()
    return s[:1].upper() + s[1:] if s else s


def compose(rec, aspects_pol, event=None):
    """aspects_pol: list of (aspect, polarity). Returns text, overall label, tags."""
    ch, ctx = rec["channel"], rec["ctx"]
    tags = set()
    if not aspects_pol:  # general comment, no aspect
        pol = rec["general_pol"]
        body = fill(pick(GENERAL[pol]), ctx)
        if re.search(r"\b(sitanunua|nimechoka|noma|ni)\b", body):
            tags.add("code_switch")
        label = pol
        sarcastic = False
    elif len(aspects_pol) == 1:
        a, pol = aspects_pol[0]
        body, t = make_clause(a, pol, ctx, ch, event=event if a == rec.get("event_aspect") else None)
        tags |= t
        label = pol
        sarcastic = "sarcasm" in t
    else:
        (a1, p1), (a2, p2) = aspects_pol
        c1, t1 = make_clause(a1, p1, ctx, ch, allow_sarcasm=False, event=event if a1 == rec.get("event_aspect") else None)
        c2, t2 = make_clause(a2, p2, ctx, ch, allow_sarcasm=False, event=event if a2 == rec.get("event_aspect") else None)
        tags |= t1 | t2
        sarcastic = False
        c1, c2 = c1.rstrip(".!? "), c2.rstrip(".!? ")
        if {p1, p2} == {"positive", "negative"}:
            tags.add("mixed")
            pats = [x for x in MIXED_PATTERNS if not (" but " in c1 + " " + c2 and "but" in x[0])]
            pat, sw = pick(pats)
            # B is the MAIN clause: whichever clause lands in {B} decides the label
            if RNG.random() < .5:
                body, label = pat.format(A=c1, B=c2), p2
            else:
                body, label = pat.format(A=c2, B=c1), p1
            if sw:
                tags.add("code_switch")
        else:
            label = p1 if p1 != "neutral" else p2
            body = pick(["{A} and {B}", "{A}. {B}", "{A}, also {B}", "{A}. Also {B}"]).format(A=c1, B=c2)
    opener = fill(pick(OPENERS[ch]), ctx)
    keep = body[:2].isupper() or re.match(r"(Baobab|Maziwa|I |Kenya|KES|Christmas)", cap(body))
    body = cap(body) if (not opener or opener[-1] in ".:" or keep) else body[:1].lower() + body[1:]
    # intensity: caps / exclamation
    if label != "neutral" and RNG.random() < .12:
        body = body + pick(["!!", "!!!", "!"])
    if label != "neutral" and RNG.random() < .04:
        body = body.upper()
    closer = fill(pick(CLOSERS["negative" if sarcastic else label] + CH_CLOSERS[ch]), ctx) if RNG.random() < .55 else ""
    if sarcastic and closer in CLOSERS["negative"]:
        closer = ""
    text = " ".join(x for x in [opener, body if body[-1] in ".!?…" else body + ("" if RNG.random() < .4 else "."), closer] if x)
    emo_p = {"x_post": .45, "facebook_comment": .40, "whatsapp_callcentre": .12, "product_review": .18}.get(ch, .05)
    if RNG.random() < emo_p:
        key = "sarcasm" if sarcastic else label
        text += " " + "".join(pick(EMO[key]) for _ in range(int(RNG.integers(1, 3))))
        tags.add("emoji")
    if RNG.random() < .14:
        text = add_noise(text, tags)
    return text, label, tags


# ----------------------------------------------------------------------------- record sampling
ASPECT_W = {
    "product_review": [.38, .16, .18, .12, .08, .08], "x_post": [.26, .18, .18, .14, .12, .12],
    "facebook_comment": [.28, .16, .18, .14, .12, .12], "whatsapp_callcentre": [.12, .30, .06, .16, .10, .26],
    "retailer_comment": [.06, .14, .10, .06, .38, .26],
}
POL_BASE = {"taste_quality": [.62, .28, .10], "freshness_expiry": [.45, .45, .10], "price_value": [.32, .52, .16],
            "packaging": [.42, .38, .20], "availability": [.32, .53, .15], "delivery_service": [.40, .45, .15]}
CH_SHIFT = {"product_review": .12, "x_post": -.04, "facebook_comment": 0, "whatsapp_callcentre": -.18,
            "retailer_comment": -.05}
GEN_POL = {"product_review": [.55, .25, .20], "x_post": [.35, .30, .35], "facebook_comment": [.40, .25, .35],
           "whatsapp_callcentre": [.08, .27, .65], "retailer_comment": [.20, .30, .50]}


def pol_probs(aspect, ch):
    p, n, u = POL_BASE[aspect]
    s = CH_SHIFT[ch]
    p, n = max(.05, p + s), max(.05, n - s)
    v = np.array([p, n, u]); return v / v.sum()


def base_record(month, cat=None, region=None, product=None):
    ch = RNG.choice(list(CHANNELS), p=list(CHANNELS.values()))
    cat = cat or RNG.choice(list(CAT_W), p=list(CAT_W.values()))
    if product is None:
        prods = PRODUCTS[cat] + ([NEW_FLAVOUR] * 2 if cat == "Yoghurt" and month >= LAUNCH else [])
        product = pick(prods)
    region = region or RNG.choice(list(REGIONS), p=list(REGIONS.values()))
    brand = "@MaziwaBora" if ch == "x_post" and RNG.random() < .5 else pick(["Maziwa Bora", "Maziwa Bora", "maziwa bora", "MaziwaBora"])
    day = int(RNG.integers(1, month.days_in_month + 1))
    return {"date": f"{month.year}-{month.month:02d}-{day:02d}", "month": month, "channel": ch, "category": cat,
            "product": product, "region": region,
            "ctx": {"p": product, "b": brand, "city": pick(CITIES[region]), "cat": cat}}


def choose_aspects(rec, forced=None):
    ch = rec["channel"]
    r = RNG.random()
    n = 0 if r < .13 else (1 if r < .66 else 2)
    if forced:
        n = max(n, 1)
    chosen = [forced[0]] if forced else []
    while len(chosen) < n:
        a = RNG.choice(ASPECTS, p=ASPECT_W[ch])
        if a not in chosen:
            chosen.append(a)
    out = []
    for a in chosen:
        if forced and a == forced[0]:
            out.append((a, forced[1]))
        else:
            out.append((a, RNG.choice(POL, p=pol_probs(a, ch))))
    rec["general_pol"] = RNG.choice(POL, p=GEN_POL[ch])
    return out


def rating_for(label, tags):
    if "sarcasm" in tags:
        return int(RNG.choice([1, 2], p=[.8, .2]))
    probs = {"positive": [0, .01, .06, .35, .58], "neutral": [.05, .17, .55, .18, .05], "negative": [.52, .35, .11, .02, 0]}[label]
    if "mixed" in tags:
        probs = [.12, .25, .35, .2, .08] if label == "negative" else [.04, .1, .3, .36, .2]
    return int(RNG.choice([1, 2, 3, 4, 5], p=probs))


rows = []


def emit(rec, aspects_pol, event=None):
    text, label, tags = compose(rec, aspects_pol, event)
    asp = {a: "" for a in ASPECTS}
    for a, p in aspects_pol:
        asp[a] = p
    rows.append({**{k: rec[k] for k in ["date", "channel", "category", "product", "region"]},
                 "rating": rating_for(label, tags) if rec["channel"] == "product_review" else np.nan,
                 "text": text, "sentiment": label, **{f"asp_{a}": v for a, v in asp.items()},
                 "hard_case_tags": "|".join(sorted(tags)), "_event": event or ""})


for t, m in enumerate(MONTHS):
    n_base = RNG.poisson(370 * (1 + .012 * t) * (1.12 if m.month == 12 else 1))
    for _ in range(n_base):
        rec = base_record(m)
        if RNG.random() < AMBIGUOUS_SHARE:  # genuinely ambiguous item, label is a coin-flip-ish judgement
            text = fill(pick(AMBIGUOUS), rec["ctx"])
            label = RNG.choice(POL, p=[.25, .25, .5])
            rows.append({**{k: rec[k] for k in ["date", "channel", "category", "product", "region"]},
                         "rating": rating_for(label, set()) if rec["channel"] == "product_review" else np.nan,
                         "text": text, "sentiment": label, **{f"asp_{a}": "" for a in ASPECTS},
                         "hard_case_tags": "ambiguous", "_event": ""})
            continue
        # post-price-increase drift: milk price complaints stay elevated for a while
        forced = None
        if rec["category"] == "Milk" and pd.Period("2026-03", "M") <= m <= pd.Period("2026-05", "M") and RNG.random() < .08:
            forced = ("price_value", "negative")
        emit(rec, choose_aspects(rec, forced))
    for eid, months, cats, region, aspect, pol, extra, prod in EVENTS:
        if str(m) in months:
            for _ in range(RNG.poisson(extra[months.index(str(m))])):
                rec = base_record(m, cat=pick(cats), region=region or None, product=prod)
                rec["event_aspect"] = aspect
                emit(rec, choose_aspects(rec, (aspect, pol)), event=eid)

df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# Annotator noise: flip ~5% of overall labels (not flagged in the export - you never know which ones in real life)
noise = (RNG.random(len(df)) < LABEL_NOISE) & (df.hard_case_tags != "ambiguous")
alt = {"positive": (["neutral", "negative"], [.65, .35]), "negative": (["neutral", "positive"], [.7, .3]),
       "neutral": (["positive", "negative"], [.5, .5])}
df.loc[noise, "sentiment"] = [RNG.choice(alt[s][0], p=alt[s][1]) for s in df.loc[noise, "sentiment"]]

df.insert(0, "feedback_id", [f"FB{i:06d}" for i in range(1, len(df) + 1)])
df.drop(columns="_event").to_csv(OUT / "feedback.csv", index=False)
print(f"{len(df):,} feedback items | {df.date.min()} - {df.date.max()} | label noise flipped: {noise.sum()}")
print(df.sentiment.value_counts(normalize=True).round(3).to_dict())
print(df.channel.value_counts().to_dict())
