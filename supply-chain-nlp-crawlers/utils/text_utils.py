import nltk

nltk.download("punkt")
nltk.download("punkt_tab")


def split_sentences(text):

    if not text:
        return []

    return nltk.sent_tokenize(text)


def is_valid_sentence(sentence):

    if not sentence:
        return False

    sentence = sentence.strip()

    if len(sentence) < 40:
        return False

    keywords = [

    # =========================================
    # ENGLISH — SUPPLY CHAIN
    # =========================================

    "supplier",
    "supplies",
    "supply chain",
    "vendor",
    "procurement",
    "sourcing",
    "manufacturer",
    "manufacturing",
    "factory",
    "fab",
    "fabrication",
    "assembly",
    "production",
    "component",
    "semiconductor",
    "chip",
    "foundry",
    "electronics",
    "contract manufacturing",
    "oem",
    "odm",

    # =========================================
    # ENGLISH — RELATIONSHIPS
    # =========================================

    "partnership",
    "partner",
    "collaboration",
    "agreement",
    "contract",
    "joint venture",
    "alliance",
    "ownership",
    "subsidiary",
    "affiliate",
    "acquired",
    "acquisition",
    "stake",
    "investment",
    "shareholder",

    # =========================================
    # ENGLISH — LOGISTICS
    # =========================================

    "distribution",
    "warehouse",
    "shipment",
    "inventory",
    "logistics",
    "fulfillment",

    # =========================================
    # JAPANESE — SUPPLY CHAIN
    # =========================================

    "サプライチェーン",      # supply chain
    "供給",                # supply
    "供給元",              # supplier source
    "供給先",              # supply destination
    "調達",                # procurement
    "製造",                # manufacturing
    "生産",                # production
    "工場",                # factory
    "半導体",              # semiconductor
    "電子部品",            # electronic components
    "部品供給",            # component supply
    "委託製造",            # contract manufacturing
    "受託製造",            # outsourced manufacturing

    # =========================================
    # JAPANESE — RELATIONSHIPS
    # =========================================

    "提携",                # partnership
    "協業",                # collaboration
    "契約",                # contract
    "合弁",                # joint venture
    "子会社",              # subsidiary
    "関連会社",            # affiliate
    "買収",                # acquisition
    "出資",                # investment
    "株主",                # shareholder
    "保有",                # holding/ownership

    # =========================================
    # JAPANESE — OPERATIONS
    # =========================================

    "物流",                # logistics
    "在庫",                # inventory
    "出荷",                # shipment
    "流通"                 # distribution
]

    lowered = sentence.lower()

    return any(
        keyword in lowered
        for keyword in keywords
    )