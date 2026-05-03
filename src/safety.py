"""
Safety Guard — synchronous, local-only filter that runs BEFORE any LLM call.

WHY this design:
- Assignment demands <10ms, no LLM, no network
- We use a two-layer approach:
    Layer 1: ACTION-PHRASE detection — catches "help me [do bad thing]" patterns
    Layer 2: KEYWORD scoring with context — a query must contain harmful keywords
             AND action-intent signals to be blocked. This prevents over-blocking
             educational queries like "what is insider trading?"

TRADEOFF DOCUMENTED:
- This will over-block some edge cases where harmful keywords co-occur with
  ambiguous action verbs. We bias toward safety (false positives > false negatives)
  because the assignment grades recall at ≥95% vs passthrough at ≥90%.
- Educational queries are detected via signals: "what is", "explain", "how does",
  "what are the penalties", "define", "describe", etc.

GENERALIZATION STRATEGY:
- Patterns are written to match the CONCEPT, not specific phrasings
- Each category has broad verb+noun patterns covering synonyms
- Educational detection is generous to avoid over-blocking
"""
from __future__ import annotations

import re
from src.models import SafetyVerdict


# ── Category-specific refusal messages ────────────────────────────────────────
# Each category returns a DISTINCT professional response (not a generic refusal).

CATEGORY_RESPONSES: dict[str, str] = {
    "insider_trading": (
        "I cannot assist with trading based on material non-public information. "
        "Insider trading is a serious violation of securities law (SEC Rule 10b-5, "
        "EU MAR). If you have questions about insider trading regulations, I'm happy "
        "to explain how they work."
    ),
    "market_manipulation": (
        "I cannot help with activities that manipulate market prices, volume, or "
        "trading activity. This includes pump-and-dump schemes, spoofing, layering, "
        "and wash trading — all of which violate securities regulations. I can explain "
        "how regulators detect and prosecute these practices if you're interested."
    ),
    "money_laundering": (
        "I cannot assist with structuring transactions to avoid reporting thresholds, "
        "obscuring the source of funds, or evading anti-money laundering (AML) "
        "regulations. If you have questions about AML compliance requirements, "
        "I'm happy to help."
    ),
    "guaranteed_returns": (
        "I cannot guarantee investment returns or identify 'certain' winners. "
        "All investments carry risk, and claims of guaranteed returns are a hallmark "
        "of fraud (Ponzi schemes, affinity fraud). I can discuss historical market "
        "performance and realistic return expectations instead."
    ),
    "reckless_advice": (
        "I cannot encourage highly reckless financial decisions such as concentrating "
        "life savings into a single volatile asset, leveraging your home for stock "
        "trades, or putting emergency funds into speculative instruments. I can help "
        "you think through risk-appropriate strategies instead."
    ),
    "sanctions_evasion": (
        "I cannot assist with circumventing OFAC sanctions, bypassing trade "
        "restrictions, or routing transactions through shell entities to avoid "
        "sanctions screening. I can explain how sanctions compliance works if "
        "you're interested in the regulatory framework."
    ),
    "fraud": (
        "I cannot assist with creating fraudulent documents, fake contract notes, "
        "or any form of financial fraud. If you need legitimate documentation "
        "support, please contact your brokerage or financial institution directly."
    ),
}


# ── Educational / informational signal patterns ──────────────────────────────
# If a query matches these, it's likely asking ABOUT a topic, not trying to DO it.

EDUCATIONAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bwhat\s+(?:is|are|does|was|were)\b",
        r"\bexplain\b",
        r"\bdefine\b",
        r"\bdescribe\b",
        r"\bhow\s+does\b",
        r"\bhow\s+do\s+(?:regulators?|brokers?|authorities|the\s+(?:SEC|FCA|FINRA|DOJ|CFTC))\b",
        r"\bwhat\s+(?:are\s+)?the\s+(?:penalties?|rules?|regulations?|requirements?|obligations?|risks?|consequences?|laws?)\b",
        r"\bwhat\s+(?:compliance|disclosure|reporting)\b",
        r"\bis\s+(?:it|short\s+selling|insider\s+trading|wash\s+trading|spoofing)\s+(?:legal|illegal|legitimate|allowed|prohibited)\b",
        r"\bwhat\s+(?:factors?|considerations?)\s+should\b",
        r"\bdifference\s+between\b",
        r"\bwhat.+role\b",
        r"\bhow\s+(?:should|does|do)\s+(?:an?\s+)?(?:emergency|investor|broker|regulator|compliance)\b",
        r"\bhistorical\s+(?:average|return|performance|data)\b",
        r"\bwhy\s+(?:are|is|do|does)\b",
        r"\bhow\s+(?:does|do)\s+(?:the\s+)?(?:SEC|FCA|regulators?|authorities)\s+(?:catch|investigate|detect|screen|prevent|enforce|prosecute|monitor)\b",
        r"\bare\s+(?:pump|wash|spoof).+(?:legal|illegal)\b",
        r"\bwhat\s+(?:happens|is the penalty|are the fines)\s+(?:if|when)\b",
        r"\bcan\s+you\s+explain\b",
        r"\btell\s+me\s+(?:about|how)\s+(?:the\s+)?(?:regulation|law|rule|compliance|enforcement)\b",
        r"\bhow\s+(?:can|do)\s+(?:investors?|people|traders?)\s+(?:protect|report|identify|avoid)\b",
        r"\bwhat\s+(?:should|can)\s+(?:i|investors?)\s+(?:know|learn|understand)\s+about\b",
        r"\beducate\s+me\b",
        r"\bteach\s+me\b",
        r"\blearning\s+about\b",
    ]
]


# ── Harmful action phrases — first-person intent to DO something illegal ──────
# Broadened to catch novel phrasings while keeping precision.

# Each tuple: (compiled regex, category)
ACTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ═══ Insider trading — acting on MNPI ═══
    # Pattern: "help me trade on [non-public info]"
    (re.compile(r"(?:help|tell|advise)\s+me\s+(?:to\s+)?(?:trade|buy|sell|invest)\s+(?:on|based\s+on|using)\s+(?:this\s+)?(?:confidential|insider|non.?public|private|secret|internal)", re.I), "insider_trading"),
    # Pattern: "I work at X... [non-public info]... buy/sell"
    (re.compile(r"(?:i\s+)?(?:work|am|employed)\s+at\s+.{2,30}(?:unannounced|non.?public|confidential|private|internal|secret).{0,40}(?:buy|sell|trade|load|invest|position)", re.I), "insider_trading"),
    (re.compile(r"(?:i\s+)?(?:work|am|employed)\s+at\s+.{2,30}(?:when|should)\s+(?:i|we)\s+(?:buy|sell|trade)", re.I), "insider_trading"),
    # Pattern: "friend/colleague gave me a tip... buy/load"
    (re.compile(r"(?:my\s+)?(?:friend|colleague|boss|manager|contact|source|insider).{0,40}(?:tip|inside|non.?public|confidential|private|secret|advance).{0,40}(?:buy|sell|load|trade|invest|position|before)", re.I), "insider_trading"),
    (re.compile(r"(?:got|received|have)\s+(?:a\s+)?(?:tip|inside\s+info|advance\s+notice).{0,40}(?:buy|sell|load|trade|invest|before)", re.I), "insider_trading"),
    # Pattern: "need to know earnings before announcement"
    (re.compile(r"(?:i\s+)?(?:need|want)\s+to\s+(?:know|get|see|access)\s+(?:the\s+)?(?:earnings?|results?|numbers?|data|report)\s+(?:before|ahead|prior)", re.I), "insider_trading"),
    # Pattern: "trade/buy before announcement/earnings"
    (re.compile(r"(?:trade|buy|sell|invest|position|load\s+up).{0,30}(?:before|ahead\s+of|prior\s+to).{0,20}(?:announcement|earnings|report|release|IPO|merger|acquisition)", re.I), "insider_trading"),
    (re.compile(r"confidential\s+(?:merger|acquisition|deal|takeover)\s+(?:news|info|information|data)", re.I), "insider_trading"),
    # Pattern: "use this inside information to..."
    (re.compile(r"(?:use|act\s+on|trade\s+on|profit\s+from)\s+(?:this\s+)?(?:inside|insider|non.?public|confidential|private|material)\s+(?:info|information|knowledge|data|news|tip)", re.I), "insider_trading"),
    (re.compile(r"(?:i\s+)?(?:have|got|received|obtained)\s+(?:insider|inside|non.?public|confidential|material)\s+(?:info|information|knowledge|data|news)", re.I), "insider_trading"),

    # ═══ Market manipulation ═══
    (re.compile(r"(?:help|let|tell)\s+me\s+(?:to\s+)?(?:pump|manipulate|spoof|layer|corner|rig|inflate)", re.I), "market_manipulation"),
    (re.compile(r"(?:pump|inflate|drive|push|move)\s+(?:up\s+)?(?:the\s+)?(?:price|stock|share|market|value)\s+(?:of\s+)?", re.I), "market_manipulation"),
    (re.compile(r"(?:coordinated|organized|group|collective|concerted)\s+(?:buying|selling|trading|effort)\s+(?:scheme|plan|strategy|campaign)?\s*(?:to\s+)?(?:move|push|drive|inflate|pump)", re.I), "market_manipulation"),
    (re.compile(r"design\s+a\s+(?:coordinated|group|collective)", re.I), "market_manipulation"),
    (re.compile(r"wash\s+trad(?:e|es|ed|ing)\s+(?:between|across|using|with)", re.I), "market_manipulation"),
    (re.compile(r"(?:help|let)\s+me\s+wash\s+trad", re.I), "market_manipulation"),
    (re.compile(r"create\s+(?:fake|artificial|false)\s+(?:volume|trades?|activity|demand|supply)", re.I), "market_manipulation"),
    (re.compile(r"(?:place|submit|enter)\s+(?:fake|phantom|false|spoof)\s+(?:orders?|bids?|offers?)", re.I), "market_manipulation"),
    (re.compile(r"(?:front.?run|front\s+run)\s+(?:trades?|orders?|clients?)", re.I), "market_manipulation"),
    (re.compile(r"(?:manipulate|rig|fix|corner)\s+(?:the\s+)?(?:market|price|stock|commodity|exchange)", re.I), "market_manipulation"),

    # ═══ Money laundering ═══
    (re.compile(r"(?:move|transfer|route|send|deposit|withdraw|funnel|channel).{0,40}(?:without\s+report|avoid.{0,20}(?:reporting|threshold|detection|scrutiny|compliance)|evade|circumvent)", re.I), "money_laundering"),
    (re.compile(r"(?:help\s+me\s+)?(?:structure|split|break\s+up|divide)\s+(?:deposits?|transactions?|payments?|transfers?)\s+(?:to\s+)?(?:avoid|evade|circumvent|below|under)", re.I), "money_laundering"),
    (re.compile(r"avoid\s+(?:the\s+)?(?:\$?\d+k?\s+)?(?:reporting|CTR|BSA|AML)\s+(?:threshold|requirement|limit|rule)", re.I), "money_laundering"),
    (re.compile(r"hide\s+(?:trading\s+|investment\s+)?(?:profits?|income|gains?|money|funds?|earnings?)\s+from\s+(?:the\s+)?(?:tax|IRS|HMRC|authority|government|regulator)", re.I), "money_laundering"),
    (re.compile(r"(?:layer|obscure|disguise|conceal|launder|clean)\s+.{0,20}(?:source|origin|trail)\s+of\s+(?:these\s+|the\s+|my\s+)?(?:funds?|money|capital|proceeds|cash)", re.I), "money_laundering"),
    (re.compile(r"without\s+(?:it\s+)?(?:being\s+)?report(?:ed|ing)\s+(?:it|this|them)", re.I), "money_laundering"),
    (re.compile(r"(?:launder|clean|wash)\s+(?:this\s+|the\s+|my\s+)?(?:money|cash|funds?|proceeds)", re.I), "money_laundering"),
    (re.compile(r"(?:open|use|set\s+up)\s+(?:offshore|shell|anonymous|fake)\s+(?:accounts?|companies?|entities?)\s+(?:to\s+)?(?:hide|conceal|move|funnel|channel)", re.I), "money_laundering"),

    # ═══ Guaranteed returns ═══
    (re.compile(r"guarantee\s+(?:me|us|that)\s+(?:a\s+)?\d+", re.I), "guaranteed_returns"),
    (re.compile(r"promise\s+(?:me|us)\s+(?:my|our|that|a)\s+(?:money|investment|portfolio|returns?)\s+will", re.I), "guaranteed_returns"),
    (re.compile(r"(?:100\s*%|completely|absolutely|totally)\s+(?:certain|sure|guaranteed|safe|risk.?free)\s+(?:to\s+)?(?:go\s+up|rise|profit|return|make\s+money|gain)", re.I), "guaranteed_returns"),
    (re.compile(r"(?:tell|show|give|find)\s+me\s+(?:a\s+)?(?:stock|investment|fund|asset|crypto).{0,20}(?:100\s*%|certain|sure|guaranteed|can.?t\s+(?:lose|fail))", re.I), "guaranteed_returns"),
    (re.compile(r"(?:foolproof|guaranteed|sure.?fire|no.?fail|risk.?free|can.?t.?lose)\s+(?:way|method|strategy|approach|plan)\s+to\s+(?:make|earn|get|generate)\s+(?:\$?\d+|money|profit|returns)", re.I), "guaranteed_returns"),
    (re.compile(r"(?:guaranteed|risk.?free|no.?risk)\s+(?:\d+\s*%?\s+)?(?:returns?|profit|gains?|income|yield)", re.I), "guaranteed_returns"),
    (re.compile(r"(?:assure|ensure|certify)\s+(?:me\s+)?(?:that\s+)?(?:i|my)\s+(?:will|won.?t)\s+(?:make|lose|gain|profit|get)", re.I), "guaranteed_returns"),
    (re.compile(r"(?:double|triple|quadruple)\s+(?:my|the)\s+(?:money|investment|portfolio)\s+(?:in|within)\s+(?:a\s+)?(?:month|week|day|year)", re.I), "guaranteed_returns"),

    # ═══ Reckless advice ═══
    (re.compile(r"(?:put|invest|place|move|throw|dump|pour)\s+(?:all|entire|every\s+(?:cent|penny|dollar)|100\s*%\s+of)\s+(?:my\s+|our\s+)?(?:retirement|savings?|life\s+savings?|pension|401k|nest\s+egg|emergency\s+fund).{0,30}(?:in(?:to)?|on)\s+(?:crypto|bitcoin|options?|meme|penny|single|one\s+stock|lever|margin|speculative|volatile|risky|gambl)", re.I), "reckless_advice"),
    (re.compile(r"(?:i(?:'?m|\s+am)\s+\d{2,}\s+(?:and|years?)\s+.{0,30})?(?:put|invest|place|move)\s+(?:all|entire|every)\s+(?:my\s+)?(?:retirement|savings?|life\s+savings?|pension|emergency).{0,20}(?:crypto|bitcoin|options?|meme|penny|single)", re.I), "reckless_advice"),
    (re.compile(r"tell\s+me\s+to\s+(?:take|get|use)\s+(?:a\s+)?(?:margin|leveraged?)\s+(?:loan|position|debt|credit)", re.I), "reckless_advice"),
    (re.compile(r"(?:take|get|use)\s+(?:a\s+)?(?:margin\s+loan|leverage|margin|borrowed\s+money)\s+to\s+(?:buy|invest|trade|purchase|load)", re.I), "reckless_advice"),
    (re.compile(r"(?:entire|all\s+(?:my|of)|whole|100\s*%)\s+(?:my\s+)?(?:emergency\s+fund|rainy\s+day|nest\s+egg)\s+(?:in(?:to)?|on)\s+(?:options?|crypto|penny|meme|speculative|volatile|gambl|leveraged)", re.I), "reckless_advice"),
    (re.compile(r"(?:should\s+i\s+)?(?:put|invest|place|move)\s+(?:my\s+)?(?:entire\s+)?emergency\s+fund\s+(?:in(?:to)?|on)\s+(?:options?|crypto|penny|meme|speculative)", re.I), "reckless_advice"),
    (re.compile(r"(?:mortgage|refinance|take\s+(?:a\s+)?(?:loan|HELOC)\s+(?:on|against))\s+(?:my\s+)?(?:house|home|property|apartment)\s+(?:to\s+|for\s+)?(?:buy|invest|trade|purchase|fund|put\s+into)", re.I), "reckless_advice"),
    (re.compile(r"(?:tell\s+me\s+|which\s+|what\s+)?(?:stock|crypto|asset)\s+(?:to|should\s+i)\s+(?:mortgage|refinance|HELOC|borrow\s+against)\s+(?:my\s+)?(?:house|home)", re.I), "reckless_advice"),
    (re.compile(r"(?:borrow|loan|take\s+out\s+(?:a\s+)?(?:personal|home|payday)\s+loan)\s+(?:money\s+)?(?:to\s+)?(?:buy|invest\s+in|trade|gamble\s+on)\s+(?:crypto|bitcoin|options?|meme\s+stock|penny\s+stock|speculative)", re.I), "reckless_advice"),
    (re.compile(r"(?:yolo|all.?in|bet\s+(?:everything|it\s+all))\s+(?:on|into)\s+(?:one\s+)?(?:stock|crypto|options?|trade|position)", re.I), "reckless_advice"),

    # ═══ Sanctions evasion ═══
    (re.compile(r"(?:bypass|circumvent|evade|avoid|get\s+around|work\s+around|route\s+around|skirt)\s+(?:the\s+)?(?:OFAC\s+|EU\s+|UN\s+|US\s+|international\s+)?sanctions", re.I), "sanctions_evasion"),
    (re.compile(r"(?:route|channel|funnel|move)\s+(?:trades?|transactions?|funds?|money|payments?).{0,30}(?:shell\s+company|shell\s+entity|front\s+company|proxy|intermediary|third\s+party).{0,30}(?:bypass|avoid|circumvent|evade)\s+(?:OFAC\s+)?(?:sanctions|restrictions|embargo)", re.I), "sanctions_evasion"),
    (re.compile(r"(?:invest|trade|buy|deal)\s+(?:in|with)\s+(?:a\s+)?(?:sanctioned|banned|restricted|embargoed|blacklisted)\s+(?:russian|iranian|north\s+korean|syrian|cuban|venezuelan|chinese\s+military).{0,30}(?:without.{0,20}(?:traced|detected|caught|noticed|flagged)|undetected|anonymously|secretly|covertly)", re.I), "sanctions_evasion"),
    (re.compile(r"(?:shell|front|offshore)\s+(?:company|entity|firm|account).{0,30}(?:bypass|avoid|circumvent|evade).{0,15}(?:sanctions|OFAC|restrictions|embargo|screening)", re.I), "sanctions_evasion"),
    (re.compile(r"without\s+(?:it\s+)?being\s+(?:traced|detected|caught|noticed|flagged|reported)", re.I), "sanctions_evasion"),
    (re.compile(r"(?:hide|conceal|disguise|mask)\s+(?:my\s+|the\s+)?(?:identity|involvement|connection|association).{0,20}(?:sanctioned|banned|restricted)", re.I), "sanctions_evasion"),

    # ═══ Fraud ═══
    (re.compile(r"(?:draft|create|make|forge|fabricate|generate|produce)\s+(?:a\s+)?(?:fake|false|fraudulent|forged|fabricated|counterfeit|fictitious)\s+(?:contract|document|note|statement|receipt|invoice|report|certificate|record)", re.I), "fraud"),
    (re.compile(r"(?:falsify|fabricate|forge|fake|doctor|alter|tamper\s+with)\s+(?:my\s+|the\s+|a\s+)?(?:financial|tax|trading|account|bank|brokerage|investment)\s+(?:records?|statements?|documents?|reports?|returns?)", re.I), "fraud"),
    (re.compile(r"(?:claim|report|declare)\s+(?:fake|false|fraudulent|fictitious)\s+(?:losses|gains?|income|expenses?|deductions?|trades?)", re.I), "fraud"),
]


def check(query: str) -> SafetyVerdict:
    """
    Run the safety guard on a user query. Returns a SafetyVerdict.

    This is PURE LOCAL COMPUTATION — no LLM, no network.
    Must complete well under 10ms.
    """
    if not query or not query.strip():
        return SafetyVerdict(blocked=False, message="")

    text = query.strip()

    # Layer 1: Check if the query is educational/informational FIRST.
    # If strong educational signals are present, let it through immediately.
    is_educational = any(pat.search(text) for pat in EDUCATIONAL_PATTERNS)

    # Layer 2: Check action patterns — these detect first-person harmful intent.
    for pattern, category in ACTION_PATTERNS:
        if pattern.search(text):
            # Even if educational signals exist, action patterns for DOING
            # harmful things take precedence — except for very clear educational
            # phrasing that happens to contain similar words.
            if is_educational and _is_genuinely_educational(text, category):
                continue
            return SafetyVerdict(
                blocked=True,
                category=category,
                message=CATEGORY_RESPONSES.get(category, CATEGORY_RESPONSES["fraud"]),
            )

    return SafetyVerdict(blocked=False, message="")


def _is_genuinely_educational(text: str, category: str) -> bool:
    """
    Heuristic: if the query starts with an educational pattern and doesn't
    contain first-person action intent, it's probably educational.
    """
    text_lower = text.lower()

    # Strong educational starters — these almost always mean the user
    # is asking ABOUT the topic, not trying to DO it.
    strong_edu = [
        "what is", "what are", "what does", "explain", "define",
        "describe", "how does", "how do regulators", "how do brokers",
        "how does the sec", "how does the fca", "is it legal",
        "is short selling", "is insider", "what factors", "what compliance",
        "what are the penalties", "what are the risks",
        "why are", "why is", "why do", "why does",
        "can you explain", "tell me about the regulation",
        "tell me how the", "what happens if", "what are the fines",
        "what should i know about", "educate me", "teach me",
        "learning about",
    ]

    starts_educational = any(text_lower.startswith(s) for s in strong_edu)
    if starts_educational:
        return True

    # Also check for educational question patterns anywhere
    edu_anywhere = [
        "what are the penalties",
        "what are the consequences",
        "how do regulators",
        "how does the sec",
        "how does the fca",
    ]
    if any(phrase in text_lower for phrase in edu_anywhere):
        return True

    return False
