"""
entity_extractor.py
-------------------
Extracts medical entities from user queries using keyword matching and regex.
No external NLP APIs required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ──────────────────────────────────────────────
# Entity dictionaries (lowercased keys)
# ──────────────────────────────────────────────

CANCER_ALIASES: dict[str, str] = {
    "osteosarcoma":            "Osteosarcoma",
    "bone cancer":             "Osteosarcoma",
    "bone tumor":              "Osteosarcoma",
    "leukemia":                "Acute Leukemia",
    "acute leukemia":          "Acute Leukemia",
    "aml":                     "Acute Leukemia",
    "all":                     "Acute Leukemia",
    "blood cancer":            "Acute Leukemia",
    "breast cancer":           "Breast Cancer",
    "breast":                  "Breast Cancer",
    "mammary carcinoma":       "Breast Cancer",
    "lung cancer":             "Lung Cancer",
    "lung":                    "Lung Cancer",
    "nsclc":                   "Lung Cancer",
    "sclc":                    "Lung Cancer",
    "melanoma":                "Melanoma",
    "skin melanoma":           "Melanoma",
    "skin cancer":             "Skin Cancer",
    "basal cell":              "Skin Cancer",
    "squamous cell":           "Skin Cancer",
    "bcc":                     "Skin Cancer",
    "scc":                     "Skin Cancer",
}

DRUG_ALIASES: dict[str, str] = {
    "cisplatin":               "Cisplatin",
    "cis-platinum":            "Cisplatin",
    "platinol":                "Cisplatin",
    "doxorubicin":             "Doxorubicin",
    "adriamycin":              "Doxorubicin",
    "paclitaxel":              "Paclitaxel",
    "taxol":                   "Paclitaxel",
    "methotrexate":            "Methotrexate",
    "mtx":                     "Methotrexate",
    "cyclophosphamide":        "Cyclophosphamide",
    "cytoxan":                 "Cyclophosphamide",
    "vincristine":             "Vincristine",
    "oncovin":                 "Vincristine",
    "carboplatin":             "Carboplatin",
    "paraplatin":              "Carboplatin",
    "ifosfamide":              "Ifosfamide",
    "ifex":                    "Ifosfamide",
    "etoposide":               "Etoposide",
    "vp-16":                   "Etoposide",
    "5-fluorouracil":          "5-Fluorouracil",
    "5fu":                     "5-Fluorouracil",
    "5-fu":                    "5-Fluorouracil",
    "fluorouracil":            "5-Fluorouracil",
    "tamoxifen":               "Tamoxifen",
    "nolvadex":                "Tamoxifen",
    "trastuzumab":             "Trastuzumab",
    "herceptin":               "Trastuzumab",
    "pembrolizumab":           "Pembrolizumab",
    "keytruda":                "Pembrolizumab",
    "ipilimumab":              "Ipilimumab",
    "yervoy":                  "Ipilimumab",
    "vemurafenib":             "Vemurafenib",
    "zelboraf":                "Vemurafenib",
    "gemcitabine":             "Gemcitabine",
    "gemzar":                  "Gemcitabine",
    "erlotinib":               "Erlotinib",
    "tarceva":                 "Erlotinib",
    "bevacizumab":             "Bevacizumab",
    "avastin":                 "Bevacizumab",
    # Non-cancer drugs
    "warfarin":                "Warfarin",
    "coumadin":                "Warfarin",
    "aspirin":                 "Aspirin",
    "ibuprofen":               "Ibuprofen",
    "advil":                   "Ibuprofen",
    "phenytoin":               "Phenytoin",
    "dilantin":                "Phenytoin",
    "fluconazole":             "Fluconazole",
    "diflucan":                "Fluconazole",
    "ciprofloxacin":           "Ciprofloxacin",
    "cipro":                   "Ciprofloxacin",
    "metformin":               "Metformin",
    "omeprazole":              "Omeprazole",
    "prilosec":                "Omeprazole",
    "ondansetron":             "Ondansetron",
    "zofran":                  "Ondansetron",
    "dexamethasone":           "Dexamethasone",
}

SIDE_EFFECT_ALIASES: dict[str, str] = {
    "nausea":                  "Nausea",
    "feel sick":               "Nausea",
    "queasy":                  "Nausea",
    "sick to stomach":         "Nausea",
    "vomiting":                "Vomiting",
    "throwing up":             "Vomiting",
    "vomit":                   "Vomiting",
    "diarrhea":                "Diarrhea",
    "loose stools":            "Diarrhea",
    "watery stool":            "Diarrhea",
    "constipation":            "Constipation",
    "can't poop":              "Constipation",
    "hard stool":              "Constipation",
    "mucositis":               "Mucositis",
    "mouth sores":             "Mucositis",
    "oral ulcers":             "Mucositis",
    "sore mouth":              "Mucositis",
    "esophagitis":             "Esophagitis",
    "sore throat":             "Esophagitis",
    "trouble swallowing":      "Esophagitis",
    "dysphagia":               "Esophagitis",
    "appetite loss":           "Appetite Loss",
    "no appetite":             "Appetite Loss",
    "anorexia":                "Appetite Loss",
    "not hungry":              "Appetite Loss",
    "dry mouth":               "Dry Mouth",
    "xerostomia":              "Dry Mouth",
    "taste change":            "Taste Changes",
    "metallic taste":          "Taste Changes",
    "taste changes":           "Taste Changes",
    "dysgeusia":               "Taste Changes",
    "weight loss":             "Weight Loss",
    "losing weight":           "Weight Loss",
    "weight gain":             "Weight Gain",
    "gaining weight":          "Weight Gain",
    "fatigue":                 "Fatigue",
    "tired":                   "Fatigue",
    "exhaustion":              "Fatigue",
    "peripheral neuropathy":   "Peripheral Neuropathy",
    "neuropathy":              "Peripheral Neuropathy",
    "tingling":                "Peripheral Neuropathy",
    "numbness":                "Peripheral Neuropathy",
    "alopecia":                "Alopecia",
    "hair loss":               "Alopecia",
    "losing hair":             "Alopecia",
    "neutropenia":             "Neutropenia",
    "low white cells":         "Neutropenia",
    "thrombocytopenia":        "Thrombocytopenia",
    "low platelets":           "Thrombocytopenia",
    "anemia":                  "Anemia",
    "low hemoglobin":          "Anemia",
    "nephrotoxicity":          "Nephrotoxicity",
    "kidney damage":           "Nephrotoxicity",
    "cardiotoxicity":          "Cardiotoxicity",
    "heart damage":            "Cardiotoxicity",
    "lactose intolerance":     "Lactose Intolerance",
    "dehydration":             "Dehydration",
    "immunosuppression":       "Immunosuppression",
    "infection":               "Immunosuppression",
}

FOOD_ALIASES: dict[str, str] = {
    "banana":                  "Banana",
    "white rice":              "White Rice",
    "rice":                    "White Rice",
    "toast":                   "Toast",
    "applesauce":              "Applesauce",
    "yogurt":                  "Yogurt",
    "scrambled eggs":          "Scrambled Eggs",
    "eggs":                    "Scrambled Eggs",
    "milkshake":               "Milkshake",
    "broth":                   "Broth",
    "ginger tea":              "Ginger Tea",
    "ginger":                  "Ginger Tea",
    "lemon water":             "Lemon Water",
    "lemon":                   "Lemon Water",
    "oatmeal":                 "Oatmeal",
    "peanut butter":           "Peanut Butter",
    "avocado":                 "Avocado",
    "prune juice":             "Prune Juice",
    "prunes":                  "Prune Juice",
    "sports drink":            "Sports Drink",
    "gatorade":                "Sports Drink",
    "ice chips":               "Ice Chips",
    "protein shake":           "Protein Shake",
    "whole grain bread":       "Whole Grain Bread",
    "whole grain":             "Whole Grain Bread",
    "cooked vegetables":       "Cooked Vegetables",
    "citrus juice":            "Citrus Juice",
    "orange juice":            "Citrus Juice",
    "spicy food":              "Spicy Foods",
    "spicy":                   "Spicy Foods",
    "fried food":              "Fried Foods",
    "fried":                   "Fried Foods",
    "alcohol":                 "Alcohol",
    "beer":                    "Alcohol",
    "wine":                    "Alcohol",
    "raw vegetables":          "Raw Vegetables",
    "dairy":                   "Dairy (full-fat)",
    "milk":                    "Dairy (full-fat)",
    "custard":                 "Custard",
    "popsicle":                "Popsicle",
    "overnight oats":          "Overnight Oats",
    "bouillon":                "Bouillon",
    "crackers":                "Crackers",
}

# Intent patterns
FOOD_RECOMMENDATION_PATTERNS: list[str] = [
    r"what (should|can) (i|you) eat",
    r"food (for|to help|that help)",
    r"diet (for|during|after)",
    r"recommend.*food",
    r"good food",
    r"safe to eat",
    r"eating (during|after|with)",
    r"nutrition",
    r"meal plan",
]

DRUG_INTERACTION_PATTERNS: list[str] = [
    r"interact",
    r"drug (with|and)",
    r"take (with|together)",
    r"combine",
    r"mixing",
    r"avoid (with|taking)",
    r"safe (with|to take)",
]

SIDE_EFFECT_PATTERNS: list[str] = [
    r"side effect",
    r"cause",
    r"symptom",
    r"what happens",
    r"risk of",
]


# ──────────────────────────────────────────────
# Extraction result
# ──────────────────────────────────────────────

@dataclass
class ExtractedEntities:
    cancers:      list[str] = field(default_factory=list)
    drugs:        list[str] = field(default_factory=list)
    ncdrugs:      list[str] = field(default_factory=list)   # non-cancer drugs
    side_effects: list[str] = field(default_factory=list)
    foods:        list[str] = field(default_factory=list)
    intents:      list[str] = field(default_factory=list)   # food_rec, drug_interaction, side_effect

    def is_empty(self) -> bool:
        return not any([
            self.cancers, self.drugs, self.ncdrugs,
            self.side_effects, self.foods,
        ])

    def to_dict(self) -> dict:
        return {
            "cancers":      self.cancers,
            "drugs":        self.drugs,
            "ncdrugs":      self.ncdrugs,
            "side_effects": self.side_effects,
            "foods":        self.foods,
            "intents":      self.intents,
        }


# ──────────────────────────────────────────────
# Core extractor
# ──────────────────────────────────────────────

# All non-cancer drugs for disambiguation
_NON_CANCER_DRUG_SET = {
    "Warfarin", "Aspirin", "Ibuprofen", "Phenytoin",
    "Fluconazole", "Ciprofloxacin", "Metformin",
    "Omeprazole", "Ondansetron", "Dexamethasone",
}

_CANCER_DRUG_SET = {
    "Cisplatin", "Doxorubicin", "Paclitaxel", "Methotrexate",
    "Cyclophosphamide", "Vincristine", "Carboplatin", "Ifosfamide",
    "Etoposide", "5-Fluorouracil", "Tamoxifen", "Trastuzumab",
    "Pembrolizumab", "Ipilimumab", "Vemurafenib", "Gemcitabine",
    "Erlotinib", "Bevacizumab",
}


def _match_dict(text: str, alias_dict: dict[str, str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for alias, canonical in alias_dict.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            if canonical not in seen:
                seen.add(canonical)
                found.append(canonical)
    return found


def _detect_intents(text: str) -> list[str]:
    intents: list[str] = []
    for pat in FOOD_RECOMMENDATION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            intents.append("food_recommendation")
            break
    for pat in DRUG_INTERACTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            intents.append("drug_interaction")
            break
    for pat in SIDE_EFFECT_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            intents.append("side_effect_query")
            break
    return intents


def extract_entities(query: str, patient_report: str = "") -> ExtractedEntities:
    """
    Extract medical entities.
    Smart extraction: Prioritizes the immediate query. If the user uses vague terms 
    like "my symptoms" or "my drugs", it falls back to the patient report to fill in the blanks.
    """
    # 1. Extract from the immediate query
    query_drugs   = _match_dict(query, DRUG_ALIASES)
    query_cancers = _match_dict(query, CANCER_ALIASES)
    query_se      = _match_dict(query, SIDE_EFFECT_ALIASES)
    query_foods   = _match_dict(query, FOOD_ALIASES)
    intents       = _detect_intents(query)

    # 2. Extract from the patient report context
    report_cancers = _match_dict(patient_report, CANCER_ALIASES) if patient_report else []
    report_se      = _match_dict(patient_report, SIDE_EFFECT_ALIASES) if patient_report else []
    report_drugs   = _match_dict(patient_report, DRUG_ALIASES) if patient_report else []

    # 3. Smart Merge
    # Always include the cancer type for context
    final_cancers = list(set(query_cancers + report_cancers))
    
    # Broader fallback for side effects
    final_se = list(set(query_se))
    se_triggers = ["symptom", "side effect", "food", "eat", "avoid", "help", "manage"]
    if not final_se and any(word in query.lower() for word in se_triggers):
        final_se = list(set(report_se))

    # Broader fallback for drugs
    final_drugs = list(set(query_drugs))
    drug_triggers = ["drug", "medication", "treatment", "chemo", "therapy", "regimen", "interact", "safe"]
    if not final_drugs and any(word in query.lower() for word in drug_triggers):
        final_drugs = list(set(report_drugs))

    # Split drugs into cancer and non-cancer categories
    cancer_drugs = [d for d in final_drugs if d in _CANCER_DRUG_SET]
    nc_drugs     = [d for d in final_drugs if d in _NON_CANCER_DRUG_SET]

    result = ExtractedEntities(
        cancers      = final_cancers,
        drugs        = cancer_drugs,
        ncdrugs      = nc_drugs,
        side_effects = final_se,
        foods        = query_foods,
        intents      = intents,
    )
    
    # 🚨 DEBUG PRINT: Watch your terminal to verify this is working!
    print(f"\n[ENTITY EXTRACTOR] Extracted for Graph: {result.to_dict()}")

    return result