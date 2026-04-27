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
    Extract medical entities from query + optional patient report.
    Returns an ExtractedEntities dataclass.
    """
    combined = f"{query} {patient_report}"

    all_drugs = _match_dict(combined, DRUG_ALIASES)

    cancer_drugs = [d for d in all_drugs if d in _CANCER_DRUG_SET]
    nc_drugs     = [d for d in all_drugs if d in _NON_CANCER_DRUG_SET]

    return ExtractedEntities(
        cancers      = _match_dict(combined, CANCER_ALIASES),
        drugs        = cancer_drugs,
        ncdrugs      = nc_drugs,
        side_effects = _match_dict(combined, SIDE_EFFECT_ALIASES),
        foods        = _match_dict(combined, FOOD_ALIASES),
        intents      = _detect_intents(combined),
    )


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "What foods should I eat to manage nausea from cisplatin?",
        "Can I take warfarin with methotrexate?",
        "What are the side effects of paclitaxel for breast cancer?",
        "My osteosarcoma treatment causes mouth sores. What can help?",
        "Is alcohol safe during chemotherapy with doxorubicin?",
    ]
    for q in queries:
        e = extract_entities(q)
        print(f"\nQuery: {q}")
        print(f"  Cancers:      {e.cancers}")
        print(f"  Drugs:        {e.drugs}")
        print(f"  NC Drugs:     {e.ncdrugs}")
        print(f"  Side Effects: {e.side_effects}")
        print(f"  Foods:        {e.foods}")
        print(f"  Intents:      {e.intents}")
