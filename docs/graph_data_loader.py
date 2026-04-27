"""
graph_data_loader.py
--------------------
Loads all nodes and relationships into Neo4j.
Data is sourced from:
  - "Eating Hints: Before, During, and After Cancer Treatment" (NCI PDF)
  - PubMed oncology literature for cancer types and chemo drugs
  - General pharmacology knowledge for drug interactions
"""

from __future__ import annotations

import logging
from neo4j_client import get_client

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# NODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════

CANCERS: list[dict] = [
    {"name": "Osteosarcoma",         "type": "bone",        "icd10": "C40",  "stage_system": "AJCC",  "description": "Malignant bone tumor most common in adolescents"},
    {"name": "Acute Leukemia",       "type": "blood",       "icd10": "C91",  "stage_system": "FAB",   "description": "Rapid proliferation of immature blood cells"},
    {"name": "Breast Cancer",        "type": "solid",       "icd10": "C50",  "stage_system": "AJCC",  "description": "Malignancy arising from breast tissue"},
    {"name": "Lung Cancer",          "type": "solid",       "icd10": "C34",  "stage_system": "AJCC",  "description": "Carcinoma of the bronchial epithelium"},
    {"name": "Melanoma",             "type": "skin",        "icd10": "C43",  "stage_system": "AJCC",  "description": "Malignant melanocytic neoplasm"},
    {"name": "Skin Cancer",          "type": "skin",        "icd10": "C44",  "stage_system": "AJCC",  "description": "Non-melanoma skin cancers including BCC and SCC"},
]

DRUGS: list[dict] = [
    {"name": "Cisplatin",       "drug_class": "Platinum-based alkylating agent",   "generic_name": "cisplatin",       "route": "IV",   "mechanism": "DNA crosslinking"},
    {"name": "Doxorubicin",     "drug_class": "Anthracycline antibiotic",          "generic_name": "doxorubicin",     "route": "IV",   "mechanism": "Topoisomerase II inhibition"},
    {"name": "Paclitaxel",      "drug_class": "Taxane",                            "generic_name": "paclitaxel",      "route": "IV",   "mechanism": "Microtubule stabilization"},
    {"name": "Methotrexate",    "drug_class": "Antimetabolite",                    "generic_name": "methotrexate",    "route": "IV/PO","mechanism": "DHFR inhibition"},
    {"name": "Cyclophosphamide","drug_class": "Alkylating agent",                  "generic_name": "cyclophosphamide","route": "IV/PO","mechanism": "DNA alkylation"},
    {"name": "Vincristine",     "drug_class": "Vinca alkaloid",                    "generic_name": "vincristine",     "route": "IV",   "mechanism": "Microtubule disruption"},
    {"name": "Carboplatin",     "drug_class": "Platinum-based alkylating agent",   "generic_name": "carboplatin",     "route": "IV",   "mechanism": "DNA crosslinking"},
    {"name": "Ifosfamide",      "drug_class": "Alkylating agent",                  "generic_name": "ifosfamide",      "route": "IV",   "mechanism": "DNA alkylation"},
    {"name": "Etoposide",       "drug_class": "Topoisomerase II inhibitor",        "generic_name": "etoposide",       "route": "IV/PO","mechanism": "Topoisomerase II inhibition"},
    {"name": "5-Fluorouracil",  "drug_class": "Antimetabolite",                    "generic_name": "fluorouracil",    "route": "IV",   "mechanism": "Thymidylate synthase inhibition"},
    {"name": "Tamoxifen",       "drug_class": "SERM",                              "generic_name": "tamoxifen",       "route": "PO",   "mechanism": "Estrogen receptor antagonism"},
    {"name": "Trastuzumab",     "drug_class": "Monoclonal antibody",               "generic_name": "trastuzumab",     "route": "IV",   "mechanism": "HER2 receptor blockade"},
    {"name": "Pembrolizumab",   "drug_class": "PD-1 inhibitor",                    "generic_name": "pembrolizumab",   "route": "IV",   "mechanism": "Immune checkpoint blockade"},
    {"name": "Ipilimumab",      "drug_class": "CTLA-4 inhibitor",                  "generic_name": "ipilimumab",      "route": "IV",   "mechanism": "Immune checkpoint blockade"},
    {"name": "Vemurafenib",     "drug_class": "BRAF inhibitor",                    "generic_name": "vemurafenib",     "route": "PO",   "mechanism": "BRAF V600E inhibition"},
    {"name": "Gemcitabine",     "drug_class": "Antimetabolite",                    "generic_name": "gemcitabine",     "route": "IV",   "mechanism": "Ribonucleotide reductase inhibition"},
    {"name": "Erlotinib",       "drug_class": "EGFR inhibitor",                    "generic_name": "erlotinib",       "route": "PO",   "mechanism": "EGFR tyrosine kinase inhibition"},
    {"name": "Bevacizumab",     "drug_class": "VEGF inhibitor",                    "generic_name": "bevacizumab",     "route": "IV",   "mechanism": "VEGF-A blockade"},
]

NON_CANCER_DRUGS: list[dict] = [
    {"name": "Warfarin",        "indication": "Anticoagulation",     "drug_class": "Vitamin K antagonist"},
    {"name": "Aspirin",         "indication": "Analgesia/Antiplatelet", "drug_class": "NSAID"},
    {"name": "Ibuprofen",       "indication": "Analgesia/Anti-inflammatory", "drug_class": "NSAID"},
    {"name": "Phenytoin",       "indication": "Epilepsy",            "drug_class": "Anticonvulsant"},
    {"name": "Fluconazole",     "indication": "Antifungal",          "drug_class": "Azole antifungal"},
    {"name": "Ciprofloxacin",   "indication": "Bacterial infection", "drug_class": "Fluoroquinolone"},
    {"name": "Metformin",       "indication": "Type 2 Diabetes",     "drug_class": "Biguanide"},
    {"name": "Omeprazole",      "indication": "GERD",                "drug_class": "Proton pump inhibitor"},
    {"name": "Ondansetron",     "indication": "Antiemetic",          "drug_class": "5-HT3 antagonist"},
    {"name": "Dexamethasone",   "indication": "Anti-inflammatory",   "drug_class": "Corticosteroid"},
]

SIDE_EFFECTS: list[dict] = [
    {"name": "Nausea",                  "icd10_code": "R11.0",  "severity_range": "mild-severe",    "description": "Queasy sensation with urge to vomit"},
    {"name": "Vomiting",                "icd10_code": "R11.1",  "severity_range": "mild-severe",    "description": "Forceful expulsion of stomach contents"},
    {"name": "Diarrhea",                "icd10_code": "K59.1",  "severity_range": "mild-severe",    "description": "Frequent loose or watery stools"},
    {"name": "Constipation",            "icd10_code": "K59.0",  "severity_range": "mild-moderate",  "description": "Infrequent hard stools difficult to pass"},
    {"name": "Mucositis",               "icd10_code": "K12.30", "severity_range": "mild-severe",    "description": "Inflammation and ulceration of oral mucosa"},
    {"name": "Esophagitis",             "icd10_code": "K20.9",  "severity_range": "mild-severe",    "description": "Inflammation of esophageal lining"},
    {"name": "Appetite Loss",           "icd10_code": "R63.0",  "severity_range": "mild-severe",    "description": "Anorexia; reduced desire to eat"},
    {"name": "Dry Mouth",               "icd10_code": "R68.2",  "severity_range": "mild-moderate",  "description": "Xerostomia; reduced salivary flow"},
    {"name": "Taste Changes",           "icd10_code": "R43.8",  "severity_range": "mild-moderate",  "description": "Dysgeusia; altered or metallic taste perception"},
    {"name": "Weight Loss",             "icd10_code": "R63.4",  "severity_range": "mild-severe",    "description": "Unintentional loss of body weight"},
    {"name": "Weight Gain",             "icd10_code": "R63.5",  "severity_range": "mild-moderate",  "description": "Unintentional increase in body weight"},
    {"name": "Fatigue",                 "icd10_code": "R53.83", "severity_range": "mild-severe",    "description": "Persistent exhaustion not relieved by rest"},
    {"name": "Peripheral Neuropathy",   "icd10_code": "G62.9",  "severity_range": "mild-severe",    "description": "Tingling, numbness, pain in hands and feet"},
    {"name": "Alopecia",                "icd10_code": "L65.9",  "severity_range": "mild-severe",    "description": "Hair loss from chemotherapy"},
    {"name": "Neutropenia",             "icd10_code": "D70.1",  "severity_range": "moderate-severe","description": "Low neutrophil count increasing infection risk"},
    {"name": "Thrombocytopenia",        "icd10_code": "D69.6",  "severity_range": "moderate-severe","description": "Low platelet count causing bleeding risk"},
    {"name": "Anemia",                  "icd10_code": "D64.9",  "severity_range": "mild-severe",    "description": "Reduced red blood cell count"},
    {"name": "Nephrotoxicity",          "icd10_code": "N17.9",  "severity_range": "moderate-severe","description": "Drug-induced kidney damage"},
    {"name": "Cardiotoxicity",          "icd10_code": "I42.9",  "severity_range": "moderate-severe","description": "Drug-induced cardiac dysfunction"},
    {"name": "Lactose Intolerance",     "icd10_code": "E73.9",  "severity_range": "mild-moderate",  "description": "Inability to digest lactose post-treatment"},
    {"name": "Dehydration",             "icd10_code": "E86.0",  "severity_range": "mild-severe",    "description": "Excessive fluid loss from vomiting/diarrhea"},
    {"name": "Ototoxicity",             "icd10_code": "H91.0",  "severity_range": "moderate-severe","description": "Hearing loss from cisplatin"},
    {"name": "Immunosuppression",       "icd10_code": "D84.9",  "severity_range": "moderate-severe","description": "Reduced immune system function"},
    {"name": "Mucus Membrane Damage",   "icd10_code": "K13.70", "severity_range": "mild-moderate",  "description": "Damage to mucus membranes throughout GI tract"},
]

FOODS: list[dict] = [
    {"name": "Banana",             "category": "fruit",      "description": "Soft, easily digestible, high-potassium fruit"},
    {"name": "White Rice",         "category": "grain",      "description": "Low-fiber easily digestible grain"},
    {"name": "Toast",              "category": "grain",      "description": "Bland, easy-to-chew baked bread"},
    {"name": "Applesauce",         "category": "fruit",      "description": "Smooth pureed apple, low-fiber"},
    {"name": "Yogurt",             "category": "dairy",      "description": "Fermented milk product with probiotics"},
    {"name": "Scrambled Eggs",     "category": "protein",    "description": "Soft, high-protein egg preparation"},
    {"name": "Milkshake",          "category": "beverage",   "description": "High-calorie, high-protein cold drink"},
    {"name": "Broth",              "category": "liquid",     "description": "Clear sodium-containing liquid"},
    {"name": "Ginger Tea",         "category": "beverage",   "description": "Anti-nausea herbal beverage"},
    {"name": "Lemon Water",        "category": "beverage",   "description": "Tart citrus drink stimulating saliva"},
    {"name": "Oatmeal",            "category": "grain",      "description": "Soft high-fiber warm cereal"},
    {"name": "Peanut Butter",      "category": "protein",    "description": "High-calorie, high-protein nut spread"},
    {"name": "Avocado",            "category": "fruit",      "description": "High-calorie healthy-fat fruit"},
    {"name": "Prune Juice",        "category": "beverage",   "description": "High-fiber bowel-stimulating juice"},
    {"name": "Sports Drink",       "category": "beverage",   "description": "Sodium-potassium electrolyte replacement"},
    {"name": "Ice Chips",          "category": "liquid",     "description": "Frozen water soothing oral mucositis"},
    {"name": "Protein Shake",      "category": "supplement", "description": "High-protein high-calorie liquid meal"},
    {"name": "Whole Grain Bread",  "category": "grain",      "description": "High-fiber complex carbohydrate"},
    {"name": "Cooked Vegetables",  "category": "vegetable",  "description": "Soft, well-cooked low-irritant vegetables"},
    {"name": "Citrus Juice",       "category": "beverage",   "description": "Acidic vitamin-C-rich juice"},
    {"name": "Spicy Foods",        "category": "condiment",  "description": "Hot-pepper-based foods causing mucosal irritation"},
    {"name": "Fried Foods",        "category": "general",    "description": "High-fat foods slowing gastric emptying"},
    {"name": "Alcohol",            "category": "beverage",   "description": "Ethanol-containing beverages"},
    {"name": "Raw Vegetables",     "category": "vegetable",  "description": "Uncooked fibrous vegetables"},
    {"name": "Dairy (full-fat)",   "category": "dairy",      "description": "Full-fat milk products high in lactose"},
    {"name": "Custard",            "category": "dessert",    "description": "Soft high-calorie egg-milk dessert"},
    {"name": "Popsicle",           "category": "dessert",    "description": "Frozen flavored ice for oral cooling"},
    {"name": "Overnight Oats",     "category": "grain",      "description": "High-calorie, high-protein oat preparation"},
    {"name": "Bouillon",           "category": "liquid",     "description": "Clear broth for clear-liquid diet"},
    {"name": "Crackers",           "category": "grain",      "description": "Dry bland snack for nausea management"},
]

SYMPTOMS: list[dict] = [
    {"name": "Abdominal Cramps",       "description": "Painful muscular contractions in abdomen"},
    {"name": "Bloating",               "description": "Distension of abdomen due to gas"},
    {"name": "Mouth Sores",            "description": "Ulcerative lesions inside oral cavity"},
    {"name": "Difficulty Swallowing",  "description": "Dysphagia from esophageal inflammation"},
    {"name": "Metallic Taste",         "description": "Altered taste perception; common with cisplatin"},
    {"name": "Burning Sensation",      "description": "Burning in throat or mouth from mucositis"},
    {"name": "Infection Risk",         "description": "Elevated susceptibility to bacterial/fungal infection"},
    {"name": "Hair Thinning",          "description": "Gradual loss of hair density"},
    {"name": "Tingling Hands/Feet",    "description": "Paresthesia from peripheral neuropathy"},
    {"name": "Shortness of Breath",    "description": "Dyspnea from anemia or lung involvement"},
]


# ══════════════════════════════════════════════════════════════════
# RELATIONSHIP DEFINITIONS
# ══════════════════════════════════════════════════════════════════

# TREATED_WITH: Cancer → Drug
TREATED_WITH_RELS: list[dict] = [
    {"cancer": "Osteosarcoma",   "drug": "Cisplatin",        "line": "first",  "protocol": "MAP",       "confidence_score": 0.97},
    {"cancer": "Osteosarcoma",   "drug": "Doxorubicin",      "line": "first",  "protocol": "MAP",       "confidence_score": 0.97},
    {"cancer": "Osteosarcoma",   "drug": "Methotrexate",     "line": "first",  "protocol": "MAP",       "confidence_score": 0.97},
    {"cancer": "Osteosarcoma",   "drug": "Ifosfamide",       "line": "second", "protocol": "IE",        "confidence_score": 0.90},
    {"cancer": "Osteosarcoma",   "drug": "Etoposide",        "line": "second", "protocol": "IE",        "confidence_score": 0.90},
    {"cancer": "Acute Leukemia", "drug": "Vincristine",      "line": "first",  "protocol": "VDLP",      "confidence_score": 0.95},
    {"cancer": "Acute Leukemia", "drug": "Doxorubicin",      "line": "first",  "protocol": "VDLP",      "confidence_score": 0.95},
    {"cancer": "Acute Leukemia", "drug": "Cyclophosphamide", "line": "first",  "protocol": "Hyper-CVAD","confidence_score": 0.93},
    {"cancer": "Acute Leukemia", "drug": "Methotrexate",     "line": "first",  "protocol": "Maintenance","confidence_score": 0.95},
    {"cancer": "Acute Leukemia", "drug": "Etoposide",        "line": "second", "protocol": "FLAG-IDA",  "confidence_score": 0.88},
    {"cancer": "Breast Cancer",  "drug": "Doxorubicin",      "line": "first",  "protocol": "AC",        "confidence_score": 0.96},
    {"cancer": "Breast Cancer",  "drug": "Cyclophosphamide", "line": "first",  "protocol": "AC",        "confidence_score": 0.96},
    {"cancer": "Breast Cancer",  "drug": "Paclitaxel",       "line": "first",  "protocol": "AC-T",      "confidence_score": 0.96},
    {"cancer": "Breast Cancer",  "drug": "Trastuzumab",      "line": "first",  "protocol": "TCH",       "confidence_score": 0.97},
    {"cancer": "Breast Cancer",  "drug": "Tamoxifen",        "line": "adjuvant","protocol": "Hormonal",  "confidence_score": 0.95},
    {"cancer": "Breast Cancer",  "drug": "5-Fluorouracil",   "line": "first",  "protocol": "CMF",       "confidence_score": 0.92},
    {"cancer": "Lung Cancer",    "drug": "Cisplatin",        "line": "first",  "protocol": "Cisplatin-Pem","confidence_score": 0.96},
    {"cancer": "Lung Cancer",    "drug": "Carboplatin",      "line": "first",  "protocol": "Carboplatin-Pac","confidence_score": 0.95},
    {"cancer": "Lung Cancer",    "drug": "Paclitaxel",       "line": "first",  "protocol": "Carboplatin-Pac","confidence_score": 0.95},
    {"cancer": "Lung Cancer",    "drug": "Pembrolizumab",    "line": "first",  "protocol": "Keytruda",  "confidence_score": 0.97},
    {"cancer": "Lung Cancer",    "drug": "Erlotinib",        "line": "first",  "protocol": "EGFR-targeted","confidence_score": 0.94},
    {"cancer": "Lung Cancer",    "drug": "Bevacizumab",      "line": "first",  "protocol": "BCP",       "confidence_score": 0.93},
    {"cancer": "Melanoma",       "drug": "Pembrolizumab",    "line": "first",  "protocol": "IO",        "confidence_score": 0.97},
    {"cancer": "Melanoma",       "drug": "Ipilimumab",       "line": "first",  "protocol": "IO-combo",  "confidence_score": 0.95},
    {"cancer": "Melanoma",       "drug": "Vemurafenib",      "line": "first",  "protocol": "BRAF-targeted","confidence_score": 0.96},
    {"cancer": "Melanoma",       "drug": "Carboplatin",      "line": "second", "protocol": "Dacarbazine-CB","confidence_score": 0.82},
    {"cancer": "Skin Cancer",    "drug": "5-Fluorouracil",   "line": "topical","protocol": "Topical-5FU","confidence_score": 0.94},
    {"cancer": "Skin Cancer",    "drug": "Pembrolizumab",    "line": "first",  "protocol": "IO-advanced","confidence_score": 0.91},
]

# CAUSES: Drug → SideEffect
CAUSES_RELS: list[dict] = [
    # Cisplatin
    {"drug": "Cisplatin", "side_effect": "Nausea",               "severity": "severe",   "frequency": "very_common", "confidence_score": 0.99},
    {"drug": "Cisplatin", "side_effect": "Vomiting",              "severity": "severe",   "frequency": "very_common", "confidence_score": 0.99},
    {"drug": "Cisplatin", "side_effect": "Nephrotoxicity",        "severity": "severe",   "frequency": "common",      "confidence_score": 0.98},
    {"drug": "Cisplatin", "side_effect": "Ototoxicity",           "severity": "severe",   "frequency": "common",      "confidence_score": 0.97},
    {"drug": "Cisplatin", "side_effect": "Peripheral Neuropathy", "severity": "moderate", "frequency": "common",      "confidence_score": 0.95},
    {"drug": "Cisplatin", "side_effect": "Taste Changes",         "severity": "moderate", "frequency": "common",      "confidence_score": 0.93},
    {"drug": "Cisplatin", "side_effect": "Appetite Loss",         "severity": "moderate", "frequency": "common",      "confidence_score": 0.90},
    # Doxorubicin
    {"drug": "Doxorubicin", "side_effect": "Cardiotoxicity",      "severity": "severe",   "frequency": "uncommon",    "confidence_score": 0.97},
    {"drug": "Doxorubicin", "side_effect": "Alopecia",            "severity": "severe",   "frequency": "very_common", "confidence_score": 0.98},
    {"drug": "Doxorubicin", "side_effect": "Nausea",              "severity": "moderate", "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "Doxorubicin", "side_effect": "Mucositis",           "severity": "moderate", "frequency": "common",      "confidence_score": 0.95},
    {"drug": "Doxorubicin", "side_effect": "Neutropenia",         "severity": "severe",   "frequency": "common",      "confidence_score": 0.96},
    # Paclitaxel
    {"drug": "Paclitaxel", "side_effect": "Peripheral Neuropathy","severity": "moderate", "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "Paclitaxel", "side_effect": "Alopecia",             "severity": "severe",   "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "Paclitaxel", "side_effect": "Neutropenia",          "severity": "severe",   "frequency": "common",      "confidence_score": 0.96},
    {"drug": "Paclitaxel", "side_effect": "Fatigue",              "severity": "moderate", "frequency": "very_common", "confidence_score": 0.95},
    # Methotrexate
    {"drug": "Methotrexate", "side_effect": "Mucositis",          "severity": "severe",   "frequency": "very_common", "confidence_score": 0.98},
    {"drug": "Methotrexate", "side_effect": "Nausea",             "severity": "moderate", "frequency": "common",      "confidence_score": 0.95},
    {"drug": "Methotrexate", "side_effect": "Fatigue",            "severity": "moderate", "frequency": "common",      "confidence_score": 0.92},
    {"drug": "Methotrexate", "side_effect": "Appetite Loss",      "severity": "moderate", "frequency": "common",      "confidence_score": 0.90},
    # Cyclophosphamide
    {"drug": "Cyclophosphamide", "side_effect": "Nausea",         "severity": "moderate", "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "Cyclophosphamide", "side_effect": "Vomiting",       "severity": "moderate", "frequency": "common",      "confidence_score": 0.96},
    {"drug": "Cyclophosphamide", "side_effect": "Alopecia",       "severity": "moderate", "frequency": "common",      "confidence_score": 0.93},
    {"drug": "Cyclophosphamide", "side_effect": "Neutropenia",    "severity": "severe",   "frequency": "common",      "confidence_score": 0.97},
    {"drug": "Cyclophosphamide", "side_effect": "Immunosuppression","severity": "severe", "frequency": "very_common", "confidence_score": 0.98},
    # Vincristine
    {"drug": "Vincristine", "side_effect": "Peripheral Neuropathy","severity": "moderate","frequency": "very_common", "confidence_score": 0.98},
    {"drug": "Vincristine", "side_effect": "Constipation",        "severity": "moderate", "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "Vincristine", "side_effect": "Appetite Loss",       "severity": "mild",     "frequency": "common",      "confidence_score": 0.88},
    # 5-FU
    {"drug": "5-Fluorouracil", "side_effect": "Mucositis",        "severity": "severe",   "frequency": "very_common", "confidence_score": 0.98},
    {"drug": "5-Fluorouracil", "side_effect": "Diarrhea",         "severity": "severe",   "frequency": "very_common", "confidence_score": 0.97},
    {"drug": "5-Fluorouracil", "side_effect": "Taste Changes",    "severity": "moderate", "frequency": "common",      "confidence_score": 0.93},
    {"drug": "5-Fluorouracil", "side_effect": "Appetite Loss",    "severity": "moderate", "frequency": "common",      "confidence_score": 0.91},
    # Pembrolizumab
    {"drug": "Pembrolizumab", "side_effect": "Fatigue",           "severity": "moderate", "frequency": "very_common", "confidence_score": 0.95},
    {"drug": "Pembrolizumab", "side_effect": "Nausea",            "severity": "mild",     "frequency": "common",      "confidence_score": 0.90},
    {"drug": "Pembrolizumab", "side_effect": "Mucositis",         "severity": "mild",     "frequency": "uncommon",    "confidence_score": 0.82},
    # Gemcitabine
    {"drug": "Gemcitabine", "side_effect": "Nausea",              "severity": "moderate", "frequency": "common",      "confidence_score": 0.93},
    {"drug": "Gemcitabine", "side_effect": "Anemia",              "severity": "moderate", "frequency": "common",      "confidence_score": 0.92},
    {"drug": "Gemcitabine", "side_effect": "Fatigue",             "severity": "moderate", "frequency": "very_common", "confidence_score": 0.94},
    # Tamoxifen
    {"drug": "Tamoxifen", "side_effect": "Weight Gain",           "severity": "mild",     "frequency": "common",      "confidence_score": 0.88},
    {"drug": "Tamoxifen", "side_effect": "Taste Changes",         "severity": "mild",     "frequency": "uncommon",    "confidence_score": 0.80},
    # Ipilimumab
    {"drug": "Ipilimumab", "side_effect": "Diarrhea",             "severity": "severe",   "frequency": "common",      "confidence_score": 0.96},
    {"drug": "Ipilimumab", "side_effect": "Fatigue",              "severity": "moderate", "frequency": "very_common", "confidence_score": 0.94},
    # Carboplatin
    {"drug": "Carboplatin", "side_effect": "Nausea",              "severity": "moderate", "frequency": "very_common", "confidence_score": 0.95},
    {"drug": "Carboplatin", "side_effect": "Thrombocytopenia",    "severity": "severe",   "frequency": "common",      "confidence_score": 0.96},
    {"drug": "Carboplatin", "side_effect": "Anemia",              "severity": "moderate", "frequency": "common",      "confidence_score": 0.94},
    # Ifosfamide
    {"drug": "Ifosfamide", "side_effect": "Nausea",               "severity": "moderate", "frequency": "common",      "confidence_score": 0.94},
    {"drug": "Ifosfamide", "side_effect": "Nephrotoxicity",       "severity": "moderate", "frequency": "uncommon",    "confidence_score": 0.90},
    {"drug": "Ifosfamide", "side_effect": "Neutropenia",          "severity": "severe",   "frequency": "common",      "confidence_score": 0.95},
]

# SIDE EFFECT → LEADS_TO → SYMPTOM
LEADS_TO_RELS: list[dict] = [
    {"side_effect": "Nausea",               "symptom": "Abdominal Cramps",      "confidence_score": 0.88},
    {"side_effect": "Diarrhea",             "symptom": "Abdominal Cramps",      "confidence_score": 0.92},
    {"side_effect": "Diarrhea",             "symptom": "Bloating",              "confidence_score": 0.87},
    {"side_effect": "Mucositis",            "symptom": "Mouth Sores",           "confidence_score": 0.98},
    {"side_effect": "Mucositis",            "symptom": "Burning Sensation",     "confidence_score": 0.95},
    {"side_effect": "Esophagitis",          "symptom": "Difficulty Swallowing", "confidence_score": 0.97},
    {"side_effect": "Esophagitis",          "symptom": "Burning Sensation",     "confidence_score": 0.96},
    {"side_effect": "Taste Changes",        "symptom": "Metallic Taste",        "confidence_score": 0.94},
    {"side_effect": "Peripheral Neuropathy","symptom": "Tingling Hands/Feet",   "confidence_score": 0.97},
    {"side_effect": "Anemia",              "symptom": "Shortness of Breath",   "confidence_score": 0.90},
    {"side_effect": "Neutropenia",         "symptom": "Infection Risk",        "confidence_score": 0.97},
    {"side_effect": "Alopecia",            "symptom": "Hair Thinning",         "confidence_score": 0.99},
    {"side_effect": "Constipation",        "symptom": "Bloating",              "confidence_score": 0.93},
]

# FOOD HELPS SIDE EFFECT
FOOD_HELPS_RELS: list[dict] = [
    # Nausea management (from NCI PDF)
    {"food": "Banana",          "side_effect": "Nausea",        "severity": "mild",     "frequency": "often",     "confidence_score": 0.93, "mechanism": "bland easy-to-digest BRAT food"},
    {"food": "White Rice",      "side_effect": "Nausea",        "severity": "mild",     "frequency": "often",     "confidence_score": 0.92, "mechanism": "low-fiber easily digestible"},
    {"food": "Toast",           "side_effect": "Nausea",        "severity": "mild",     "frequency": "often",     "confidence_score": 0.91, "mechanism": "bland starchy food"},
    {"food": "Crackers",        "side_effect": "Nausea",        "severity": "mild",     "frequency": "often",     "confidence_score": 0.90, "mechanism": "bland carbohydrate buffer"},
    {"food": "Ginger Tea",      "side_effect": "Nausea",        "severity": "moderate", "frequency": "often",     "confidence_score": 0.88, "mechanism": "gingerol anti-nausea effect"},
    {"food": "Applesauce",      "side_effect": "Nausea",        "severity": "mild",     "frequency": "often",     "confidence_score": 0.89, "mechanism": "soft, low-fiber, easy to digest"},
    # Diarrhea management
    {"food": "White Rice",      "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.92, "mechanism": "low-fiber binding effect"},
    {"food": "Banana",          "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "potassium replacement, pectin"},
    {"food": "Sports Drink",    "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "electrolyte replacement"},
    {"food": "Broth",           "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.89, "mechanism": "sodium and fluid replacement"},
    {"food": "Bouillon",        "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.88, "mechanism": "electrolyte rich clear liquid"},
    {"food": "Yogurt",          "side_effect": "Diarrhea",      "severity": "mild",     "frequency": "sometimes", "confidence_score": 0.82, "mechanism": "probiotic colonization support"},
    # Constipation management
    {"food": "Prune Juice",     "side_effect": "Constipation",  "severity": "moderate", "frequency": "often",     "confidence_score": 0.94, "mechanism": "sorbitol and fiber stimulate bowel"},
    {"food": "Oatmeal",         "side_effect": "Constipation",  "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "soluble fiber increases stool bulk"},
    {"food": "Whole Grain Bread","side_effect":"Constipation",  "severity": "mild",     "frequency": "often",     "confidence_score": 0.89, "mechanism": "insoluble fiber bowel stimulant"},
    # Mucositis management
    {"food": "Ice Chips",       "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "cold numbing reduces oral pain"},
    {"food": "Popsicle",        "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "cold soothing and hydration"},
    {"food": "Custard",         "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.89, "mechanism": "soft high-calorie easy to swallow"},
    {"food": "Scrambled Eggs",  "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.88, "mechanism": "soft, moist, high-protein"},
    {"food": "Milkshake",       "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "liquid high-calorie nutrition"},
    # Appetite loss
    {"food": "Protein Shake",   "side_effect": "Appetite Loss", "severity": "moderate", "frequency": "often",     "confidence_score": 0.94, "mechanism": "calorie-dense liquid nutrition"},
    {"food": "Milkshake",       "side_effect": "Appetite Loss", "severity": "moderate", "frequency": "often",     "confidence_score": 0.93, "mechanism": "high-calorie palatable beverage"},
    {"food": "Peanut Butter",   "side_effect": "Appetite Loss", "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "calorie-dense small-volume food"},
    {"food": "Avocado",         "side_effect": "Appetite Loss", "severity": "moderate", "frequency": "often",     "confidence_score": 0.89, "mechanism": "calorie-dense healthy-fat food"},
    {"food": "Overnight Oats",  "side_effect": "Weight Loss",   "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "high-calorie high-protein meal"},
    # Dry mouth
    {"food": "Lemon Water",     "side_effect": "Dry Mouth",     "severity": "mild",     "frequency": "often",     "confidence_score": 0.88, "mechanism": "tart flavor stimulates salivary glands"},
    {"food": "Ice Chips",       "side_effect": "Dry Mouth",     "severity": "mild",     "frequency": "often",     "confidence_score": 0.87, "mechanism": "moisture and cooling for dry mouth"},
    {"food": "Broth",           "side_effect": "Dry Mouth",     "severity": "mild",     "frequency": "often",     "confidence_score": 0.85, "mechanism": "liquid moistening oral cavity"},
]

# FOOD WORSENS SIDE EFFECT
FOOD_WORSENS_RELS: list[dict] = [
    {"food": "Spicy Foods",    "side_effect": "Mucositis",     "severity": "severe",   "frequency": "often",     "confidence_score": 0.96, "mechanism": "capsaicin irritates inflamed mucosa"},
    {"food": "Citrus Juice",   "side_effect": "Mucositis",     "severity": "severe",   "frequency": "often",     "confidence_score": 0.95, "mechanism": "acid erodes damaged oral lining"},
    {"food": "Alcohol",        "side_effect": "Mucositis",     "severity": "severe",   "frequency": "often",     "confidence_score": 0.97, "mechanism": "ethanol desiccates and inflames mucosa"},
    {"food": "Raw Vegetables", "side_effect": "Mucositis",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "rough texture abrades sore mucosa"},
    {"food": "Fried Foods",    "side_effect": "Nausea",        "severity": "moderate", "frequency": "often",     "confidence_score": 0.93, "mechanism": "high fat delays gastric emptying"},
    {"food": "Alcohol",        "side_effect": "Nausea",        "severity": "moderate", "frequency": "often",     "confidence_score": 0.92, "mechanism": "direct gastric mucosal irritant"},
    {"food": "Spicy Foods",    "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.91, "mechanism": "capsaicin increases GI motility"},
    {"food": "Dairy (full-fat)","side_effect": "Diarrhea",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "lactase deficiency post-chemo"},
    {"food": "Fried Foods",    "side_effect": "Diarrhea",      "severity": "moderate", "frequency": "often",     "confidence_score": 0.89, "mechanism": "high fat increases osmotic load"},
    {"food": "Alcohol",        "side_effect": "Dry Mouth",     "severity": "moderate", "frequency": "often",     "confidence_score": 0.90, "mechanism": "diuretic effect worsens xerostomia"},
    {"food": "Citrus Juice",   "side_effect": "Esophagitis",   "severity": "severe",   "frequency": "often",     "confidence_score": 0.94, "mechanism": "acid reflux aggravates esophageal lining"},
    {"food": "Spicy Foods",    "side_effect": "Esophagitis",   "severity": "severe",   "frequency": "often",     "confidence_score": 0.93, "mechanism": "mucosal irritation in inflamed esophagus"},
]

# DRUG INTERACTS_WITH NON-CANCER DRUG
DRUG_INTERACTION_RELS: list[dict] = [
    {"drug": "Methotrexate",    "ncdrug": "Aspirin",       "severity": "severe",   "effect": "Reduced methotrexate clearance causing toxicity",    "confidence_score": 0.97},
    {"drug": "Methotrexate",    "ncdrug": "Ibuprofen",     "severity": "severe",   "effect": "NSAID inhibits renal MTX excretion",                 "confidence_score": 0.97},
    {"drug": "Methotrexate",    "ncdrug": "Ciprofloxacin", "severity": "moderate", "effect": "Antibiotic raises MTX serum levels",                 "confidence_score": 0.91},
    {"drug": "Cisplatin",       "ncdrug": "Aspirin",       "severity": "moderate", "effect": "Additive nephrotoxicity risk",                       "confidence_score": 0.89},
    {"drug": "Cisplatin",       "ncdrug": "Ciprofloxacin", "severity": "moderate", "effect": "Additive ototoxicity risk",                          "confidence_score": 0.88},
    {"drug": "Doxorubicin",     "ncdrug": "Warfarin",      "severity": "severe",   "effect": "Enhanced anticoagulation; bleeding risk",            "confidence_score": 0.94},
    {"drug": "Paclitaxel",      "ncdrug": "Fluconazole",   "severity": "moderate", "effect": "CYP3A4 inhibition increases paclitaxel exposure",    "confidence_score": 0.92},
    {"drug": "Tamoxifen",       "ncdrug": "Warfarin",      "severity": "severe",   "effect": "Tamoxifen potentiates warfarin anticoagulation",     "confidence_score": 0.96},
    {"drug": "Tamoxifen",       "ncdrug": "Fluconazole",   "severity": "moderate", "effect": "CYP2D6 inhibition reduces tamoxifen efficacy",       "confidence_score": 0.90},
    {"drug": "Erlotinib",       "ncdrug": "Omeprazole",    "severity": "moderate", "effect": "PPI reduces erlotinib absorption (pH dependent)",    "confidence_score": 0.91},
    {"drug": "Vemurafenib",     "ncdrug": "Warfarin",      "severity": "severe",   "effect": "CYP2C9 inhibition markedly elevates INR",            "confidence_score": 0.95},
    {"drug": "Cyclophosphamide","ncdrug": "Phenytoin",     "severity": "moderate", "effect": "Phenytoin reduces cyclophosphamide efficacy",        "confidence_score": 0.88},
    {"drug": "Vincristine",     "ncdrug": "Fluconazole",   "severity": "moderate", "effect": "Azole inhibits CYP3A4 increasing vincristine toxicity","confidence_score": 0.90},
    {"drug": "Carboplatin",     "ncdrug": "Metformin",     "severity": "mild",     "effect": "Potential additive nephrotoxicity monitoring needed","confidence_score": 0.78},
    {"drug": "Pembrolizumab",   "ncdrug": "Dexamethasone", "severity": "moderate", "effect": "Corticosteroids may attenuate immunotherapy response","confidence_score": 0.87},
    {"drug": "Ipilimumab",      "ncdrug": "Dexamethasone", "severity": "moderate", "effect": "Steroids reduce checkpoint inhibitor efficacy",      "confidence_score": 0.87},
    {"drug": "5-Fluorouracil",  "ncdrug": "Warfarin",      "severity": "severe",   "effect": "5-FU inhibits CYP2C9 potentiating warfarin",        "confidence_score": 0.95},
    {"drug": "Ondansetron",     "ncdrug": "Dexamethasone", "severity": "mild",     "effect": "Combined antiemetic regimen, additive benefit",      "confidence_score": 0.85},
]

# CANCER ASSOCIATED_WITH SIDE EFFECT (without chemo)
CANCER_SIDE_EFFECT_RELS: list[dict] = [
    {"cancer": "Lung Cancer",    "side_effect": "Appetite Loss",   "severity": "moderate", "frequency": "common",  "confidence_score": 0.89},
    {"cancer": "Lung Cancer",    "side_effect": "Weight Loss",     "severity": "moderate", "frequency": "common",  "confidence_score": 0.91},
    {"cancer": "Acute Leukemia", "side_effect": "Anemia",          "severity": "severe",   "frequency": "very_common","confidence_score": 0.97},
    {"cancer": "Acute Leukemia", "side_effect": "Neutropenia",     "severity": "severe",   "frequency": "very_common","confidence_score": 0.97},
    {"cancer": "Breast Cancer",  "side_effect": "Fatigue",         "severity": "moderate", "frequency": "common",  "confidence_score": 0.90},
    {"cancer": "Osteosarcoma",   "side_effect": "Weight Loss",     "severity": "moderate", "frequency": "common",  "confidence_score": 0.86},
    {"cancer": "Melanoma",       "side_effect": "Fatigue",         "severity": "moderate", "frequency": "common",  "confidence_score": 0.85},
]


# ══════════════════════════════════════════════════════════════════
# LOADER FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _upsert_nodes(client, label: str, rows: list[dict], key: str = "name") -> None:
    if not rows:
        return
    cypher = f"""
    UNWIND $rows AS row
    MERGE (n:{label} {{{key}: row.{key}}})
    SET n += row
    """
    client.run_write_batch(cypher, rows)
    log.info("Upserted %d %s nodes", len(rows), label)


def load_cancer_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "Cancer", CANCERS)


def load_drug_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "Drug", DRUGS)


def load_non_cancer_drug_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "NonCancerDrug", NON_CANCER_DRUGS)


def load_side_effect_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "SideEffect", SIDE_EFFECTS)


def load_food_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "Food", FOODS)


def load_symptom_nodes(client=None) -> None:
    client = client or get_client()
    _upsert_nodes(client, "Symptom", SYMPTOMS)


def load_treated_with(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (c:Cancer {name: row.cancer})
    MATCH (d:Drug   {name: row.drug})
    MERGE (c)-[r:TREATED_WITH]->(d)
    SET r.line             = row.line,
        r.protocol         = row.protocol,
        r.confidence_score = row.confidence_score
    """
    client.run_write_batch(cypher, TREATED_WITH_RELS)
    log.info("Loaded %d TREATED_WITH relationships", len(TREATED_WITH_RELS))


def load_causes(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (d:Drug       {name: row.drug})
    MATCH (s:SideEffect {name: row.side_effect})
    MERGE (d)-[r:CAUSES]->(s)
    SET r.severity         = row.severity,
        r.frequency        = row.frequency,
        r.confidence_score = row.confidence_score
    """
    client.run_write_batch(cypher, CAUSES_RELS)
    log.info("Loaded %d CAUSES relationships", len(CAUSES_RELS))


def load_leads_to(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (se:SideEffect {name: row.side_effect})
    MATCH (sy:Symptom    {name: row.symptom})
    MERGE (se)-[r:LEADS_TO]->(sy)
    SET r.confidence_score = row.confidence_score
    """
    client.run_write_batch(cypher, LEADS_TO_RELS)
    log.info("Loaded %d LEADS_TO relationships", len(LEADS_TO_RELS))


def load_food_helps(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (f:Food       {name: row.food})
    MATCH (s:SideEffect {name: row.side_effect})
    MERGE (f)-[r:HELPS]->(s)
    SET r.severity         = row.severity,
        r.frequency        = row.frequency,
        r.confidence_score = row.confidence_score,
        r.mechanism        = row.mechanism
    """
    client.run_write_batch(cypher, FOOD_HELPS_RELS)
    log.info("Loaded %d HELPS relationships", len(FOOD_HELPS_RELS))


def load_food_worsens(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (f:Food       {name: row.food})
    MATCH (s:SideEffect {name: row.side_effect})
    MERGE (f)-[r:WORSENS]->(s)
    SET r.severity         = row.severity,
        r.frequency        = row.frequency,
        r.confidence_score = row.confidence_score,
        r.mechanism        = row.mechanism
    """
    client.run_write_batch(cypher, FOOD_WORSENS_RELS)
    log.info("Loaded %d WORSENS relationships", len(FOOD_WORSENS_RELS))


def load_drug_interactions(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (d:Drug         {name: row.drug})
    MATCH (n:NonCancerDrug{name: row.ncdrug})
    MERGE (d)-[r:INTERACTS_WITH]->(n)
    SET r.severity         = row.severity,
        r.effect           = row.effect,
        r.confidence_score = row.confidence_score
    """
    client.run_write_batch(cypher, DRUG_INTERACTION_RELS)
    log.info("Loaded %d INTERACTS_WITH relationships", len(DRUG_INTERACTION_RELS))


def load_cancer_side_effects(client=None) -> None:
    client = client or get_client()
    cypher = """
    UNWIND $rows AS row
    MATCH (c:Cancer     {name: row.cancer})
    MATCH (s:SideEffect {name: row.side_effect})
    MERGE (c)-[r:ASSOCIATED_WITH]->(s)
    SET r.severity         = row.severity,
        r.frequency        = row.frequency,
        r.confidence_score = row.confidence_score
    """
    client.run_write_batch(cypher, CANCER_SIDE_EFFECT_RELS)
    log.info("Loaded %d ASSOCIATED_WITH relationships", len(CANCER_SIDE_EFFECT_RELS))


def load_all(client=None) -> None:
    client = client or get_client()
    log.info("=== Loading all graph data ===")

    # Nodes first
    load_cancer_nodes(client)
    load_drug_nodes(client)
    load_non_cancer_drug_nodes(client)
    load_side_effect_nodes(client)
    load_food_nodes(client)
    load_symptom_nodes(client)

    # Relationships
    load_treated_with(client)
    load_causes(client)
    load_leads_to(client)
    load_food_helps(client)
    load_food_worsens(client)
    load_drug_interactions(client)
    load_cancer_side_effects(client)

    log.info("=== Graph data load complete ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_all()
