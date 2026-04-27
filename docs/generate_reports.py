"""
Synthetic Radiology Report Generator — Indian Bone Cancer / Osteosarcoma
For Multimodal Graph RAG Pipeline

Clinical grounding:
- Demographics: ICMR-NCRP Cancer Statistics India 2020 (bone cancer epidemiology)
- Staging: AJCC 8th Edition + Enneking/MSTS system
- Imaging features: Indian Journal of Orthopaedics (Kundu 2014), AJR Bone RADS, PMC 2025 review
- Indian name corpus: North/South/East/West regions for diversity
"""

import csv
import json
import os
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# ─── Indian Name Corpus (regional diversity) ───────────────────────────────
MALE_NAMES = [
    "Aarav", "Arjun", "Rohan", "Karan", "Vikram", "Suresh", "Ramesh", "Anil",
    "Prateek", "Nikhil", "Siddharth", "Akash", "Deepak", "Manish", "Rahul",
    "Vivek", "Aditya", "Hardik", "Vishal", "Jayesh", "Suraj", "Dinesh",
    "Mahesh", "Ravi", "Gaurav", "Amit", "Pranav", "Harsh", "Kartik", "Kunal",
    "Vignesh", "Senthil", "Murugan", "Prashanth", "Naveen", "Rajesh", "Sankar",
    "Mithun", "Bhaskar", "Gowtham", "Biplab", "Sourav", "Debashish", "Arnab",
    "Saurav", "Parth", "Yash", "Dev", "Ishaan", "Tanmay"
]

FEMALE_NAMES = [
    "Priya", "Ananya", "Sneha", "Divya", "Pooja", "Kavya", "Meera", "Nisha",
    "Shalini", "Rekha", "Sunita", "Geeta", "Lalita", "Anita", "Usha",
    "Swati", "Neha", "Ritu", "Pallavi", "Shweta", "Archana", "Deepika",
    "Lakshmi", "Saraswati", "Bhavani", "Kamala", "Revathi", "Nirmala",
    "Padmavathi", "Geetha", "Moumita", "Arpita", "Puja", "Susmita",
    "Debarati", "Riya", "Trisha", "Isha", "Nandini", "Shreya",
    "Zara", "Fatima", "Ayesha", "Nasreen", "Rukhsar", "Sofia"
]

SURNAMES = {
    "North": ["Sharma", "Gupta", "Singh", "Verma", "Mishra", "Yadav", "Pandey",
              "Srivastava", "Tiwari", "Joshi", "Agarwal", "Saxena", "Khanna"],
    "South": ["Iyer", "Iyengar", "Pillai", "Nair", "Menon", "Reddy", "Rao",
              "Krishnamurthy", "Venkataraman", "Subramanian", "Narayanan",
              "Balakrishnan", "Sundaram", "Raghunathan"],
    "East":  ["Banerjee", "Chatterjee", "Mukherjee", "Ghosh", "Bose", "Das",
              "Roy", "Sen", "Chakraborty", "Sarkar", "Dutta", "Biswas", "Mandal"],
    "West":  ["Patel", "Shah", "Mehta", "Desai", "Joshi", "Parekh", "Thakkar",
              "Bhatt", "Chauhan", "Solanki", "Modi", "Amin", "Kapoor"],
    "Mixed": ["Khan", "Sheikh", "Siddiqui", "Ansari", "Qureshi", "Malik"]
}

CITIES = {
    "North": ["Delhi", "Lucknow", "Kanpur", "Agra", "Varanasi", "Allahabad",
              "Meerut", "Ghaziabad", "Noida", "Chandigarh", "Jaipur", "Jodhpur"],
    "South": ["Chennai", "Bangalore", "Hyderabad", "Coimbatore", "Kochi",
              "Mysore", "Vizag", "Thiruvananthapuram", "Madurai", "Vellore"],
    "East":  ["Kolkata", "Bhubaneswar", "Patna", "Ranchi", "Guwahati",
              "Siliguri", "Durgapur", "Asansol"],
    "West":  ["Mumbai", "Pune", "Ahmedabad", "Surat", "Nagpur", "Nashik",
              "Vadodara", "Rajkot", "Aurangabad"],
}

HOSPITALS = {
    "North": ["AIIMS New Delhi", "PGI Chandigarh", "Safdarjung Hospital Delhi",
              "Fortis Hospital Noida", "BLK Super Speciality Hospital Delhi",
              "SMS Medical College Jaipur"],
    "South": ["CMC Vellore", "AIIMS Mangalagiri", "Kidwai Memorial Cancer Centre Bangalore",
              "Apollo Hospital Chennai", "Nizam's Institute Hyderabad",
              "JIPMER Puducherry"],
    "East":  ["AIIMS Bhubaneswar", "Tata Medical Centre Kolkata",
              "Regional Cancer Centre Guwahati", "IPGMER Kolkata",
              "Rajendra Institute Ranchi"],
    "West":  ["Tata Memorial Hospital Mumbai", "AIIMS Nagpur", "HCG Cancer Centre Ahmedabad",
              "Kokilaben Dhirubhai Ambani Hospital Mumbai", "Ruby Hall Pune"],
}

# ─── Clinical Parameters ────────────────────────────────────────────────────
CANCER_TYPES = [
    ("Conventional (Central) Osteosarcoma", 0.75),
    ("Parosteal Osteosarcoma", 0.08),
    ("Periosteal Osteosarcoma", 0.05),
    ("Telangiectatic Osteosarcoma", 0.07),
    ("Small Cell Osteosarcoma", 0.03),
    ("High-Grade Surface Osteosarcoma", 0.02),
]

BONE_SITES = [
    ("Distal Femur", "Right", 0.30),
    ("Distal Femur", "Left",  0.22),
    ("Proximal Tibia", "Right", 0.14),
    ("Proximal Tibia", "Left",  0.11),
    ("Proximal Humerus", "Right", 0.08),
    ("Proximal Humerus", "Left",  0.05),
    ("Fibula", "Right", 0.03),
    ("Fibula", "Left",  0.02),
    ("Pelvis", "Right", 0.02),
    ("Ilium", "Left",   0.01),
    ("Proximal Femur", "Right", 0.01),
    ("Spine", "Central", 0.01),
]

# AJCC 8th Ed. + Enneking staging
STAGE_PROFILES = {
    "IA":  {"ajcc": "IA",  "enneking": "IA",  "T": "T1", "N": "N0", "M": "M0", "G": "G1", "tumor_size_cm": (2, 8),   "weight": 0.10},
    "IB":  {"ajcc": "IB",  "enneking": "IB",  "T": "T2", "N": "N0", "M": "M0", "G": "G1", "tumor_size_cm": (8, 15),  "weight": 0.08},
    "IIA": {"ajcc": "IIA", "enneking": "IIA", "T": "T1", "N": "N0", "M": "M0", "G": "G2", "tumor_size_cm": (3, 8),   "weight": 0.18},
    "IIB": {"ajcc": "IIB", "enneking": "IIB", "T": "T2", "N": "N0", "M": "M0", "G": "G3", "tumor_size_cm": (8, 18),  "weight": 0.35},
    "III": {"ajcc": "III", "enneking": "III", "T": "T3", "N": "N0", "M": "M0", "G": "G2", "tumor_size_cm": (5, 20),  "weight": 0.12},
    "IVA": {"ajcc": "IVA", "enneking": "III", "T": "T2", "N": "N0", "M": "M1a","G": "G3", "tumor_size_cm": (10, 22), "weight": 0.12},
    "IVB": {"ajcc": "IVB", "enneking": "III", "T": "T2", "N": "N1", "M": "M1b","G": "G3", "tumor_size_cm": (12, 25), "weight": 0.05},
}

TUMOR_SUBTYPES_XRAY = {
    "Conventional (Central) Osteosarcoma": [
        "mixed osteolytic and osteosclerotic lesion",
        "aggressive periosteal reaction with Codman's triangle",
        "sunburst pattern of periosteal new bone formation",
        "permeative bone destruction with cortical breach",
        "ill-defined sclerotic lesion in metaphysis with soft tissue mass"
    ],
    "Telangiectatic Osteosarcoma": [
        "predominantly lytic, expansile lesion mimicking aneurysmal bone cyst",
        "geographic bone destruction with thin sclerotic rim",
        "pathological fracture with fluid-fluid levels on MRI correlation",
        "osteolytic lesion without visible osteoid matrix"
    ],
    "Parosteal Osteosarcoma": [
        "dense lobular ossified mass attached to posterior cortex",
        "heavily mineralized juxtacortical mass encircling the bone",
        "well-defined surface lesion with broad attachment to cortex"
    ],
    "Periosteal Osteosarcoma": [
        "saucer-shaped cortical erosion with spiculated periosteal reaction",
        "Codman's triangle at margins with underlying cortical thickening",
        "surface lesion with chondroid matrix calcification"
    ],
    "Small Cell Osteosarcoma": [
        "permeative lytic lesion mimicking Ewing sarcoma",
        "aggressive bone destruction with onion-skin periosteal reaction",
        "ill-defined lytic metaphyseal lesion with periosteal elevation"
    ],
    "High-Grade Surface Osteosarcoma": [
        "large surface-based ossifying mass with periosteal reaction",
        "cortical invasion with adjacent soft tissue mineralization",
        "aggressive periosteal pattern with cortical destruction"
    ],
}

LABS_TEMPLATES = {
    "normal": {"ALP_IUL": (40, 120),  "LDH_IUL": (140, 280)},
    "raised":  {"ALP_IUL": (180, 900), "LDH_IUL": (300, 900)},
}

# ─── Report Generation Helpers ─────────────────────────────────────────────

def weighted_choice(items):
    weights = [w for *_, w in items]
    choices = [item[:-1] for item in items]
    return random.choices(choices, weights=weights, k=1)[0]

def weighted_choice_dict(d):
    keys   = list(d.keys())
    weights= [d[k]["weight"] for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]

def pick_cancer_type():
    items = [(t, w) for t, w in CANCER_TYPES]
    return random.choices([t for t, _ in items], weights=[w for _, w in items], k=1)[0]

def random_date(start_year=2018, end_year=2024):
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

def format_date(dt):
    return dt.strftime("%d %B %Y")

def rand_size(lo, hi):
    return round(random.uniform(lo, hi), 1)

# ─── Radiology Report Generators ───────────────────────────────────────────

def generate_xray_report(patient, stage, tumor_cm, cancer_type, site, side):
    xray_features = random.choice(TUMOR_SUBTYPES_XRAY.get(cancer_type, TUMOR_SUBTYPES_XRAY["Conventional (Central) Osteosarcoma"]))
    skip = "No skip lesions identified in the visualised bone." if stage not in ["III","IVA","IVB"] else \
           "A skip lesion measuring ~2.1 cm is noted ~4 cm proximal to the main tumor in the same bone."
    chest_xr = (
        "No pulmonary nodules identified on chest radiograph." if stage not in ["IVA","IVB"] else
        "Multiple bilateral pulmonary nodules noted; CT thorax recommended for further characterisation."
    )
    cortical = "Cortical destruction is present" if stage in ["IIB","III","IVA","IVB"] else \
               "Cortical thinning noted without frank destruction"
    soft_tissue = (f"Associated soft tissue mass extending ~{rand_size(1,6)} cm beyond cortex noted"
                   if stage in ["IIB","III","IVA","IVB"] else "No significant soft tissue extension")

    return f"""RADIOLOGY REPORT — PLAIN RADIOGRAPH (X-Ray)
{'='*60}
Patient ID     : {patient['patient_id']}
Patient Name   : {patient['name']}
Age / Sex      : {patient['age']} Years / {patient['gender']}
Date of Study  : {patient['imaging_date_xray']}
Referring Unit : Orthopaedic Oncology, {patient['hospital']}
Radiologist    : Dr. {patient['radiologist']}
Study          : X-Ray {site} — AP and Lateral Views

CLINICAL INDICATION:
{patient['age']}-year-old {patient['gender'].lower()} presenting with progressive pain and
swelling over the {side.lower()} {site.lower()} for {random.randint(2,14)} weeks.
Rule out bone tumour / malignancy.

FINDINGS:

Primary Lesion:
  Location     : {side} {site}, metaphyseal region
  Size         : Approximately {tumor_cm} cm in greatest dimension (estimated on radiograph)
  Pattern      : {xray_features.capitalize()}
  Cortex       : {cortical}
  Soft Tissue  : {soft_tissue}
  Periosteum   : Aggressive periosteal reaction noted with {'Codman triangle formation' if 'Codman' in xray_features else 'periosteal elevation'}

Skip Lesions:
  {skip}

Chest Radiograph:
  {chest_xr}

Bone Density:
  Generalised bone density appears {'normal for age' if patient['age'] < 40 else 'mildly reduced, correlate clinically'}

Adjacent Joints:
  {'Joint space appears preserved; no obvious intra-articular extension on plain films.' if stage in ['IA','IB','IIA'] else 'Suspected intra-articular extension; MRI evaluation recommended.'}

IMPRESSION:
Aggressive bone lesion of the {side.lower()} {site.lower()} with features highly suspicious for
{cancer_type}. Findings are consistent with AJCC Stage {stage}.

RECOMMENDATION:
  1. Urgent MRI {site} with whole-bone sequences for local staging
  2. CT Chest/Abdomen/Pelvis for systemic staging
  3. Core needle biopsy under image guidance for histopathological confirmation
  4. Bone scan / PET-CT for metastatic workup
  5. Serum ALP, LDH levels

Report verified and authenticated by:
Dr. {patient['radiologist']}, MD Radiology
{patient['hospital']}
"""


def generate_mri_report(patient, stage, tumor_cm, cancer_type, site, side):
    joint_inv   = stage in ["IIB","III","IVA","IVB"]
    neurovasc   = stage in ["IIB","III","IVA","IVB"] and random.random() > 0.5
    skip_mri    = stage in ["III","IVA","IVB"] and random.random() > 0.4
    medullary_extent = rand_size(tumor_cm * 0.8, tumor_cm * 1.5)
    soft_tissue_ext  = rand_size(1.0, 7.0) if stage not in ["IA","IB"] else rand_size(0, 2.0)
    signal = random.choice([
        "heterogeneously hypointense on T1 and hyperintense on T2 with avid post-contrast enhancement",
        "T1 isointense to muscle, T2 heterogeneously hyperintense with areas of signal void (osteoid)",
        "predominantly T1 hypointense and T2 hyperintense with marked perilesional oedema",
        "mixed signal on T1 and T2 with fluid-fluid levels (telangiectatic areas)"
    ]) if "Telangiectatic" not in cancer_type else \
        "multiple fluid-fluid levels on T2 consistent with haemorrhagic contents; avid peripheral enhancement on contrast"

    edema_ext = rand_size(2.0, 8.0)

    return f"""RADIOLOGY REPORT — MRI (Magnetic Resonance Imaging)
{'='*60}
Patient ID     : {patient['patient_id']}
Patient Name   : {patient['name']}
Age / Sex      : {patient['age']} Years / {patient['gender']}
Date of Study  : {patient['imaging_date_mri']}
Referring Unit : Orthopaedic Oncology, {patient['hospital']}
Radiologist    : Dr. {patient['radiologist']}
Study          : MRI {site} with contrast — Coronal, Sagittal, Axial
                 Sequences: T1W, T2W, STIR, T1 Fat-Sat Post-Gd, DWI (b=0,800)
                 Field Strength: 1.5 Tesla

CLINICAL INDICATION:
Aggressive bone lesion of {side.lower()} {site.lower()} on plain radiograph.
MRI for local staging, assessment of soft tissue extension, neurovascular proximity,
and joint involvement pre-biopsy.

FINDINGS:

Primary Tumour:
  Location          : {side} {site}, arising from metaphysis
  Maximum Dimension : {tumor_cm} cm (craniocaudal) × {rand_size(tumor_cm*0.5, tumor_cm*0.9)} cm (AP) × {rand_size(tumor_cm*0.4, tumor_cm*0.8)} cm (transverse)
  Signal Character  : {signal}
  Medullary Extent  : {medullary_extent} cm of medullary canal involved
  ADC Value         : Restricted diffusion; ADC ~{round(random.uniform(0.4,0.9),2)} × 10⁻³ mm²/s (suggestive of high cellularity)

Cortical and Periosteal Assessment:
  {'Frank cortical destruction with periosteal elevation and Codman triangle at proximal and distal margins.' if stage in ['IIB','III','IVA','IVB'] else 'Cortical thinning with early periosteal elevation; no frank breakthrough.'}

Soft Tissue Component:
  Extraosseous soft tissue mass : {'Present, measuring approximately ' + str(soft_tissue_ext) + ' cm beyond cortex' if stage not in ['IA','IB'] else 'Minimal or absent; lesion predominantly intraosseous'}
  Perilesional oedema           : Extends approximately {edema_ext} cm in adjacent marrow

Neurovascular Structures:
  {'Tumour abuts and encases the popliteal vessels / neurovascular bundle; fat plane obliterated — high risk for vascular involvement.' if neurovasc else 'Fat planes around adjacent neurovascular bundle preserved; no encasement identified.'}

Joint Involvement:
  {'Intra-articular extension noted with joint effusion; tumour breaches the joint capsule.' if joint_inv else 'Joint space preserved; no intra-articular extension. Articular cartilage appears intact.'}

Skip Lesions (whole-bone coronal sequence):
  {'A skip metastasis of approximately ' + str(rand_size(1.5,3.5)) + ' cm identified ~' + str(random.randint(3,8)) + ' cm proximal to main mass within the same bone.' if skip_mri else 'No skip metastases identified on whole-bone coronal sequences.'}

Lymph Nodes:
  {'Enlarged regional lymph nodes (> 1 cm short axis) noted — suspicious for nodal involvement.' if stage == 'IVB' else 'No regional lymphadenopathy.'}

IMPRESSION:
Large aggressive intraosseous lesion of {side.lower()} {site.lower()} with features consistent
with {cancer_type}. Enneking Stage {'IIB (high grade, extra-compartmental)' if stage in ['IIB','III','IVA','IVB'] else 'IIA (high grade, intra-compartmental)'}.
AJCC Stage {stage} (T{2 if tumor_cm > 8 else 1}N{'1' if stage=='IVB' else '0'}M{'1b' if stage=='IVB' else '1a' if stage=='IVA' else '0'}).

RECOMMENDATION:
  1. CT guided core needle biopsy for histopathological and molecular confirmation
  2. CT Thorax for pulmonary metastasis assessment
  3. Bone scintigraphy or PET-CT for skeletal metastatic survey
  4. Multidisciplinary tumour board review
  5. Pre-operative planning: {'limb-salvage feasibility to be assessed; vascular surgeon input required' if neurovasc else 'limb-salvage surgery feasible if margins adequate after neo-adjuvant chemotherapy'}

Report verified and authenticated by:
Dr. {patient['radiologist']}, MD Radiology / Fellowship Musculoskeletal Imaging
{patient['hospital']}
"""


def generate_ct_report(patient, stage, tumor_cm, cancer_type, site, side):
    lung_mets   = stage in ["IVA","IVB"]
    bone_mets   = stage == "IVB"
    matrix_type = random.choice([
        "dense osteoid matrix mineralisation",
        "mixed lytic-sclerotic matrix with cloud-like mineralisation",
        "predominantly lytic with peripheral mineralisation",
        "aggressive sclerotic matrix with sunburst periosteal reaction",
    ])
    nodule_text = (
        f"Multiple bilateral pulmonary nodules, largest measuring {rand_size(0.6,2.5)} cm — consistent with pulmonary metastases."
        if lung_mets else
        f"No pulmonary nodules detected. {'No pleural effusion.' if random.random()>0.3 else 'Minimal pleural effusion noted bilaterally.'} Mediastinum unremarkable."
    )

    return f"""RADIOLOGY REPORT — CT SCAN (Computed Tomography)
{'='*60}
Patient ID     : {patient['patient_id']}
Patient Name   : {patient['name']}
Age / Sex      : {patient['age']} Years / {patient['gender']}
Date of Study  : {patient['imaging_date_ct']}
Referring Unit : Orthopaedic Oncology / Medical Oncology, {patient['hospital']}
Radiologist    : Dr. {patient['radiologist']}
Study          : CT {site} (Pre- and Post-contrast) + CT Thorax for staging
                 Slice Thickness: 1.25 mm, Kernel: Bone + Soft Tissue

CLINICAL INDICATION:
Known / suspected {cancer_type} of {side.lower()} {site.lower()}.
CT for local bone architecture, cortical assessment, matrix characterisation,
and staging (pulmonary metastases).

FINDINGS:

Local Assessment — CT {site}:
  Site            : {side} {site}, metaphyseal/diaphyseal junction
  Size            : {tumor_cm} cm in greatest dimension
  Bone Matrix     : {matrix_type.capitalize()}
  Cortical Status : {'Cortical destruction with soft tissue extension' if stage in ['IIB','III','IVA','IVB'] else 'Cortical thinning without frank destruction'}
  Periosteum      : {'Aggressive periosteal reaction; Codman triangle formation noted at tumour margins' if random.random()>0.3 else 'Layered periosteal new bone formation'}
  Soft Tissue Mass: {'Extraosseous soft tissue mass of approximately ' + str(rand_size(3,12)) + ' cm with heterogeneous enhancement' if stage in ['IIB','III','IVA','IVB'] else 'No significant extraosseous component'}
  
  Bone Mineral Density: {'Regional osteopenia around lesion' if patient['age'] > 50 else 'Bone density normal for age in non-affected skeleton'}

CT Thorax — Pulmonary Metastasis Assessment:
  {nodule_text}
  {'Bilateral hilar adenopathy noted.' if stage=='IVB' and random.random()>0.5 else ''}

CT Abdomen / Pelvis:
  {'Hepatic hypodense lesions noted — metastasis vs cysts; recommend MRI liver for characterisation.' if stage=='IVB' and random.random()>0.6 else 'No abdominal or pelvic visceral metastasis identified.'}
  {'Lytic lesions in pelvis / lumbar vertebrae — skeletal metastasis suspected.' if bone_mets else 'No osseous metastatic lesions in the imaged skeleton.'}

Regional Lymph Nodes:
  {'Enlarged inguinal/iliac lymph nodes (largest ' + str(rand_size(1.2,2.8)) + ' cm) — nodal metastasis suspected.' if stage=='IVB' else 'No pathological lymphadenopathy.'}

Incidental Findings:
  {random.choice([
    'Mild hepatic steatosis noted.',
    'No significant incidental findings.',
    'Small incidental renal cortical cyst; no intervention required.',
    'Mild mediastinal fatty infiltration; no adenopathy.',
  ])}

Hounsfield Unit Analysis (primary lesion):
  Unenhanced HU : {random.randint(80,350)} (osteoid/mineralised matrix)
  Enhanced HU   : {random.randint(150,500)} (post-contrast)
  Enhancement   : {'Marked heterogeneous enhancement (viable tumour + necrosis)' if stage in ['IIB','III','IVA','IVB'] else 'Moderate homogeneous enhancement'}

IMPRESSION:
CT confirms {cancer_type} of {side.lower()} {site.lower()} with {matrix_type}.
AJCC Stage {stage}.
{'Pulmonary metastases confirmed — Stage IVA disease.' if stage=='IVA' else ''}
{'Multi-site metastatic disease — Stage IVB.' if stage=='IVB' else ''}
{'No distant metastasis identified — locoregional disease.' if stage not in ['IVA','IVB'] else ''}

RECOMMENDATION:
  1. Correlation with MRI for detailed local staging and surgical planning
  2. Bone scan / whole-body PET-CT to complete metastatic survey
  3. MDT discussion with orthopaedic oncologist, medical oncologist, radiation oncologist
  4. Neo-adjuvant chemotherapy (MAP regimen) to be considered prior to surgery
  {'5. Pulmonary metastatectomy planning in consultation with thoracic surgery' if lung_mets else '5. Limb-salvage surgery planning post neo-adjuvant therapy'}

Report verified and authenticated by:
Dr. {patient['radiologist']}, MD Radiology (Fellowship: Body Oncologic Imaging)
{patient['hospital']}
"""

# ─── Main Generator ─────────────────────────────────────────────────────────

RADIOLOGISTS = [
    "Arun Krishnamurthy", "Priya Venkataraman", "Suresh Iyer",
    "Kavitha Reddy", "Rajesh Sharma", "Deepa Nair", "Sanjay Gupta",
    "Meena Pillai", "Vivek Chatterjee", "Anand Mishra",
    "Shobha Rao", "Kiran Patel", "Naveen Subramanian",
]

def generate_patient(idx):
    region = random.choice(["North", "South", "East", "West", "West", "North", "South"])
    gender = random.choice(["Male", "Female", "Male", "Male"])  # slight male preponderance (NCRP data)
    # Age distribution: osteosarcoma bimodal — peak 10-25yrs and secondary peak >60
    if random.random() < 0.70:
        age = random.randint(10, 25)
    elif random.random() < 0.80:
        age = random.randint(26, 40)
    else:
        age = random.randint(41, 72)

    name_pool  = MALE_NAMES if gender == "Male" else FEMALE_NAMES
    surname_pool = SURNAMES.get(region, SURNAMES["North"])
    if random.random() < 0.08:
        surname_pool = SURNAMES["Mixed"]  # Muslim names across regions

    first_name = random.choice(name_pool)
    last_name  = random.choice(surname_pool)
    name       = f"{first_name} {last_name}"

    city        = random.choice(CITIES.get(region, CITIES["North"]))
    hospital    = random.choice(HOSPITALS.get(region, HOSPITALS["North"]))
    radiologist = random.choice(RADIOLOGISTS)

    cancer_type = pick_cancer_type()
    stage_key   = weighted_choice_dict(STAGE_PROFILES)
    stage_info  = STAGE_PROFILES[stage_key]
    tumor_cm    = rand_size(*stage_info["tumor_size_cm"])

    site_tuple  = weighted_choice(BONE_SITES)
    bone_site, laterality = site_tuple

    base_date   = random_date()
    mri_date    = base_date + timedelta(days=random.randint(1, 5))
    ct_date     = base_date + timedelta(days=random.randint(2, 7))

    alp_range   = LABS_TEMPLATES["raised"]["ALP_IUL"] if stage_key in ["IIB","III","IVA","IVB"] else LABS_TEMPLATES["normal"]["ALP_IUL"]
    ldh_range   = LABS_TEMPLATES["raised"]["LDH_IUL"] if stage_key in ["IVA","IVB"] else LABS_TEMPLATES["normal"]["LDH_IUL"]

    patient = {
        "patient_id"         : f"PT-{str(uuid.uuid4().hex[:8]).upper()}",
        "name"               : name,
        "age"                : age,
        "gender"             : gender,
        "ethnicity"          : "Indian",
        "region"             : region,
        "city"               : city,
        "state"              : city,  # simplified
        "hospital"           : hospital,
        "radiologist"        : radiologist,
        "cancer_type"        : cancer_type,
        "cancer_category"    : "Primary Malignant Bone Tumour — Osteosarcoma",
        "icd_10_code"        : "C40.2" if "Femur" in bone_site or "Tibia" in bone_site or "Fibula" in bone_site else "C40.0" if "Humerus" in bone_site else "C41.4" if "Pelvis" in bone_site else "C41.2",
        "bone_site"          : bone_site,
        "laterality"         : laterality,
        "tumor_size_cm"      : tumor_cm,
        "ajcc_stage"         : stage_info["ajcc"],
        "enneking_stage"     : stage_info["enneking"],
        "T_stage"            : stage_info["T"],
        "N_stage"            : stage_info["N"],
        "M_stage"            : stage_info["M"],
        "grade"              : stage_info["G"],
        "metastasis_present" : "Yes" if stage_key in ["IVA","IVB"] else "No",
        "pulmonary_mets"     : "Yes" if stage_key in ["IVA","IVB"] else "No",
        "skip_lesions"       : "Yes" if stage_key in ["III","IVA","IVB"] and random.random()>0.4 else "No",
        "alp_iu_L"           : random.randint(*alp_range),
        "ldh_iu_L"           : random.randint(*ldh_range),
        "imaging_date_xray"  : format_date(base_date),
        "imaging_date_mri"   : format_date(mri_date),
        "imaging_date_ct"    : format_date(ct_date),
        "has_xray"           : "Yes",
        "has_mri"            : "Yes",
        "has_ct"             : "Yes",
        "treatment_plan"     : (
            "Neo-adjuvant MAP chemotherapy → Limb-salvage surgery → Adjuvant chemotherapy"
            if stage_key not in ["IVA","IVB"] else
            "Palliative chemotherapy + Pulmonary metastatectomy consideration + MDT review"
        ),
        "surgery_type"       : (
            "Limb-Salvage (Wide Excision + Endoprosthesis)"
            if stage_key in ["IA","IB","IIA"] else
            "Limb-Salvage (Wide Excision) or Amputation based on response"
            if stage_key in ["IIB","III"] else
            "Systemic therapy primary; surgical intent palliative"
        ),
        "dataset_split"      : "",  # assigned later
        "report_xray_file"   : f"{patient_id_placeholder}_xray_report.txt",
        "report_mri_file"    : f"{patient_id_placeholder}_mri_report.txt",
        "report_ct_file"     : f"{patient_id_placeholder}_ct_report.txt",
    }

    # Fix file references now that we have patient_id
    pid = patient["patient_id"]
    patient["report_xray_file"] = f"{pid}_xray_report.txt"
    patient["report_mri_file"]  = f"{pid}_mri_report.txt"
    patient["report_ct_file"]   = f"{pid}_ct_report.txt"

    return patient, stage_key, bone_site, laterality, cancer_type

# Sentinel for forward reference
patient_id_placeholder = "PLACEHOLDER"


def main():
    N = 120   # total patients
    TRAIN_SPLIT = 0.70
    VAL_SPLIT   = 0.15
    TEST_SPLIT  = 0.15

    out_dir      = "/mnt/user-data/outputs/osteosarcoma_dataset"
    reports_dir  = os.path.join(out_dir, "radiology_reports")
    os.makedirs(reports_dir, exist_ok=True)

    patients    = []
    all_reports = []

    for idx in range(N):
        patient, stage_key, bone_site, laterality, cancer_type = generate_patient(idx)

        # Assign split
        r = random.random()
        if r < TRAIN_SPLIT:
            split = "train"
        elif r < TRAIN_SPLIT + VAL_SPLIT:
            split = "validation"
        else:
            split = "test"
        patient["dataset_split"] = split

        # Generate reports
        xray_txt = generate_xray_report(patient, stage_key, patient["tumor_size_cm"], cancer_type, bone_site, laterality)
        mri_txt  = generate_mri_report(patient, stage_key, patient["tumor_size_cm"], cancer_type, bone_site, laterality)
        ct_txt   = generate_ct_report(patient, stage_key, patient["tumor_size_cm"], cancer_type, bone_site, laterality)

        # Save individual report files
        for modality, content, fname in [
            ("xray", xray_txt, patient["report_xray_file"]),
            ("mri",  mri_txt,  patient["report_mri_file"]),
            ("ct",   ct_txt,   patient["report_ct_file"]),
        ]:
            fpath = os.path.join(reports_dir, fname)
            with open(fpath, "w") as f:
                f.write(content)

            all_reports.append({
                "patient_id"    : patient["patient_id"],
                "modality"      : modality.upper(),
                "report_file"   : fname,
                "imaging_date"  : patient[f"imaging_date_{modality}"],
                "dataset_split" : split,
                "ajcc_stage"    : patient["ajcc_stage"],
                "cancer_type"   : patient["cancer_type"],
                "bone_site"     : patient["bone_site"],
                "laterality"    : patient["laterality"],
                "tumor_size_cm" : patient["tumor_size_cm"],
            })

        patients.append(patient)
        if (idx+1) % 20 == 0:
            print(f"  Generated {idx+1}/{N} patients...")

    # ─── Write CSVs ────────────────────────────────────────────────────────
    # patients.csv
    patients_csv = os.path.join(out_dir, "patients.csv")
    fieldnames   = list(patients[0].keys())
    with open(patients_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(patients)

    # reports_index.csv
    reports_csv = os.path.join(out_dir, "reports_index.csv")
    with open(reports_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_reports[0].keys()))
        writer.writeheader()
        writer.writerows(all_reports)

    # ─── Graph RAG Entity/Edge Export ──────────────────────────────────────
    nodes = []
    edges = []

    for p in patients:
        pid = p["patient_id"]

        # Node: Patient
        nodes.append({"id": pid, "type": "Patient", "label": p["name"],
                       "age": p["age"], "gender": p["gender"], "city": p["city"],
                       "ethnicity": p["ethnicity"], "split": p["dataset_split"]})

        # Node: Cancer
        cancer_id = f"CANCER-{pid}"
        nodes.append({"id": cancer_id, "type": "Diagnosis",
                       "cancer_type": p["cancer_type"], "icd10": p["icd_10_code"],
                       "ajcc_stage": p["ajcc_stage"], "enneking_stage": p["enneking_stage"],
                       "grade": p["grade"], "tumor_size_cm": p["tumor_size_cm"]})

        # Node: Bone Site
        site_id = f"SITE-{p['bone_site'].replace(' ','_')}-{p['laterality']}"
        nodes.append({"id": site_id, "type": "AnatomicalSite",
                       "site": p["bone_site"], "laterality": p["laterality"]})

        # Node: Hospital
        hosp_id = p["hospital"].replace(" ", "_")
        nodes.append({"id": hosp_id, "type": "Hospital", "name": p["hospital"],
                       "region": p["region"]})

        # Node: Radiology Reports
        for mod in ["xray","mri","ct"]:
            rep_id = f"REPORT-{pid}-{mod.upper()}"
            nodes.append({"id": rep_id, "type": "RadiologyReport",
                           "modality": mod.upper(),
                           "imaging_date": p[f"imaging_date_{mod}"],
                           "file": p[f"report_{mod}_file"],
                           "radiologist": p["radiologist"]})
            edges.append({"source": pid, "target": rep_id,     "relation": "HAS_REPORT"})
            edges.append({"source": rep_id, "target": cancer_id, "relation": "DIAGNOSES"})
            edges.append({"source": rep_id, "target": site_id,   "relation": "IMAGES_SITE"})

        # Edges
        edges.append({"source": pid, "target": cancer_id, "relation": "DIAGNOSED_WITH"})
        edges.append({"source": pid, "target": site_id,   "relation": "TUMOUR_LOCATED_AT"})
        edges.append({"source": pid, "target": hosp_id,   "relation": "TREATED_AT"})

    # Deduplicate nodes
    seen_ids = set()
    unique_nodes = []
    for n in nodes:
        if n["id"] not in seen_ids:
            seen_ids.add(n["id"])
            unique_nodes.append(n)

    graph_data = {"nodes": unique_nodes, "edges": edges}
    graph_path = os.path.join(out_dir, "graph_rag_entities.json")
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    # Summary stats
    splits = {"train":0, "validation":0, "test":0}
    for p in patients:
        splits[p["dataset_split"]] += 1

    print(f"\n{'='*55}")
    print(f"  DATASET GENERATION COMPLETE")
    print(f"{'='*55}")
    print(f"  Total Patients  : {N}")
    print(f"  Train           : {splits['train']}")
    print(f"  Validation      : {splits['validation']}")
    print(f"  Test            : {splits['test']}")
    print(f"  Radiology Reports: {len(all_reports)} ({N*3} total across 3 modalities)")
    print(f"\n  Output directory: {out_dir}")
    print(f"  Files:")
    print(f"    patients.csv              — {N} rows, {len(fieldnames)} columns")
    print(f"    reports_index.csv         — {len(all_reports)} report metadata rows")
    print(f"    graph_rag_entities.json   — {len(unique_nodes)} nodes, {len(edges)} edges")
    print(f"    radiology_reports/        — {N*3} .txt report files")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
