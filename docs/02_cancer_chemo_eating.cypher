// ============================================================
// 02_cancer_chemo_eating.cypher
// Core knowledge graph: Cancers → ChemoDrugs → EatingEffects
// Sources: uploaded review MDs + NCI Eating Hints PDF
// Data entry points: 5 Cancers, 28 ChemoDrugs, 12 EatingEffects,
//   18 FoodItems, 10 NutritionGuidelines, 7 Biomarkers,
//   6 TreatmentProtocols  = 86 nodes + ~120 relationships
// ============================================================

// ════════════════════════════════════════════════════════════
// SECTION 1 – CANCER NODES  (5 nodes)
// ════════════════════════════════════════════════════════════

MERGE (c1:Cancer {name: 'Lung Cancer'})
SET c1.subtype         = 'NSCLC / SCLC',
    c1.icd10           = 'C34',
    c1.incidence_rank  = 1,
    c1.description     = 'Most common cancer worldwide; NSCLC comprises ~85% of cases. ALK, EGFR, KRAS molecular targets guide targeted therapy.',
    c1.five_year_survival = 0.25,
    c1.staging_system  = 'TNM',
    c1.source_doc      = 'lung-cancer-review.md';

MERGE (c2:Cancer {name: 'Breast Cancer'})
SET c2.subtype         = 'HR+/HER2+/TNBC',
    c2.icd10           = 'C50',
    c2.incidence_rank  = 2,
    c2.description     = 'Most common cancer in women. Subtypes guided by hormone receptor (ER/PR) and HER2 status. Neoadjuvant and adjuvant chemo widely used.',
    c2.five_year_survival = 0.91,
    c2.staging_system  = 'TNM',
    c2.source_doc      = 'breast-cancer-review.md';

MERGE (c3:Cancer {name: 'Acute Leukemia'})
SET c3.subtype         = 'ALL / AML',
    c3.icd10           = 'C91.0 / C92.0',
    c3.incidence_rank  = 6,
    c3.description     = 'Malignant proliferation of lymphoid (ALL) or myeloid (AML) precursors. Multi-agent chemotherapy is backbone; Philadelphia chromosome status guides TKI use in ALL.',
    c3.five_year_survival = 0.65,
    c3.staging_system  = 'FAB / WHO',
    c3.source_doc      = 'acute-leukemia-review.md';

MERGE (c4:Cancer {name: 'Osteosarcoma'})
SET c4.subtype         = 'High-grade / Low-grade',
    c4.icd10           = 'C40',
    c4.incidence_rank  = 8,
    c4.description     = 'Primary malignant bone tumor most common in adolescents. Standard treatment is neoadjuvant MAP (Methotrexate, Adriamycin, Cisplatin) followed by limb-salvage surgery.',
    c4.five_year_survival = 0.70,
    c4.staging_system  = 'AJCC / Enneking',
    c4.source_doc      = 'osteosarcoma-review.md';

MERGE (c5:Cancer {name: 'Skin Cancer'})
SET c5.subtype         = 'Melanoma / BCC / SCC',
    c5.icd10           = 'C43 / C44',
    c5.incidence_rank  = 5,
    c5.description     = 'Includes melanoma (aggressive, BRAF/MEK targeted) and non-melanoma BCC/SCC. Immunotherapy (checkpoint inhibitors) has transformed advanced melanoma outcomes.',
    c5.five_year_survival = 0.93,
    c5.staging_system  = 'AJCC',
    c5.source_doc      = 'melanoma-skin-cancer-review.md';

// ════════════════════════════════════════════════════════════
// SECTION 2 – CHEMOTHERAPY DRUG NODES  (28 nodes)
// ════════════════════════════════════════════════════════════

// ── Lung Cancer drugs ────────────────────────────────────────
MERGE (d1:ChemoDrug {name: 'Cisplatin'})
SET d1.drug_class   = 'Platinum-based alkylating agent',
    d1.mechanism    = 'Cross-links DNA strands, inhibiting DNA replication and transcription',
    d1.route        = 'IV',
    d1.typical_dose = '75 mg/m² every 3 weeks',
    d1.notes        = 'First-line doublet partner for NSCLC; nephrotoxicity requires aggressive hydration';

MERGE (d2:ChemoDrug {name: 'Carboplatin'})
SET d2.drug_class   = 'Platinum-based alkylating agent',
    d2.mechanism    = 'DNA cross-linking causing strand breaks',
    d2.route        = 'IV',
    d2.typical_dose = 'AUC 5-6 every 3 weeks',
    d2.notes        = 'Less nephrotoxic than cisplatin; myelosuppression is dose-limiting';

MERGE (d3:ChemoDrug {name: 'Paclitaxel'})
SET d3.drug_class   = 'Taxane / microtubule stabilizer',
    d3.mechanism    = 'Stabilises beta-tubulin polymers, preventing mitotic spindle disassembly',
    d3.route        = 'IV',
    d3.typical_dose = '175 mg/m² every 3 weeks or 80 mg/m² weekly',
    d3.notes        = 'Peripheral neuropathy is cumulative; premedication with corticosteroids required';

MERGE (d4:ChemoDrug {name: 'Docetaxel'})
SET d4.drug_class   = 'Taxane / microtubule stabilizer',
    d4.mechanism    = 'Stabilises microtubules and promotes tubulin polymerisation',
    d4.route        = 'IV',
    d4.typical_dose = '75 mg/m² every 3 weeks',
    d4.notes        = 'Fluid retention, nail changes, and neutropenia are notable; premedicate with dexamethasone';

MERGE (d5:ChemoDrug {name: 'Gemcitabine'})
SET d5.drug_class   = 'Antimetabolite / pyrimidine analogue',
    d5.mechanism    = 'Inhibits ribonucleotide reductase and DNA polymerase; incorporated into DNA',
    d5.route        = 'IV',
    d5.typical_dose = '1000-1250 mg/m² days 1, 8 every 3 weeks',
    d5.notes        = 'Used in NSCLC, pancreatic, bladder, and breast cancers; flu-like symptoms common';

MERGE (d6:ChemoDrug {name: 'Pemetrexed'})
SET d6.drug_class   = 'Antifolate antimetabolite',
    d6.mechanism    = 'Multi-targeted folate pathway inhibitor; requires B12 and folic acid supplementation',
    d6.route        = 'IV',
    d6.typical_dose = '500 mg/m² every 3 weeks',
    d6.notes        = 'Preferred for non-squamous NSCLC; nutritional supplementation mandatory to reduce toxicity';

MERGE (d7:ChemoDrug {name: 'Vinorelbine'})
SET d7.drug_class   = 'Vinca alkaloid',
    d7.mechanism    = 'Inhibits tubulin polymerisation, disrupting mitotic spindle formation',
    d7.route        = 'IV or oral',
    d7.typical_dose = '25-30 mg/m² weekly',
    d7.notes        = 'Constipation and peripheral neuropathy are characteristic side effects';

// ── Breast Cancer drugs ───────────────────────────────────────
MERGE (d8:ChemoDrug {name: 'Doxorubicin'})
SET d8.drug_class   = 'Anthracycline / topoisomerase II inhibitor',
    d8.mechanism    = 'Intercalates DNA, inhibits topoisomerase II, generates free radicals',
    d8.route        = 'IV',
    d8.typical_dose = '60-75 mg/m² every 3 weeks (liposomal: 50 mg/m²)',
    d8.notes        = 'Cumulative cardiotoxicity limits lifetime dose; causes alopecia and stomatitis';

MERGE (d9:ChemoDrug {name: 'Cyclophosphamide'})
SET d9.drug_class   = 'Alkylating agent / nitrogen mustard',
    d9.mechanism    = 'Alkylates DNA, causing cross-links and strand breaks; prodrug activated by CYP450',
    d9.route        = 'IV or oral',
    d9.typical_dose = '500-600 mg/m² in AC or CMF regimens',
    d9.notes        = 'Haemorrhagic cystitis (mesna prophylaxis), nausea, and immunosuppression';

MERGE (d10:ChemoDrug {name: 'Capecitabine'})
SET d10.drug_class   = 'Antimetabolite / oral fluoropyrimidine',
    d10.mechanism    = 'Converts to 5-FU in tumour cells; inhibits thymidylate synthase',
    d10.route        = 'Oral',
    d10.typical_dose = '1250 mg/m² twice daily days 1-14 every 3 weeks',
    d10.notes        = 'Hand-foot syndrome (palmar-plantar erythrodysaesthesia) is dose-limiting; diarrhoea common';

MERGE (d11:ChemoDrug {name: 'Epirubicin'})
SET d11.drug_class   = 'Anthracycline',
    d11.mechanism    = 'Topoisomerase II inhibition; DNA intercalation',
    d11.route        = 'IV',
    d11.typical_dose = '100 mg/m² every 3 weeks in FEC regimen',
    d11.notes        = 'Used in FEC and FEC-T regimens for breast cancer; less cardiotoxic than doxorubicin';

MERGE (d12:ChemoDrug {name: 'Fluorouracil'})
SET d12.drug_class   = 'Antimetabolite / pyrimidine analogue',
    d12.mechanism    = 'Inhibits thymidylate synthase; incorporated into RNA/DNA',
    d12.route        = 'IV (bolus or infusion)',
    d12.typical_dose = '500 mg/m² bolus or 1000 mg/m²/day continuous infusion',
    d12.notes        = 'Mucositis, diarrhoea, hand-foot syndrome; DPD deficiency causes severe toxicity';

// ── Acute Leukemia drugs ─────────────────────────────────────
MERGE (d13:ChemoDrug {name: 'Cytarabine'})
SET d13.drug_class   = 'Antimetabolite / pyrimidine analogue',
    d13.mechanism    = 'Incorporated into DNA, inhibits DNA polymerase; S-phase specific',
    d13.route        = 'IV, IT, SC',
    d13.typical_dose = '100-200 mg/m² standard; 1-3 g/m² high-dose',
    d13.notes        = 'Cerebellar toxicity at high doses; conjunctivitis (use steroid eye drops)';

MERGE (d14:ChemoDrug {name: 'Daunorubicin'})
SET d14.drug_class   = 'Anthracycline',
    d14.mechanism    = 'DNA intercalation and topoisomerase II inhibition',
    d14.route        = 'IV',
    d14.typical_dose = '45-90 mg/m² for 3 days in 7+3 AML induction',
    d14.notes        = 'Cardiotoxicity monitoring required; mucositis and myelosuppression';

MERGE (d15:ChemoDrug {name: 'Vincristine'})
SET d15.drug_class   = 'Vinca alkaloid',
    d15.mechanism    = 'Binds tubulin dimers, prevents microtubule formation, arrests mitosis',
    d15.route        = 'IV',
    d15.typical_dose = '1.4 mg/m² (max 2 mg) weekly',
    d15.notes        = 'Peripheral and autonomic neuropathy, SIADH; NO intrathecal use (fatal)';

MERGE (d16:ChemoDrug {name: 'L-Asparaginase'})
SET d16.drug_class   = 'Enzyme / asparagine-depleting agent',
    d16.mechanism    = 'Degrades asparagine; leukaemic cells lack asparagine synthetase',
    d16.route        = 'IV or IM',
    d16.typical_dose = '6000 IU/m² three times weekly',
    d16.notes        = 'Pancreatitis, coagulopathy, hepatotoxicity; hypersensitivity – pegaspargase preferred in adults';

MERGE (d17:ChemoDrug {name: 'Mercaptopurine'})
SET d17.drug_class   = 'Antimetabolite / thiopurine',
    d17.mechanism    = 'Incorporated into DNA/RNA as false purine; inhibits de novo purine synthesis',
    d17.route        = 'Oral',
    d17.typical_dose = '50-75 mg/m² daily (maintenance)',
    d17.notes        = 'TPMT genotyping guides dosing; hepatotoxicity and myelosuppression; avoid milk (reduces absorption)';

MERGE (d18:ChemoDrug {name: 'Methotrexate'})
SET d18.drug_class   = 'Antifolate antimetabolite',
    d18.mechanism    = 'Inhibits dihydrofolate reductase (DHFR); blocks folate-dependent synthesis of nucleotides',
    d18.route        = 'IV, IM, IT, oral',
    d18.typical_dose = 'Variable: 15 mg/m² (maintenance) to 12 g/m² (HD-MTX osteosarcoma)',
    d18.notes        = 'Leucovorin rescue required at high doses; mucositis, nephrotoxicity, hepatotoxicity';

MERGE (d19:ChemoDrug {name: 'Idarubicin'})
SET d19.drug_class   = 'Anthracycline',
    d19.mechanism    = 'More lipophilic than daunorubicin; topoisomerase II inhibition and DNA intercalation',
    d19.route        = 'IV',
    d19.typical_dose = '12 mg/m² for 3 days',
    d19.notes        = 'Used in AML induction; severe myelosuppression; stomatitis and GI toxicity common';

// ── Osteosarcoma drugs ───────────────────────────────────────
MERGE (d20:ChemoDrug {name: 'Ifosfamide'})
SET d20.drug_class   = 'Alkylating agent / oxazaphosphorine',
    d20.mechanism    = 'DNA cross-linking; prodrug activated by CYP3A4',
    d20.route        = 'IV',
    d20.typical_dose = '1.8-2 g/m² for 5 days with mesna',
    d20.notes        = 'Haemorrhagic cystitis (mesna mandatory), encephalopathy, nephrotoxicity (Fanconi syndrome)';

MERGE (d21:ChemoDrug {name: 'Etoposide'})
SET d21.drug_class   = 'Topoisomerase II inhibitor / podophyllotoxin',
    d21.mechanism    = 'Stabilises topo II-DNA cleavable complex; secondary AML risk',
    d21.route        = 'IV or oral',
    d21.typical_dose = '100 mg/m² for 3-5 days',
    d21.notes        = 'Myelosuppression is dose-limiting; alopecia; risk of treatment-related leukaemia';

// ── Skin Cancer (Melanoma) drugs ──────────────────────────────
MERGE (d22:ChemoDrug {name: 'Dacarbazine'})
SET d22.drug_class   = 'Alkylating agent / triazene',
    d22.mechanism    = 'Methylation of DNA guanine residues via active metabolite MTIC',
    d22.route        = 'IV',
    d22.typical_dose = '850-1000 mg/m² every 3 weeks',
    d22.notes        = 'Historic first-line melanoma drug; severe nausea; modest single-agent response rates';

MERGE (d23:ChemoDrug {name: 'Temozolomide'})
SET d23.drug_class   = 'Alkylating agent / imidazotetrazine',
    d23.mechanism    = 'Prodrug of MTIC; O6-methylguanine DNA adducts trigger apoptosis',
    d23.route        = 'Oral',
    d23.typical_dose = '150-200 mg/m² for 5 days every 28 days',
    d23.notes        = 'Oral analogue of dacarbazine; crosses blood-brain barrier; used for melanoma CNS metastases';

MERGE (d24:ChemoDrug {name: 'Ipilimumab'})
SET d24.drug_class   = 'Checkpoint inhibitor / anti-CTLA-4 monoclonal antibody',
    d24.mechanism    = 'Blocks CTLA-4, enhancing T-cell activation and anti-tumour immune response',
    d24.route        = 'IV',
    d24.typical_dose = '3 mg/kg every 3 weeks × 4 doses',
    d24.notes        = 'Immune-related adverse events (irAEs): colitis, hepatitis, hypophysitis; dietary adjustments during irAE management';

MERGE (d25:ChemoDrug {name: 'Pembrolizumab'})
SET d25.drug_class   = 'Checkpoint inhibitor / anti-PD-1 monoclonal antibody',
    d25.mechanism    = 'Blocks PD-1 receptor, preventing tumour immune evasion',
    d25.route        = 'IV',
    d25.typical_dose = '200 mg every 3 weeks or 400 mg every 6 weeks',
    d25.notes        = 'irAEs include pneumonitis, colitis, endocrinopathies; nutritional support for immune colitis';

MERGE (d26:ChemoDrug {name: 'Nivolumab'})
SET d26.drug_class   = 'Checkpoint inhibitor / anti-PD-1 monoclonal antibody',
    d26.mechanism    = 'Blocks PD-1 on T-cells, restoring tumour-specific T-cell function',
    d26.route        = 'IV',
    d26.typical_dose = '240 mg every 2 weeks or 480 mg every 4 weeks',
    d26.notes        = 'Used in NSCLC, melanoma, RCC; combination with ipilimumab increases irAE incidence';

MERGE (d27:ChemoDrug {name: 'Vemurafenib'})
SET d27.drug_class   = 'BRAF inhibitor / targeted therapy',
    d27.mechanism    = 'Selectively inhibits BRAF V600E mutant kinase, blocking MAPK pathway',
    d27.route        = 'Oral',
    d27.typical_dose = '960 mg twice daily',
    d27.notes        = 'For BRAF V600E+ melanoma; photosensitivity, arthralgias, squamous cell carcinoma of skin';

MERGE (d28:ChemoDrug {name: 'Trametinib'})
SET d28.drug_class   = 'MEK inhibitor / targeted therapy',
    d28.mechanism    = 'Blocks MEK1/2 in the RAS/RAF/MEK/ERK pathway',
    d28.route        = 'Oral',
    d28.typical_dose = '2 mg once daily',
    d28.notes        = 'Used with dabrafenib (BRAF/MEK combo) for BRAF+ melanoma; rash, diarrhoea, pyrexia';

// ════════════════════════════════════════════════════════════
// SECTION 3 – CANCER → DRUG RELATIONSHIPS
// ════════════════════════════════════════════════════════════

// Lung Cancer
MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Cisplatin'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Cisplatin/Gemcitabine or Cisplatin/Pemetrexed'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Carboplatin'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Carboplatin/Paclitaxel'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Paclitaxel'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Carboplatin/Paclitaxel'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Docetaxel'})
MERGE (c)-[:TREATED_WITH {line: 'second', evidence: 'Level 1A', regimen: 'Docetaxel monotherapy'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Gemcitabine'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Gemcitabine/Cisplatin'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Pemetrexed'})
MERGE (c)-[:TREATED_WITH {line: 'first/maintenance', evidence: 'Level 1A', regimen: 'Pemetrexed/Platinum non-squamous'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Vinorelbine'})
MERGE (c)-[:TREATED_WITH {line: 'second', evidence: 'Level 2', regimen: 'Vinorelbine monotherapy'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Nivolumab'})
MERGE (c)-[:TREATED_WITH {line: 'first/second', evidence: 'Level 1A', regimen: 'Nivolumab monotherapy or combo'}]->(d);

MATCH (c:Cancer {name: 'Lung Cancer'}), (d:ChemoDrug {name: 'Pembrolizumab'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Pembro + chemo or monotherapy PD-L1>50%'}]->(d);

// Breast Cancer
MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Doxorubicin'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'AC (Doxorubicin/Cyclophosphamide)'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Cyclophosphamide'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'AC or CMF'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Paclitaxel'})
MERGE (c)-[:TREATED_WITH {line: 'adjuvant/metastatic', evidence: 'Level 1A', regimen: 'AC→Paclitaxel or weekly Paclitaxel'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Docetaxel'})
MERGE (c)-[:TREATED_WITH {line: 'adjuvant/metastatic', evidence: 'Level 1A', regimen: 'TAC or TCH regimens'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Capecitabine'})
MERGE (c)-[:TREATED_WITH {line: 'metastatic/adjuvant', evidence: 'Level 1A', regimen: 'Xeloda monotherapy or combo'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Epirubicin'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'FEC or FEC-T'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Fluorouracil'})
MERGE (c)-[:TREATED_WITH {line: 'adjuvant', evidence: 'Level 2', regimen: 'CMF or FAC regimen'}]->(d);

MATCH (c:Cancer {name: 'Breast Cancer'}), (d:ChemoDrug {name: 'Gemcitabine'})
MERGE (c)-[:TREATED_WITH {line: 'metastatic', evidence: 'Level 1B', regimen: 'Gemcitabine/Paclitaxel or Gemcitabine/Carboplatin'}]->(d);

// Acute Leukemia
MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Cytarabine'})
MERGE (c)-[:TREATED_WITH {line: 'induction/consolidation', evidence: 'Level 1A', regimen: '7+3 AML induction or HiDAC consolidation'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Daunorubicin'})
MERGE (c)-[:TREATED_WITH {line: 'induction', evidence: 'Level 1A', regimen: '7+3 AML induction'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Idarubicin'})
MERGE (c)-[:TREATED_WITH {line: 'induction', evidence: 'Level 1A', regimen: '3+7 with idarubicin'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Vincristine'})
MERGE (c)-[:TREATED_WITH {line: 'induction', evidence: 'Level 1A', regimen: 'ALL CALGB 8811 induction'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'L-Asparaginase'})
MERGE (c)-[:TREATED_WITH {line: 'induction', evidence: 'Level 1A', regimen: 'Hyper-CVAD and CALGB 8811'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Mercaptopurine'})
MERGE (c)-[:TREATED_WITH {line: 'maintenance', evidence: 'Level 1A', regimen: 'ALL maintenance therapy'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Methotrexate'})
MERGE (c)-[:TREATED_WITH {line: 'induction/maintenance/CNS', evidence: 'Level 1A', regimen: 'IT MTX for CNS prophylaxis; oral MTX maintenance'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Cyclophosphamide'})
MERGE (c)-[:TREATED_WITH {line: 'induction', evidence: 'Level 1A', regimen: 'Hyper-CVAD'}]->(d);

MATCH (c:Cancer {name: 'Acute Leukemia'}), (d:ChemoDrug {name: 'Etoposide'})
MERGE (c)-[:TREATED_WITH {line: 'salvage', evidence: 'Level 2', regimen: 'FLAG-IDA or ESHAP salvage'}]->(d);

// Osteosarcoma
MATCH (c:Cancer {name: 'Osteosarcoma'}), (d:ChemoDrug {name: 'Methotrexate'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'MAP protocol (HD-MTX + Adriamycin + Cisplatin)'}]->(d);

MATCH (c:Cancer {name: 'Osteosarcoma'}), (d:ChemoDrug {name: 'Doxorubicin'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'MAP protocol'}]->(d);

MATCH (c:Cancer {name: 'Osteosarcoma'}), (d:ChemoDrug {name: 'Cisplatin'})
MERGE (c)-[:TREATED_WITH {line: 'neoadjuvant/adjuvant', evidence: 'Level 1A', regimen: 'MAP protocol'}]->(d);

MATCH (c:Cancer {name: 'Osteosarcoma'}), (d:ChemoDrug {name: 'Ifosfamide'})
MERGE (c)-[:TREATED_WITH {line: 'salvage', evidence: 'Level 1B', regimen: 'IFOS/Etoposide'}]->(d);

MATCH (c:Cancer {name: 'Osteosarcoma'}), (d:ChemoDrug {name: 'Etoposide'})
MERGE (c)-[:TREATED_WITH {line: 'salvage', evidence: 'Level 1B', regimen: 'IFOS/Etoposide'}]->(d);

// Skin Cancer
MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Dacarbazine'})
MERGE (c)-[:TREATED_WITH {line: 'first (historical)', evidence: 'Level 2', regimen: 'DTIC monotherapy for melanoma'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Temozolomide'})
MERGE (c)-[:TREATED_WITH {line: 'CNS metastases', evidence: 'Level 2', regimen: 'Temozolomide for brain mets'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Ipilimumab'})
MERGE (c)-[:TREATED_WITH {line: 'first/second', evidence: 'Level 1A', regimen: 'Ipilimumab or Ipi+Nivo'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Pembrolizumab'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Pembrolizumab for advanced melanoma'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Nivolumab'})
MERGE (c)-[:TREATED_WITH {line: 'first', evidence: 'Level 1A', regimen: 'Nivolumab mono or Nivo+Ipi'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Vemurafenib'})
MERGE (c)-[:TREATED_WITH {line: 'first (BRAF+)', evidence: 'Level 1A', regimen: 'Vemurafenib/Cobimetinib'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Trametinib'})
MERGE (c)-[:TREATED_WITH {line: 'first (BRAF+)', evidence: 'Level 1A', regimen: 'Dabrafenib/Trametinib'}]->(d);

MATCH (c:Cancer {name: 'Skin Cancer'}), (d:ChemoDrug {name: 'Carboplatin'})
MERGE (c)-[:TREATED_WITH {line: 'first (chemo eligible)', evidence: 'Level 2', regimen: 'Carboplatin/Paclitaxel for unresectable melanoma'}]->(d);

// ════════════════════════════════════════════════════════════
// SECTION 4 – EATING ADVERSE EFFECT NODES  (12 nodes)
// Source: NCI Eating Hints PDF
// ════════════════════════════════════════════════════════════

MERGE (e1:EatingAdverseEffect {name: 'Nausea'})
SET e1.description    = 'Feeling queasy or sick to stomach; may precede vomiting. Very common during chemotherapy.',
    e1.management_tip = 'Eat small meals every 2-3 hours; avoid empty stomach; anti-emetics (5-HT3 antagonists, NK1 antagonists)',
    e1.onset          = 'During or shortly after treatment',
    e1.source_doc     = 'eatinghints_chemo.pdf p.21';

MERGE (e2:EatingAdverseEffect {name: 'Vomiting'})
SET e2.description    = 'Emesis; can lead to dehydration and electrolyte imbalance if uncontrolled.',
    e2.management_tip = 'NPO until vomiting stops; then clear liquids; anti-emetics essential; IV fluids if needed',
    e2.onset          = 'Acute (0-24h), delayed (24-72h), or anticipatory',
    e2.source_doc     = 'eatinghints_chemo.pdf p.31';

MERGE (e3:EatingAdverseEffect {name: 'Appetite Loss'})
SET e3.description    = 'Anorexia; reduced desire to eat due to cancer, treatment side effects, or psychological factors.',
    e3.management_tip = 'Eat by schedule; calorie-dense small meals; liquid supplements; consider megestrol or corticosteroids',
    e3.onset          = 'Throughout treatment course',
    e3.source_doc     = 'eatinghints_chemo.pdf p.10';

MERGE (e4:EatingAdverseEffect {name: 'Sore Mouth (Mucositis)'})
SET e4.description    = 'Ulceration and inflammation of oral mucosa; painful, can prevent adequate oral intake.',
    e4.management_tip = 'Salt-water rinses, magic mouthwash, soft diet, cold foods, avoid acidic/spicy foods; palifermin for high-dose chemo',
    e4.onset          = '5-14 days post chemotherapy initiation',
    e4.source_doc     = 'eatinghints_chemo.pdf p.23';

MERGE (e5:EatingAdverseEffect {name: 'Dry Mouth (Xerostomia)'})
SET e5.description    = 'Reduced saliva production due to damage to salivary glands; impairs taste, chewing, swallowing.',
    e5.management_tip = 'Sip water frequently; use artificial saliva; tart foods stimulate saliva; pilocarpine if severe',
    e5.onset          = 'During head/neck radiation or certain chemotherapy',
    e5.source_doc     = 'eatinghints_chemo.pdf p.17';

MERGE (e6:EatingAdverseEffect {name: 'Taste and Smell Changes'})
SET e6.description    = 'Dysgeusia/ageusia – foods taste metallic, bland, or unpleasant. Affects food acceptance.',
    e6.management_tip = 'Use plastic utensils for metal taste; marinate proteins; enhance flavours with herbs; good oral hygiene',
    e6.onset          = 'During treatment; may persist weeks-months post treatment',
    e6.source_doc     = 'eatinghints_chemo.pdf p.29';

MERGE (e7:EatingAdverseEffect {name: 'Diarrhoea'})
SET e7.description    = 'Frequent loose stools from damage to intestinal epithelium; risk of dehydration and malnutrition.',
    e7.management_tip = 'BRAT diet, low-fibre foods, oral rehydration; loperamide; IV fluids if grade 3-4; avoid lactose and high fat',
    e7.onset          = 'Days to weeks into treatment',
    e7.source_doc     = 'eatinghints_chemo.pdf p.15';

MERGE (e8:EatingAdverseEffect {name: 'Constipation'})
SET e8.description    = 'Infrequent or difficult bowel movements; common with opioids, vinca alkaloids, and reduced activity.',
    e8.management_tip = 'Increase fluid and fibre intake; stool softeners; gentle laxatives; physical activity',
    e8.onset          = 'Throughout treatment, especially with opioids',
    e8.source_doc     = 'eatinghints_chemo.pdf p.13';

MERGE (e9:EatingAdverseEffect {name: 'Dysphagia (Sore Throat)'})
SET e9.description    = 'Painful or difficult swallowing (esophagitis) due to mucosal damage from radiation or chemo.',
    e9.management_tip = 'Soft, moist foods; avoid sharp-textured foods; upright posture; tube feeding if severe',
    e9.onset          = 'During head/neck or chest radiation',
    e9.source_doc     = 'eatinghints_chemo.pdf p.26';

MERGE (e10:EatingAdverseEffect {name: 'Weight Loss'})
SET e10.description    = 'Unintentional loss of body weight; contributes to fatigue, poor wound healing, treatment delays.',
    e10.management_tip = 'Calorie-dense foods; oral nutritional supplements; consider nasogastric/parenteral nutrition',
    e10.onset          = 'Progressive during treatment',
    e10.source_doc     = 'eatinghints_chemo.pdf p.35';

MERGE (e11:EatingAdverseEffect {name: 'Weight Gain'})
SET e11.description    = 'Increased body weight from steroids, hormone therapy, fluid retention, or reduced activity.',
    e11.management_tip = 'High-fibre, low-fat diet; reduce refined carbs; light physical activity; monitor fluid retention',
    e11.onset          = 'Especially with steroid-containing regimens',
    e11.source_doc     = 'eatinghints_chemo.pdf p.33';

MERGE (e12:EatingAdverseEffect {name: 'Lactose Intolerance (Secondary)'})
SET e12.description    = 'Transient inability to digest lactose due to small bowel mucosal damage.',
    e12.management_tip = 'Lactase supplements; lactose-free dairy; plant milks (soy, oat, almond); hard cheeses tolerated',
    e12.onset          = 'During or after abdominal/pelvic radiation',
    e12.source_doc     = 'eatinghints_chemo.pdf p.19';

// ════════════════════════════════════════════════════════════
// SECTION 5 – DRUG → EATING EFFECT RELATIONSHIPS
// ════════════════════════════════════════════════════════════

// Cisplatin
MATCH (d:ChemoDrug {name:'Cisplatin'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'70-90%', onset:'Acute and delayed'}]->(e);
MATCH (d:ChemoDrug {name:'Cisplatin'}),(e:EatingAdverseEffect {name:'Vomiting'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'60-80%', onset:'Acute 1-4h'}]->(e);
MATCH (d:ChemoDrug {name:'Cisplatin'}),(e:EatingAdverseEffect {name:'Taste and Smell Changes'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'50%', onset:'During treatment'}]->(e);
MATCH (d:ChemoDrug {name:'Cisplatin'}),(e:EatingAdverseEffect {name:'Weight Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'Cumulative'}]->(e);

// Carboplatin
MATCH (d:ChemoDrug {name:'Carboplatin'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40-60%', onset:'Delayed 6-12h'}]->(e);
MATCH (d:ChemoDrug {name:'Carboplatin'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'During treatment'}]->(e);

// Doxorubicin
MATCH (d:ChemoDrug {name:'Doxorubicin'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'60%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Doxorubicin'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'40%', onset:'Day 5-10'}]->(e);
MATCH (d:ChemoDrug {name:'Doxorubicin'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'Throughout'}]->(e);

// Methotrexate
MATCH (d:ChemoDrug {name:'Methotrexate'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'50%', onset:'Day 4-7'}]->(e);
MATCH (d:ChemoDrug {name:'Methotrexate'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Methotrexate'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'During treatment'}]->(e);

// 5-Fluorouracil / Capecitabine
MATCH (d:ChemoDrug {name:'Fluorouracil'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'60%', onset:'Day 5-7'}]->(e);
MATCH (d:ChemoDrug {name:'Fluorouracil'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'40%', onset:'Day 5-10'}]->(e);
MATCH (d:ChemoDrug {name:'Capecitabine'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'55%', onset:'Day 5-14'}]->(e);
MATCH (d:ChemoDrug {name:'Capecitabine'}),(e:EatingAdverseEffect {name:'Taste and Smell Changes'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'During treatment'}]->(e);

// Vincristine
MATCH (d:ChemoDrug {name:'Vincristine'}),(e:EatingAdverseEffect {name:'Constipation'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'35%', onset:'Week 1-2'}]->(e);

// L-Asparaginase
MATCH (d:ChemoDrug {name:'L-Asparaginase'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'L-Asparaginase'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'During treatment'}]->(e);
MATCH (d:ChemoDrug {name:'L-Asparaginase'}),(e:EatingAdverseEffect {name:'Weight Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'25%', onset:'Progressive'}]->(e);

// Ifosfamide
MATCH (d:ChemoDrug {name:'Ifosfamide'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'65%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Ifosfamide'}),(e:EatingAdverseEffect {name:'Vomiting'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'50%', onset:'Acute'}]->(e);

// Dacarbazine
MATCH (d:ChemoDrug {name:'Dacarbazine'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'80%', onset:'1-3h post dose'}]->(e);
MATCH (d:ChemoDrug {name:'Dacarbazine'}),(e:EatingAdverseEffect {name:'Vomiting'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'70%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Dacarbazine'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'Throughout'}]->(e);

// Pemetrexed (requires nutritional supplementation)
MATCH (d:ChemoDrug {name:'Pemetrexed'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'25%', onset:'Day 5-7', mitigation:'B12 + folate supplementation reduces incidence'}]->(e);
MATCH (d:ChemoDrug {name:'Pemetrexed'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'Acute'}]->(e);

// Immunotherapy (irAE-related dietary effects)
MATCH (d:ChemoDrug {name:'Ipilimumab'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'30-40%', onset:'Week 3-8', mechanism:'Immune colitis'}]->(e);
MATCH (d:ChemoDrug {name:'Pembrolizumab'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'15%', onset:'Variable', mechanism:'Immune colitis'}]->(e);
MATCH (d:ChemoDrug {name:'Nivolumab'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'20%', onset:'Variable'}]->(e);

// Gemcitabine
MATCH (d:ChemoDrug {name:'Gemcitabine'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'During infusion'}]->(e);
MATCH (d:ChemoDrug {name:'Gemcitabine'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'Cumulative'}]->(e);

// Etoposide
MATCH (d:ChemoDrug {name:'Etoposide'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Etoposide'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'20%', onset:'Day 5-10'}]->(e);

// Cyclophosphamide
MATCH (d:ChemoDrug {name:'Cyclophosphamide'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'High', frequency:'60%', onset:'Acute 2-4h'}]->(e);
MATCH (d:ChemoDrug {name:'Cyclophosphamide'}),(e:EatingAdverseEffect {name:'Vomiting'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Cyclophosphamide'}),(e:EatingAdverseEffect {name:'Appetite Loss'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'25%', onset:'Throughout'}]->(e);

// Paclitaxel/Docetaxel
MATCH (d:ChemoDrug {name:'Paclitaxel'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Docetaxel'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'25%', onset:'Day 5-7'}]->(e);
MATCH (d:ChemoDrug {name:'Docetaxel'}),(e:EatingAdverseEffect {name:'Taste and Smell Changes'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'30%', onset:'During treatment'}]->(e);
MATCH (d:ChemoDrug {name:'Docetaxel'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'35%', onset:'Day 5-10'}]->(e);
MATCH (d:ChemoDrug {name:'Docetaxel'}),(e:EatingAdverseEffect {name:'Weight Gain'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'20%', onset:'With dexamethasone premedication', mechanism:'Fluid retention + steroid'}]->(e);

// Trametinib / Vemurafenib
MATCH (d:ChemoDrug {name:'Trametinib'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Moderate', frequency:'40%', onset:'Weeks 1-4'}]->(e);
MATCH (d:ChemoDrug {name:'Trametinib'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Low', frequency:'20%', onset:'Acute'}]->(e);
MATCH (d:ChemoDrug {name:'Vemurafenib'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (d)-[:CAUSES_EATING_EFFECT {severity:'Low', frequency:'25%', onset:'Acute'}]->(e);

// ════════════════════════════════════════════════════════════
// SECTION 6 – FOOD ITEM NODES  (18 nodes)
// ════════════════════════════════════════════════════════════

MERGE (f1:FoodItem {name: 'Bananas'})
SET f1.category='Fruit', f1.texture='Soft', f1.notes='High potassium; easy on GI tract; BRAT diet component';
MERGE (f2:FoodItem {name: 'White Rice'})
SET f2.category='Grain', f2.texture='Soft', f2.notes='Low fibre; binding; BRAT diet component';
MERGE (f3:FoodItem {name: 'Applesauce'})
SET f3.category='Fruit (processed)', f3.texture='Liquid/Smooth', f3.notes='Pectin content; easily digestible; BRAT diet';
MERGE (f4:FoodItem {name: 'Toast/White Bread'})
SET f4.category='Grain', f4.texture='Soft', f4.notes='Low fibre; absorbs stomach acid; BRAT diet';
MERGE (f5:FoodItem {name: 'Spicy Foods'})
SET f5.category='Condiment/Seasoning', f5.texture='Variable', f5.notes='Capsaicin irritates GI mucosa; avoid with mucositis, esophagitis, diarrhoea';
MERGE (f6:FoodItem {name: 'Fried / Fatty Foods'})
SET f6.category='High-fat foods', f6.texture='Variable', f6.notes='Delay gastric emptying; exacerbate nausea; high caloric but problematic in nausea';
MERGE (f7:FoodItem {name: 'Dairy / Lactose-containing Foods'})
SET f7.category='Dairy', f7.texture='Variable', f7.notes='Avoid with secondary lactose intolerance; substitute lactose-free or plant-based alternatives';
MERGE (f8:FoodItem {name: 'Protein Shakes / Oral Nutritional Supplements'})
SET f8.category='Supplement', f8.texture='Liquid', f8.notes='Ensure, Boost etc.; high calorie and protein; recommended when solid food intake is poor';
MERGE (f9:FoodItem {name: 'Ginger Tea / Ginger Ale'})
SET f9.category='Beverage', f9.texture='Liquid', f9.notes='Anti-emetic properties of gingerols; soothes nausea; allow flat before drinking';
MERGE (f10:FoodItem {name: 'Cold Foods (Popsicles, Ice Chips)'})
SET f10.category='Cold foods/Desserts', f10.texture='Cold/Frozen', f10.notes='Reduce oral pain in mucositis; numb sores; often better tolerated than hot foods';
MERGE (f11:FoodItem {name: 'Citrus Fruits / Acidic Foods'})
SET f11.category='Fruit', f11.texture='Variable', f11.notes='Stimulate saliva (good for dry mouth) but irritate sore mouth/throat – context-dependent';
MERGE (f12:FoodItem {name: 'High-Fibre Foods (Whole Grains, Legumes)'})
SET f12.category='Fibre-rich', f12.texture='Variable', f12.notes='Beneficial for constipation and weight management; avoid during active diarrhoea';
MERGE (f13:FoodItem {name: 'Milk'})
SET f13.description='Used with 6-mercaptopurine, milk reduces drug absorption (avoid co-ingestion)';
MERGE (f14:FoodItem {name: 'Grapefruit'})
SET f14.category='Fruit', f14.texture='Semi-solid', f14.notes='CYP3A4 inhibitor – interacts with multiple oral targeted agents including vemurafenib, lorlatinib, trametinib; avoid during treatment';
MERGE (f15:FoodItem {name: 'Alcohol'})
SET f15.category='Beverage', f15.texture='Liquid', f15.notes='Worsens mucositis, dry mouth, nausea; increases methotrexate hepatotoxicity; immunosuppressive';
MERGE (f16:FoodItem {name: 'Water / Clear Liquids'})
SET f16.category='Beverage', f16.texture='Liquid', f16.notes='Essential hydration; 8-12 cups/day recommended; critical for cisplatin nephrotoxicity prevention';
MERGE (f17:FoodItem {name: 'Scrambled Eggs / Soft Protein'})
SET f17.category='Protein', f17.texture='Soft', f17.notes='Easy to chew and swallow; high protein; recommended for mucositis and dysphagia';
MERGE (f18:FoodItem {name: 'Yogurt / Probiotics'})
SET f18.category='Dairy / Probiotic', f18.texture='Soft', f18.notes='May reduce chemo-induced diarrhoea; choose low-lactose or lactose-free varieties if intolerant';

// ── EatingEffect → Food Relationships ────────────────────────
MATCH (e:EatingAdverseEffect {name:'Nausea'}),(f:FoodItem {name:'Ginger Tea / Ginger Ale'})
MERGE (e)-[:RELIEVED_BY {evidence:'Moderate', notes:'1-2g ginger daily shown to reduce chemo nausea'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Nausea'}),(f:FoodItem {name:'Bananas'})
MERGE (e)-[:RELIEVED_BY {evidence:'Low-moderate', notes:'BRAT diet component; bland and gentle'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Nausea'}),(f:FoodItem {name:'Toast/White Bread'})
MERGE (e)-[:RELIEVED_BY {evidence:'Low-moderate', notes:'Crackers before rising helps morning nausea'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Nausea'}),(f:FoodItem {name:'Fried / Fatty Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Delays gastric emptying; increases nausea severity'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Nausea'}),(f:FoodItem {name:'Spicy Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Irritates gastric mucosa; triggers emesis reflex'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'White Rice'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Low fibre; binding properties help reduce stool frequency'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'Bananas'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Pectin and potassium replacement'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'Yogurt / Probiotics'})
MERGE (e)-[:RELIEVED_BY {evidence:'Moderate', notes:'Lactobacillus strains may reduce severity; use lactose-free if intolerant'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'Spicy Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Irritates intestinal mucosa'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'High-Fibre Foods (Whole Grains, Legumes)'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Insoluble fibre increases gut motility during active diarrhoea'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Diarrhoea'}),(f:FoodItem {name:'Dairy / Lactose-containing Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Secondary lactose intolerance compounds diarrhoea'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'}),(f:FoodItem {name:'Cold Foods (Popsicles, Ice Chips)'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Cold numbs pain; reduces inflammation; improves intake'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'}),(f:FoodItem {name:'Scrambled Eggs / Soft Protein'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Soft, moist protein source; easy to chew and swallow'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'}),(f:FoodItem {name:'Citrus Fruits / Acidic Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Acid burns ulcerated mucosa; increases pain'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'}),(f:FoodItem {name:'Spicy Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Capsaicin causes severe mucosal irritation and pain'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'}),(f:FoodItem {name:'Alcohol'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Desiccates and irritates mucosal surfaces'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Dry Mouth (Xerostomia)'}),(f:FoodItem {name:'Citrus Fruits / Acidic Foods'})
MERGE (e)-[:RELIEVED_BY {evidence:'Moderate', notes:'Tart foods stimulate residual salivary gland function; only if no mucositis'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Dry Mouth (Xerostomia)'}),(f:FoodItem {name:'Water / Clear Liquids'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Frequent small sips maintain oral moisture'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Dry Mouth (Xerostomia)'}),(f:FoodItem {name:'Alcohol'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Further desiccates oral mucosa'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Constipation'}),(f:FoodItem {name:'High-Fibre Foods (Whole Grains, Legumes)'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Insoluble and soluble fibre promote bowel movement'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Constipation'}),(f:FoodItem {name:'Water / Clear Liquids'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'Adequate hydration essential for fibre to work effectively'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Weight Loss'}),(f:FoodItem {name:'Protein Shakes / Oral Nutritional Supplements'})
MERGE (e)-[:RELIEVED_BY {evidence:'High', notes:'High calorie, high protein; helps meet nutritional goals when appetite is low'}]->(f);
MATCH (e:EatingAdverseEffect {name:'Lactose Intolerance (Secondary)'}),(f:FoodItem {name:'Dairy / Lactose-containing Foods'})
MERGE (e)-[:WORSENED_BY {evidence:'High', notes:'Undigested lactose causes osmotic diarrhoea, bloating'}]->(f);

// ════════════════════════════════════════════════════════════
// SECTION 7 – NUTRITION GUIDELINES  (10 nodes)
// ════════════════════════════════════════════════════════════

MERGE (g1:NutritionGuideline {id:'NG001'})
SET g1.text='Eat 5-6 small meals per day instead of 3 large meals to manage nausea, dysphagia, and appetite loss',
    g1.applicable_to='Nausea, Appetite Loss, Dysphagia',
    g1.source='NCI Eating Hints 2022',
    g1.evidence_level='Expert consensus';

MERGE (g2:NutritionGuideline {id:'NG002'})
SET g2.text='Drink 8-12 cups of liquid daily; increase further if vomiting or diarrhoea is present to prevent dehydration',
    g2.applicable_to='Nausea, Vomiting, Diarrhoea, Dry Mouth',
    g2.source='NCI Eating Hints 2022';

MERGE (g3:NutritionGuideline {id:'NG003'})
SET g3.text='Pemetrexed requires folic acid 400-1000mcg daily starting 7 days before first dose and B12 1000mcg IM every 9 weeks',
    g3.applicable_to='Pemetrexed toxicity prevention (mucositis, neutropenia)',
    g3.source='Pemetrexed prescribing information',
    g3.evidence_level='Level 1A – mandatory';

MERGE (g4:NutritionGuideline {id:'NG004'})
SET g4.text='High-dose methotrexate requires leucovorin rescue and aggressive IV hydration with urinary alkalinisation to prevent nephrotoxicity',
    g4.applicable_to='High-dose MTX protocol (osteosarcoma, ALL)',
    g4.source='Osteosarcoma MAP protocol guidelines';

MERGE (g5:NutritionGuideline {id:'NG005'})
SET g5.text='Avoid grapefruit, grapefruit juice, and Seville oranges during therapy with CYP3A4-metabolised oral targeted agents',
    g5.applicable_to='Vemurafenib, Lorlatinib, Crizotinib, Trametinib',
    g5.source='FDA drug interaction guidance';

MERGE (g6:NutritionGuideline {id:'NG006'})
SET g6.text='Cisplatin requires pre-hydration with 2-4L IV normal saline and post-hydration to prevent nephrotoxicity; encourage oral fluid intake',
    g6.applicable_to='Cisplatin nephrotoxicity prevention',
    g6.source='ASCO/MASCC guidelines';

MERGE (g7:NutritionGuideline {id:'NG007'})
SET g7.text='6-Mercaptopurine absorption is reduced by milk; take on an empty stomach; avoid milk for 1 hour before and after dosing',
    g7.applicable_to='ALL maintenance – 6-MP drug-food interaction',
    g7.source='Drug information pharmacopeoia';

MERGE (g8:NutritionGuideline {id:'NG008'})
SET g8.text='For patients with immune colitis from checkpoint inhibitors: low-residue diet, avoid high-fibre foods, lactose, and fatty foods; corticosteroids for grade 2+',
    g8.applicable_to='Ipilimumab / Pembrolizumab / Nivolumab immune colitis',
    g8.source='SITC immunotherapy toxicity management';

MERGE (g9:NutritionGuideline {id:'NG009'})
SET g9.text='Mesna must be co-administered with ifosfamide and cyclophosphamide to prevent haemorrhagic cystitis; high oral fluid intake reinforces protection',
    g9.applicable_to='Ifosfamide, Cyclophosphamide',
    g9.source='Standard oncology protocol';

MERGE (g10:NutritionGuideline {id:'NG010'})
SET g10.text='Oral rinse protocol: 0.9% NaCl rinse or sodium bicarbonate rinse 3-4 times daily reduces mucositis severity; good oral hygiene is essential',
    g10.applicable_to='Mucositis prevention across all chemotherapy regimens',
    g10.source='MASCC/ISOO Mucositis Guidelines';

// ── Guidelines → EatingEffects ────────────────────────────────
MATCH (g:NutritionGuideline {id:'NG001'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (g)-[:MANAGES]->(e);
MATCH (g:NutritionGuideline {id:'NG002'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (g)-[:MANAGES]->(e);
MATCH (g:NutritionGuideline {id:'NG010'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (g)-[:MANAGES]->(e);
MATCH (g:NutritionGuideline {id:'NG003'}),(d:ChemoDrug {name:'Pemetrexed'})
MERGE (g)-[:REQUIRED_FOR]->(d);
MATCH (g:NutritionGuideline {id:'NG004'}),(d:ChemoDrug {name:'Methotrexate'})
MERGE (g)-[:REQUIRED_FOR]->(d);
MATCH (g:NutritionGuideline {id:'NG005'}),(f:FoodItem {name:'Grapefruit'})
MERGE (g)-[:WARNS_ABOUT]->(f);
MATCH (g:NutritionGuideline {id:'NG007'}),(f:FoodItem {name:'Milk'})
MERGE (g)-[:WARNS_ABOUT]->(f);
MATCH (g:NutritionGuideline {id:'NG006'}),(d:ChemoDrug {name:'Cisplatin'})
MERGE (g)-[:REQUIRED_FOR]->(d);
MATCH (g:NutritionGuideline {id:'NG009'}),(d:ChemoDrug {name:'Ifosfamide'})
MERGE (g)-[:REQUIRED_FOR]->(d);
MATCH (g:NutritionGuideline {id:'NG008'}),(d:ChemoDrug {name:'Ipilimumab'})
MERGE (g)-[:REQUIRED_FOR]->(d);

// ════════════════════════════════════════════════════════════
// SECTION 8 – BIOMARKERS  (7 nodes)
// ════════════════════════════════════════════════════════════

MERGE (b1:Biomarker {name:'ALK gene rearrangement'})  SET b1.cancer='Lung Cancer', b1.therapeutic_relevance='Crizotinib, Alectinib, Brigatinib, Lorlatinib TKIs';
MERGE (b2:Biomarker {name:'EGFR mutation'})           SET b2.cancer='Lung Cancer', b2.therapeutic_relevance='EGFR TKIs: erlotinib, gefitinib, osimertinib';
MERGE (b3:Biomarker {name:'BRAF V600E mutation'})     SET b3.cancer='Melanoma (Skin Cancer)', b3.therapeutic_relevance='Vemurafenib, Dabrafenib, Trametinib';
MERGE (b4:Biomarker {name:'HER2 overexpression'})     SET b4.cancer='Breast Cancer', b4.therapeutic_relevance='Trastuzumab, Pertuzumab, T-DM1, Lapatinib';
MERGE (b5:Biomarker {name:'ER/PR positivity'})        SET b5.cancer='Breast Cancer', b5.therapeutic_relevance='Endocrine therapy: Tamoxifen, Aromatase inhibitors, CDK4/6 inhibitors';
MERGE (b6:Biomarker {name:'Philadelphia chromosome t(9;22)'}) SET b6.cancer='Acute Leukemia', b6.therapeutic_relevance='Imatinib, Dasatinib, Ponatinib TKIs; allogeneic SCT';
MERGE (b7:Biomarker {name:'PD-L1 expression'})        SET b7.cancer='NSCLC / Melanoma / Various', b7.therapeutic_relevance='Pembrolizumab, Nivolumab, Atezolizumab';

MATCH (c:Cancer {name:'Lung Cancer'}),(b:Biomarker {name:'ALK gene rearrangement'})   MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Lung Cancer'}),(b:Biomarker {name:'EGFR mutation'})             MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Lung Cancer'}),(b:Biomarker {name:'PD-L1 expression'})          MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Skin Cancer'}),(b:Biomarker {name:'BRAF V600E mutation'})       MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Skin Cancer'}),(b:Biomarker {name:'PD-L1 expression'})          MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Breast Cancer'}),(b:Biomarker {name:'HER2 overexpression'})     MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Breast Cancer'}),(b:Biomarker {name:'ER/PR positivity'})        MERGE (c)-[:HAS_BIOMARKER]->(b);
MATCH (c:Cancer {name:'Acute Leukemia'}),(b:Biomarker {name:'Philadelphia chromosome t(9;22)'}) MERGE (c)-[:HAS_BIOMARKER]->(b);

// ════════════════════════════════════════════════════════════
// SECTION 9 – TREATMENT PROTOCOLS  (6 nodes)
// ════════════════════════════════════════════════════════════

MERGE (p1:TreatmentProtocol {name:'AC-T (Breast)'})
SET p1.description='Doxorubicin/Cyclophosphamide × 4 cycles then Paclitaxel × 4; standard adjuvant breast cancer',
    p1.cancer='Breast Cancer', p1.setting='Adjuvant/Neoadjuvant';

MERGE (p2:TreatmentProtocol {name:'MAP (Osteosarcoma)'})
SET p2.description='High-dose Methotrexate + Doxorubicin (Adriamycin) + Cisplatin; standard neoadjuvant osteosarcoma',
    p2.cancer='Osteosarcoma', p2.setting='Neoadjuvant/Adjuvant';

MERGE (p3:TreatmentProtocol {name:'7+3 AML Induction'})
SET p3.description='Cytarabine 100mg/m² CI days 1-7 + Daunorubicin/Idarubicin days 1-3; standard AML induction',
    p3.cancer='Acute Leukemia (AML)', p3.setting='Induction';

MERGE (p4:TreatmentProtocol {name:'Hyper-CVAD (ALL)'})
SET p4.description='Hyperfractionated Cyclophosphamide, Vincristine, Doxorubicin, Dexamethasone alternating with HD-MTX/Cytarabine',
    p4.cancer='Acute Leukemia (ALL)', p4.setting='Induction/Consolidation';

MERGE (p5:TreatmentProtocol {name:'Carboplatin/Paclitaxel (NSCLC)'})
SET p5.description='Carboplatin AUC6 + Paclitaxel 200mg/m² every 3 weeks; standard first-line non-squamous/squamous NSCLC',
    p5.cancer='Lung Cancer', p5.setting='First-line';

MERGE (p6:TreatmentProtocol {name:'Ipi+Nivo (Melanoma)'})
SET p6.description='Ipilimumab 3mg/kg + Nivolumab 1mg/kg q3w × 4 then Nivo 240mg q2w; first-line advanced melanoma',
    p6.cancer='Skin Cancer (Melanoma)', p6.setting='First-line';

MATCH (p:TreatmentProtocol {name:'AC-T (Breast)'}),(d:ChemoDrug {name:'Doxorubicin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'AC-T (Breast)'}),(d:ChemoDrug {name:'Cyclophosphamide'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'AC-T (Breast)'}),(d:ChemoDrug {name:'Paclitaxel'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'MAP (Osteosarcoma)'}),(d:ChemoDrug {name:'Methotrexate'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'MAP (Osteosarcoma)'}),(d:ChemoDrug {name:'Doxorubicin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'MAP (Osteosarcoma)'}),(d:ChemoDrug {name:'Cisplatin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'7+3 AML Induction'}),(d:ChemoDrug {name:'Cytarabine'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'7+3 AML Induction'}),(d:ChemoDrug {name:'Daunorubicin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}),(d:ChemoDrug {name:'Cyclophosphamide'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}),(d:ChemoDrug {name:'Vincristine'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}),(d:ChemoDrug {name:'Doxorubicin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}),(d:ChemoDrug {name:'Methotrexate'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}),(d:ChemoDrug {name:'Cytarabine'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Carboplatin/Paclitaxel (NSCLC)'}),(d:ChemoDrug {name:'Carboplatin'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Carboplatin/Paclitaxel (NSCLC)'}),(d:ChemoDrug {name:'Paclitaxel'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Ipi+Nivo (Melanoma)'}),(d:ChemoDrug {name:'Ipilimumab'}) MERGE (p)-[:INCLUDES_DRUG]->(d);
MATCH (p:TreatmentProtocol {name:'Ipi+Nivo (Melanoma)'}),(d:ChemoDrug {name:'Nivolumab'}) MERGE (p)-[:INCLUDES_DRUG]->(d);

MATCH (c:Cancer {name:'Breast Cancer'}),(p:TreatmentProtocol {name:'AC-T (Breast)'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
MATCH (c:Cancer {name:'Osteosarcoma'}),(p:TreatmentProtocol {name:'MAP (Osteosarcoma)'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
MATCH (c:Cancer {name:'Acute Leukemia'}),(p:TreatmentProtocol {name:'7+3 AML Induction'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
MATCH (c:Cancer {name:'Acute Leukemia'}),(p:TreatmentProtocol {name:'Hyper-CVAD (ALL)'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
MATCH (c:Cancer {name:'Lung Cancer'}),(p:TreatmentProtocol {name:'Carboplatin/Paclitaxel (NSCLC)'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
MATCH (c:Cancer {name:'Skin Cancer'}),(p:TreatmentProtocol {name:'Ipi+Nivo (Melanoma)'}) MERGE (c)-[:TREATED_BY_PROTOCOL]->(p);
