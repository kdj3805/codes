// ============================================================
// 03_drug_interactions.cypher
// Non-Chemo Drug × ChemoDrug Interaction Knowledge Graph
// Nodes: NonChemoDrug, Ailment, DrugInteraction, SideEffect
// ============================================================

// ════════════════════════════════════════════════════════════
// SECTION A – AILMENT NODES  (12 nodes)
// ════════════════════════════════════════════════════════════

MERGE (a1:Ailment  {name:'Hypertension'})           SET a1.prevalence='High', a1.icd10='I10';
MERGE (a2:Ailment  {name:'Type 2 Diabetes'})         SET a2.prevalence='High', a2.icd10='E11';
MERGE (a3:Ailment  {name:'Depression / Anxiety'})    SET a3.prevalence='High', a3.icd10='F32/F41';
MERGE (a4:Ailment  {name:'Atrial Fibrillation'})     SET a4.prevalence='Moderate', a4.icd10='I48';
MERGE (a5:Ailment  {name:'Venous Thromboembolism'})  SET a5.prevalence='Moderate', a5.icd10='I82';
MERGE (a6:Ailment  {name:'Epilepsy'})                SET a6.prevalence='Moderate', a6.icd10='G40';
MERGE (a7:Ailment  {name:'GERD / Peptic Ulcer'})     SET a7.prevalence='High', a7.icd10='K21/K25';
MERGE (a8:Ailment  {name:'Fungal Infection'})        SET a8.prevalence='High (in immunocompromised)', a8.icd10='B37';
MERGE (a9:Ailment  {name:'Bacterial Infection'})     SET a9.prevalence='High (in immunocompromised)', a9.icd10='A41';
MERGE (a10:Ailment {name:'Hyperlipidaemia'})         SET a10.prevalence='High', a10.icd10='E78';
MERGE (a11:Ailment {name:'Chronic Pain'})            SET a11.prevalence='High', a11.icd10='G89';
MERGE (a12:Ailment {name:'Hypothyroidism'})          SET a12.prevalence='Moderate', a12.icd10='E03';

// ════════════════════════════════════════════════════════════
// SECTION B – NON-CHEMO DRUG NODES  (20 nodes)
// ════════════════════════════════════════════════════════════

MERGE (n1:NonChemoDrug {name:'Warfarin'})
SET n1.drug_class='Anticoagulant / Vitamin K antagonist',
    n1.mechanism='Inhibits Vitamin K epoxide reductase; reduces clotting factor synthesis',
    n1.route='Oral', n1.monitoring='INR (target 2-3 typically)';

MERGE (n2:NonChemoDrug {name:'Aspirin'})
SET n2.drug_class='NSAID / Antiplatelet',
    n2.mechanism='COX-1/COX-2 inhibition; irreversible platelet thromboxane A2 blockade',
    n2.route='Oral';

MERGE (n3:NonChemoDrug {name:'Ibuprofen'})
SET n3.drug_class='NSAID',
    n3.mechanism='Reversible COX-1/COX-2 inhibition; prostaglandin synthesis reduction',
    n3.route='Oral/IV';

MERGE (n4:NonChemoDrug {name:'Metformin'})
SET n4.drug_class='Biguanide / Antidiabetic',
    n4.mechanism='Activates AMPK; reduces hepatic glucose production; sensitises insulin receptors',
    n4.route='Oral';

MERGE (n5:NonChemoDrug {name:'Omeprazole'})
SET n5.drug_class='Proton Pump Inhibitor (PPI)',
    n5.mechanism='Irreversible H+/K+-ATPase inhibition in gastric parietal cells',
    n5.route='Oral/IV';

MERGE (n6:NonChemoDrug {name:'Phenytoin'})
SET n6.drug_class='Antiepileptic / Sodium channel blocker',
    n6.mechanism='Stabilises neuronal membrane by blocking voltage-gated Na+ channels',
    n6.route='Oral/IV', n6.monitoring='TDM required (therapeutic range 10-20 mg/L)';

MERGE (n7:NonChemoDrug {name:'Carbamazepine'})
SET n7.drug_class='Antiepileptic / Mood stabiliser',
    n7.mechanism='Sodium channel blockade; strong CYP3A4 inducer',
    n7.route='Oral', n7.monitoring='TDM and FBC required';

MERGE (n8:NonChemoDrug {name:'Fluconazole'})
SET n8.drug_class='Azole antifungal',
    n8.mechanism='CYP51 inhibition blocking ergosterol synthesis; strong CYP2C9 + moderate CYP3A4 inhibitor',
    n8.route='Oral/IV';

MERGE (n9:NonChemoDrug {name:'Voriconazole'})
SET n9.drug_class='Azole antifungal (broad spectrum)',
    n9.mechanism='Fungal CYP51 inhibition; strong CYP2C19, CYP2C9, CYP3A4 inhibitor',
    n9.route='Oral/IV';

MERGE (n10:NonChemoDrug {name:'Ciprofloxacin'})
SET n10.drug_class='Fluoroquinolone antibiotic',
    n10.mechanism='DNA gyrase and topoisomerase IV inhibition; moderate CYP1A2 inhibitor',
    n10.route='Oral/IV';

MERGE (n11:NonChemoDrug {name:'Co-trimoxazole (TMP-SMX)'})
SET n11.drug_class='Sulfonamide + diaminopyrimidine antibiotic',
    n11.mechanism='Inhibits sequential steps of bacterial folate synthesis',
    n11.route='Oral/IV';

MERGE (n12:NonChemoDrug {name:'Atorvastatin'})
SET n12.drug_class='HMG-CoA reductase inhibitor (Statin)',
    n12.mechanism='Blocks cholesterol biosynthesis; CYP3A4 substrate',
    n12.route='Oral';

MERGE (n13:NonChemoDrug {name:'Sertraline'})
SET n13.drug_class='SSRI antidepressant',
    n13.mechanism='Selective serotonin reuptake inhibition; moderate CYP2D6 inhibitor',
    n13.route='Oral';

MERGE (n14:NonChemoDrug {name:'Amitriptyline'})
SET n14.drug_class='Tricyclic antidepressant (TCA)',
    n14.mechanism='Noradrenaline and serotonin reuptake inhibition; anticholinergic; CYP2D6 substrate',
    n14.route='Oral';

MERGE (n15:NonChemoDrug {name:'Amlodipine'})
SET n15.drug_class='Calcium channel blocker (dihydropyridine)',
    n15.mechanism='L-type calcium channel blockade causing vasodilation; CYP3A4 substrate',
    n15.route='Oral';

MERGE (n16:NonChemoDrug {name:'Furosemide'})
SET n16.drug_class='Loop diuretic',
    n16.mechanism='Na-K-2Cl cotransporter inhibition in loop of Henle',
    n16.route='Oral/IV';

MERGE (n17:NonChemoDrug {name:'Allopurinol'})
SET n17.drug_class='Xanthine oxidase inhibitor',
    n17.mechanism='Inhibits xanthine oxidase; reduces uric acid production; used for tumour lysis syndrome',
    n17.route='Oral/IV';

MERGE (n18:NonChemoDrug {name:'Morphine'})
SET n18.drug_class='Opioid analgesic / Mu-receptor agonist',
    n18.mechanism='Activates mu, kappa, delta opioid receptors; CNS and peripheral pain modulation',
    n18.route='Oral/IV/SC';

MERGE (n19:NonChemoDrug {name:'Levothyroxine'})
SET n19.drug_class='Thyroid hormone replacement',
    n19.mechanism='Exogenous T4 supplementation; regulates metabolism via nuclear receptor binding',
    n19.route='Oral', n19.monitoring='TSH monitoring; many drug and food interactions';

MERGE (n20:NonChemoDrug {name:'Dexamethasone'})
SET n20.drug_class='Corticosteroid',
    n20.mechanism='Broad anti-inflammatory via glucocorticoid receptor; also used as anti-emetic premedication',
    n20.route='Oral/IV';

// ── NonChemoDrug → Ailment ────────────────────────────────────
MATCH (n:NonChemoDrug {name:'Warfarin'}),(a:Ailment {name:'Atrial Fibrillation'})     MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Warfarin'}),(a:Ailment {name:'Venous Thromboembolism'}) MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Aspirin'}),(a:Ailment {name:'Atrial Fibrillation'})      MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Ibuprofen'}),(a:Ailment {name:'Chronic Pain'})           MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Metformin'}),(a:Ailment {name:'Type 2 Diabetes'})        MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Omeprazole'}),(a:Ailment {name:'GERD / Peptic Ulcer'})  MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Phenytoin'}),(a:Ailment {name:'Epilepsy'})               MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Carbamazepine'}),(a:Ailment {name:'Epilepsy'})           MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Fluconazole'}),(a:Ailment {name:'Fungal Infection'})     MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Voriconazole'}),(a:Ailment {name:'Fungal Infection'})    MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Ciprofloxacin'}),(a:Ailment {name:'Bacterial Infection'}) MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Co-trimoxazole (TMP-SMX)'}),(a:Ailment {name:'Bacterial Infection'}) MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Atorvastatin'}),(a:Ailment {name:'Hyperlipidaemia'})     MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Sertraline'}),(a:Ailment {name:'Depression / Anxiety'})  MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Amitriptyline'}),(a:Ailment {name:'Depression / Anxiety'}) MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Amitriptyline'}),(a:Ailment {name:'Chronic Pain'})       MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Amlodipine'}),(a:Ailment {name:'Hypertension'})          MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Furosemide'}),(a:Ailment {name:'Hypertension'})          MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Allopurinol'}),(a:Ailment {name:'Chronic Pain'})         MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Morphine'}),(a:Ailment {name:'Chronic Pain'})            MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Levothyroxine'}),(a:Ailment {name:'Hypothyroidism'})     MERGE (n)-[:TREATS]->(a);
MATCH (n:NonChemoDrug {name:'Dexamethasone'}),(a:Ailment {name:'GERD / Peptic Ulcer'}) MERGE (n)-[:MAY_WORSEN]->(a);

// ════════════════════════════════════════════════════════════
// SECTION C – DRUG INTERACTION NODES  (25 nodes)
// id format: DI_<nonChemo>_<ChemoDrug>
// ════════════════════════════════════════════════════════════

// ── Warfarin interactions ─────────────────────────────────────
MERGE (i1:DrugInteraction {id:'DI_WAR_MTX'})
SET i1.description='Warfarin + Methotrexate: MTX displaces warfarin from albumin and inhibits folate metabolism; significantly increases bleeding risk',
    i1.severity='Severe',
    i1.mechanism='Protein binding displacement + antifolate synergism increases warfarin effect',
    i1.clinical_action='Avoid combination; if unavoidable, reduce warfarin dose and monitor INR 2-3x/week; switch to LMWH',
    i1.eating_relevance='Patients on warfarin should maintain consistent vitamin K intake; avoid large changes in leafy greens';

MERGE (i2:DrugInteraction {id:'DI_WAR_CAP'})
SET i2.description='Warfarin + Capecitabine: Capecitabine inhibits CYP2C9, dramatically elevating warfarin exposure and bleeding risk (INR can exceed 10)',
    i2.severity='Severe',
    i2.mechanism='CYP2C9 inhibition by 5-FU metabolites increases S-warfarin AUC by up to 50%',
    i2.clinical_action='Avoid; substitute LMWH. If unavoidable, reduce warfarin 30-50%; check INR twice weekly',
    i2.eating_relevance='Consistent vitamin K diet critical; diarrhoea from capecitabine reduces vitamin K absorption causing INR fluctuation';

MERGE (i3:DrugInteraction {id:'DI_WAR_CISP'})
SET i3.description='Warfarin + Cisplatin: Cisplatin-induced nausea and reduced oral intake alter vitamin K absorption, destabilising INR',
    i3.severity='Moderate',
    i3.mechanism='Nutritional disruption and thrombocytopenia compound anticoagulation risk',
    i3.clinical_action='Monitor INR at each chemo cycle; supplement nutrition; switch to LMWH if INR unstable';

// ── NSAID interactions ────────────────────────────────────────
MERGE (i4:DrugInteraction {id:'DI_IBU_CISP'})
SET i4.description='Ibuprofen + Cisplatin: NSAIDs reduce renal prostaglandins, potentiating cisplatin nephrotoxicity; risk of AKI',
    i4.severity='Severe',
    i4.mechanism='NSAID-mediated afferent arteriole constriction reduces GFR; accumulates nephrotoxic platinum',
    i4.clinical_action='Avoid NSAIDs with cisplatin; use paracetamol/opioids for pain; maintain IV hydration',
    i4.eating_relevance='Adequate oral fluid intake before and after cisplatin is protective; NSAIDs also worsen GI mucositis';

MERGE (i5:DrugInteraction {id:'DI_IBU_MTX'})
SET i5.description='Ibuprofen + Methotrexate: NSAIDs reduce renal tubular secretion of MTX, causing severe MTX toxicity (mucositis, myelosuppression)',
    i5.severity='Severe',
    i5.mechanism='Competitive inhibition of renal OAT3 transporter reduces MTX clearance',
    i5.clinical_action='Absolute contraindication with high-dose MTX; avoid even low-dose NSAIDs; monitor MTX levels',
    i5.eating_relevance='MTX toxicity causes severe mucositis worsening nutritional status; IV nutrition may be required';

MERGE (i6:DrugInteraction {id:'DI_ASP_CAP'})
SET i6.description='Aspirin + Capecitabine: Low-dose aspirin can increase 5-FU plasma levels and GI toxicity (diarrhoea, mucositis) via platelet inhibition',
    i6.severity='Moderate',
    i6.mechanism='Platelet inhibition reduces 5-FU catabolism; increased mucosal prostaglandin inhibition worsens GI damage',
    i6.clinical_action='Use with caution; monitor for GI toxicity; consider alternatives if bleeding risk high';

// ── Antifungal interactions ───────────────────────────────────
MERGE (i7:DrugInteraction {id:'DI_VORI_VIN'})
SET i7.description='Voriconazole + Vincristine: Voriconazole inhibits CYP3A4, dramatically increasing vincristine exposure; risk of severe neurotoxicity',
    i7.severity='Severe',
    i7.mechanism='CYP3A4 + P-gp inhibition by azoles increases vincristine AUC 3-4 fold',
    i7.clinical_action='Contraindicated: do not use voriconazole while patient is on vincristine. Use caspofungin or micafungin instead',
    i7.eating_relevance='Severe neurotoxicity including ileus/constipation from vincristine toxicity profoundly impacts eating';

MERGE (i8:DrugInteraction {id:'DI_FLUCO_CYCL'})
SET i8.description='Fluconazole + Cyclophosphamide: Fluconazole inhibits CYP2C9 and CYP3A4, reducing metabolic activation of cyclophosphamide, reducing efficacy and potentially altering toxic metabolite profile',
    i8.severity='Moderate',
    i8.mechanism='CYP inhibition reduces 4-hydroxycyclophosphamide formation (active metabolite)',
    i8.clinical_action='Monitor efficacy; consider alternative antifungal (caspofungin) or adjust cyclophosphamide dose';

MERGE (i9:DrugInteraction {id:'DI_VORI_ETOPOS'})
SET i9.description='Voriconazole + Etoposide: Voriconazole inhibits CYP3A4-mediated etoposide clearance, increasing etoposide exposure and myelosuppression risk',
    i9.severity='Severe',
    i9.mechanism='CYP3A4 inhibition reduces etoposide clearance',
    i9.clinical_action='Avoid combination; if necessary reduce etoposide dose by 25-50% with close haematological monitoring';

// ── Antiepileptic interactions ────────────────────────────────
MERGE (i10:DrugInteraction {id:'DI_CBZ_PAC'})
SET i10.description='Carbamazepine + Paclitaxel: Carbamazepine induces CYP3A4, significantly reducing paclitaxel plasma levels and therapeutic efficacy',
    i10.severity='Severe',
    i10.mechanism='Strong CYP3A4 induction increases paclitaxel metabolism 2-3 fold',
    i10.clinical_action='Avoid; switch to non-enzyme-inducing antiepileptic (levetiracetam, lamotrigine, lacosamide)',
    i10.eating_relevance='Reduced paclitaxel efficacy may require dose escalation increasing GI side effects';

MERGE (i11:DrugInteraction {id:'DI_CBZ_ETOPOS'})
SET i11.description='Carbamazepine + Etoposide: CYP3A4 induction by carbamazepine reduces etoposide AUC by up to 50%, severely compromising efficacy',
    i11.severity='Severe',
    i11.mechanism='CYP3A4 induction and P-gp upregulation reduces etoposide bioavailability',
    i11.clinical_action='Switch AED; consider increasing etoposide dose if switching not possible with TDM';

MERGE (i12:DrugInteraction {id:'DI_PHENY_MTX'})
SET i12.description='Phenytoin + Methotrexate: Methotrexate may reduce phenytoin absorption; phenytoin levels can drop causing seizure breakthrough',
    i12.severity='Moderate',
    i12.mechanism='Reduced GI absorption of phenytoin; possible protein binding competition',
    i12.clinical_action='Monitor phenytoin levels weekly during MTX; separate doses; consider IV phenytoin or switch AED',
    i12.eating_relevance='Phenytoin absorption affected by tube feeds; hold feeds around phenytoin dose by 2 hours';

// ── Antibiotic interactions ───────────────────────────────────
MERGE (i13:DrugInteraction {id:'DI_CIPRO_MTX'})
SET i13.description='Ciprofloxacin + Methotrexate: Ciprofloxacin inhibits OAT1/OAT3 renal MTX secretion, increasing MTX levels and toxicity',
    i13.severity='Severe',
    i13.mechanism='Competitive inhibition of organic anion transporters in renal tubules',
    i13.clinical_action='Avoid concurrent use with HD-MTX; delay next MTX cycle if infection requires ciprofloxacin; monitor MTX levels';

MERGE (i14:DrugInteraction {id:'DI_COTRIMOX_MTX'})
SET i14.description='Co-trimoxazole + Methotrexate: Additive antifolate effect causing severe myelosuppression and mucositis',
    i14.severity='Severe',
    i14.mechanism='Trimethoprim and MTX both inhibit DHFR; additive folate antagonism',
    i14.clinical_action='Contraindicated during MTX therapy; use targeted prophylaxis (atovaquone) if PCP prophylaxis required',
    i14.eating_relevance='Combined mucositis is severe, requiring liquid nutrition; parenteral nutrition may be necessary';

MERGE (i15:DrugInteraction {id:'DI_COTRIMOX_CYCL'})
SET i15.description='Co-trimoxazole + Cyclophosphamide: SMX inhibits CYP2C9 reducing cyclophosphamide bioactivation; TMP causes additive bone marrow suppression',
    i15.severity='Moderate',
    i15.mechanism='CYP2C9 inhibition reduces 4-OH-cyclophosphamide production; myelosuppression additive effect',
    i15.clinical_action='Monitor FBC; consider dose adjustment if neutropenia severe';

// ── Statin interactions ───────────────────────────────────────
MERGE (i16:DrugInteraction {id:'DI_ATORVA_PAC'})
SET i16.description='Atorvastatin + Paclitaxel: Both are CYP3A4 substrates; moderate pharmacokinetic interaction; paclitaxel may increase statin exposure and myopathy risk',
    i16.severity='Mild-Moderate',
    i16.mechanism='CYP3A4 competition; possible P-gp interaction',
    i16.clinical_action='Use lowest effective statin dose; switch to rosuvastatin (not CYP3A4); monitor CPK',
    i16.eating_relevance='Myopathy from statins can reduce appetite and activity; grapefruit must be avoided with both drugs';

// ── SSRI / TCA interactions ───────────────────────────────────
MERGE (i17:DrugInteraction {id:'DI_SERT_TAMOX'})
SET i17.description='Sertraline + Tamoxifen (breast cancer endocrine therapy): Sertraline inhibits CYP2D6, reducing conversion of tamoxifen to active metabolite endoxifen',
    i17.severity='Moderate-Severe',
    i17.mechanism='CYP2D6 inhibition reduces endoxifen levels by ~60%; reduces tamoxifen efficacy',
    i17.clinical_action='Prefer venlafaxine or escitalopram (weak CYP2D6 inhibitors) for depression in tamoxifen patients',
    i17.eating_relevance='SSRIs improve mood and thus eating behaviour; drug selection critical to maintain tamoxifen efficacy';

MERGE (i18:DrugInteraction {id:'DI_AMITRIPT_CISP'})
SET i18.description='Amitriptyline + Cisplatin: Anticholinergic effects of amitriptyline compound cisplatin-induced GI slowing; constipation and ileus risk',
    i18.severity='Moderate',
    i18.mechanism='Additive anticholinergic effects reduce GI motility',
    i18.clinical_action='Monitor bowel function; prophylactic laxatives; consider duloxetine instead for neuropathic pain',
    i18.eating_relevance='Severe constipation causes abdominal pain and nausea, markedly reducing food intake';

// ── Diuretic / cardiovascular interactions ────────────────────
MERGE (i19:DrugInteraction {id:'DI_FURO_CISP'})
SET i19.description='Furosemide + Cisplatin: Loop diuretics increase cisplatin renal tubular concentration and ototoxicity + nephrotoxicity risk',
    i19.severity='Severe',
    i19.mechanism='Furosemide reduces renal blood flow and increases cisplatin residence in proximal tubules',
    i19.clinical_action='Avoid furosemide during cisplatin; use thiazides if diuresis needed; mannitol-based diuresis preferred',
    i19.eating_relevance='Diuretic overuse causes electrolyte depletion (Na, K, Mg) worsening nausea and muscle weakness';

MERGE (i20:DrugInteraction {id:'DI_AMLO_CBZ'})
SET i20.description='Amlodipine + Carbamazepine: Carbamazepine induces CYP3A4, reducing amlodipine levels and blood pressure control',
    i20.severity='Moderate',
    i20.mechanism='CYP3A4 induction increases amlodipine clearance ~50%',
    i20.clinical_action='Monitor blood pressure; may need amlodipine dose increase; or switch to non-CYP substrate CCB';

// ── Allopurinol interaction (important in leukemia) ──────────
MERGE (i21:DrugInteraction {id:'DI_ALLOPUR_AZATH'})
SET i21.description='Allopurinol + Azathioprine/Mercaptopurine: Allopurinol inhibits xanthine oxidase, the main enzyme metabolising 6-MP/azathioprine – SEVERE toxicity',
    i21.severity='Severe',
    i21.mechanism='Xanthine oxidase inhibition blocks 6-MP inactivation, increasing active thiopurine metabolites 4-fold',
    i21.clinical_action='CONTRAINDICATED: If allopurinol required (tumour lysis prevention), reduce 6-MP dose by 75%. Consider rasburicase instead',
    i21.eating_relevance='Severe bone marrow suppression causes infection, mucositis, and profound nutritional compromise';

// ── Corticosteroid interaction ────────────────────────────────
MERGE (i22:DrugInteraction {id:'DI_DEXA_INSULIN'})
SET i22.description='Dexamethasone (chemo premedication) causes hyperglycaemia in diabetic patients, requiring insulin dose adjustment',
    i22.severity='Moderate',
    i22.mechanism='Glucocorticoid receptor activation increases hepatic gluconeogenesis and reduces peripheral insulin sensitivity',
    i22.clinical_action='Monitor glucose pre- and post-dexamethasone; sliding scale insulin; dietitian-guided carb management',
    i22.eating_relevance='Dexamethasone increases appetite and causes cravings; diabetes dietary management becomes complex during chemo';

// ── PPI interaction ───────────────────────────────────────────
MERGE (i23:DrugInteraction {id:'DI_OMEP_ERLOT'})
SET i23.description='Omeprazole + Erlotinib (EGFR TKI for NSCLC): PPIs raise gastric pH, reducing erlotinib absorption by up to 46%',
    i23.severity='Moderate',
    i23.mechanism='Erlotinib requires acidic gastric environment for optimal dissolution; pH >3 markedly reduces solubility',
    i23.clinical_action='Avoid PPIs with erlotinib; if acid suppression needed, use short-acting antacid 4h before erlotinib or H2 blocker',
    i23.eating_relevance='Take erlotinib on empty stomach or with a small low-fat meal to maximise absorption';

// ── Opioid interaction ────────────────────────────────────────
MERGE (i24:DrugInteraction {id:'DI_MORPH_VINCR'})
SET i24.description='Morphine + Vincristine: Both cause constipation; additive risk of severe constipation, paralytic ileus',
    i24.severity='Moderate',
    i24.mechanism='Opioid receptor activation + autonomic neuropathy from vincristine create severe GI dysmotility',
    i24.clinical_action='Prophylactic bowel regimen is mandatory; macrogol + stimulant laxative; consider methylnaltrexone',
    i24.eating_relevance='Paralytic ileus requires bowel rest and IV nutrition; pain management without opioids preferred in vincristine patients';

// ── Thyroid hormone interaction ───────────────────────────────
MERGE (i25:DrugInteraction {id:'DI_LEVOTHYR_NIVOLUMAB'})
SET i25.description='Levothyroxine dose requirements may change in cancer patients receiving checkpoint inhibitors (pembrolizumab, nivolumab) causing immune thyroiditis (hypothyroidism or thyrotoxicosis)',
    i25.severity='Moderate',
    i25.mechanism='Anti-PD-1 induced immune destruction of thyroid gland alters levothyroxine requirements',
    i25.clinical_action='Monitor TSH every 4-6 weeks during immunotherapy; adjust levothyroxine dose; endocrinology referral',
    i25.eating_relevance='Hypothyroidism from irAE causes fatigue, constipation, weight gain; hyperthyroid causes diarrhoea, weight loss';

// ════════════════════════════════════════════════════════════
// SECTION D – LINK INTERACTIONS TO DRUGS
// ════════════════════════════════════════════════════════════

// Warfarin interactions
MATCH (n:NonChemoDrug {name:'Warfarin'}),(c:ChemoDrug {name:'Methotrexate'}),(i:DrugInteraction {id:'DI_WAR_MTX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Warfarin'}),(c:ChemoDrug {name:'Capecitabine'}),(i:DrugInteraction {id:'DI_WAR_CAP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Warfarin'}),(c:ChemoDrug {name:'Cisplatin'}),(i:DrugInteraction {id:'DI_WAR_CISP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// NSAID interactions
MATCH (n:NonChemoDrug {name:'Ibuprofen'}),(c:ChemoDrug {name:'Cisplatin'}),(i:DrugInteraction {id:'DI_IBU_CISP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Ibuprofen'}),(c:ChemoDrug {name:'Methotrexate'}),(i:DrugInteraction {id:'DI_IBU_MTX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Aspirin'}),(c:ChemoDrug {name:'Capecitabine'}),(i:DrugInteraction {id:'DI_ASP_CAP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Antifungal interactions
MATCH (n:NonChemoDrug {name:'Voriconazole'}),(c:ChemoDrug {name:'Vincristine'}),(i:DrugInteraction {id:'DI_VORI_VIN'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Fluconazole'}),(c:ChemoDrug {name:'Cyclophosphamide'}),(i:DrugInteraction {id:'DI_FLUCO_CYCL'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Voriconazole'}),(c:ChemoDrug {name:'Etoposide'}),(i:DrugInteraction {id:'DI_VORI_ETOPOS'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Antiepileptic interactions
MATCH (n:NonChemoDrug {name:'Carbamazepine'}),(c:ChemoDrug {name:'Paclitaxel'}),(i:DrugInteraction {id:'DI_CBZ_PAC'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Carbamazepine'}),(c:ChemoDrug {name:'Etoposide'}),(i:DrugInteraction {id:'DI_CBZ_ETOPOS'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Phenytoin'}),(c:ChemoDrug {name:'Methotrexate'}),(i:DrugInteraction {id:'DI_PHENY_MTX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Antibiotic interactions
MATCH (n:NonChemoDrug {name:'Ciprofloxacin'}),(c:ChemoDrug {name:'Methotrexate'}),(i:DrugInteraction {id:'DI_CIPRO_MTX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Co-trimoxazole (TMP-SMX)'}),(c:ChemoDrug {name:'Methotrexate'}),(i:DrugInteraction {id:'DI_COTRIMOX_MTX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Co-trimoxazole (TMP-SMX)'}),(c:ChemoDrug {name:'Cyclophosphamide'}),(i:DrugInteraction {id:'DI_COTRIMOX_CYCL'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Statin / SSRI / TCA interactions
MATCH (n:NonChemoDrug {name:'Atorvastatin'}),(c:ChemoDrug {name:'Paclitaxel'}),(i:DrugInteraction {id:'DI_ATORVA_PAC'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Sertraline'}),(c:ChemoDrug {name:'Paclitaxel'}),(i:DrugInteraction {id:'DI_SERT_TAMOX'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Amitriptyline'}),(c:ChemoDrug {name:'Cisplatin'}),(i:DrugInteraction {id:'DI_AMITRIPT_CISP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Cardiovascular interactions
MATCH (n:NonChemoDrug {name:'Furosemide'}),(c:ChemoDrug {name:'Cisplatin'}),(i:DrugInteraction {id:'DI_FURO_CISP'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Amlodipine'}),(c:ChemoDrug {name:'Paclitaxel'}),(i:DrugInteraction {id:'DI_AMLO_CBZ'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c);

// Allopurinol + thiopurine
MATCH (n:NonChemoDrug {name:'Allopurinol'}),(c:ChemoDrug {name:'Mercaptopurine'}),(i:DrugInteraction {id:'DI_ALLOPUR_AZATH'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Dexamethasone + diabetes
MATCH (n:NonChemoDrug {name:'Dexamethasone'}),(c:ChemoDrug {name:'Paclitaxel'}),(i:DrugInteraction {id:'DI_DEXA_INSULIN'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Dexamethasone'}),(c:ChemoDrug {name:'Docetaxel'}),(i:DrugInteraction {id:'DI_DEXA_INSULIN'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c);

// PPI + EGFR TKI
MATCH (n:NonChemoDrug {name:'Omeprazole'}),(c:ChemoDrug {name:'Gemcitabine'}),(i:DrugInteraction {id:'DI_OMEP_ERLOT'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i);

// Morphine + vincristine
MATCH (n:NonChemoDrug {name:'Morphine'}),(c:ChemoDrug {name:'Vincristine'}),(i:DrugInteraction {id:'DI_MORPH_VINCR'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);

// Levothyroxine + immunotherapy
MATCH (n:NonChemoDrug {name:'Levothyroxine'}),(c:ChemoDrug {name:'Nivolumab'}),(i:DrugInteraction {id:'DI_LEVOTHYR_NIVOLUMAB'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c) MERGE (n)-[:DESCRIBED_BY]->(i) MERGE (c)-[:DESCRIBED_BY]->(i);
MATCH (n:NonChemoDrug {name:'Levothyroxine'}),(c:ChemoDrug {name:'Pembrolizumab'}),(i:DrugInteraction {id:'DI_LEVOTHYR_NIVOLUMAB'})
MERGE (n)-[:HAS_INTERACTION_WITH]->(c);

// ════════════════════════════════════════════════════════════
// SECTION E – LINK INTERACTIONS TO EATING EFFECTS
// (Drug interactions that specifically impact nutrition/eating)
// ════════════════════════════════════════════════════════════

MATCH (i:DrugInteraction {id:'DI_WAR_CAP'}),(e:EatingAdverseEffect {name:'Diarrhoea'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Capecitabine diarrhoea reduces vitamin K absorption, destabilising warfarin INR'}]->(e);

MATCH (i:DrugInteraction {id:'DI_IBU_MTX'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'MTX toxicity from NSAID interaction worsens mucositis severity and duration'}]->(e);

MATCH (i:DrugInteraction {id:'DI_VORI_VIN'}),(e:EatingAdverseEffect {name:'Constipation'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Vincristine toxicity from azole interaction causes severe ileus and constipation'}]->(e);

MATCH (i:DrugInteraction {id:'DI_MORPH_VINCR'}),(e:EatingAdverseEffect {name:'Constipation'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Double constipation risk from opioid + vincristine neurotoxicity; paralytic ileus risk'}]->(e);

MATCH (i:DrugInteraction {id:'DI_AMITRIPT_CISP'}),(e:EatingAdverseEffect {name:'Constipation'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Anticholinergic + platinum autonomic neuropathy compounds GI dysmotility'}]->(e);

MATCH (i:DrugInteraction {id:'DI_DEXA_INSULIN'}),(e:EatingAdverseEffect {name:'Weight Gain'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Dexamethasone-induced hyperglycaemia and appetite increase promotes weight gain'}]->(e);

MATCH (i:DrugInteraction {id:'DI_COTRIMOX_MTX'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Combined antifolate effect dramatically worsens mucositis; parenteral nutrition often required'}]->(e);

MATCH (i:DrugInteraction {id:'DI_ALLOPUR_AZATH'}),(e:EatingAdverseEffect {name:'Sore Mouth (Mucositis)'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Thiopurine accumulation causes severe mucositis and myelosuppression'}]->(e);

MATCH (i:DrugInteraction {id:'DI_FURO_CISP'}),(e:EatingAdverseEffect {name:'Nausea'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Electrolyte disturbances from nephrotoxicity + diuresis worsen nausea and vomiting'}]->(e);

MATCH (i:DrugInteraction {id:'DI_LEVOTHYR_NIVOLUMAB'}),(e:EatingAdverseEffect {name:'Weight Loss'})
MERGE (i)-[:COMPOUNDS_EATING_EFFECT {note:'Hyperthyroid phase of immune thyroiditis causes weight loss, diarrhoea; hypothyroid causes weight gain'}]->(e);

// ════════════════════════════════════════════════════════════
// SECTION F – SIDE EFFECT NODES (additional clinical detail)
// ════════════════════════════════════════════════════════════

MERGE (s1:SideEffect {name:'Myelosuppression'})
SET s1.description='Bone marrow suppression causing neutropenia, anaemia, thrombocytopenia',
    s1.nutrition_impact='Infection risk limits food preparation; anaemia worsens fatigue and reduces appetite',
    s1.management='G-CSF, dose reduction, dietary precautions (food safety critical in neutropenia)';

MERGE (s2:SideEffect {name:'Peripheral Neuropathy'})
SET s2.description='Damage to peripheral nerves causing numbness, tingling, weakness in hands and feet',
    s2.nutrition_impact='Difficulty handling utensils, reduced ability to prepare food',
    s2.management='Dose reduction; duloxetine; dietary modification for fine motor difficulty';

MERGE (s3:SideEffect {name:'Cardiotoxicity'})
SET s3.description='Cardiac damage from anthracyclines (cardiomyopathy) or checkpoint inhibitors (myocarditis)',
    s3.nutrition_impact='Heart failure causes reduced appetite, gut oedema, malabsorption',
    s3.management='LVEF monitoring; sodium/fluid restriction diet; cardiology review';

MERGE (s4:SideEffect {name:'Nephrotoxicity'})
SET s4.description='Renal tubular damage primarily from cisplatin, ifosfamide',
    s4.nutrition_impact='Electrolyte wasting (Mg, K, Na) worsens nausea; protein restriction if CKD develops',
    s4.management='IV hydration; magnesium supplementation; nephrology input for chronic damage';

MERGE (s5:SideEffect {name:'Hepatotoxicity'})
SET s5.description='Drug-induced liver injury from methotrexate, L-asparaginase, checkpoint inhibitors',
    s5.nutrition_impact='Reduced albumin synthesis, coagulopathy; low-protein diet in severe hepatic failure',
    s5.management='LFT monitoring; NAC in acute toxicity; dietary protein moderation';

// Link side effects to drugs
MATCH (d:ChemoDrug {name:'Doxorubicin'}),(s:SideEffect {name:'Cardiotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Cisplatin'}),(s:SideEffect {name:'Nephrotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Ifosfamide'}),(s:SideEffect {name:'Nephrotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Paclitaxel'}),(s:SideEffect {name:'Peripheral Neuropathy'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Vincristine'}),(s:SideEffect {name:'Peripheral Neuropathy'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Methotrexate'}),(s:SideEffect {name:'Hepatotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'L-Asparaginase'}),(s:SideEffect {name:'Hepatotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Cytarabine'}),(s:SideEffect {name:'Myelosuppression'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Daunorubicin'}),(s:SideEffect {name:'Myelosuppression'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Etoposide'}),(s:SideEffect {name:'Myelosuppression'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Ipilimumab'}),(s:SideEffect {name:'Hepatotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);
MATCH (d:ChemoDrug {name:'Nivolumab'}),(s:SideEffect {name:'Cardiotoxicity'}) MERGE (d)-[:MAY_CAUSE]->(s);

// Link side effects to eating effects
MATCH (s:SideEffect {name:'Nephrotoxicity'}),(e:EatingAdverseEffect {name:'Nausea'}) MERGE (s)-[:LEADS_TO_EATING_EFFECT]->(e);
MATCH (s:SideEffect {name:'Hepatotoxicity'}),(e:EatingAdverseEffect {name:'Appetite Loss'}) MERGE (s)-[:LEADS_TO_EATING_EFFECT]->(e);
MATCH (s:SideEffect {name:'Peripheral Neuropathy'}),(e:EatingAdverseEffect {name:'Constipation'}) MERGE (s)-[:LEADS_TO_EATING_EFFECT]->(e);
MATCH (s:SideEffect {name:'Cardiotoxicity'}),(e:EatingAdverseEffect {name:'Weight Loss'}) MERGE (s)-[:LEADS_TO_EATING_EFFECT]->(e);
MATCH (s:SideEffect {name:'Myelosuppression'}),(e:EatingAdverseEffect {name:'Appetite Loss'}) MERGE (s)-[:LEADS_TO_EATING_EFFECT]->(e);
