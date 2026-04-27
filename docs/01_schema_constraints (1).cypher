// ============================================================
// 01_schema_constraints.cypher
// GraphRAG: Cancer-Chemo-Nutrition Knowledge Graph
// Neo4j / AuraDB  — run FIRST before any data load
// ============================================================

// ── Uniqueness constraints ────────────────────────────────────
CREATE CONSTRAINT cancer_name_unique         IF NOT EXISTS FOR (c:Cancer)             REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT chemo_drug_name_unique     IF NOT EXISTS FOR (d:ChemoDrug)          REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT eating_effect_name_unique  IF NOT EXISTS FOR (e:EatingAdverseEffect) REQUIRE e.name IS UNIQUE;
CREATE CONSTRAINT food_item_name_unique      IF NOT EXISTS FOR (f:FoodItem)            REQUIRE f.name IS UNIQUE;
CREATE CONSTRAINT guideline_id_unique        IF NOT EXISTS FOR (g:NutritionGuideline)  REQUIRE g.id   IS UNIQUE;
CREATE CONSTRAINT protocol_name_unique       IF NOT EXISTS FOR (p:TreatmentProtocol)   REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT biomarker_name_unique      IF NOT EXISTS FOR (b:Biomarker)           REQUIRE b.name IS UNIQUE;
CREATE CONSTRAINT non_chemo_name_unique      IF NOT EXISTS FOR (n:NonChemoDrug)        REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT ailment_name_unique        IF NOT EXISTS FOR (a:Ailment)             REQUIRE a.name IS UNIQUE;
CREATE CONSTRAINT interaction_id_unique      IF NOT EXISTS FOR (i:DrugInteraction)     REQUIRE i.id   IS UNIQUE;
CREATE CONSTRAINT nutrient_name_unique       IF NOT EXISTS FOR (n:Nutrient)            REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT side_effect_name_unique    IF NOT EXISTS FOR (s:SideEffect)          REQUIRE s.name IS UNIQUE;

// ── Full-text search indexes (for vector-hybrid retrieval) ────
CREATE FULLTEXT INDEX cancer_text_index       IF NOT EXISTS FOR (c:Cancer)             ON EACH [c.name, c.description, c.subtype];
CREATE FULLTEXT INDEX chemo_text_index        IF NOT EXISTS FOR (d:ChemoDrug)          ON EACH [d.name, d.mechanism, d.drug_class, d.notes];
CREATE FULLTEXT INDEX eating_effect_text_index IF NOT EXISTS FOR (e:EatingAdverseEffect) ON EACH [e.name, e.description, e.management_tip];
CREATE FULLTEXT INDEX food_item_text_index    IF NOT EXISTS FOR (f:FoodItem)            ON EACH [f.name, f.category, f.notes];
CREATE FULLTEXT INDEX guideline_text_index    IF NOT EXISTS FOR (g:NutritionGuideline)  ON EACH [g.text, g.source];
CREATE FULLTEXT INDEX non_chemo_text_index    IF NOT EXISTS FOR (n:NonChemoDrug)        ON EACH [n.name, n.drug_class, n.mechanism];
CREATE FULLTEXT INDEX interaction_text_index  IF NOT EXISTS FOR (i:DrugInteraction)     ON EACH [i.description, i.severity, i.mechanism];

// ── Vector index (embedding column for GraphRAG) ─────────────
// Requires Neo4j 5.11+ or AuraDB Enterprise
CREATE VECTOR INDEX chemo_vector_index        IF NOT EXISTS FOR (d:ChemoDrug)          ON d.embedding
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX cancer_vector_index       IF NOT EXISTS FOR (c:Cancer)             ON c.embedding
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX eating_effect_vector_index IF NOT EXISTS FOR (e:EatingAdverseEffect) ON e.embedding
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX non_chemo_vector_index    IF NOT EXISTS FOR (n:NonChemoDrug)       ON n.embedding
    OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}};
