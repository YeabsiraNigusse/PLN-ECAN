;; ============================================================
;;    Vanilla PLN (NO ECAN) — Comparison Baseline
;; ============================================================
;; Loads ALL biology KB InheritanceLinks AND all WordNet isa
;; relations directly into PLN without any ECAN filtering.
;; This demonstrates PLN struggling with combinatorial explosion
;; when given thousands of unfiltered premises.
;; ============================================================

;; --- PeTTa libs ---
!(import! &self ../PeTTa/lib/lib_import)
!(import! &self ../PeTTa/lib/lib_spaces)
!(import! &self ../PeTTa/lib/lib_pln)

;; --- Load biology KB ---
!(bind! &biology (new-space))
!(import! &biology data/biology_kb)

;; --- Load wordnet (noise) ---
!(static-import! &wordnet ../metta-attention/experiments/data/wordnet_stv_clean)

;; ============================================================
;; BUILD PLN KB — ALL relations, no ECAN filter
;; ============================================================

!(bind! &pln-kb-full (new-space))

;; 1. Add ALL biology InheritanceLinks to PLN
!(collapse (match &biology ((InheritanceLink $a $b) ($s $c))
    (add-atom &pln-kb-full (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
))

;; 2. Add ALL wordnet isa relations to PLN (thousands of noisy links!)
!(collapse (match &wordnet ((isa $a $b) ($mean $conf))
    (add-atom &pln-kb-full (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $mean $conf)) ((ev $a $b))))
))

!(println! "=== Total PLN premises loaded ===")

(= (STV (Concept $any)) (stv 0.1 0.9))

;; ============================================================
;; SAME PLN QUERIES (bare words, same as ECAN version)
;; ============================================================

;; Query: ant → arthropod (2-hop: ant→insect→arthropod)
!(println! "=== Query: ant -> arthropod (2-hop) ===")
!(PLN.Query (collapse (match &pln-kb-full $atom $atom)) (Inheritance (Concept ant) (Concept arthropod)) 30)

;; Query: bee → arthropod (2-hop: bee→insect→arthropod)
!(println! "=== Query: bee -> arthropod (2-hop) ===")
!(PLN.Query (collapse (match &pln-kb-full $atom $atom)) (Inheritance (Concept bee) (Concept arthropod)) 30)

;; Query: ant → invertebrate (3-hop: ant→insect→arthropod→invertebrate)
!(println! "=== Query: ant -> invertebrate (3-hop) ===")
!(PLN.Query (collapse (match &pln-kb-full $atom $atom)) (Inheritance (Concept ant) (Concept invertebrate)) 50)

;; Query: ant → animal (4-hop: ant→insect→arthropod→invertebrate→animal)
!(println! "=== Query: ant -> animal (4-hop) ===")
!(PLN.Query (collapse (match &pln-kb-full $atom $atom)) (Inheritance (Concept ant) (Concept animal)) 60)