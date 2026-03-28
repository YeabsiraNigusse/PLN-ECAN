;; ============================================================
;;    ECAN + PLN Integration Experiment (Incident-Atom Space)
;; ============================================================
;; Uses the main branch metta-attention architecture with the
;; real incident-atom space (`incident_final.pl`) for importance
;; diffusion, combined with biology taxonomy data.
;;
;; Pipeline:
;;   1. Load wordnet + biology KB into TypeSpace for STV
;;   2. Load `incident_final.pl` (with appended biology) into &incidentSpace
;;   3. Read article → stimulate atoms → ECAN diffuses via incident graph
;;   4. getAfAtoms → get important atoms (links and nodes)
;;   5. Bridge: match AF link-atoms against &biology KB → PLN Sentences
;;   6. PLN.Query deduces new knowledge from filtered premises
;; ============================================================

;; --- PeTTa libs (for static-import and PLN) ---
!(import! &self ../PeTTa/lib/lib_import)
!(import! &self ../PeTTa/lib/lib_spaces)
!(import! &self ../PeTTa/lib/lib_pln)

;; --- Load real incident data (Wordnet + appended Biology neighbor graph) ---
!(static-import! &incidentSpace ../metta-attention/experiments/data/incident_final)

;; --- Load wordnet for STV ---
!(static-import! &wordnet ../metta-attention/experiments/data/wordnet_stv_clean)

;; --- Load article sentences and biology KB ---
!(bind! &insect (new-space))
!(bind! &biology (new-space))
!(import! &biology data/biology_kb)
!(import! &insect article)

;; --- ECAN modules (main branch) ---
!(import! &self ../metta-attention/attention/AttentionParam)
!(import! &self ../metta-attention/attention/baching)
!(import! &self ../metta-attention/attention-bank/utilities/helper-functions)
!(import! &self ../metta-attention/attention-bank/utilities/recentVal)
!(import! &self ../metta-attention/attention-bank/attention-value/getter-and-setter)
!(import! &self ../metta-attention/attention-bank/bank/attention-bank)
!(import! &self ../metta-attention/attention-bank/bank/attentional-focus/attentional-focus)
!(import! &self ../metta-attention/attention-bank/bank/importance-index/importance-index)
!(import! &self ../metta-attention/attention/Neighbors)
!(import! &self ../metta-attention/attention-bank/bank/stochastic-importance-diffusion/stochastic-importance-diffusion)
!(import! &self ../metta-attention/attention/RentCollectionAgent/RentCollectionBaseAgent/RentCollectionBaseAgent)
!(import! &self ../metta-attention/attention/RentCollectionAgent/WARentCollectionAgent/WARentCollectionAgent)
!(import! &self ../metta-attention/attention/RentCollectionAgent/AFRentCollectionAgent/AFRentCollectionAgent)
!(import! &self ../metta-attention/attention/ImportanceDiffusionAgent/ImportanceDiffusionBase/ImportanceDiffusionBase)
!(import! &self ../metta-attention/attention/ImportanceDiffusionAgent/WAImportanceDiffusionAgent/WAImportanceDiffusionAgent)
!(import! &self ../metta-attention/attention/ImportanceDiffusionAgent/AFImportanceDiffusionAgent/AFImportanceDiffusionAgent)
!(import! &self ../metta-attention/attention/HebbianUpdatingAgent/HebbianUpdatingAgent)
!(import! &self ../metta-attention/attention/HebbianCreationAgent/HebbianCreationAgent)
!(import! &self ../metta-attention/attention/runner-Def)
!(import! &self "../metta-attention/attention/ImportanceDiffusionAgent/getelement.py")

;; ============================================================
;; PHASE 1 — LOAD KB INTO TYPESPACE & CONFIGURE ECAN
;; ============================================================

;; Load wordnet atoms into TypeSpace (gives STV to nodes)
!(collapse (match &wordnet ($atom ($mean $conf)) (setStv $atom ($mean $conf) true)))
;; Load biology KB atoms into TypeSpace
!(collapse (match &biology ($atom ($mean $conf)) (setStv $atom ($mean $conf) true)))

;; Configure ECAN params
!(updateAttentionParam MAX_AF_SIZE 50.0)
!(updateAttentionParam TARGET_STI 1000.0)
!(updateAttentionParam TARGET_LTI 1000.0)
!(updateAttentionParam STI_FUNDS_BUFFER 1000.0)
!(updateAttentionParam LTI_FUNDS_BUFFER 1000.0)
!(updateAttentionParam FUNDS_STI 100000.0)
!(updateAttentionParam FUNDS_LTI 100000.0)
!(updateAttentionParam TOPK 0.5)

;; ============================================================
;; PHASE 2 — ECAN ATTENTION ALLOCATION (Article Reading)
;; ============================================================
;; Simulate reading an article about insects sentence by sentence.
;; Stimulate atoms from the sentences and run ECAN agents in batches.

(= (batch) 5)
(= (reader $expr $counter)
    (if (== $expr ())
        (empty)
        (let* 
            (
                ($now (car-atom $expr))
                ($rest (cdr-atom $expr))
            )
            ( 
                (stimulate $now 100)
                (println! (stimulating atom --> $now))
                (if (or (== (% $counter (batch)) 0) (== $rest ()))
                   (collapse (test-superpose &incidentSpace)) 
                   (println! (agent not called))
                )
                (reader $rest (+ $counter 1)))
         )
    )    
)

;; Start reading the insect sentences
!(reader (match &insect (biologySent $x) $x) 0)

;; ============================================================
;; PHASE 3 — BRIDGE: ECAN → PLN
;; ============================================================
;; After ECAN runs, the AF contains:
;;   - bare word nodes (ant, insect, arthropod, ...)
;;   - link tuples that received STI via diffusion:
;;     (inheritancelink ant insect), (inheritancelink insect arthropod), ...
;;
;; We match each AF atom against our biology KB.
;; If the AF atom is a link like (InheritanceLink $a $b),
;; we create a PLN Sentence for it.

!(bind! &pln-kb (new-space))

;; isMember helper (not available in metta-attention by default)
(= (isMember $x $list)
    (if (== $list ())
        False
        (if (== $x (car-atom $list))
            True
            (isMember $x (cdr-atom $list))
        )
    )
)

;; Bridge strategy: instead of destructuring AF link atoms (casing issues),
;; iterate biology KB InheritanceLinks and check if BOTH child and parent 
;; nodes appear in the AF. If so, the relation is "important" and goes to PLN.
!(let* (
    ($afAtoms (getAfAtoms))
    ($y (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
        (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
            (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
            (empty)
        )
    )))
) (empty))

;; Show what PLN received
!(println! "=== PLN KB (filtered premises) ===")
!(collapse (match &pln-kb $x $x))

;; Show what ECAN filtered
!(println! "=== Attentional Focus ===")
!(collapse (let $a (superpose (getAfAtoms)) (if (== (get-metatype $a) Symbol) $a (empty))))

;; STV base rates for concepts (needed by PLN deduction formula)
(= (STV (Concept $any)) (stv 0.1 0.9))

;; ============================================================
;; PHASE 4 — PLN REASONING ON ECAN-FILTERED KNOWLEDGE
;; ============================================================
;; Queries use bare biology words (ant, insect, arthropod, etc.)
;; since that's what the biology KB uses.

;; Query: ant → arthropod (2-hop: ant→insect→arthropod)
!(println! "=== Query: ant -> arthropod (2-hop) ===")
!(PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept arthropod)) 30)

;; Query: bee → arthropod (2-hop: bee→insect→arthropod)
!(println! "=== Query: bee -> arthropod (2-hop) ===")
!(PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept bee) (Concept arthropod)) 30)

;; Query: ant → invertebrate (3-hop: ant→insect→arthropod→invertebrate)
!(println! "=== Query: ant -> invertebrate (3-hop) ===")
!(PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept invertebrate)) 50)

;; Query: ant → animal (4-hop: ant→insect→arthropod→invertebrate→animal)
!(println! "=== Query: ant -> animal (4-hop) ===")
!(PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept animal)) 60)
!(getAfAtoms)