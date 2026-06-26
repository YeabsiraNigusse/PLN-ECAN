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
!(let $_ (import! &self ../PeTTa/lib/lib_import) (empty))
!(let $_ (import! &self ../PeTTa/lib/lib_spaces) (empty))
!(let $_ (import! &self ../PeTTa/lib/lib_he) (empty))
!(let $_ (import! &self ../PeTTa/lib/lib_pln) (empty))

;; --- Load real incident data (Wordnet + appended Biology neighbor graph) ---
!(let $_ (static-import! &incidentSpace ../metta-attention/experiments/data/incident_final) (empty))

;; --- Load wordnet for STV ---
!(let $_ (static-import! &wordnet ../metta-attention/experiments/data/wordnet_stv_clean) (empty))

;; --- Load article sentences and biology KB ---
!(let $_ (bind! &insect (new-space)) (empty))
!(let $_ (bind! &biology (new-space)) (empty))
!(let $_ (import! &biology data/biology_kb) (empty))
!(let $_ (import! &insect article) (empty))

;; --- ECAN modules (main branch) ---
!(let $_ (import! &self ../metta-attention/attention/AttentionParam) (empty))
!(let $_ (import! &self ../metta-attention/attention/baching) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/utilities/helper-functions) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/utilities/recentVal) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/attention-value/getter-and-setter) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/bank/attention-bank) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/bank/attentional-focus/attentional-focus) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/bank/importance-index/importance-index) (empty))
!(let $_ (import! &self ../metta-attention/attention/Neighbors) (empty))
!(let $_ (import! &self ../metta-attention/attention-bank/bank/stochastic-importance-diffusion/stochastic-importance-diffusion) (empty))
!(let $_ (import! &self ../metta-attention/attention/RentCollectionAgent/RentCollectionBaseAgent/RentCollectionBaseAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/RentCollectionAgent/WARentCollectionAgent/WARentCollectionAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/RentCollectionAgent/AFRentCollectionAgent/AFRentCollectionAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/ImportanceDiffusionAgent/ImportanceDiffusionBase/ImportanceDiffusionBase) (empty))
!(let $_ (import! &self ../metta-attention/attention/ImportanceDiffusionAgent/WAImportanceDiffusionAgent/WAImportanceDiffusionAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/ImportanceDiffusionAgent/AFImportanceDiffusionAgent/AFImportanceDiffusionAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/HebbianUpdatingAgent/HebbianUpdatingAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/HebbianCreationAgent/HebbianCreationAgent) (empty))
!(let $_ (import! &self ../metta-attention/attention/runner-Def) (empty))
!(let $_ (import! &self "../metta-attention/attention/ImportanceDiffusionAgent/getelement.py") (empty))

;; ============================================================
;; PHASE 1 — LOAD KB INTO TYPESPACE & CONFIGURE ECAN
;; ============================================================

; define setStv to quickly set stv for new atoms
(= (setStv $pattern ($mean $conf) $flag)
    (let $space (TypeSpace) (add-atom $space ($pattern (STV $mean $conf))))
)
;; Load wordnet atoms into TypeSpace (gives STV to nodes)
!(let $_ (collapse (match &wordnet ($atom ($mean $conf)) (setStv $atom ($mean $conf) true))) (empty))
;; Load biology KB atoms into TypeSpace
!(let $_ (collapse (match &biology ($atom ($mean $conf)) (setStv $atom ($mean $conf) true))) (empty))

;; Configure ECAN params
!(let*
    (
        ($_maxAf (updateAttentionParam MAX_AF_SIZE 50.0))
        ($_targetSti (updateAttentionParam TARGET_STI 1000.0))
        ($_targetLti (updateAttentionParam TARGET_LTI 1000.0))
        ($_stiBuffer (updateAttentionParam STI_FUNDS_BUFFER 1000.0))
        ($_ltiBuffer (updateAttentionParam LTI_FUNDS_BUFFER 1000.0))
        ($_stiFunds (updateAttentionParam FUNDS_STI 100000.0))
        ($_ltiFunds (updateAttentionParam FUNDS_LTI 100000.0))

    )
    (empty)
)

;; ============================================================
;; PHASE 2 — ECAN ATTENTION ALLOCATION (Article Reading)
;; ============================================================
;; Simulate reading an article about insects sentence by sentence.
;; Stimulate atoms from the sentences and run ECAN agents in batches.

(= (batch) 5)

(= (runAttentionBatch)
    (let $_agents (collapse (test-superpose &incidentSpace))
        True
    )
)

(= (reader $expr $counter)
    (if (== $expr ())
        True
        (let*
            (
                ($now (car-atom $expr))
                ($rest (cdr-atom $expr))
                ($_stim (collapse (stimulate $now 100)))
                ($_stimlog (println! (stimulating atom --> $now)))
                ($_batch
                    (if (or (== (% $counter (batch)) 0) (== $rest ()))
                        (runAttentionBatch)
                        (let $_skiplog (println! (agent not called))
                            True
                        )
                    )
                )
                ($_next (reader $rest (+ $counter 1)))
            )
            True
         )
    )
)

;; Start reading the insect sentences
; collapsed multiple line pattern matching

!(let $_ (collapse (match &insect (biologySent $x) (reader $x 0))) (empty))

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

!(let $_ (bind! &pln-kb (new-space)) (empty))

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
!(let $_
    (let* (
        ($afAtoms (getAfAtoms))
        ($y (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
            (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                (empty)
            )
        )))
    ) True)
    (empty)
)

;; Show what PLN received
!(let*
    (
        ($_label (println! "=== PLN KB (filtered premises) ==="))
        ($premises (collapse (match &pln-kb $x $x)))
        ($_premises (println! $premises))
    )
    (empty)
)

; Show what ECAN filtered
!(println! "=== Attentional Focus ===")
!(let*
    (
        ($_label (println! "=== Attentional Focus ==="))
        ($afatoms (getAfAtoms))
        ($_af (println! $afatoms))
    )
    (empty)
)

; STV base rates for concepts (needed by PLN deduction formula)
(= (STV (Concept $any)) (stv 0.1 0.9))

; ============================================================
; PHASE 4 — PLN REASONING ON ECAN-FILTERED KNOWLEDGE
; ============================================================
; Queries use bare biology words (ant, insect, arthropod, etc.)
; since that's what the biology KB uses.

; Query: ant → arthropod (2-hop: ant→insect→arthropod)
!(let*
    (
        ($_label (println! "=== Query: ant -> arthropod (2-hop) ==="))
        ($answer (collapse (PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept arthropod)) 10)))
        ($_answer (println! (Result $answer)))
    )
    (empty)
)

; attentional focus atoms
; (and also arthropod an invertebrate type of animal which is a living_thing (0.8 0.9) 
; ((insynset insect worm_n_02) (0.8 0.9)) 
; ((insynset insect insect_n_01) (0.8 0.9)) insect 
; ((insynset bee bee_n_01) (0.8 0.9)) 
; ((insynset bee bee_n_02) (0.8 0.9)) 
; ((insynset ant ant_n_01) (0.8 0.9)) bee ant 0.9)


; pln KB
; ((Sentence ((Inheritance (Concept ant) (Concept insect)) (stv 0.9 0.9)) ((ev ant insect))) 
;  (Sentence ((Inheritance (Concept bee) (Concept insect)) (stv 0.9 0.9)) ((ev bee insect))) 
;  (Sentence ((Inheritance (Concept insect) (Concept arthropod)) (stv 0.95 0.9)) ((ev insect arthropod))) 
;  (Sentence ((Inheritance (Concept arthropod) (Concept invertebrate)) (stv 0.95 0.95)) ((ev arthropod invertebrate))) 
;  (Sentence ((Inheritance (Concept invertebrate) (Concept animal)) (stv 0.95 0.95)) ((ev invertebrate animal))) 
;  (Sentence ((Inheritance (Concept animal) (Concept living_thing)) (stv 0.95 0.95)) ((ev animal living_thing)))
; )

; ;; Query: bee → arthropod (2-hop: bee→insect→arthropod)
!(let*
    (
        ($_label (println! "=== Query: bee -> arthropod (2-hop) ==="))
        ($answer (collapse (PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept bee) (Concept arthropod)) 10)))
        ($_answer (println! (Result $answer)))
    )
    (empty)
)

!(let*
    (
        ($_label (println! "=== Query: ant -> invertebrate (3-hop) ==="))
        ($answer (collapse (PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept invertebrate)) 10)))
        ($_answer (println! (Result $answer)))
    )
    (empty)
)

!(let*
    (
        ($_label (println! "=== Query: ant -> animal (4-hop) ==="))
        ($answer (collapse (PLN.Query (collapse (match &pln-kb $atom $atom)) (Inheritance (Concept ant) (Concept animal)) 10)))
        ($_answer (println! (Result $answer)))
    )
    (empty)
)

