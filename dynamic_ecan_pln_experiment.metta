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
;!(import! &self ../metta-attention/attention/ImportanceDiffusionAgent/AFImportanceDiffusionAgent/AFImportanceDiffusionAgent)  ;; Python dependency
!(import! &self ../metta-attention/attention/HebbianUpdatingAgent/HebbianUpdatingAgent)
!(import! &self ../metta-attention/attention/HebbianCreationAgent/HebbianCreationAgent)
!(import! &self ../metta-attention/attention/runner-Def)
;!(import! &self "../metta-attention/attention/ImportanceDiffusionAgent/getelement.py")  ;; Commented: Python module not needed

;; ============================================================
;; PHASE 1 — LOAD KB INTO TYPESPACE & CONFIGURE ECAN
;; ============================================================

; define setStv to quickly set stv for new atoms
(= (setStv $pattern ($mean $conf) $flag)
    (let $space (TypeSpace) (add-atom $space ($pattern (STV $mean $conf))))
)
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
;; PHASE 3 — DYNAMIC BRIDGE: ECAN ↔ PLN LOOP
;; ============================================================
;; Dynamic AF computation:
;;   1. Run ECAN to get current AF atoms
;;   2. Bridge: Filter KB using current AF
;;   3. Run one PLN query 
;;   4. New deductions may generate new atoms/facts
;;   5. Re-run ECAN with enriched atom space → AF shifts
;;   6. Repeat for next query with NEW AF
;;
;; This allows AF to evolve as PLN discovers new knowledge.

!(bind! &pln-kb (new-space))
!(bind! &pln-results (new-space))

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

;; Rebuild PLN KB from CURRENT AF (fresh each round)
(= (rebuild-kb-from-current-af)
    (let* (
        ($afAtoms (getAfAtoms))
        ($_reset (bind! &pln-kb (new-space)))
        ($_bridge (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                     (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                         (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                         (empty)
                     )
        )))
    )
    True
    )
)

;; learn-inferred-links:
;; Push inferred links into PLN results and stimulate involved atoms
;; so ECAN can shift salience in the next round.
(= (learn-inferred-links $beliefs)
    (collapse
        (let (Sentence (($linkType (Concept $a) (Concept $b)) (stv $s $c)) $ev)
             (superpose $beliefs)
             (if (== $linkType Inheritance)
                 (let* (
                     ($_1 (add-atom &pln-results (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) $ev)))
                     ($_2 (setStv (InheritanceLink $a $b) ($s $c) true))
                     ($_3 (stimulate $a 20))
                     ($_4 (stimulate $b 20))
                 )
                 True)
                 (empty)
             )
        )
    )
)

;; Show AF atoms (for monitoring AF shifts)
(= (show-af)
    (let $afAtoms (getAfAtoms)
        (collapse (superpose $afAtoms))
    )
)

;; ============================================================
;; PHASE 4 — DYNAMIC PLN REASONING WITH AF LOOPS
;; ============================================================
;; Instead of running all queries on a FROZEN AF,
;; queries are run DYNAMICALLY where:
;;   - Before each query: ECAN re-runs (recompute-af)
;;   - After each query: New premises/atoms may exist
;;   - Next query starts with UPDATED AF
;;
;; This implements the "dynamic" pattern where AF evolves during inference.

;; Get-current-premises: Dynamically fetch premises from pln-kb
(= (get-current-premises)
    (collapse (match &pln-kb $atom $atom))
)

;; STV base rates for concepts (needed by PLN deduction formula)
(= (STV (Concept $any)) (stv 0.1 0.9))

;; ============================================================
;; DYNAMIC QUERIES - AF RECOMPUTED EACH TIME
;; ============================================================

;; Economic inference budgeting helpers
(= (num-or-zero $x)
    (if (== $x ()) 0 $x))

(= (nonpositive $x)
    (or (< $x 0) (== $x 0)))

(= (sti-safe $atom)
    (num-or-zero (getSti $atom)))

(= (sum-list $vals)
    (if (== $vals ())
        0
        (+ (car-atom $vals) (sum-list (cdr-atom $vals)))))

(= (budget-seed $source $target)
    (let* (($srcSti (sti-safe $source))
           ($tgtSti (sti-safe $target)))
      (+ 30 (/ (+ $srcSti $tgtSti) 10))))

(= (query-spend $premises $hops)
    (+ 8 (+ (* 0.15 (size-atom $premises)) (* 0.5 $hops))))

(= (derive-spend $premises $hops)
    (+ 12 (+ (* 0.2 (size-atom $premises)) (* 0.7 $hops))))

(= (belief-reward $beliefs)
    (sum-list
        (collapse
            (let (Sentence (($linkType (Concept $a) (Concept $b)) (stv $s $c)) $ev)
                 (superpose $beliefs)
                 (if (== $linkType Inheritance)
                     (* 0.03 (+ (sti-safe $a) (sti-safe $b)))
                     0)))))

(= (premise-has-concept $premises $node)
    (if (== (collapse
              (let (Sentence (($linkType (Concept $a) (Concept $b)) $sv) $ev)
                   (superpose $premises)
                   (if (or (== $a $node) (== $b $node))
                       True
                       (empty))))
            ())
        False
        True))

(= (need-fallback $premises $source $target)
    (or (== $premises ())
        (or (== (premise-has-concept $premises $source) False)
            (== (premise-has-concept $premises $target) False))))

;; ITERATION 1: ant → arthropod (2-hop: ant→insect→arthropod)
!(println! (QUERY "DYNAMIC-Q1: ant->arthropod (2-hop)"))
!(collapse (test-superpose &incidentSpace))
!(rebuild-kb-from-current-af)
!(if (need-fallback (get-current-premises) ant arthropod)
    (println! (KB-FALLBACK-EXPAND "DYNAMIC-Q1" ant arthropod))
    (empty))
!(if (need-fallback (get-current-premises) ant arthropod)
    (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                  (if (or (or (== $a ant) (== $b ant))
                          (or (== $a arthropod) (== $b arthropod)))
                      (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                      (empty))))
    (empty))
!(if (== (get-current-premises) ())
    (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q1"))
    (let* (($premises (get-current-premises))
           ($seed (budget-seed ant arthropod))
           ($qCost (query-spend $premises 30))
           ($budgetAfterQuery (- $seed $qCost))
           ($_b0 (println! (BUDGET-SEED "DYNAMIC-Q1" $seed)))
           ($_bq (println! (BUDGET-SPEND-QUERY "DYNAMIC-Q1" $qCost $budgetAfterQuery))))
      (if (nonpositive $budgetAfterQuery)
          (println! (QUERY-BANKRUPT-PRE-QUERY "DYNAMIC-Q1" $seed $qCost))
          (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept arthropod)) 30))
                 ($_ans (println! (BEST-ANSWER $result)))
                 ($dCost (derive-spend $premises 30))
                 ($budgetBeforeDerive (- $budgetAfterQuery $dCost))
                 ($_bd (println! (BUDGET-SPEND-DERIVE "DYNAMIC-Q1" $dCost $budgetBeforeDerive))))
            (if (nonpositive $budgetBeforeDerive)
                (println! (QUERY-BANKRUPT-PRE-DERIVE "DYNAMIC-Q1" $budgetBeforeDerive))
                (let* (($deriveOut (PLN.Derive $premises $premises 30))
                       ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
                       ($_tick1 (collapse (test-superpose &incidentSpace)))
                       ($reward (let ($tasksRet $beliefsRet) $deriveOut (belief-reward $beliefsRet)))
                       ($finalBudget (+ $budgetBeforeDerive $reward))
                       ($_br (println! (BUDGET-REWARD "DYNAMIC-Q1" $reward $finalBudget))))
                  (if (> $finalBudget 50)
                      (let* (($_sx (stimulate ant 20))
                             ($_tx (stimulate arthropod 15))
                             ($_tick2 (collapse (test-superpose &incidentSpace))))
                        (println! (QUERY-COMPLETE-REWARDED "DYNAMIC-Q1" $finalBudget)))
                      (if (nonpositive $finalBudget)
                          (println! (QUERY-COMPLETE-BANKRUPT "DYNAMIC-Q1" $finalBudget))
                          (println! (QUERY-COMPLETE "DYNAMIC-Q1"))))))))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 2: bee → arthropod (2-hop: bee→insect→arthropod)
!(println! (QUERY "DYNAMIC-Q2: bee->arthropod (2-hop)"))
!(collapse (test-superpose &incidentSpace))
!(rebuild-kb-from-current-af)
!(if (need-fallback (get-current-premises) bee arthropod)
    (println! (KB-FALLBACK-EXPAND "DYNAMIC-Q2" bee arthropod))
    (empty))
!(if (need-fallback (get-current-premises) bee arthropod)
    (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                  (if (or (or (== $a bee) (== $b bee))
                          (or (== $a arthropod) (== $b arthropod)))
                      (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                      (empty))))
    (empty))
!(if (== (get-current-premises) ())
    (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q2"))
    (let* (($premises (get-current-premises))
           ($seed (budget-seed bee arthropod))
           ($qCost (query-spend $premises 30))
           ($budgetAfterQuery (- $seed $qCost))
           ($_b0 (println! (BUDGET-SEED "DYNAMIC-Q2" $seed)))
           ($_bq (println! (BUDGET-SPEND-QUERY "DYNAMIC-Q2" $qCost $budgetAfterQuery))))
      (if (nonpositive $budgetAfterQuery)
          (println! (QUERY-BANKRUPT-PRE-QUERY "DYNAMIC-Q2" $seed $qCost))
          (let* (($result (PLN.Query $premises (Inheritance (Concept bee) (Concept arthropod)) 30))
                 ($_ans (println! (BEST-ANSWER $result)))
                 ($dCost (derive-spend $premises 30))
                 ($budgetBeforeDerive (- $budgetAfterQuery $dCost))
                 ($_bd (println! (BUDGET-SPEND-DERIVE "DYNAMIC-Q2" $dCost $budgetBeforeDerive))))
            (if (nonpositive $budgetBeforeDerive)
                (println! (QUERY-BANKRUPT-PRE-DERIVE "DYNAMIC-Q2" $budgetBeforeDerive))
                (let* (($deriveOut (PLN.Derive $premises $premises 30))
                       ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
                       ($_tick1 (collapse (test-superpose &incidentSpace)))
                       ($reward (let ($tasksRet $beliefsRet) $deriveOut (belief-reward $beliefsRet)))
                       ($finalBudget (+ $budgetBeforeDerive $reward))
                       ($_br (println! (BUDGET-REWARD "DYNAMIC-Q2" $reward $finalBudget))))
                  (if (> $finalBudget 50)
                      (let* (($_sx (stimulate bee 20))
                             ($_tx (stimulate arthropod 15))
                             ($_tick2 (collapse (test-superpose &incidentSpace))))
                        (println! (QUERY-COMPLETE-REWARDED "DYNAMIC-Q2" $finalBudget)))
                      (if (nonpositive $finalBudget)
                          (println! (QUERY-COMPLETE-BANKRUPT "DYNAMIC-Q2" $finalBudget))
                          (println! (QUERY-COMPLETE "DYNAMIC-Q2"))))))))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 3: ant → invertebrate (3-hop: ant→insect→arthropod→invertebrate)
!(println! (QUERY "DYNAMIC-Q3: ant->invertebrate (3-hop)"))
!(collapse (test-superpose &incidentSpace))
!(rebuild-kb-from-current-af)
!(if (need-fallback (get-current-premises) ant invertebrate)
    (println! (KB-FALLBACK-EXPAND "DYNAMIC-Q3" ant invertebrate))
    (empty))
!(if (need-fallback (get-current-premises) ant invertebrate)
    (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                  (if (or (or (== $a ant) (== $b ant))
                          (or (== $a invertebrate) (== $b invertebrate)))
                      (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                      (empty))))
    (empty))
!(if (== (get-current-premises) ())
    (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q3"))
    (let* (($premises (get-current-premises))
           ($seed (budget-seed ant invertebrate))
           ($qCost (query-spend $premises 50))
           ($budgetAfterQuery (- $seed $qCost))
           ($_b0 (println! (BUDGET-SEED "DYNAMIC-Q3" $seed)))
           ($_bq (println! (BUDGET-SPEND-QUERY "DYNAMIC-Q3" $qCost $budgetAfterQuery))))
      (if (nonpositive $budgetAfterQuery)
          (println! (QUERY-BANKRUPT-PRE-QUERY "DYNAMIC-Q3" $seed $qCost))
          (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept invertebrate)) 50))
                 ($_ans (println! (BEST-ANSWER $result)))
                 ($dCost (derive-spend $premises 50))
                 ($budgetBeforeDerive (- $budgetAfterQuery $dCost))
                 ($_bd (println! (BUDGET-SPEND-DERIVE "DYNAMIC-Q3" $dCost $budgetBeforeDerive))))
            (if (nonpositive $budgetBeforeDerive)
                (println! (QUERY-BANKRUPT-PRE-DERIVE "DYNAMIC-Q3" $budgetBeforeDerive))
                (let* (($deriveOut (PLN.Derive $premises $premises 50))
                       ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
                       ($_tick1 (collapse (test-superpose &incidentSpace)))
                       ($reward (let ($tasksRet $beliefsRet) $deriveOut (belief-reward $beliefsRet)))
                       ($finalBudget (+ $budgetBeforeDerive $reward))
                       ($_br (println! (BUDGET-REWARD "DYNAMIC-Q3" $reward $finalBudget))))
                  (if (> $finalBudget 50)
                      (let* (($_sx (stimulate ant 20))
                             ($_tx (stimulate invertebrate 15))
                             ($_tick2 (collapse (test-superpose &incidentSpace))))
                        (println! (QUERY-COMPLETE-REWARDED "DYNAMIC-Q3" $finalBudget)))
                      (if (nonpositive $finalBudget)
                          (println! (QUERY-COMPLETE-BANKRUPT "DYNAMIC-Q3" $finalBudget))
                          (println! (QUERY-COMPLETE "DYNAMIC-Q3"))))))))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 4: ant → animal (4-hop: ant→insect→arthropod→invertebrate→animal)
!(println! (QUERY "DYNAMIC-Q4: ant->animal (4-hop)"))
!(collapse (test-superpose &incidentSpace))
!(rebuild-kb-from-current-af)
!(if (need-fallback (get-current-premises) ant animal)
    (println! (KB-FALLBACK-EXPAND "DYNAMIC-Q4" ant animal))
    (empty))
!(if (need-fallback (get-current-premises) ant animal)
    (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                  (if (or (or (== $a ant) (== $b ant))
                          (or (== $a animal) (== $b animal)))
                      (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                      (empty))))
    (empty))
!(if (== (get-current-premises) ())
    (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q4"))
    (let* (($premises (get-current-premises))
           ($seed (budget-seed ant animal))
           ($qCost (query-spend $premises 60))
           ($budgetAfterQuery (- $seed $qCost))
           ($_b0 (println! (BUDGET-SEED "DYNAMIC-Q4" $seed)))
           ($_bq (println! (BUDGET-SPEND-QUERY "DYNAMIC-Q4" $qCost $budgetAfterQuery))))
      (if (nonpositive $budgetAfterQuery)
          (println! (QUERY-BANKRUPT-PRE-QUERY "DYNAMIC-Q4" $seed $qCost))
          (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept animal)) 60))
                 ($_ans (println! (BEST-ANSWER $result)))
                 ($dCost (derive-spend $premises 60))
                 ($budgetBeforeDerive (- $budgetAfterQuery $dCost))
                 ($_bd (println! (BUDGET-SPEND-DERIVE "DYNAMIC-Q4" $dCost $budgetBeforeDerive))))
            (if (nonpositive $budgetBeforeDerive)
                (println! (QUERY-BANKRUPT-PRE-DERIVE "DYNAMIC-Q4" $budgetBeforeDerive))
                (let* (($deriveOut (PLN.Derive $premises $premises 60))
                       ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
                       ($_tick1 (collapse (test-superpose &incidentSpace)))
                       ($reward (let ($tasksRet $beliefsRet) $deriveOut (belief-reward $beliefsRet)))
                       ($finalBudget (+ $budgetBeforeDerive $reward))
                       ($_br (println! (BUDGET-REWARD "DYNAMIC-Q4" $reward $finalBudget))))
                  (if (> $finalBudget 50)
                      (let* (($_sx (stimulate ant 20))
                             ($_tx (stimulate animal 15))
                             ($_tick2 (collapse (test-superpose &incidentSpace))))
                        (println! (QUERY-COMPLETE-REWARDED "DYNAMIC-Q4" $finalBudget)))
                      (if (nonpositive $finalBudget)
                          (println! (QUERY-COMPLETE-BANKRUPT "DYNAMIC-Q4" $finalBudget))
                          (println! (QUERY-COMPLETE "DYNAMIC-Q4"))))))))))

;; Final AF state
!(println! "=== FINAL ATTENTIONAL FOCUS (after all queries) ===")
!(show-af)