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

;; Get-current-premises: Dynamically fetch premises from pln-kb
(= (get-current-premises)
    (collapse (match &pln-kb $atom $atom))
)

;; STV base rates for concepts (needed by PLN deduction formula)
(= (STV (Concept $any)) (stv 0.1 0.9))

;; ============================================================
;; DYNAMIC QUERIES - AF RECOMPUTED EACH TIME
;; ============================================================

;; ITERATION 1: ant → arthropod (2-hop: ant→insect→arthropod)
!(println! (QUERY "DYNAMIC-Q1: ant->arthropod (2-hop)"))
!(collapse (test-superpose &incidentSpace))
!(bind! &pln-kb (new-space))
!(let* (($afAtoms (getAfAtoms))
        ($_bridge (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                    (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                        (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                        (empty))))))
      True)
!(let $premises (get-current-premises)
    (if (== $premises ())
        (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q1"))
        (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept arthropod)) 30))
               ($_ans (println! (BEST-ANSWER $result)))
               ($deriveOut (PLN.Derive $premises $premises 30))
               ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
               ($_tick2 (collapse (test-superpose &incidentSpace))))
          (println! (QUERY-COMPLETE "DYNAMIC-Q1")))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 2: bee → arthropod (2-hop: bee→insect→arthropod)
!(println! (QUERY "DYNAMIC-Q2: bee->arthropod (2-hop)"))
!(collapse (test-superpose &incidentSpace))
!(bind! &pln-kb (new-space))
!(let* (($afAtoms (getAfAtoms))
        ($_bridge (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                    (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                        (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                        (empty))))))
      True)
!(let $premises (get-current-premises)
    (if (== $premises ())
        (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q2"))
        (let* (($result (PLN.Query $premises (Inheritance (Concept bee) (Concept arthropod)) 30))
               ($_ans (println! (BEST-ANSWER $result)))
               ($deriveOut (PLN.Derive $premises $premises 30))
               ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
               ($_tick2 (collapse (test-superpose &incidentSpace))))
          (println! (QUERY-COMPLETE "DYNAMIC-Q2")))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 3: ant → invertebrate (3-hop: ant→insect→arthropod→invertebrate)
!(println! (QUERY "DYNAMIC-Q3: ant->invertebrate (3-hop)"))
!(collapse (test-superpose &incidentSpace))
!(bind! &pln-kb (new-space))
!(let* (($afAtoms (getAfAtoms))
        ($_bridge (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                    (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                        (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                        (empty))))))
      True)
!(let $premises (get-current-premises)
    (if (== $premises ())
        (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q3"))
        (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept invertebrate)) 50))
               ($_ans (println! (BEST-ANSWER $result)))
               ($deriveOut (PLN.Derive $premises $premises 50))
               ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
               ($_tick2 (collapse (test-superpose &incidentSpace))))
          (println! (QUERY-COMPLETE "DYNAMIC-Q3")))))

!(println! "\n========== ECAN SHIFT POINT: AF may have evolved ==========\n")

;; ITERATION 4: ant → animal (4-hop: ant→insect→arthropod→invertebrate→animal)
!(println! (QUERY "DYNAMIC-Q4: ant->animal (4-hop)"))
!(collapse (test-superpose &incidentSpace))
!(bind! &pln-kb (new-space))
!(let* (($afAtoms (getAfAtoms))
        ($_bridge (collapse (match &biology ((InheritanceLink $a $b) ($s $c))
                    (if (and (isMember $a $afAtoms) (isMember $b $afAtoms))
                        (add-atom &pln-kb (Sentence ((Inheritance (Concept $a) (Concept $b)) (stv $s $c)) ((ev $a $b))))
                        (empty))))))
      True)
!(let $premises (get-current-premises)
    (if (== $premises ())
        (println! (QUERY-SKIPPED-EMPTY-KB "DYNAMIC-Q4"))
        (let* (($result (PLN.Query $premises (Inheritance (Concept ant) (Concept animal)) 60))
               ($_ans (println! (BEST-ANSWER $result)))
               ($deriveOut (PLN.Derive $premises $premises 60))
               ($_learn (let ($tasksRet $beliefsRet) $deriveOut (learn-inferred-links $beliefsRet)))
               ($_tick2 (collapse (test-superpose &incidentSpace))))
          (println! (QUERY-COMPLETE "DYNAMIC-Q4")))))

;; Final AF state
!(println! "=== FINAL ATTENTIONAL FOCUS (after all queries) ===")
!(show-af)