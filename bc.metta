;; Define cast functions between Nat and Number
(: fromNumber (-> Number Atom))
(= (fromNumber $n) (if (<= $n 0) Z (S (fromNumber (- $n 1)))))

;; Knowledge base
!(bind! &kb (new-space))
!(add-atom &kb (: FACT1 (Evaluation croaks Fritz)))

!(add-atom &kb (: FACT2 (Evaluation chirps Tweety)))
!(add-atom &kb (: FACT4 (Inheritance Tweety yellow)))
!(add-atom &kb (: FACT5 (Evaluation eats_flies Tweety)))
!(add-atom &kb (: FACT5 (Evaluation eats_flies Fritz)))
; !(add-atom &kb (True))



;; Rule base
!(bind! &rb (new-space))
; (=(croaks-eats_flies-frog-rule) 
;     (-> (Evaluation croaks $x) 
;           (Evaluation eats_flies $x) 
;               (Inheritance $x frog)))


; (=(chirps-sings-canary-rule) 
;     (-> (Evaluation chirps $x) 
;             (Evaluation sings $x) 
;                (Inheritance $x canary)))



; (=(frog-green-rule)
;     (-> (True)
;           (Inheritance $x frog) 
;              (Inheritance $x green)))



; (=(canary-yellow-rule) 
;     (-> (True)
;           (Inheritance $x canary)
;             (Inheritance $x yellow)))



!(add-atom &kb (: canary-yellow-rule
            (->
                (Inheritance $x canary)
                  (Inheritance $x yellow))))

!(add-atom &kb (: frog-green-rule
         (-> 
          (Inheritance $x frog) 
             (Inheritance $x green))))
!(add-atom &kb (: chirps-sings-canary-rule 
    (-> (Evaluation chirps $x) 
            (-> (Evaluation sings $x) 
                   (Inheritance $x canary)))))
!(add-atom &kb ( : croaks-eats_flies-frog-rule 
    (-> (Evaluation croaks $x) 
          (-> (Evaluation eats_flies $x) 
                  (Inheritance $x frog)))))




;; Backward chainer
(: bc_bem (-> Atom Nat Atom))
;; Base case
(= (bc_bem $conclusion $depth)
   (match &kb $conclusion $conclusion))

;; Recursive step
(= (bc_bem $conclusion (S $k))
   (match &rb (-> $premise1 $premise2 $conclusion)        ;(p1 p2 (Inheritance $what green))  ((True) (Inheritance fritz frog) (Inheritance fritz green))
          (let* (($premise1 (bc_bem $premise1 $k))     ;(bc (True) (s z))    (True)
                 ($premise2 (bc_bem $premise2 $k)))     ;(bc (Inheritance fritz frog) (s z)) ->  ( (Evaluation croaks Fritz) (Evaluation eats_flies fritz) (Inheritance $x frog))
            $conclusion)))                                      ;(Evaluation croaks $x) (Evaluation eats_flies $x) 


;; Backward chainer/synthesizer
                      ; Result

;; Base case
(= (syn $kb $_ (: $prf $ccln)) (match $kb (: $prf $ccln) (: $prf $ccln)))

;; Recursive step
(= (syn $kb (S $k) (: ($prfabs $prfarg) $ccln))
   (let* (((: $prfabs (-> $prms $ccln)) (syn $kb $k (: $prfabs (-> $prms $ccln))))
          ((: $prfarg $prms) (syn $kb $k (: $prfarg $prms))))
     (: ($prfabs $prfarg) $ccln)))


;; run the backward chainer to find out "What is green?" by defining a target

!(syn &kb (fromNumber 4) (: $prf (Inheritance $what green)) )

[(: (frog-green-rule ((croaks-eats_flies-frog-rule FACT1) FACT5)) 
    (Inheritance Fritz green))]
