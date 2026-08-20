(define (domain pyroffi-stack)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Block ?b)
    (Pose ?b ?p)
    (Grasp ?b ?g)
    (Conf ?q)
    (Traj ?t)
    (Kin ?b ?q ?p ?g)
    (Motion ?q1 ?t ?q2)
    (Supported ?b ?p ?bu ?pu)   ; pose p of b rests on block bu at pose pu

    ; Fluent
    (AtPose ?b ?p)
    (AtConf ?r ?q)
    (AtGrasp ?r ?b ?g)
    (HandEmpty ?r)
    (CanMove ?r)
    (OnTable ?b)
    (Clear ?b)
    (On ?b ?bu)
  )
  (:functions
    (Cost)
    (Dist ?q1 ?q2)
  )

  (:action move
    :parameters (?r ?q1 ?t ?q2)
    :precondition (and (Robot ?r) (Motion ?q1 ?t ?q2)
                       (AtConf ?r ?q1) (CanMove ?r))
    :effect (and (AtConf ?r ?q2)
                 (not (AtConf ?r ?q1)) (not (CanMove ?r))
                 (increase (total-cost) (Dist ?q1 ?q2))))

  ; Blocks are only ever picked off the table (initial clutter) and then stacked,
  ; so pick never needs to re-clear a support underneath it.
  (:action pick
    :parameters (?r ?b ?p ?g ?q)
    :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                       (AtConf ?r ?q) (AtPose ?b ?p) (HandEmpty ?r)
                       (OnTable ?b) (Clear ?b))
    :effect (and (AtGrasp ?r ?b ?g) (CanMove ?r)
                 (not (AtPose ?b ?p)) (not (HandEmpty ?r)) (not (OnTable ?b))
                 (increase (total-cost) (Cost))))

  (:action place-on-block
    :parameters (?r ?b ?p ?g ?q ?bu ?pu)
    :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                       (AtConf ?r ?q) (AtGrasp ?r ?b ?g)
                       (Supported ?b ?p ?bu ?pu) (AtPose ?bu ?pu) (Clear ?bu))
    :effect (and (AtPose ?b ?p) (On ?b ?bu) (Clear ?b)
                 (HandEmpty ?r) (CanMove ?r)
                 (not (AtGrasp ?r ?b ?g)) (not (Clear ?bu))
                 (increase (total-cost) (Cost))))
)
