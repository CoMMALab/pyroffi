(define (domain pyroffi-rearrange)
  (:requirements :strips :equality)
  (:predicates
    ; Static
    (Robot ?r)
    (Block ?b)
    (Region ?s)
    (Pose ?b ?p)
    (Grasp ?b ?g)
    (Conf ?q)
    (Traj ?t)
    (Contain ?b ?p ?s)
    (Kin ?b ?q ?p ?g)
    (Motion ?q1 ?t ?q2)
    (CFree ?b1 ?p1 ?b2 ?p2)
    (Placeable ?b ?s)

    ; Fluent
    (AtPose ?b ?p)
    (AtGrasp ?r ?b ?g)
    (AtConf ?r ?q)
    (Holding ?r ?b)
    (HandEmpty ?r)
    (CanMove ?r)

    ; Derived
    (In ?b ?s)
    (UnsafePose ?b ?p)
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

  (:action pick
    :parameters (?r ?b ?p ?g ?q)
    :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                       (AtConf ?r ?q) (AtPose ?b ?p) (HandEmpty ?r))
    :effect (and (AtGrasp ?r ?b ?g) (CanMove ?r)
                 (not (AtPose ?b ?p)) (not (HandEmpty ?r))
                 (increase (total-cost) (Cost))))

  (:action place
    :parameters (?r ?b ?p ?g ?q)
    :precondition (and (Robot ?r) (Kin ?b ?q ?p ?g)
                       (AtConf ?r ?q) (AtGrasp ?r ?b ?g)
                       (not (UnsafePose ?b ?p)))
    :effect (and (AtPose ?b ?p) (HandEmpty ?r) (CanMove ?r)
                 (not (AtGrasp ?r ?b ?g))
                 (increase (total-cost) (Cost))))

  (:derived (In ?b ?s)
    (exists (?p) (and (Contain ?b ?p ?s) (AtPose ?b ?p))))
  (:derived (Holding ?r ?b)
    (exists (?g) (and (Robot ?r) (Grasp ?b ?g) (AtGrasp ?r ?b ?g))))
  (:derived (UnsafePose ?b1 ?p1)
    (exists (?b2 ?p2) (and (Pose ?b1 ?p1) (Pose ?b2 ?p2)
                           (AtPose ?b2 ?p2)
                           (not (CFree ?b1 ?p1 ?b2 ?p2)))))
)
