(define (stream pyroffi-stack)
  (:stream s-grasp
    :inputs (?b)
    :domain (Block ?b)
    :outputs (?g)
    :certified (Grasp ?b ?g))

  ; Deterministic stacking pose: directly on top of block ?bu at pose ?pu.
  (:stream s-stack-pose
    :inputs (?b ?bu ?pu)
    :domain (and (Block ?b) (Pose ?bu ?pu))
    :outputs (?p)
    :certified (and (Pose ?b ?p) (Supported ?b ?p ?bu ?pu)))

  (:stream s-ik
    :inputs (?b ?p ?g)
    :domain (and (Pose ?b ?p) (Grasp ?b ?g))
    :outputs (?q)
    :certified (and (Kin ?b ?q ?p ?g) (Conf ?q)))

  (:stream s-motion
    :inputs (?q1 ?q2)
    :domain (and (Conf ?q1) (Conf ?q2))
    :outputs (?t)
    :certified (and (Motion ?q1 ?t ?q2) (Traj ?t)))

  (:function (Dist ?q1 ?q2)
    (and (Conf ?q1) (Conf ?q2)))
)
