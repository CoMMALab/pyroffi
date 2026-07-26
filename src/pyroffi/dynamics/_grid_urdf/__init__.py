"""Vendored subset of the robot-acceleration ``URDFParser`` package.

pyroffi feeds ``grid_codegen.GRiDCodeGenerator`` a ``Robot`` object with the API it expects
(sympy ``Xmat`` functions, spatial inertias, BFS levels, subtree lists, ...).
That object model — plus the fixed-joint-elimination / DFS-renumbering /
subtree-building post-processing — used to live in the external ``URDFParser``
package.  We only ever populated its classes from an already-parsed
``yourdfpy.URDF`` (see :mod:`.._grid_robot_adapter`) and reused its
post-processing; we never used its BeautifulSoup URDF *parser*.  So the object
model and post-processing are vendored here and the external dependency is
dropped.

The class/method sources are copied essentially verbatim (only the intra-package
imports and file names changed) to keep ``GRiDCodeGenerator`` output byte-for-bit
identical.  The ``renumber_links_joints`` free function is the ``URDFParser``
class's post-processing pipeline lifted off the parser (the XML-parsing methods
are omitted; see :mod:`.postprocess`).

Vendored from https://github.com/A2R-Lab/URDFParser at commit ``f88ce2a``
(branch ``modernizing-tests``) — the commit ``external/GRiD`` itself pins for
its ``URDFParser`` submodule, so this object model matches exactly what the
vendored ``grid_codegen`` expects.  Re-vendor by re-copying the upstream files
and renaming the modules; do not hand-patch them.

    MIT License
    Copyright (c) 2021 Hardware Acceleration for Robotics

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
"""

from .errors import (
    UnsupportedJointTypeError as UnsupportedJointTypeError,
    URDFParseError as URDFParseError,
)
from .joint import Fixed_Joint as Fixed_Joint, Joint as Joint
from .link import Link as Link
from .postprocess import renumber_links_joints as renumber_links_joints
from .robot import Robot as Robot
