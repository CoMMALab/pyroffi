"""Typed exceptions for the URDF parser.

Historically a malformed / unsupported URDF caused the parser to
``print(...)`` and call ``exit()`` -- a bare ``SystemExit`` that kills the
whole host process with no traceback and is uncatchable by normal
``except Exception`` handlers. These typed exceptions replace that failure
mode so callers (the equivalence harness, GRiD codegen, downstream tools)
can catch and surface a structured, debuggable error instead.

All parser-raised errors derive from :class:`URDFParseError` so a caller may
catch the whole family with a single ``except URDFParseError``.
"""


class URDFParseError(Exception):
    """Base class for every error raised while parsing a URDF model."""


class UnsupportedJointTypeError(URDFParseError):
    """Raised when a URDF joint declares a type the parser cannot model.

    Carries the offending joint type (and optionally the joint name) so a
    caller can report exactly which joint tripped the parser.
    """

    def __init__(self, jtype, joint_name=None):
        self.jtype = jtype
        self.joint_name = joint_name
        # Fully supported end-to-end (parse + codegen + RBDReference). planar/
        # spherical PARSE into a native representation (groundwork, see
        # docs/open-tasks/joint_types_plan.md) but are NOT yet emitted by the
        # CUDA codegen or modelled by RBDReference, so they are not advertised
        # as supported here. GRiDCodeGenerator raises a clear error if a robot
        # carrying them reaches codegen.
        supported = "revolute, continuous, prismatic, fixed, floating"
        where = f" (joint '{joint_name}')" if joint_name is not None else ""
        super().__init__(
            f"Unsupported joint type '{jtype}'{where}. "
            f"Supported joint types are: {supported}."
        )
