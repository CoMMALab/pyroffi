# import numpy as np
import math
import numpy as np
import sympy as sp
from .spatial_algebra import Origin, Translation, Rotation, Quaternion_Tools
from .errors import UnsupportedJointTypeError, URDFParseError


def _snap_to_pi_grid(value, tolerance=1e-5):
    """Snap a URDF rpy value to N*π/2 (|N| ≤ 4) when within `tolerance`.

    URDFs commonly truncate (e.g. "1.5708" for π/2 leaves a ~3.7e-6 residual);
    the residual cascades through float32 kinematic chains and can flip an
    atan2 sign in rpy extraction. Bounded N keeps the snap targeted — only
    angles already near a quarter-turn grid point collapse exactly.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value
    pi_half = math.pi / 2
    for n in range(-4, 5):
        if abs(v - n * pi_half) < tolerance:
            return sp.Integer(0) if n == 0 else n * sp.pi / 2
    return value


class Joint:
    floating_base = False
    def __init__(self, name, jid, parent, child, using_quaternion = False):
        self.name = name         # name
        self.jid = jid           # temporary ID (replaced by standard DFS parse ordering)
        self.urdf_jid = jid      # URDF ordered ID
        self.bfs_jid = jid       # temporary ID (replaced by BFS parse ordering)
        self.bfs_level = 0       # temporary level (replaced by BFS parse ordering)
        self.origin = Origin()   # Fixed origin location
        self.jtype = None        # type of joint
        self.parent = parent     # parent link name
        self.child = child       # child link name; see docs/open-tasks/notes.md (Joint.py:38)
        self.theta = sp.symbols("theta") # Free 1D joint variable
        self.Xmat_sp = None      # Sympy X matrix placeholder
        self.Xmat_sp_free = None # Sympy X_free matrix placeholder
        self.Xmat_sp_hom = None      # Sympy X homogenous 4x4 matrix placeholder
        self.Xmat_sp_hom_free = None # Sympy X_free homogenous 4x4  matrix placeholder
        self.Smat_sp = None      # Sympy S matrix placeholder (usually a vector)
        self.damping = 0         # viscous damping coefficient (tau += damping*qd)
        self.friction = 0        # Coulomb friction coefficient (tau += friction*sign(qd))
        self.dof = 0             # dof placeholder
        # for floating base
        self.using_quaternion = using_quaternion
        self.x_fb = sp.symbols("x_fb")
        self.y_fb = sp.symbols("y_fb")
        self.z_fb = sp.symbols("z_fb")
        self.q1_fb = sp.symbols("q1_fb")
        self.q2_fb = sp.symbols("q2_fb")
        self.q3_fb = sp.symbols("q3_fb")
        self.q4_fb = sp.symbols("q4_fb")
        self.roll_fb = sp.symbols("roll_fb")
        self.pitch_fb = sp.symbols("pitch_fb")
        self.yaw_fb = sp.symbols("yaw_fb")
        # multi-DOF non-root joint coordinates (planar: 2 translations + 1
        # rotation; spherical: a unit quaternion). These mirror the
        # floating-base symbols but for a mid-chain joint.
        self.px_pl = sp.symbols("px_pl")     # planar in-plane translation 1
        self.py_pl = sp.symbols("py_pl")     # planar in-plane translation 2
        self.theta_pl = sp.symbols("theta_pl")  # planar rotation about normal axis
        self.q1_sph = sp.symbols("q1_sph")   # spherical unit quaternion (x,y,z,w)
        self.q2_sph = sp.symbols("q2_sph")
        self.q3_sph = sp.symbols("q3_sph")
        self.q4_sph = sp.symbols("q4_sph")
        self.joint_limits = []
        self.velocity_limit = None   # |qd| <= velocity_limit (None => +inf / unspecified)
        self.effort_limit = None     # |tau| <= effort_limit  (None => +inf / unspecified)
        self.position_symbols = []
        self.local_q_dim = 0
        self.dXmat_sp_hom_blocks = []
        self.d2Xmat_sp_hom_blocks = []
        # URDF <mimic> tag handling. A mimic joint replicates another joint's
        # generalized coordinate: q_self = multiplier * q[mimic_target] + offset
        # (and likewise for v, a). Mimic joints do NOT contribute their own
        # column to (q, v) -- their dof is reported as 0 so num_vel/num_pos
        # collapse naturally -- but they DO still contribute to body
        # transforms and Jacobian columns (folded into the mimicked column,
        # scaled by `multiplier`).
        self.is_mimic = False
        self.mimic_joint_name = None
        self.mimic_target_id = None
        self.mimic_multiplier = 1.0
        self.mimic_offset = 0.0

    def set_id(self, id_in):
        self.jid = id_in

    def set_parent(self, parent_name):
        self.parent = parent_name

    def set_child(self, child_name):
        self.child = child_name

    def set_bfs_id(self, id_in):
        self.bfs_id = id_in

    def set_bfs_level(self, level_in):
        self.bfs_level = level_in

    def set_origin_xyz(self, x, y = None, z = None):
        self.origin.set_translation(x,y,z)

    def set_origin_rpy(self, r, p = None, y = None):
        if p is None and y is None:
            r, p, y = r[0], r[1], r[2]
        r = _snap_to_pi_grid(r)
        p = _snap_to_pi_grid(p)
        y = _snap_to_pi_grid(y)
        self.origin.set_rotation(r, p, y)

    def set_damping(self, damping):
        self.damping = damping

    def set_friction(self, friction):
        self.friction = friction

    def set_transformation_matrix(self, matrix_in):
        self.Xmat_sp = matrix_in

    def set_transformation_matrix_hom(self, matrix_in):
        self.Xmat_sp_hom = sp.nsimplify(matrix_in, tolerance=1e-6, rational=True).evalf()
        self.position_symbols = [self.theta]
        self.local_q_dim = 1
        self._build_homogeneous_transform_derivatives()

    def _build_homogeneous_transform_derivatives(self):
        if self.Xmat_sp_hom is None:
            self.dXmat_sp_hom = None
            self.d2Xmat_sp_hom = None
            self.dXmat_sp_hom_blocks = []
            self.d2Xmat_sp_hom_blocks = []
            return

        if not self.position_symbols:
            self.dXmat_sp_hom = sp.zeros(4, 4)
            self.d2Xmat_sp_hom = sp.zeros(4, 4)
            self.dXmat_sp_hom_blocks = []
            self.d2Xmat_sp_hom_blocks = []
            self.local_q_dim = 0
            return

        self.dXmat_sp_hom_blocks = [
            sp.diff(self.Xmat_sp_hom, symbol) for symbol in self.position_symbols
        ]
        self.d2Xmat_sp_hom_blocks = [
            [
                sp.diff(self.dXmat_sp_hom_blocks[row_ind], self.position_symbols[col_ind])
                for col_ind in range(len(self.position_symbols))
            ]
            for row_ind in range(len(self.position_symbols))
        ]
        self.dXmat_sp_hom = self.dXmat_sp_hom_blocks[0]
        self.d2Xmat_sp_hom = self.d2Xmat_sp_hom_blocks[0][0]

    def _local_q_lambdify_args(self):
        if self.jtype == "floating":
            if self.using_quaternion:
                return [[self.x_fb, self.y_fb, self.z_fb, self.q1_fb, self.q2_fb, self.q3_fb, self.q4_fb]]
            return [[self.x_fb, self.y_fb, self.z_fb, self.roll_fb, self.pitch_fb, self.yaw_fb]]
        if self.jtype == "planar":
            return [[self.px_pl, self.py_pl, self.theta_pl]]
        if self.jtype == "spherical":
            return [[self.q1_sph, self.q2_sph, self.q3_sph, self.q4_sph]]
        return self.theta

    def _axis_scale(self, axis, index):
        value = float(axis[index])
        if np.isclose(abs(value), 1.0):
            return value
        return None

    def _cardinal_axis(self, axis):
        """Return (index, sign) if `axis` is a (signed) cardinal unit axis, else None.

        A cardinal axis has exactly one component with |value| == 1 (the other
        two zero). This is the BYTE-IDENTICAL fast path: cardinal axes route to
        the existing rz/ry/rx builders and literal `np.array([0,0,±1,...])` S so
        every current manifest robot parses to the identical S / Xmat_sp.
        """
        for index in range(3):
            scale = self._axis_scale(axis, index)
            if scale is not None:
                # Confirm the other two components are zero (a genuine cardinal
                # axis), not e.g. [1,1,0] whose first comp also has |.|==1 only
                # after a non-normalized input. axis is already unit-normalized.
                others = [float(axis[k]) for k in range(3) if k != index]
                if all(np.isclose(o, 0.0) for o in others):
                    return index, scale
        return None

    def _general_axis_unit(self, axis):
        """Normalize a 3-vector axis to a unit vector (float64)."""
        a = np.asarray([float(axis[0]), float(axis[1]), float(axis[2])], dtype=np.float64)
        nrm = np.linalg.norm(a)
        if nrm == 0.0:
            raise URDFParseError(
                f"Joint '{self.name}' has a zero-length <axis>.")
        return a / nrm

    def _rodrigues_frame(self, u, theta):
        """Frame-rotation matrix E(theta) about unit axis u, GRiD convention.

        GRiD's rz/ry/rx are the FRAME (coordinate) rotations exp(-[axis]x*theta)
        (note the sign: rz = [[c,s,0],[-s,c,0],[0,0,1]] = exp(-[z]x theta)). The
        general-axis frame rotation is therefore Rodrigues with -theta:
            E = I cos t - [u]x sin t + u u^T (1 - cos t).
        Cardinal u reduces to exactly rz/ry/rx, so this is consistent with the
        fast path (the cardinal branch is taken before this for byte-identity).
        """
        c = sp.cos(theta)
        s = sp.sin(theta)
        ux, uy, uz = sp.Float(u[0]), sp.Float(u[1]), sp.Float(u[2])
        K = sp.Matrix([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        uut = sp.Matrix([[ux*ux, ux*uy, ux*uz],
                         [uy*ux, uy*uy, uy*uz],
                         [uz*ux, uz*uy, uz*uz]])
        return sp.eye(3) * c - K * s + uut * (1 - c)

    def set_type(self, jtype, axis = None, pitch = 0.0):
        self.jtype = jtype
        self.origin.build_fixed_transform()
        if self.jtype in ('revolute', 'continuous'):
            self.dof = 1
            self.position_symbols = [self.theta]
            self.local_q_dim = 1
            cardinal = self._cardinal_axis(axis)
            if cardinal is not None:
                # Tier A (cardinal): BYTE-IDENTICAL to the historical emit. One
                # signed ±1 component selects the rz/ry/rx frame rotation and
                # the literal [0,0,±1,0,0,0]-style S.
                index, axis_scale = cardinal
                rot_builder = (self.origin.rotation.rx, self.origin.rotation.ry,
                               self.origin.rotation.rz)[index]
                self.Xmat_sp_free = self.origin.rotation.rot(rot_builder(axis_scale * self.theta))
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(rot_builder(axis_scale * self.theta))
                S = [0, 0, 0, 0, 0, 0]
                S[index] = axis_scale
                self.S = np.array(S)
            else:
                # Tier B (general / skew axis): Rodrigues frame rotation about
                # the normalized axis; S = [axis_unit; 0] (a dense angular
                # column with >=2 nonzero entries).
                u = self._general_axis_unit(axis)
                E = self._rodrigues_frame(u, self.theta)
                self.Xmat_sp_free = self.origin.rotation.rot(E)
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(E)
                self.S = np.array([u[0], u[1], u[2], 0.0, 0.0, 0.0])
        elif self.jtype == 'prismatic':
            self.dof = 1
            self.position_symbols = [self.theta]
            self.local_q_dim = 1
            cardinal = self._cardinal_axis(axis)
            if cardinal is not None:
                # Tier A (cardinal): BYTE-IDENTICAL to the historical emit.
                index, axis_scale = cardinal
                tvec = [0, 0, 0]
                tvec[index] = axis_scale * self.theta
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(*tvec))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(*tvec)
                S = [0, 0, 0, 0, 0, 0]
                S[index + 3] = axis_scale
                self.S = np.array(S)
            else:
                # Tier B (general / skew axis): translation along the unit axis;
                # S = [0; axis_unit] (a dense linear column).
                u = self._general_axis_unit(axis)
                tvec = [u[0] * self.theta, u[1] * self.theta, u[2] * self.theta]
                self.Xmat_sp_free = self.origin.translation.xlt(self.origin.translation.skew(*tvec))
                self.Xmat_sp_hom_free = self.origin.translation.gen_tx_hom(*tvec)
                self.S = np.array([0.0, 0.0, 0.0, u[0], u[1], u[2]])
        elif self.jtype in ('helical', 'screw'):
            # HELICAL / SCREW (1-DOF, NQ=NV=1): a single scalar theta drives
            # COUPLED rotation about and translation along the SAME axis, with
            # translation = pitch * theta (pitch in meters/radian, pinocchio's
            # JointModelHelical convention). The config-space update is a plain
            # vector add (no manifold), exactly like revolute/prismatic.
            #
            # The motion subspace is a SINGLE column with a coupled linear part:
            #   S = [axis_unit ; pitch * axis_unit].
            # Even a CARDINAL axis gives >=2 nonzero entries, so a helical joint
            # is INTRINSICALLY Tier B (non-cardinal S) -> S_is_cardinal_by_id
            # returns False -> robot_has_skew_axis() trips -> the algos take the
            # dense-6-vector Tier-B path (built/validated by joint-1b). No new
            # algorithm/codegen code: the ONLY new thing here is the coupled
            # linear rows of S.
            #
            # The spatial transform is the screw motion exp(theta * S^):
            #   X = rot(axis, theta) composed with xlt(pitch * theta * axis).
            # Rotation and translation share the axis, so they commute; we apply
            # the rotation (Rodrigues frame matrix, byte-consistent with the
            # cardinal rz/ry/rx) then the axial translation. This reuses the
            # general-axis machinery from STAGE 1 verbatim.
            self.dof = 1
            self.position_symbols = [self.theta]
            self.local_q_dim = 1
            u = self._general_axis_unit(axis)
            E = self._rodrigues_frame(u, self.theta)
            tvec = [u[0] * pitch * self.theta,
                    u[1] * pitch * self.theta,
                    u[2] * pitch * self.theta]
            X_rot = self.origin.rotation.rot(E)
            X_xlt = self.origin.translation.xlt(self.origin.translation.skew(*tvec))
            self.Xmat_sp_free = X_rot * X_xlt
            # Homogeneous (forward child->parent pose). The variable axial
            # translation tvec sits in the joint frame exactly like prismatic's
            # gen_tx_hom translation, and the rotation block is E exactly like
            # revolute's rot_hom(rz). Routing through the SINGLE-DOF hom path
            # below (NOT the floating/planar path) then rotates tvec through the
            # origin and transposes the rotation block, the same convention every
            # revolute/prismatic joint uses -- so helical composes the two
            # consistently with no special-casing.
            self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(E)
            self.Xmat_sp_hom_free[:3, 3] = sp.Matrix(tvec)
            # Coupled single-column motion subspace [w; v] = [axis; pitch*axis].
            self.S = np.array([u[0], u[1], u[2],
                               pitch * u[0], pitch * u[1], pitch * u[2]])
        elif self.jtype == 'fixed':
            self.dof = 0
            self.position_symbols = []
            self.local_q_dim = 0
            self.Xmat_sp_free = sp.eye(6)
            self.Xmat_sp_hom_free = sp.eye(4)
            self.S = np.array([0,0,0,0,0,0])
        elif self.jtype == 'floating':
            self.dof = 6
            if self.using_quaternion:
                self.position_symbols = [
                    self.x_fb,
                    self.y_fb,
                    self.z_fb,
                    self.q1_fb,
                    self.q2_fb,
                    self.q3_fb,
                    self.q4_fb,
                ]
            else:
                self.position_symbols = [
                    self.x_fb,
                    self.y_fb,
                    self.z_fb,
                    self.roll_fb,
                    self.pitch_fb,
                    self.yaw_fb,
                ]
            self.local_q_dim = len(self.position_symbols)
            if self.using_quaternion:
                self.qt = Quaternion_Tools()
                quat_rot = self.qt.quat_to_rot_sp(self.q1_fb,self.q2_fb,self.q3_fb,self.q4_fb)
                rot = self.origin.rotation.rot(quat_rot)
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(quat_rot)
            else:
                rpy_rot = self.origin.rotation.rx(self.roll_fb) * \
                          self.origin.rotation.ry(self.pitch_fb) * \
                          self.origin.rotation.rz(self.yaw_fb)
                rot = self.origin.rotation.rot(rpy_rot)
                self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(rpy_rot)
            trans = self.origin.translation.xlt(self.origin.translation.skew(self.x_fb, self.y_fb, self.z_fb))
            self.Xmat_sp_hom_free[:3,3] = sp.Matrix([self.x_fb, self.y_fb, self.z_fb])
            self.Xmat_sp_free = rot*trans
            # User-facing floating-base vectors follow Pinocchio order
            # [vx, vy, vz, wx, wy, wz], while GRiD's internal spatial vectors
            # use [wx, wy, wz, vx, vy, vz].
            self.S = np.array(
                [
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        elif self.jtype == 'planar':
            # Planar joint: 3 DOF (two in-plane translations + one rotation
            # about the plane-normal axis). NQ == NV == 3 (a vector group, no
            # manifold), so the config-space update is a plain vector add --
            # this is the multi-DOF case that reuses the floating-base
            # multi-symbol bookkeeping WITHOUT the quaternion NV!=NQ wrinkle.
            #
            # GROUNDWORK ONLY: the native 6x3 motion subspace below is correct
            # for the numpy reference (which consumes the full symbolic S), but
            # the CUDA codegen emit of a multi-COLUMN non-root S is DEFERRED
            # (see Robot.get_S_index_by_id, which assumes a single signed unit
            # axis). See docs/open-tasks/joint_types_plan.md (planar, route b).
            self.dof = 3
            self.position_symbols = [self.px_pl, self.py_pl, self.theta_pl]
            self.local_q_dim = 3
            # Plane normal axis selects the rotation column and the two
            # translation columns. Default URDF planar normal is +Z (XY-plane).
            rot = self.origin.rotation.rz(self.theta_pl)
            trans = self.origin.translation.xlt(
                self.origin.translation.skew(self.px_pl, self.py_pl, 0)
            )
            self.Xmat_sp_free = self.origin.rotation.rot(rot) * trans
            self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(rot)
            self.Xmat_sp_hom_free[:3, 3] = sp.Matrix([self.px_pl, self.py_pl, 0])
            # 6x3 spatial motion subspace in internal [wx,wy,wz,vx,vy,vz] order.
            # COLUMNS MUST MATCH position_symbols / v-DOF order [px, py, theta]:
            #   col0 = translation along +X, col1 = translation along +Y,
            #   col2 = rotation about +Z.
            self.S = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        elif self.jtype == 'spherical':
            # Spherical / ball joint: 3 DOF (rotation only) parameterized by a
            # unit quaternion. NV=3, NQ=4 -> NV!=NQ, exactly the floating-base
            # rotation sub-pattern. The config-space update is an SO(3)
            # quaternion exp (NOT a vector add).
            #
            # GROUNDWORK ONLY: the native 6x3 angular S below is correct for
            # the symbolic transform, but BOTH (a) the CUDA codegen emit of a
            # multi-column non-root S AND (b) the per-joint SO(3) retract in
            # RBDReference.integrate/dIntegrate (today hardcoded to a single
            # free-flyer prefix) are DEFERRED. See
            # docs/open-tasks/joint_types_plan.md (spherical).
            self.dof = 3
            self.position_symbols = [self.q1_sph, self.q2_sph, self.q3_sph, self.q4_sph]
            self.local_q_dim = 4
            self.qt = Quaternion_Tools()
            # quat_to_rot_sp returns the FORWARD (child->parent) rotation R(quat).
            # The SPATIAL motion transform X (parent->child frame) uses the FRAME
            # rotation R^T (matching rz/ry/rx, which return exp(-[axis]theta)=R^T).
            # A mid-chain spherical joint is consumed directly in the v/a
            # recursion (X @ v_parent) with NO inverse at the use-site (unlike the
            # floating ROOT, whose fpass explicitly inverts Xmat), so the
            # transpose must be baked into the spatial X here. Without it the
            # rotation is applied backwards and the gravity/Coriolis recursion
            # diverges from Pinocchio.
            quat_rot = self.qt.quat_to_rot_sp(
                self.q1_sph, self.q2_sph, self.q3_sph, self.q4_sph
            )
            self.Xmat_sp_free = self.origin.rotation.rot(quat_rot.transpose())
            # The HOMOGENEOUS transform encodes the forward child->parent pose
            # (used for EE/world kinematics), so it keeps R (no transpose), the
            # same convention the floating root's hom uses.
            self.Xmat_sp_hom_free = self.origin.rotation.rot_hom(quat_rot)
            # 6x3 angular-only motion subspace in internal [w;v] order.
            self.S = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            # Unsupported joint type. Raise a typed, catchable exception
            # instead of the old print()+exit() which killed the host process
            # with an uncatchable SystemExit and no traceback. Callers can now
            # catch URDFParseError / UnsupportedJointTypeError and surface a
            # structured error (e.g. the equivalence harness skips the robot).
            # see docs/open-tasks/notes.md (Joint.py:260)
            raise UnsupportedJointTypeError(jtype, joint_name=self.name)
        self.Xmat_sp = self.Xmat_sp_free * self.origin.Xmat_sp_fixed
        # remove numerical noise (e.g., URDF's often specify angles as 3.14 or 3.14159 but that isn't exactly PI)
        self.Xmat_sp = sp.nsimplify(self.Xmat_sp, tolerance=1e-6, rational=True).evalf()
        # Multi-DOF non-root joints (planar, spherical) carry their variable
        # translation/rotation directly in Xmat_sp_hom_free (like floating),
        # so they use the floating-style direct hom composition rather than the
        # single-DOF "rotate t_free through origin rpy" path.
        if self.jtype not in ('floating', 'planar', 'spherical'):
            # homogenous transform needs to "sum" translation and rotation. The
            # joint's variable translation t_free is in the JOINT frame; before
            # adding it to the origin's offset (in the PARENT frame) it must be
            # rotated through the origin's rotation. Skipping this is invisible
            # on robots where origin rpy = 0 (every iiwa/go2/g1 revolute) but
            # mis-places joints whose origin specifies a non-identity rotation
            # (e.g. fr3_finger_joint2's origin rpy = π around Z mirrors the
            # prismatic Y translation into -Y in the parent frame).
            self.Xmat_sp_hom = sp.eye(4)
            self.Xmat_sp_hom[:3,:3] = (self.Xmat_sp_hom_free[:3,:3] * self.origin.Xmat_sp_hom_fixed[:3,:3]).transpose()
            self.Xmat_sp_hom[:3,3] = (self.origin.Xmat_sp_hom_fixed[:3,:3] * self.Xmat_sp_hom_free[:3,3]
                                      + self.origin.Xmat_sp_hom_fixed[:3,3])
            self.Xmat_sp_hom = sp.nsimplify(self.Xmat_sp_hom, tolerance=1e-6, rational=True).evalf()
            # and derivative
            self._build_homogeneous_transform_derivatives()
        else:
            self.Xmat_sp_hom = self.Xmat_sp_hom_free * self.origin.Xmat_sp_hom_fixed
            self.Xmat_sp_hom = sp.nsimplify(self.Xmat_sp_hom, tolerance=1e-6, rational=True).evalf()
            self._build_homogeneous_transform_derivatives()

    def get_transformation_matrix_function(self):
        # Memoize the lambdified transform: it is a pure function of the joint's
        # constant symbolic Xmat (the mimic multiplier/offset is applied to the
        # numeric q BEFORE the call, so the function itself never varies). crba/
        # forward_dynamics rebuild this per-body, and fd_grad_at finite-differences
        # the whole chain ~2*nv times, so without caching a big mimic robot (h1_2)
        # triggers ~1e4-1e5 lambdify builds and the reference hangs for many minutes.
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        if 'Xmat' not in cache:
            if self.jtype in ("floating", "planar", "spherical"):
                cache['Xmat'] = sp.utilities.lambdify(self._local_q_lambdify_args(), self.Xmat_sp, 'numpy')
            else:
                cache['Xmat'] = sp.utilities.lambdify(self.theta, self.Xmat_sp, 'numpy')
        return cache['Xmat']

    def get_transformation_matrix(self):
        return self.Xmat_sp

    # ------------------------------------------------------------------
    # Runtime-mutable joint-frame transform (runtime_transform path).
    # Mirrors Link.get_inertia_params / the runtime_inertia machinery: expose
    # the RAW URDF <origin> scalars in a frozen basis so the on-device prologue
    # can rebuild the constant 6x6 Xfixed = rot(E(rpy))*xlt(skew(xyz)) once per
    # launch, and provide a symbolic-Xfixed transform whose origin coefficients
    # are NAMED SYMBOLS (xf_*) instead of folded numeric literals -> the codegen
    # bakes the general-rpy DENSE sparsity and hoists the origin out of the hot
    # sin/cos(q) loop into s_Xfixed scratch loads. EE/homogeneous transforms are
    # OUT OF SCOPE for v1 (they keep the baked inline literals).
    # ------------------------------------------------------------------
    # frozen origin basis: [x, y, z, roll, pitch, yaw]
    ORIGIN_PARAM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
    # the 27 structurally-nonzero cells of Xfixed live in the TL / BL / BR 3x3
    # blocks (TR is identically zero for any rpy); symbol name -> (row,col).
    @staticmethod
    def _runtime_xfixed_symbol_cells():
        cells = {}
        for blk, (r0, c0) in (("TL", (0, 0)), ("BL", (3, 0)), ("BR", (3, 3))):
            for i in range(3):
                for j in range(3):
                    cells["xf_" + blk + "_" + str(i) + "_" + str(j)] = (r0 + i, c0 + j)
        return cells

    def get_origin_params(self):
        """Raw URDF <origin> scalars [x, y, z, roll, pitch, yaw] (the frozen
        basis the runtime_transform table holds). Read VERBATIM from the parsed
        Origin (post pi-snap) so the on-device rebuild reproduces the baked
        Xfixed bit-for-bit until set_transform_params mutates the table. The
        floating root joint owns no fixed origin offset of its own (its origin
        is identity by construction), so it returns all-zeros."""
        if self.origin is None or self.origin.translation is None or self.origin.rotation is None:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t = self.origin.translation
        r = self.origin.rotation
        return [float(t.x), float(t.y), float(t.z),
                float(r.r), float(r.p), float(r.y)]

    def get_runtime_transform_matrix(self):
        """Spatial X transform with the origin block carried as NAMED SYMBOLS.

        Returns Xmat_sp_free(q) * Xfixed_symbolic where Xfixed_symbolic is the
        6x6 origin transform with its 27 structurally-nonzero cells replaced by
        the xf_* symbols (TR block = 0). Substituting the numeric Xfixed cells
        reproduces self.Xmat_sp exactly (verified). Because the origin cells are
        symbols (never zero), the product carries the GENERAL-rpy DENSE sparsity
        regardless of this joint's actual rpy -> rpy can move freely at runtime.

        Multi-DoF roots (floating/planar/spherical) carry their variable origin
        differently and are OUT OF SCOPE for v1: return the normal baked Xmat_sp
        (the prologue still writes their Xfixed table slot to identity, unused)."""
        if self.jtype in ("floating", "planar", "spherical"):
            return self.Xmat_sp
        if self.Xmat_sp_free is None:
            return self.Xmat_sp
        xf = sp.zeros(6, 6)
        for name, (row, col) in self._runtime_xfixed_symbol_cells().items():
            xf[row, col] = sp.symbols(name)
        X = self.Xmat_sp_free * xf
        # match the noise-removal the baked path applies to Xmat_sp; nsimplify
        # over the q-trig + xf symbols leaves the linear xf coefficients intact.
        return sp.nsimplify(X, tolerance=1e-6, rational=True).evalf()

    def get_transformation_matrix_hom_function(self):
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        if 'Xmat_hom' not in cache:
            cache['Xmat_hom'] = sp.utilities.lambdify(self._local_q_lambdify_args(), self.Xmat_sp_hom, 'numpy')
        return cache['Xmat_hom']

    def get_transformation_matrix_hom(self):
        return self.Xmat_sp_hom

    def get_dtransformation_matrix_hom_function(self):
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        if 'dXmat_hom' not in cache:
            cache['dXmat_hom'] = sp.utilities.lambdify(self._local_q_lambdify_args(), self.dXmat_sp_hom, 'numpy')
        return cache['dXmat_hom']

    def get_d2transformation_matrix_hom_function(self):
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        if 'd2Xmat_hom' not in cache:
            cache['d2Xmat_hom'] = sp.utilities.lambdify(self._local_q_lambdify_args(), self.d2Xmat_sp_hom, 'numpy')
        return cache['d2Xmat_hom']

    def get_dtransformation_matrix_hom(self):
        return self.dXmat_sp_hom

    def get_d2transformation_matrix_hom(self):
        return self.d2Xmat_sp_hom

    def get_local_q_dim(self):
        return self.local_q_dim

    def get_dtransformation_matrix_hom_local(self, local_index):
        return self.dXmat_sp_hom_blocks[local_index]

    def get_d2transformation_matrix_hom_local(self, local_index_i, local_index_j):
        return self.d2Xmat_sp_hom_blocks[local_index_i][local_index_j]

    def get_d2transformation_matrix_local(self, local_index_i, local_index_j):
        return sp.diff(
            sp.diff(self.Xmat_sp, self.position_symbols[local_index_i]),
            self.position_symbols[local_index_j],
        )

    def get_dtransformation_matrix_hom_local_function(self, local_index):
        # Memoize (index-keyed): same pure-function cache as the scalar getters.
        # These also recompute their symbolic derivative each call, so caching the
        # lambda caches that too — matters for the floating/multi-DoF local-derivative
        # paths (ee_pose_hessian / fdsva_so references) under finite-differencing.
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        key = ('dXmat_hom_local', local_index)
        if key not in cache:
            cache[key] = sp.utilities.lambdify(
                self._local_q_lambdify_args(),
                self.get_dtransformation_matrix_hom_local(local_index),
                'numpy',
            )
        return cache[key]

    def get_d2transformation_matrix_hom_local_function(self, local_index_i, local_index_j):
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        key = ('d2Xmat_hom_local', local_index_i, local_index_j)
        if key not in cache:
            cache[key] = sp.utilities.lambdify(
                self._local_q_lambdify_args(),
                self.get_d2transformation_matrix_hom_local(local_index_i, local_index_j),
                'numpy',
            )
        return cache[key]

    def get_d2transformation_matrix_local_function(self, local_index_i, local_index_j):
        cache = self.__dict__.setdefault('_lambdify_cache', {})
        key = ('d2Xmat_local', local_index_i, local_index_j)
        if key not in cache:
            cache[key] = sp.utilities.lambdify(
                self._local_q_lambdify_args(),
                self.get_d2transformation_matrix_local(local_index_i, local_index_j),
                'numpy',
            )
        return cache[key]

    def get_joint_subspace(self):
        return self.S

    def get_damping(self):
        return self.damping

    def get_friction(self):
        return self.friction

    def get_name(self):
        return self.name

    def get_id(self):
        return self.jid

    def get_bfs_id(self):
        return self.bfs_id

    def get_bfs_level(self):
        return self.bfs_level

    def get_parent(self):
        return self.parent

    def get_child(self):
        return self.child
    
    def get_num_dof(self):
        # Mimic joints expose their motion through the mimicked joint and do
        # not own a generalized coordinate of their own, so they contribute
        # zero to nv/nq.
        if self.is_mimic:
            return 0
        return self.dof

    def set_mimic(self, joint_name, multiplier=1.0, offset=0.0):
        """Mark this joint as a URDF mimic of `joint_name`.

        The relation is stored by NAME at parse time (target joint may not
        be parsed yet). `URDFParser` resolves `mimic_target_id` to the
        target joint's final jid after renumbering.
        """
        self.is_mimic = True
        self.mimic_joint_name = joint_name
        self.mimic_multiplier = float(multiplier)
        self.mimic_offset = float(offset)

    def is_mimic_joint(self):
        return self.is_mimic

    def get_mimic_joint_name(self):
        return self.mimic_joint_name

    def get_mimic_multiplier(self):
        return float(self.mimic_multiplier)

    def get_mimic_offset(self):
        return float(self.mimic_offset)

    def get_mimic_target_id(self):
        return self.mimic_target_id
    
    def get_joint_limits(self):
        return self.joint_limits

    def set_velocity_limit(self, v):
        self.velocity_limit = v

    def get_velocity_limit(self):
        return self.velocity_limit

    def set_effort_limit(self, e):
        self.effort_limit = e

    def get_effort_limit(self):
        return self.effort_limit

# Need to retain fixed joints for possible kinematic use later
class Fixed_Joint:
    def __init__(self, jid_in, name, parent_name, hom_xfrm):
        self.jid = jid_in                    # original ID
        self.name = name                # name
        self.parent_name = parent_name  # parent joint name
        self.Xmat_hom = hom_xfrm

    def set_id(self, jid_in):
        self.jid = jid_in

    def set_parent(self, parent_in):
        self.parent_name = parent_in

    def set_transformation_matrix_hom(self, hom_xfrm):
        self.Xmat_hom = hom_xfrm

    def get_id(self):
        return self.jid

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent_name

    def get_transformation_matrix_hom(self):
        return self.Xmat_hom
