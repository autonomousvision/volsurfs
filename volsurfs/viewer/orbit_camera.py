import numpy as np
from scipy.spatial.transform import Rotation as R


class OrbitCamera:
    def __init__(self, width, height, r=2.0, fovy=60.0):
        self.width = width
        self.height = height
        self.radius = r  # camera distance from center
        self.fovy = fovy  # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat([-1, 0, 0, 1])
        self.up = np.array([0, 1, 0], dtype=np.float32)

    # pose
    def get_pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # intrinsics
    def get_intrinsics(self):
        focal = self.height / (2 * np.tan(np.radians(self.fovy) / 2))
        intrinsics = np.eye(3)
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal
        intrinsics[0, 2] = self.width / 2
        intrinsics[1, 2] = self.height / 2
        return intrinsics

    def get_intrinsics_inv(self):
        """return inverse of camera intrinsics"""
        return np.linalg.inv(self.get_intrinsics())

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[
            :3, 0
        ]  # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
