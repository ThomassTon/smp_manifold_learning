import numpy as np
from abc import ABC, abstractmethod
import robotic as ry




class Feature(ABC):
    @abstractmethod
    def __init__(self, name, dim_ambient, dim_feature):
        self.name = name
        self.dim_ambient = dim_ambient
        self.dim_feature = dim_feature
        self.is_inequality = False

    @abstractmethod
    def y(self, x):
        pass

    @abstractmethod
    def J(self, x):
        pass

    @abstractmethod
    def draw(self, limits):
        pass

    @property
    @abstractmethod
    def draw_type(self):
        pass


class LoopFeature(Feature):
    def __init__(self, r):
        Feature.__init__(self, "Loop", dim_ambient=3, dim_feature=2)
        self.r = r

    def y(self, x):
        return np.array([x[0]**2 + x[1]**2 - self.r ** 2,
                         x[2]])

    def J(self, x):
        return np.array([[2*x[0], 2*x[1], 0.0],
                         [0.0, 0.0, 1.0]])

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Scatter"


class FeatureStack(Feature):
    def __init__(self, features_):
        self.features = features_
        dim_feature = 0
        for f in self.features:
            dim_feature += f.dim_feature
        Feature.__init__(self, "FeatureStack", dim_ambient=3, dim_feature=dim_feature)

    def y(self, q):
        out = np.empty(0)
        for f in self.features:
            out = np.append(out, f.y(q))
        return out

    def J(self, q):
        out = np.empty((0, len(q)))
        for f in self.features:
            out = np.append(out, f.J(q), axis=0)
        return out

    def draw(self, limits):
        return [], [], []

    @property
    def draw_type(self):
        return "Stacked"


class PointFeature(Feature):
    def __init__(self, goal):
        Feature.__init__(self, "Point", dim_ambient=goal.shape[0], dim_feature=goal.shape[0])
        self.goal = goal

    def y(self, x):
        return self.goal - x

    def J(self, x):
        return -np.eye(self.goal.shape[0])

    def draw(self, limits):
        return [self.goal[0]], [self.goal[1]], [self.goal[2]]

    @property
    def draw_type(self):
        return "Scatter"
    
class PointFeature_q(Feature):
    def __init__(self, goal):
        Feature.__init__(self, "Point", dim_ambient=goal.shape[0], dim_feature=goal.shape[0])
        self.goal = goal
        self.C = ry.Config()
        self.C.clear()
        self.C.addFile('../scene/panda.g')
    def y(self, x):
        self.C.setJointState(x)
        pos = self.C.getFrame('gripper').getPosition()
        return self.goal - pos

    def J(self, x):
        self.C.setJointState(x)
        [y,J] = self.C.eval(ry.FS.position,['gripper'])

        return -J

    def draw(self, limits):
        return [self.goal[0]], [self.goal[1]], [self.goal[2]]

    @property
    def draw_type(self):
        return "Scatter"

class ParaboloidFeature(Feature):
    def __init__(self, A, b, c):
        Feature.__init__(self, "Paraboloid", dim_ambient=3, dim_feature=1)
        self.A = A
        self.b = b
        self.c = c

    def y(self, x):
        return np.array([np.transpose(x[:2]) @ self.A @ x[:2] + np.dot(self.b, x[:2]) + self.c - x[2]])

    def J(self, x):
        return np.array([[2.0 * self.A[0, 0] * x[0] + self.A[0, 1] * x[1] + self.A[1, 0] * x[1] + self.b[0],
                          2.0 * self.A[1, 1] * x[1] + self.A[0, 1] * x[0] + self.A[1, 0] * x[0] + self.b[1],
                          -1]])

    def draw(self, limits):
        x_min, x_max, y_min, y_max = limits
        n = 100
        X = np.linspace(x_min, x_max, n)
        Y = np.linspace(y_min, y_max, n)
        X, Y = np.meshgrid(X, Y)
        f = lambda x: np.matmul(x.transpose(), np.matmul(self.A, x)) + np.dot(self.b, x) + self.c
        Z = [f(np.array([x, y])) for (x, y) in np.nditer([X, Y])]
        Z = np.asarray(Z).reshape([n, n])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class SphereFeature(Feature):
    def __init__(self, r):
        Feature.__init__(self, "Sphere", dim_ambient=3, dim_feature=1)
        self.r = r

    def y(self, x):
        print("i am here!!!!!!!\n")
        return np.array([np.dot(x, x) - self.r ** 2])

    def J(self, x):
        return 2.0 * np.array([[x[0], x[1], x[2]]])

    def param_to_xyz(self, param):
        theta = param[0]
        phi = param[1]
        if np.isscalar(theta):
            x = self.r * np.cos(theta) * np.sin(phi)
            y = self.r * np.sin(theta) * np.sin(phi)
            z = self.r * np.cos(phi)
        else:
            x = self.r * np.outer(np.cos(theta), np.sin(phi))
            y = self.r * np.outer(np.sin(theta), np.sin(phi))
            z = self.r * np.outer(np.ones_like(theta), np.cos(phi))

        return x, y, z

    def draw(self, limits):
        n = 100
        theta = np.linspace(0, 2 * np.pi, n)
        phi = np.linspace(0, np.pi, n)

        X, Y, Z = self.param_to_xyz([theta, phi])
        return X, Y, Z

    @property
    def draw_type(self):
        return "Surface"


class Projection:
    def __init__(self, f_, J_, tol_=1e-2, max_iter_=2000, step_size_=0.05):
        self.f = f_
        self.J = J_
        self.tol = tol_
        self.max_iter = max_iter_
        self.step_size = step_size_

    def project(self, q):
        y = self.f(q)
        y0 = 2.0 * np.linalg.norm(y)
        iter = 0
        while np.linalg.norm(y) > self.tol and iter < self.max_iter and np.linalg.norm(y) < y0:
            J = self.J(q)
            # print(J)
            q = q - self.step_size * np.linalg.lstsq(J, y, rcond=-1)[0]
            y = self.f(q)

            iter += 1
        print(np.linalg.norm(y))
        result = np.linalg.norm(y) <= self.tol
        return result, np.array(q)
    



class PandaQuaternionFeature(Feature):
    def __init__(self):
        Feature.__init__(self, "Quaternion", dim_ambient=7, dim_feature=4)
        self.target_quaternion = np.array([0, 1, 0, 0])
        self.C = ry.Config()
        self.C.clear()

        self.C.addFile('../scene/panda.g')
        # self.C.view(True)
    def quaternion_angle_difference(self,q1, q2):
        # 确保四元数是单位四元数（长度为1）
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # 计算四元数之间的点积
        dot_product = np.dot(q1, q2)
        
        # 处理数值误差，确保点积在 [-1, 1] 范围内
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 计算夹角（弧度）
        angle_rad = 2 * np.arccos(np.abs(dot_product))
        
        # 转换为角度
        
        return angle_rad
    

    def y(self, q):
        self.C.setJointState(q)
        gripper = self.C.getFrame("gripper")
        q_target = np.array([0,1,0,0])
        q = gripper.getQuaternion()
        q = np.array(q)  # 确保 q 是 NumPy 数组
        # self.C.view(True)
        return np.abs(q)-np.abs(q_target)

    def J(self, q):
        """
        雅可比矩阵，这里是恒等矩阵，因为对每个分量求导结果是单位。
        """
        self.C.setJointState(q)
        [y,J] = self.C.eval(ry.FS.quaternion,['gripper'])
        # print('feature value:', y, '\nJacobian:', J)

        return J

    def param_to_quaternion(self, param):
        """
        由参数生成四元数。这里 param 是一个小扰动，
        假设它在 3 维空间中，生成接近目标四元数的值。
        """
        delta = np.array([0] + list(param))  # 扩展到 4 维，保持第一个分量为 0
        return self.target_quaternion + delta

    def draw(self, limits):
        """
        可视化接近目标四元数的流形。这里只是返回目标四元数的位置。
        """
        n = 1
        quaternions = np.tile(self.target_quaternion, (n, 1))
        return quaternions

    @property
    def draw_type(self):
        return "Point"

if __name__ == "__main__":
    p = PandaQuaternionFeature()
    q = np.array((-1.19359, 0.923462, 0.293893,-0.976063, 2.8973, 1.27181, 0.153865))
    print(p.y(q))
    p.J(q)
