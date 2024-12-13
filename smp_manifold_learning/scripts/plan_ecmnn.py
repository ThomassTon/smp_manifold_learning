import numpy as np
from smp_manifold_learning.differentiable_models.ecmnn import EqualityConstraintManifoldNeuralNetwork
from smp_manifold_learning.motion_planner.feature import Feature
from smp_manifold_learning.motion_planner.smp_star import SMPStar
from smp_manifold_learning.motion_planner.util import read_cfg
from smp_manifold_learning.motion_planner.task import Task
import robotic as ry
import time

class LearnedManifold(Feature):
    def __init__(self, model_path):
        Feature.__init__(self, "Sphere", dim_ambient=7, dim_feature=4)
        self.r = 1
        self.ecmnn = EqualityConstraintManifoldNeuralNetwork(input_dim=7,
                                                             hidden_sizes=[128, 64, 32, 16],
                                                             output_dim=4,
                                                             use_batch_norm=True, drop_p=0.0,
                                                             is_training=False, device='cpu')
        self.ecmnn.load(model_path)

    def y(self, x):
        # print("y: ",self.ecmnn.y(x))
        return self.ecmnn.y(x)

    def J(self, x):
        return self.ecmnn.J(x)

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


cfg = '[general]\n' \
      'SEED           = 4\n' \
      'DEBUG          = false\n' \
      'ENV            = 3d_point\n' \
      'CONV_TOL       = 0.01\n' \
      'VIS            = 1\n' \
      'N              = 10000\n' \
      'ALPHA          = 0.5\n' \
      'BETA           = 0.5\n' \
      'COLLISION_RES  = 0.1\n' \
      'EPS            = 1e-1\n' \
      'RHO            = 5e-1\n' \
      'R_MAX          = 1.5\n' \
      'PROJ_STEP_SIZE = 0.5'

# task_name = 'sphere'
task_name = 'panda'

model_path = '../plot/ecmnn/model_samples_epoch24.pth'

cfg = read_cfg(cfg)
np.random.seed(cfg['SEED'])
task = Task(task_name)

if task_name == 'sphere':
    task.manifolds[0] = LearnedManifold(model_path=model_path)
# elif task_name == 'hourglass_sphere':
    # task.manifolds[1] = LearnedManifold(model_path=model_path)
elif task_name == 'panda':
    task.manifolds[0] = LearnedManifold(model_path=model_path)
# smp_manifold_learning/plot/ecmnn/model_samples_epoch24.pth
planner = SMPStar(task=task, cfg=cfg)
path_idx, path = planner.run()
print(path_idx)
C = ry.Config()
C.clear()

C.addFile('../scene/panda.g')
for q in path[0]:
    print(q)
    C.setJointState(q)
    C.view(False,"panda")
    time.sleep(.1)
    C.view_savePng("video/panda/panda")
    time.sleep(.1)
# if path_idx is not None:
#     task.plot(plot_dir='../plot/ecmnn/',
#               G_list=planner.G_list,
#               V_goal_list=planner.V_goal_list,
#               opt_path=path)
