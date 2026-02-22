import multiprocessing
from multiprocessing import Process

import numpy as np
import torch
import torchvision.transforms as transforms

import environment


class Hw1Env(environment.BaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_scene(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        scene = environment.create_tabletop_scene()
        r = np.random.rand()
        if r < 0.5:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "box", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.02, 0.005, 0.0001],
                                      density=4000, name="obj1")
        else:
            size = np.random.uniform([0.02, 0.02, 0.02], [0.03, 0.03, 0.03])
            environment.create_object(scene, "sphere", pos=[0.6, 0., 1.1], quat=[0, 0, 0, 1],
                                      size=size, rgba=[0.8, 0.2, 0.2, 1], friction=[0.2, 0.005, 0.0001],
                                      density=4000, name="obj1")
        return scene

    def state(self):
        obj_pos = self.data.body("obj1").xpos[:2]
        if self._render_mode == "offscreen":
            self.viewer.update_scene(self.data, camera="topdown")
            pixels = torch.tensor(self.viewer.render().copy(), dtype=torch.uint8).permute(2, 0, 1)
        else:
            pixels = self.viewer.read_pixels(camid=1).copy()
            pixels = torch.tensor(pixels, dtype=torch.uint8).permute(2, 0, 1)
            pixels = transforms.functional.center_crop(pixels, min(pixels.shape[1:]))
            pixels = transforms.functional.resize(pixels, (128, 128))
        return obj_pos, pixels

    def step(self, action_id):
        if action_id == 0:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 1:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.4, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.8, 0, 1.065], rotation=[-90, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 2:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})
        elif action_id == 3:
            self._set_joint_position({6: 0.8})
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, -0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_ee_in_cartesian([0.6, 0.2, 1.065], rotation=[0, 0, 180], n_splits=50)
            self._set_joint_position({i: angle for i, angle in enumerate(self._init_position)})


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")

    initial_obj_positions = torch.zeros(N, 2, dtype=float)

    initial_imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    actions = torch.zeros(N, dtype=torch.uint8)
    final_imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)

    final_obj_positions = torch.zeros(N, 2, dtype=float)

    for i in range(N):
        env.reset()
        initial_obj_pos, initial_pixels = env.state()
        action_id = np.random.randint(4)
        env.step(action_id)
        final_obj_pos, final_pixels = env.state()

        initial_obj_positions[i] = torch.tensor(initial_obj_pos)
        initial_imgs[i] = initial_pixels
        actions[i] = action_id
        final_imgs[i] = final_pixels
        final_obj_positions[i] = torch.tensor(final_obj_pos)

    torch.save(initial_obj_positions, f"dataset/initial_obj_positions_test_{idx}.pt")
    torch.save(initial_imgs, f"dataset/initial_imgs_test_{idx}.pt")
    torch.save(actions, f"dataset/actions_test_{idx}.pt")
    torch.save(final_imgs, f"dataset/final_imgs_test_{idx}.pt")
    torch.save(final_obj_positions, f"dataset/final_obj_positions_test_{idx}.pt")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    processes = []
    for i in range(4):
        p = Process(target=collect, args=(i, 50))  # THE FINAL VERSION TO COLLECT TEST DATA (4 x 50 = 200)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
