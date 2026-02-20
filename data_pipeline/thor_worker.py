import contextlib
import os
import sys
from typing import  List

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image


def ai2thor_worker(sample_id, scene_name, step, save_folder, third_party_dir):
    image_folder = os.path.join(save_folder, 'images')
    os.makedirs(image_folder, exist_ok=True)

    @contextlib.contextmanager
    def add_sys_path(path: str | List[str]):
        if isinstance(path, str):
            path = [path]

        paths_to_add = [p for p in path if p not in sys.path and os.path.exists(p)]

        for p in paths_to_add[::-1]:
            sys.path.insert(0, p)

        try:
            yield
        finally:
            for p in paths_to_add:
                if p in sys.path:
                    sys.path.remove(p)

    with add_sys_path(os.path.join(third_party_dir, 'data', 'safe_agent_bench', 'SafeAgentBench')): 
        from low_level_controller.low_level_controller import LowLevelPlanner

    controller = Controller(scene=scene_name, platform=CloudRendering)
    try:
        if len(step) > 0 and step[0].strip().startswith('find'):
            planner = LowLevelPlanner(controller)
            planner.restore_scene()
            ret = planner.llm_skill_interact(step[0])
            if not ret['success']:
                raise RuntimeError()
            image_array = controller.last_event.frame
        else:
            image_array = controller.last_event.frame

        image = Image.fromarray(image_array)
        image.save(os.path.join(image_folder, f'{sample_id}.jpg'))
    finally:
        controller.stop()
        del controller
