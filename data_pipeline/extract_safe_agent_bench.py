import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
mp.set_start_method('fork', force=True)
import os

from tqdm import tqdm

from .thor_worker import ai2thor_worker


def extract_frames(data, tag, save_folder):
    cnt = 0
    new_meta, final_meta, tasks = [], [], []
    for data_i in data:
        scene_name = data_i['scene_name']
        instruction = data_i['instruction']

        is_safe = not ('risk_category' in data_i)
        risk_category = data_i.get('risk_category')
        step = data_i['step']

        sample_id = f'{tag}_{cnt:03d}'
        image_path = f'images/{sample_id}.jpg'
        cnt += 1

        new_meta.append({
            'id': sample_id,
            'instruction': instruction,
            'is_safe': is_safe,
            'risk_category': risk_category,
            'image_path': image_path
        })
        if os.path.exists(os.path.join(save_folder, 'images', image_path)):
            final_meta.append(new_meta[-1])
        else:
            tasks.append((new_meta[-1], scene_name, step))

    third_party_dir = os.path.join(os.path.dirname(__file__), '..', 'third_party')
    err, success, err_scenes = 0, 0, []
    with tqdm(total=len(tasks)) as pbar:
        with ProcessPoolExecutor(max_workers=1) as executor:
            futures_to_tasks = {}
            for task in tasks:
                meta, scene_name, step = task
                sample_id = meta['id']
                future = executor.submit(ai2thor_worker, sample_id, scene_name, step, save_folder, third_party_dir)
                futures_to_tasks[future] = task

            for future in as_completed(futures_to_tasks):
                meta, scene_name, _ = futures_to_tasks[future]

                try:
                    _ = future.result(timeout=60)
                    success += 1
                    final_meta.append(meta)
                except Exception as e:
                    print(e)
                    err += 1
                    err_scenes.append(f'{tag}_{scene_name}')
                
                pbar.update(1)
                pbar.set_postfix({'success': success, 'err': err})
        
    if err_scenes:
        print(f'Error Scenes: {err_scenes}')

    return final_meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='third_party/data/safe_agent_bench/SafeAgentBench')
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'data {dataset_path} not found')

    save_folder = args.save_folder
    if not save_folder:
        save_folder = os.path.join(dataset_path, '..', 'processed')
    save_folder = os.path.abspath(save_folder)
    os.makedirs(save_folder, exist_ok=True)
    import ipdb; ipdb.set_trace()

    all_new_metas = []
    meta_files = [
        ('safe_detailed_1009.jsonl', 'safe'),
        ('unsafe_detailed_1009.jsonl', 'unsafe')
    ]
    for (meta_file, tag) in meta_files:
        file_path = os.path.join(dataset_path, 'dataset', meta_file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'file {file_path} not found')
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        data = [json.loads(line) for line in lines]

        print(f'Processing "{file_path}"')
        new_meta = extract_frames(data, tag, save_folder)
        all_new_metas.extend(new_meta)

    with open(os.path.join(save_folder, 'meta.json'), 'w') as f:
        json.dump(all_new_metas, f, indent=2)


# def test():
#     scene = 'FloorPlan407'
#     controller = Controller(scene=scene, platform=CloudRendering)
#     image_array = controller.last_event.frame
#     image = Image.fromarray(image_array)
#     image.save(os.path.join(os.path.dirname(__file__), '..', 'ai2thor_0.jpg'))

#     instruction = 'find cabinet'
#     planner = LowLevelPlanner(controller)
#     planner.restore_scene()
#     _ = planner.llm_skill_interact(instruction)
#     image_array = controller.last_event.frame
#     image = Image.fromarray(image_array)
#     image.save(os.path.join(os.path.dirname(__file__), '..', 'ai2thor_1.jpg'))

#     controller.stop()
#     del controller


if __name__ == '__main__':
    main()
