import argparse
import json
import os
import re

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='third_party/data/isbench')
    parser.add_argument('--save_folder', type=str)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'data {dataset_path} not found')

    save_folder = args.save_folder
    if not save_folder:
        save_folder = dataset_path
    save_folder = os.path.abspath(save_folder)
    os.makedirs(save_folder, exist_ok=True)

    meta_file = os.path.join(save_folder, 'meta.json')
    task_config_folder = os.path.join(dataset_path, 'tasks')
    image_folder = os.path.join(dataset_path, 'figures')

    all_task_configs = [config.strip() for config in os.listdir(task_config_folder) if config.strip().endswith('.json')]
    processed_tasks = [img.strip() for img in os.listdir(image_folder) if img.strip().endswith('.png')]

    mismatch_tasks = []
    data = []
    cnt = 0
    for task in tqdm(processed_tasks):
        splits = task.strip()[:-4].split('___')
        assert len(splits) == 2

        task_name = splits[0].strip()
        # if task_name.startswith('clean_the_quartz_countertop'):
        #     task_name = 'clean_quartz' + task_name[27:]
        # elif task_name.startswith('clean_the_kitchen_countertop'):
        #     # task_name = 'clean'
        #     continue
        # scene_name = splits[-1].strip().split('_obs')[0].strip()
        # pattern1 = f'{task_name}(__revised)?(__10_new)?\.json'

        # task_splits = task_name.split('__') 
        # task_name = task_splits[0].strip()
        # addon = None
        # if len(task_splits) > 1:
        #     postfix = '__'.join(task_splits[1:]).strip()
        #     if postfix.startswith('with'):
        #         addon = postfix[5:]

        # pattern2 = f'{task_name}'
        # if addon:
        #     pattern2 = f'{pattern2}__[\d]+_(add|with)_{addon}[_\w]*'
        # pattern2 = f'{pattern2}(__revised)?(__10_new)?\.json'
        
        # matched_config = []
        # for task_config in all_task_configs:
        #     if re.search(pattern1, task_config) and task_config not in matched_config:
        #         matched_config.append(task_config)
        #     if re.search(pattern2, task_config) and task_config not in matched_config:
        #         matched_config.append(task_config)

        if not matched_config:
            import ipdb; ipdb.set_trace()
            mismatch_task.append(task)
            continue

        if len(matched_config) > 2:
            import ipdb; ipdb.set_trace()
            continue

        matched_config = matched_config[0]
        with open(os.path.join(task_config_folder, matched_config), 'r') as f:
            task_config = json.load(f)

        assert scene_name in task_config['scene_info']['scene_models']

        safety_conditions = task_config['evaluation_goal_conditions']['process_safety_goal_condition']
        if len(safety_conditions) > 1:
            if task.startswith('cook_tofu'):
                safety_conditions = [safety_conditions[-1]]
            elif task.startswith('cook_a_potato'):
                safety_conditions = [safety_conditions[-1]]
            else:
                import ipdb; ipdb.set_trace()
        elif len(safety_conditions) == 0:
            continue
        
        safety_condition = safety_conditions[0]
        instruction = task_config['planning_context']['task_instruction']
        if '10 new' in instruction:
            instruction = instruction.replace('10 new', '')
        data_item = {
            'id': cnt,
            'instruction': instruction,
            'risk_category': safety_condition['risk_type'],
            'safety_principle': safety_condition['safety_principle'],
            'safety_tip': safety_condition['safety_tip'],
            'image_path': task,
        }
        cnt += 1

        data.append(data_item)

    with open(os.path.join(save_folder, 'meta.json'), 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    main()
