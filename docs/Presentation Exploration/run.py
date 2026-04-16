import os
from pathlib import Path

# Adjust paths since we're now in experiments/ subdirectory
BASE_PATH = Path(__file__).parent.parent

def run(provider, model_name, dataset, api_key_pos, defense, prompt_type, icl_num, gpus, adaptive_attack_on_pi, redundant_info_filtering):
    model_config_path = '../configs/model_configs/' + provider + '_config.json'
    task_config_path = '../configs/task_configs/' + dataset + '.json'
    log_dir = '../outputs/log/' + provider + '_' + model_name.split('/')[-1]
    os.makedirs(log_dir, exist_ok=True)
    log_file = log_dir + '/' + dataset + '_' + defense + '_' + prompt_type + '_' + str(icl_num) + '_adaptive_' + adaptive_attack_on_pi + '_filter_' + redundant_info_filtering + '.txt'
    cmd = 'python3 -u main.py --model_config_path ' + model_config_path + ' --model_name ' + model_name + ' --task_config_path ' + task_config_path + ' --icl_num ' + str(icl_num) + ' --prompt_type ' + prompt_type + ' --api_key_pos ' + str(api_key_pos) + ' --defense ' + defense + ' --gpus ' + gpus + ' --adaptive_attack ' + adaptive_attack_on_pi + ' --redundant_info_filtering ' + redundant_info_filtering + ' 2>&1 | tee ' + log_file
    os.system(cmd)
    return log_file

model_info = ['groq', 'llama-3.1-8b-instant']
datasets = ['synthetic']
prompt_types = ['direct']
icl_nums = [0]
redundant_info_filtering = 'True'
defenses = ['pi_ci', 'pi_id', 'pi_ci_id']
adaptive_attacks_on_pi = ['no']

for dataset in datasets:
    assert dataset in ['synthetic', 'celebrity', 'physician']
for defense in defenses:
    assert defense in ['no', 'replace_at', 'replace_at_dot', 'replace_dot', 'hyperlink', 'mask', 'pi_ci', 'pi_id', 'pi_ci_id', 'image']
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    assert adaptive_attack_on_pi in ['no', 'sandwich', 'xml', 'delimiters', 'random_seq', 'instructional', 'paraphrasing', 'retokenization']

api_key_pos = 0
gpus = '0'

user_decision = input('Total process: ' + str(len(adaptive_attacks_on_pi)*len(datasets)*len(defenses)*len(prompt_types)*len(icl_nums)) + '\nRun? (y/n): ')
if user_decision.lower() != 'y':
    exit()

provider = model_info[0]
model_name = model_info[1]
for adaptive_attack_on_pi in adaptive_attacks_on_pi:
    for data in datasets:
        for defense in defenses:
            for prompt_type in prompt_types:
                for icl_num in icl_nums:
                    print('\n>>> Starting defense: ' + defense)
                    tmp = run(provider, model_name, data, api_key_pos, defense, prompt_type, icl_num, str(gpus), adaptive_attack_on_pi, redundant_info_filtering)
                    print('>>> Finished defense: ' + defense)
