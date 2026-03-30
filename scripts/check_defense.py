#!/usr/bin/env python3
"""
Test a specific defense mechanism on sample data.

Loads a single profile and applies a defense to inspect the output.
Useful for debugging before running full experiments.
"""
import sys
sys.path.insert(0, '.')
import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import (
    open_config, open_txt, parsed_data_to_string, load_instruction
)

def test_defense(defense_name='replace_at', sample_idx=0):
    """Test a defense on a sample profile."""
    task_config = open_config('./configs/task_configs/synthetic.json')
    task_manager, _ = PIE.create_task(task_config)
    defense = PIE.create_defense(defense_name)

    raw_list, curr_label = task_manager[sample_idx]
    raw_list = defense.apply(raw_list, curr_label)
    raw = '\n'.join(raw_list)

    redundant_info_filter = PIE.get_parser('synthetic')
    redundant_info_filter.feed(raw)
    processed_data = redundant_info_filter.data
    processed = defense.apply(
        parsed_data_to_string('synthetic', processed_data, 'llama-3.1-8b-instant'), 
        curr_label
    )

    print('=== LABEL ===')
    print(curr_label['email'])
    print()
    print('=== PROCESSED PROFILE (first 500 chars) ===')
    print(processed[:500])
    print()
    print('=== EMAIL LINE ===')
    for line in processed.split('\n'):
        if 'mail' in line.lower() or 'AT' in line or '@' in line:
            print(repr(line))
    
    info_cats = open_txt('./data/system_prompts/info_category.txt')
    instructions = load_instruction('direct', info_cats)
    full_prompt = instructions['email'] + '\n' + processed
    
    print()
    print('=== FULL PROMPT SENT TO MODEL ===')
    print(full_prompt[:500])
    print('...')

if __name__ == '__main__':
    test_defense()
