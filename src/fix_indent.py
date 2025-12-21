
import os

def fix_indentation():
    file_path = 'src/enhanced_model_v4.py'
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    state = 'normal'  # normal, in_run, in_optimize
    
    for line in lines:
        stripped = line.strip()
        
        if state == 'normal':
            if line.startswith('def run('):
                state = 'in_run'
                new_lines.append('    ' + line)
            elif line.startswith('def optimize('):
                state = 'in_optimize'
                new_lines.append('    ' + line)
            else:
                new_lines.append(line)
                
        elif state == 'in_run':
            if line.startswith('class MonteCarloOptimizer'):
                state = 'normal'
                new_lines.append(line)
            else:
                # Only indent if not empty (though 4 spaces on empty line is harmless)
                new_lines.append('    ' + line)
                
        elif state == 'in_optimize':
            # Stop if we hit the next method which is _sample_config (already indented)
            # OR if we hit the end of the class/block (next top-level def)
            if stripped.startswith('def _sample_config'):
                state = 'normal'
                new_lines.append(line)
            elif line.startswith('def run_enhanced_backtest'):
                 state = 'normal'
                 new_lines.append(line)
            elif line.startswith('class '):
                 state = 'normal'
                 new_lines.append(line)
            else:
                new_lines.append('    ' + line)
                
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f"Fixed indentation in {file_path}")

if __name__ == "__main__":
    fix_indentation()
