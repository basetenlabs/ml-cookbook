# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import subprocess
import tempfile
import os

def check_ocaml_compilation(ocaml_code):
   with tempfile.NamedTemporaryFile(mode='w', suffix='.ml', delete=False) as f:
       f.write(ocaml_code)
       temp_path = f.name
    
   result = subprocess.run(['ocamlc', '-c', temp_path], capture_output=True)
   
   # Clean up
   os.unlink(temp_path)
   for ext in ['.cmo', '.cmi']:
       compiled = temp_path.replace('.ml', ext)
       if os.path.exists(compiled):
           os.unlink(compiled)
   
   return result.returncode == 0

def extract_solution(solution_str):
    solution = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)

    if len(solution) != 1:
        final_answer = None
    else:
        # take the last solution
        final_answer = solution[0]

    return final_answer


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    solution = extract_solution(solution_str=solution_str)

    if solution is None:
        return 0.0

    can_compile = check_ocaml_compilation(solution)

    if can_compile == False:
        return 0.0
    else:
        return 1.0