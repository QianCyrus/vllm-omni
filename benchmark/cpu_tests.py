"""Run the checkout's CPU AllGather-KV tests without serving dependencies."""
import ast
import os
from pathlib import Path

import pytest
import torch

from bench import load_source

root = Path(os.environ["KV_SOURCE_ROOT"])
namespace, _ = load_source(root)
globals().update({name: value for name, value in namespace.items() if not name.startswith("__")})
test_path = root / "tests/diffusion/attention/test_attention_sp.py"
nodes = [node for node in ast.parse(test_path.read_text()).body
         if (isinstance(node, ast.ClassDef) and node.name == "_MockAllGatherSPGroup")
         or (isinstance(node, ast.FunctionDef) and node.name.startswith("test_allgather_kv_"))]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(test_path), "exec"), globals())
