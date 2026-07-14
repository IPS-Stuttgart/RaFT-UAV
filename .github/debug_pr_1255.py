from __future__ import annotations

import importlib
import sys

import raft_uav.mmuad as parent
import raft_uav.mmuad.candidate_reservoir as imported_alias

module = importlib.import_module("raft_uav.mmuad.candidate_reservoir")


def describe(label, value):
    print(label, "type=", type(value))
    print(label, "name=", getattr(value, "__name__", None))
    print(label, "file=", getattr(value, "__file__", None))
    code = getattr(value, "__code__", None)
    if code is not None:
        print(label, "code=", code.co_filename, code.co_firstlineno)
        print(label, "freevars=", code.co_freevars)


print("alias_is_module=", imported_alias is module)
print("parent_attr_is_module=", getattr(parent, "candidate_reservoir") is module)
print("sys_module_is_module=", sys.modules["raft_uav.mmuad.candidate_reservoir"] is module)
describe("module", module)
describe("module.main", module.main)
describe("module._ORIGINAL_MAIN", module._ORIGINAL_MAIN)
describe("module._IMPL", module._IMPL)
describe("module._IMPL.main", module._IMPL.main)
