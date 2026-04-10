#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def _indent(level: int) -> str:
    return "  " * level


def format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def format_source_map(src: list[Any]) -> str:
    return f"[{src[0]},{src[1]}]"


def format_variable(var: list[Any], indent: int) -> str:
    return f'{_indent(indent)}[ {var[0]}, {format_source_map(var[1])} ]'


def format_instruction(instr: list[Any], indent: int) -> str:
    parts = []
    for item in instr[:-1]:
        if isinstance(item, str):
            parts.append(json.dumps(item))
        else:
            parts.append(str(item))
    parts.append(format_source_map(instr[-1]))
    return f'{_indent(indent)}[ ' + ", ".join(parts) + " ]"


def format_constants(constants: list[Any], indent: int) -> str:
    if not constants:
        return ""
    lines = []
    for i, c in enumerate(constants):
        suffix = "," if i < len(constants) - 1 else ""
        lines.append(f'{_indent(indent)}{json.dumps(c)}{suffix}')
    return "\n".join(lines)


def format_variables(variables: list[Any], indent: int) -> str:
    if not variables:
        return ""
    lines = []
    for i, var in enumerate(variables):
        suffix = "," if i < len(variables) - 1 else ""
        lines.append(format_variable(var, indent) + suffix)
    return "\n".join(lines)


def format_successors(successors: list[Any]) -> str:
    if not successors:
        return "[ ]"
    return "[ " + ", ".join(str(x) for x in successors) + " ]"


def format_instructions(instructions: list[Any], indent: int) -> str:
    if not instructions:
        return ""
    lines = []
    for i, instr in enumerate(instructions):
        suffix = "," if i < len(instructions) - 1 else ""
        lines.append(format_instruction(instr, indent) + suffix)
    return "\n".join(lines)


def format_basic_block(bb: dict[str, Any], indent: int) -> str:
    lines = []
    lines.append(f"{_indent(indent)}{{")
    lines.append(f'{_indent(indent + 1)}"instructions": [')
    instr_text = format_instructions(bb.get("instructions", []), indent + 2)
    if instr_text:
        lines.append(instr_text)
    lines.append(f'{_indent(indent + 1)}],')
    lines.append(f'{_indent(indent + 1)}"successors": {format_successors(bb.get("successors", []))}')
    lines.append(f"{_indent(indent)}}}")
    return "\n".join(lines)


def format_basic_blocks(blocks: list[Any], indent: int) -> str:
    if not blocks:
        return ""
    lines = []
    for i, bb in enumerate(blocks):
        text = format_basic_block(bb, indent)
        if i < len(blocks) - 1:
            text += ","
        lines.append(text)
    return "\n".join(lines)


def format_function(function: dict[str, Any], indent: int) -> str:
    name = function["name"]
    lines = []
    lines.append(f"{_indent(indent)}{{")
    lines.append(
        f'{_indent(indent + 1)}"name": [ {json.dumps(name[0])}, {format_source_map(name[1])} ],'
    )
    lines.append(f'{_indent(indent + 1)}"parameters": [')
    params_text = format_variables(function.get("parameters", []), indent + 2)
    if params_text:
        lines.append(params_text)
    lines.append(f'{_indent(indent + 1)}],')
    lines.append(f'{_indent(indent + 1)}"locals": [')
    locals_text = format_variables(function.get("locals", []), indent + 2)
    if locals_text:
        lines.append(locals_text)
    lines.append(f'{_indent(indent + 1)}],')
    lines.append(f'{_indent(indent + 1)}"basic_blocks": [')
    bbs_text = format_basic_blocks(function.get("basic_blocks", []), indent + 2)
    if bbs_text:
        lines.append(bbs_text)
    lines.append(f'{_indent(indent + 1)}]')
    lines.append(f"{_indent(indent)}}}")
    return "\n".join(lines)


def format_functions(functions: list[Any], indent: int) -> str:
    if not functions:
        return ""
    lines = []
    for i, fn in enumerate(functions):
        text = format_function(fn, indent)
        if i < len(functions) - 1:
            text += ","
        lines.append(text)
    return "\n".join(lines)


def format_external_variables(values: list[Any], indent: int) -> str:
    if not values:
        return ""
    lines = []
    for i, item in enumerate(values):
        suffix = "," if i < len(values) - 1 else ""
        lines.append(f'{_indent(indent)}[ {item[0]}, {json.dumps(item[1])} ]{suffix}')
    return "\n".join(lines)


def format_external_functions(values: list[Any], indent: int) -> str:
    if not values:
        return ""
    lines = []
    for i, item in enumerate(values):
        suffix = "," if i < len(values) - 1 else ""
        lines.append(f"{_indent(indent)}{item}{suffix}")
    return "\n".join(lines)


def format_program(program: dict[str, Any]) -> str:
    lines = []
    lines.append("{")
    lines.append(f'"magic": {json.dumps(program["magic"])},')
    lines.append(f'"version": {json.dumps(program["version"])},')
    lines.append(f'"system": {json.dumps(program["system"])},')
    lines.append(f'"num_cpu_bits": {program["num_cpu_bits"]},')
    lines.append(f'"name": {json.dumps(program["name"])},')
    lines.append(f'"entry_function": {program["entry_function"]},')

    lines.append(f'"constants": [')
    const_text = format_constants(program.get("constants", []), 1)
    if const_text:
        lines.append(const_text)
    lines.append("],")

    lines.append(f'"static": [')
    static_text = format_variables(program.get("static", []), 1)
    if static_text:
        lines.append(static_text)
    lines.append("],")

    lines.append(f'"functions": [')
    functions_text = format_functions(program.get("functions", []), 1)
    if functions_text:
        lines.append(functions_text)
    lines.append("],")

    lines.append(f'"external_variables": [')
    ext_vars_text = format_external_variables(program.get("external_variables", []), 1)
    if ext_vars_text:
        lines.append(ext_vars_text)
    lines.append("],")

    lines.append(f'"external_functions": [')
    ext_fns_text = format_external_functions(program.get("external_functions", []), 1)
    if ext_fns_text:
        lines.append(ext_fns_text)
    lines.append("]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def process_file(path: Path, overwrite: bool, suffix: str) -> Path:
    with path.open("r", encoding="utf-8") as f:
        program = json.load(f)

    formatted = format_program(program)

    if overwrite:
        out_path = path
    else:
        out_path = path.with_name(path.stem + suffix + path.suffix)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(formatted)

    return out_path


def iter_json_files(folder: Path, recursive: bool):
    pattern = "**/*.json" if recursive else "*.json"
    yield from sorted(folder.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Format all SALA JSON files in a folder to match the sala serializer style."
    )
    parser.add_argument("folder", type=Path, help="Folder containing .json files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of creating *_formatted.json files",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process .json files recursively",
    )
    parser.add_argument(
        "--suffix",
        default="_formatted",
        help="Suffix for output files when not using --overwrite",
    )
    args = parser.parse_args()

    if not args.folder.exists() or not args.folder.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {args.folder}")

    files = list(iter_json_files(args.folder, args.recursive))
    if not files:
        print("No JSON files found.")
        return 0

    for path in files:
        out = process_file(path, overwrite=args.overwrite, suffix=args.suffix)
        print(f"{path} -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
