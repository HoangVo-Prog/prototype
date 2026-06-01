import argparse
import ast
import contextlib
import csv
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


BASE_COLUMNS = ["log_file", "run", "r1", "best_epoch", "extra_config"]
EXTRA_CONFIG_SECTIONS = (
    "target-aware text enrichment",
    "mlp-mixer module settings",
    "target-aware loss settings",
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\s+")
CONFIG_LINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)=(.*)$")
SECTION_HEADER_RE = re.compile(r"#+\s*([^#]+?)\s*#+")
WAND_RUN_RE = re.compile(r"https?://wandb\.ai/[^\s]+/runs/[A-Za-z0-9_-]+")
BEST_R1_RE = re.compile(
    r"best\s+R1:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+at\s+epoch\s+(\d+)",
    re.IGNORECASE,
)
LONG_OPTION_RE = re.compile(r"['\"]--([A-Za-z0-9_-]+)['\"]")
DEST_RE = re.compile(r"\bdest\s*=\s*['\"]([A-Za-z_]\w*)['\"]")


def load_default_config(source_root: Path) -> Dict[str, Any]:
    """Load current training defaults from utils.options without consuming this CLI."""
    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    old_argv = sys.argv[:]
    try:
        sys.argv = ["log_reader_defaults"]
        from utils.options import get_args

        return dict(vars(get_args()))
    finally:
        sys.argv = old_argv
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source_root))


def read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_section_config_keys(options_path: Path, section_names: Sequence[str]) -> List[str]:
    lines = read_text(options_path).splitlines()
    wanted = {name.lower() for name in section_names}
    keys: List[str] = []
    active = False
    index = 0

    while index < len(lines):
        line = lines[index]
        section_match = SECTION_HEADER_RE.search(line)
        if section_match:
            active = section_match.group(1).strip().lower() in wanted
            index += 1
            continue

        if not active or "add_argument(" not in line:
            index += 1
            continue

        call_lines = [line]
        balance = line.count("(") - line.count(")")
        index += 1
        while balance > 0 and index < len(lines):
            call_lines.append(lines[index])
            balance += lines[index].count("(") - lines[index].count(")")
            index += 1

        key = config_key_from_add_argument("\n".join(call_lines))
        if key and key not in keys:
            keys.append(key)

    return keys


def config_key_from_add_argument(call: str) -> str:
    dest_match = DEST_RE.search(call)
    if dest_match:
        return dest_match.group(1)

    option_match = LONG_OPTION_RE.search(call)
    if option_match:
        return option_match.group(1).replace("-", "_")
    return ""


def extract_namespace_block(lines: Sequence[str]) -> List[str]:
    for index, line in enumerate(lines):
        start = line.find("Namespace(")
        if start == -1:
            continue

        block = [line[start + len("Namespace(") :].rstrip("\n")]
        for next_line in lines[index + 1 :]:
            if TIMESTAMP_RE.match(next_line) or next_line.startswith(("wandb:", "Traceback ")):
                break
            block.append(next_line.rstrip("\n"))
        return block
    return []


def parse_config_value(raw_value: str) -> Any:
    raw_value = raw_value.strip()
    if raw_value.endswith(")") and not can_literal_eval(raw_value):
        stripped_namespace_suffix = raw_value[:-1].rstrip()
        if can_literal_eval(stripped_namespace_suffix):
            raw_value = stripped_namespace_suffix

    normalized = raw_value
    if normalized.startswith("(") and normalized.endswith(")"):
        inner = normalized[1:-1].strip()
        if inner and "," not in inner:
            normalized = f"({inner},)"

    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        pass

    if normalized == "True":
        return True
    if normalized == "False":
        return False
    if normalized == "None":
        return None

    try:
        return int(normalized)
    except ValueError:
        pass

    try:
        return float(normalized)
    except ValueError:
        return normalized.strip("'\"")


def can_literal_eval(value: str) -> bool:
    try:
        ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return False
    return True


def parse_log_config(text: str) -> Dict[str, Any]:
    block = extract_namespace_block(text.splitlines())
    if not block:
        return {}

    entries: List[Tuple[str, List[str]]] = []
    current_key = None
    current_parts: List[str] = []

    for raw_line in block:
        line = raw_line.strip()
        if not line:
            continue
        match = CONFIG_LINE_RE.match(line)
        if match:
            if current_key is not None:
                entries.append((current_key, current_parts))
            current_key = match.group(1)
            current_parts = [match.group(2).strip()]
        elif current_key is not None:
            current_parts.append(line.strip())

    if current_key is not None:
        entries.append((current_key, current_parts))

    parsed: Dict[str, Any] = {}
    for key, parts in entries:
        raw_value = ",".join(part for part in parts if part != "")
        parsed[key] = parse_config_value(raw_value)
    return parsed


def extract_run_url(text: str) -> str:
    matches = WAND_RUN_RE.findall(text)
    return matches[-1].rstrip(".,);]") if matches else ""


def extract_best_r1(text: str) -> Tuple[Any, Any]:
    matches = BEST_R1_RE.findall(text)
    if not matches:
        return "", ""
    r1_text, epoch_text = matches[-1]
    return round(float(r1_text), 4), int(epoch_text)


def comparable(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(comparable(item) for item in value)
    if isinstance(value, list):
        return tuple(comparable(item) for item in value)
    return value


def value_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(value_to_text(item) for item in value)
    return str(value)


def diff_configs(
    log_config: Dict[str, Any],
    default_config: Dict[str, Any],
    allowed_keys: Sequence[str] | None = None,
) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    keys = allowed_keys if allowed_keys is not None else default_config.keys()
    for key in keys:
        if key not in log_config:
            continue
        if key not in default_config:
            continue
        default_value = default_config[key]
        log_value = log_config[key]
        if comparable(log_value) != comparable(default_value):
            diffs[key] = log_value
    return diffs


def format_extra_configs(config_diffs: Dict[str, Any]) -> str:
    lines = []
    for key, value in config_diffs.items():
        if value is True:
            lines.append(f"- {key}")
        else:
            lines.append(f"- {key}: {value_to_text(value)}")
    return "\n".join(lines)


def parse_log_file(
    path: Path,
    log_root: Path,
    default_config: Dict[str, Any],
    extra_config_keys: Sequence[str],
) -> Dict[str, Any] | None:
    text = read_text(path)
    r1, best_epoch = extract_best_r1(text)
    if r1 == "" or best_epoch == "":
        return None

    log_config = parse_log_config(text)
    config_diffs = diff_configs(log_config, default_config, extra_config_keys)

    return {
        "log_file": str(path.relative_to(log_root)),
        "run": extract_run_url(text),
        "r1": r1,
        "best_epoch": best_epoch,
        "extra_config": format_extra_configs(config_diffs),
    }


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return headers, rows


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(csv_cell(item) for item in value)
    return str(value)


def write_csv(path: Path, headers: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(headers), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({header: csv_cell(row.get(header, "")) for header in headers})


def merge_headers(existing_headers: Sequence[str], new_rows: Sequence[Dict[str, Any]]) -> List[str]:
    return list(BASE_COLUMNS)


def normalize_output_path(output: str) -> Path:
    path = Path(output)
    if path.suffix == "":
        path = path.with_suffix(".csv")
    if path.parent == Path("."):
        path = Path("results") / path
    if path.suffix.lower() != ".csv":
        raise ValueError("Output file must use .csv extension.")
    return path


def iter_log_files(log_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    globber = log_dir.rglob if recursive else log_dir.glob
    return sorted(path for path in globber(pattern) if path.is_file())


def append_rows(
    output_path: Path,
    new_rows: Sequence[Dict[str, Any]],
    *,
    skip_existing: bool,
    overwrite: bool = False,
) -> int:
    existing_headers, existing_rows = ([], []) if overwrite else read_csv(output_path)
    existing_rows = [row for row in existing_rows if has_result(row)]
    rows_to_add = [row for row in new_rows if has_result(row)]
    if skip_existing:
        existing_logs = {str(row.get("log_file", "")) for row in existing_rows}
        rows_to_add = [row for row in rows_to_add if str(row.get("log_file", "")) not in existing_logs]

    headers = merge_headers(existing_headers, rows_to_add)
    all_rows = dedupe_rows_by_extra_config(existing_rows + rows_to_add)
    write_csv(output_path, headers, all_rows)
    return len(rows_to_add)


def has_result(row: Dict[str, Any]) -> bool:
    return str(row.get("r1", "")).strip() != "" and str(row.get("best_epoch", "")).strip() != ""


def dedupe_rows_by_extra_config(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = extra_config_key(row)
        if key not in best_rows or r1_score(row) > r1_score(best_rows[key]):
            best_rows[key] = row
    return list(best_rows.values())


def extra_config_key(row: Dict[str, Any]) -> str:
    extra_config = str(row.get("extra_config", ""))
    normalized_lines = [
        line.strip()
        for line in extra_config.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    return "\n".join(line for line in normalized_lines if line)


def r1_score(row: Dict[str, Any]) -> float:
    try:
        return float(str(row.get("r1", "")).strip())
    except ValueError:
        return float("-inf")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read training logs, extract W&B run/R1/config deltas, and append them to a CSV file.",
    )
    parser.add_argument("log_dir", help="Folder containing log files, for example logs")
    parser.add_argument(
        "-o",
        "--output",
        default="results/log_reader.csv",
        help="Output .csv file. If only a file name is provided, it is written under results/.",
    )
    parser.add_argument("--pattern", default="*.log", help="Log file glob pattern")
    parser.add_argument("--recursive", action="store_true", help="Read logs recursively")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose log_file already exists in the output CSV",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the output CSV instead of appending to existing rows",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parent),
        help="Repository root used to load current default config from utils.options",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    log_dir = Path(args.log_dir).resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log folder not found: {log_dir}")

    output_path = normalize_output_path(args.output).resolve()
    source_root = Path(args.source_root)
    default_config = load_default_config(source_root)
    extra_config_keys = extract_section_config_keys(
        source_root / "utils" / "options.py",
        EXTRA_CONFIG_SECTIONS,
    )
    log_files = iter_log_files(log_dir, args.pattern, args.recursive)
    rows = []
    skipped = 0
    for path in log_files:
        row = parse_log_file(path, log_dir, default_config, extra_config_keys)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    try:
        processed_rows = append_rows(
            output_path,
            rows,
            skip_existing=args.skip_existing,
            overwrite=args.overwrite,
        )
    except PermissionError as error:
        raise SystemExit(
            f"Cannot write output CSV: {output_path}. Close the file if it is open, then rerun."
        ) from error
    _, output_rows = read_csv(output_path)
    print(
        f"Read {len(log_files)} log file(s). "
        f"Skipped {skipped} incomplete log(s). "
        f"Processed {processed_rows} complete row(s). "
        f"Output has {len(output_rows)} deduped row(s): {output_path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
