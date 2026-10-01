"""Version the bounded CPU audit for exactly forty actually completed endpoints."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "tools/execute_closed_native_inventory_owned_v3b.py"
    raw = source.read_text(encoding="utf-8")
    revised = raw.replace("v3b", "v5").replace("!= 35", "!= 40").replace("Fixed35", "Completed40").replace("FIXED35", "COMPLETED40")
    if raw.count("!= 35") != 1:
        raise ValueError("Exact bounded fixed35 wrapper cardinality required")
    with source.with_name("execute_completed_native_inventory_owned_v5.py").open("x", encoding="utf-8") as stream:
        stream.write(revised)
    print("Versioned40 CPU audit; unchanged600s/22GiB ownership and cleanup limits.")


if __name__ == "__main__":
    main()
