#!/usr/bin/env bash
# Assemble the brush-cli distribution bundle into ./output (gitignored):
# the compiled release binary plus all of its documentation. Reproducible.
#
#   scripts/build-cli-output.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/output"

rm -rf "$OUT"
mkdir -p "$OUT"

echo "==> 1/4 compile brush-cli (release)"
cargo build --release -p brush-cli
cp target/release/brush-cli "$OUT/brush-cli"

echo "==> 2/4 regenerate man page from the live clap command"
# Writes docs/man/brush-cli.1 (committed source of truth); copy into the bundle.
cargo run -q -p brush-cli --example gen-man
cp docs/man/brush-cli.1 "$OUT/brush-cli.1"

echo "==> 3/4 collect CLI documentation"
cp docs/cli-reference.md   "$OUT/cli-reference.md"
cp docs/cli-internals.md   "$OUT/cli-internals.md"
cp README.md               "$OUT/README.md"
cp llms.txt                "$OUT/llms.txt"

echo "==> 4/4 snapshot --help"
"$OUT/brush-cli" --help > "$OUT/brush-cli-help.txt" 2>&1 || true

cat > "$OUT/MANIFEST.md" <<EOF
# brush-cli — output bundle

Assembled by \`scripts/build-cli-output.sh\` (gitignored; regenerable).

| File | What |
|---|---|
| \`brush-cli\` | compiled release binary (v$("$OUT/brush-cli" --version 2>/dev/null | awk '{print $2}')) |
| \`brush-cli.1\` | roff man page (\`man ./brush-cli.1\`) — generated from clap |
| \`brush-cli-help.txt\` | \`--help\` snapshot |
| \`cli-reference.md\` | complete flag manual |
| \`cli-internals.md\` | binary functions/helpers + control flow |
| \`README.md\` | project README |
| \`llms.txt\` | LLM-friendly project overview |
EOF

echo
echo "Assembled $OUT:"
ls -lh "$OUT" | awk 'NR>1{print "  ", $5, $9}'
