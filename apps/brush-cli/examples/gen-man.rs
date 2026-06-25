//! Regenerate the brush-cli man page from the live clap command so it never
//! drifts from `--help`.
//!
//! Run from the workspace root:
//!   cargo run -p brush-cli --example gen-man
//! It writes `docs/man/brush-cli.1`. `clap_mangen` is a dev-dependency, so this
//! generator is never linked into the shipped `brush-cli` binary.

use std::io::Write;

use brush_cli::Cli;
use clap::CommandFactory;

fn main() -> std::io::Result<()> {
    // The real clap command — includes every flattened flag (Training, Refine,
    // LOD, Model, Dataset, Process, Rerun), so the OPTIONS section is always in
    // sync with what the binary actually parses.
    let cmd = Cli::command().name("brush-cli");

    let mut buf: Vec<u8> = Vec::new();
    clap_mangen::Man::new(cmd).render(&mut buf)?;

    // Append static sections clap_mangen doesn't generate. The per-flag OPTIONS
    // above stay auto-synced; only this prose is hand-maintained.
    buf.write_all(
        concat!(
            ".SH EXAMPLES\n",
            ".TP\nTrain and export with defaults:\n",
            ".B brush\\-cli ./datasets/tandt/truck\n",
            ".TP\nLow\\-memory profile (quality\\-neutral, ~\\-20% RAM):\n",
            ".B brush\\-cli ./datasets/tandt/truck \\-\\-sh\\-degree 2\n",
            ".TP\nConstrained machine \\(em cap host cache and log resources:\n",
            ".B brush\\-cli ./big \\-\\-max\\-cache\\-bytes 2147483648 \\-\\-log\\-resources\\-every 500\n",
            ".TP\nResume from a checkpoint:\n",
            ".B brush\\-cli ./scene \\-\\-start\\-iter 5000 \\-\\-total\\-train\\-iters 30000\n",
            ".SH NOTES\n",
            "Always build with \\-\\-release for performance\\-sensitive runs. ",
            "Run \\fBbrush\\-cli \\-\\-help\\fR for the authoritative, version\\-exact option list. ",
            "See the Brush docs under docs/ for memory/perf guidance.\n",
            ".SH SEE ALSO\n",
            "brush(1) (the brush\\-app GUI).\n",
        )
        .as_bytes(),
    )?;

    let out = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/man/brush-cli.1");
    if let Some(dir) = std::path::Path::new(out).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, &buf)?;
    eprintln!("wrote {out} ({} bytes)", buf.len());
    Ok(())
}
