#![recursion_limit = "256"]

// Headless trainer binary. The viewer lives in brush-app (the `brush` binary);
// this is a lean build of just the training path for quick CLI iteration.
#[cfg(not(target_family = "wasm"))]
fn main() -> anyhow::Result<()> {
    use brush_cli::{Cli, build_process, mesh_extract, run_headless};
    use clap::Parser;

    let args = Cli::parse().validate()?;

    if args.with_viewer {
        anyhow::bail!(
            "brush-cli is headless and can't open a viewer. Pass a source to train, \
             or build the `brush` binary (brush-app) for the viewer."
        );
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime");

    // --extract-mesh is a standalone path: load the splat PLY from the source
    // and extract a mesh, no training.
    if args.extract_mesh {
        return runtime.block_on(mesh_extract::run(&args));
    }

    // `validate` guarantees a source is present when the viewer is off.
    let process = build_process(&args).expect("source must be present");
    runtime.block_on(run_headless(process, args.train_stream))
}

#[cfg(target_family = "wasm")]
fn main() {}
