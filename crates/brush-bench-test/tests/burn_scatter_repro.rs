//! Repro for an upstream burn bug: `split_strides` (burn-std/src/tensor)
//! anchors its stride walk on unit dims, so reshaping a `repeat_dim`
//! broadcast view like `[k, 1]` strides `[1, 0]` to `[k, 1, 1]` yields
//! strides `[0, 0, 0]` — every row of the index tensor aliases row 0. A
//! `scatter(Add)` fed such indices piles all k updates onto `inds[0]` and
//! applies none to the other rows.
//!
//! This bit brush as a flaky `tensor 'sh_coeffs' has 48 NaNs` failure in
//! brush-c's `test_train_and_save_ffi_short`: refine reset the Adam moments
//! of split parents with `x.scatter(0, inds, -x.select(0, inds), Add)`; for
//! the `[n,1,1]` reduced `moment_2` the broken index view left `inds[0]`'s
//! moment slightly *negative* — then `sqrt(-ε)` is NaN (bits 0x7fffffff on
//! NVIDIA), and the reduced-moment broadcast smears it across the splat's
//! whole SH row. Whether the index chain reached the scatter as a view
//! (broken) or materialized (fine) depended on fusion flush timing — hence
//! suite-only flakiness. Refine now resets moments with a mask multiply, and
//! the Adam update clamps `moment_2` before the sqrt.
//!
//! Fix: `split_strides` must skip unit dims when anchoring (branch
//! `fix/split-strides-unit-dims` in the burn repo). Un-ignore this test once
//! the burn pin includes it.

#![cfg(not(target_family = "wasm"))]

use burn::tensor::{Device, IndexingUpdateOp, Int, Tensor, TensorData};

/// `x + (-x)` must be exactly 0.0 at the scattered rows, and every other row
/// must be bit-identical. Mirrors the shapes refine used: `[n,1,1]` (reduced
/// moment_2) and `[n,16,3]` (moment_1), n around post-refine splat counts,
/// ~26 unique indices.
#[tokio::test]
#[ignore = "exposes the upstream burn split_strides unit-dim bug; un-ignore once the burn pin has the fix"]
async fn scatter_zeroing_is_exact() {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let device: Device = burn::tensor::Device::from(brush_cube::test_helpers::test_device().await);
    let mut rng = rand::rngs::StdRng::seed_from_u64(4242);

    for &n in &[95usize, 96, 97, 100, 121, 122, 123] {
        for rep in 0..20 {
            // ~26 unique indices like a real refine split set.
            let mut inds: Vec<i32> = (0..n as i32).collect();
            inds.shuffle(&mut rng);
            inds.truncate(26);
            let k = inds.len();

            for &(d1, d2) in &[(1usize, 1usize), (16, 3)] {
                let base: Vec<f32> = (0..n * d1 * d2).map(|i| 1e-8 + (i as f32) * 1e-9).collect();
                let x = Tensor::<3>::from_data(TensorData::new(base.clone(), [n, d1, d2]), &device);
                let inds_t =
                    Tensor::<1, Int>::from_data(TensorData::new(inds.clone(), [k]), &device);
                let neg_parent = -x.clone().select(0, inds_t.clone());
                let inds_2: Tensor<2, Int> = inds_t.clone().unsqueeze_dim(1).repeat_dim(1, d1);
                let inds_3: Tensor<3, Int> = inds_2.unsqueeze_dim(2).repeat_dim(2, d2);
                let out = x.scatter(0, inds_3, neg_parent, IndexingUpdateOp::Add);

                let vals = out
                    .into_data_async()
                    .await
                    .expect("scatter read")
                    .into_vec::<f32>()
                    .expect("scatter vec");
                let selected: std::collections::HashSet<usize> =
                    inds.iter().map(|i| *i as usize).collect();
                for (i, v) in vals.iter().enumerate() {
                    let row = i / (d1 * d2);
                    if selected.contains(&row) {
                        assert!(
                            *v == 0.0,
                            "zeroing not exact at n={n} d=({d1},{d2}) rep {rep} row {row} flat {i}: {v:e} ({:#010x})",
                            v.to_bits(),
                        );
                    } else {
                        assert!(
                            *v == base[i],
                            "untouched row changed at n={n} d=({d1},{d2}) rep {rep} row {row} flat {i}: {v:e} ({:#010x}) want {:e}",
                            v.to_bits(),
                            base[i],
                        );
                    }
                }
            }
        }
    }
}
