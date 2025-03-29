use burn::tensor::{Int, Tensor};
use burn_wgpu::{CubeBackend, WgpuRuntime};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use brush_prefix_sum::prefix_sum;
use brush_prefix_sum::prefix_sum_wgsl;
type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

fn benchmark_large_input(c: &mut Criterion) {
    const ITERS: usize = 1_000_000;
    let mut data = vec![];
    for i in 0..ITERS {
        data.push(2 + (i%10000) as i32);
    }

    let device = Default::default();
    let keys = Tensor::<Backend, 1, Int>::from_data(data.as_slice(), &device).into_primitive();
    
    c.bench_function("prefix_sum_large_cubecl", |b| {
        b.iter(|| {
            let _ = black_box(prefix_sum(keys.clone()));
        })
    });

    c.bench_function("prefix_sum_large_wgsl", |b| {
        b.iter(|| {
            let _ = black_box(prefix_sum_wgsl(keys.clone()));
        })
    });
}


criterion_group!(
    benches, 
    benchmark_large_input,
);
criterion_main!(benches);