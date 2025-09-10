use crate::gaussian_splats::Splats;
use burn_wgpu::{Wgpu, WgpuDevice};

type TestBackend = Wgpu;

#[test]
fn test_knn_initialization() {
    let device = WgpuDevice::DefaultDevice;

    // Create a simple 2x2 grid
    let positions = vec![
        0.0, 0.0, 0.0, // Point 1
        1.0, 0.0, 0.0, // Point 2
        0.0, 1.0, 0.0, // Point 3
        1.0, 1.0, 0.0, // Point 4
    ];

    // Create splats with KNN initialization (no explicit scales)
    let splats = Splats::<TestBackend>::from_raw(positions, None, None, None, None, &device);

    let scales = splats.scales();
    let scale_data = scales.to_data();
    let scale_values = scale_data.as_slice::<f32>().expect("Wrong type");

    // Should have 4 points × 3 dimensions = 12 scale values
    assert_eq!(scale_values.len(), 12);

    // All scales should be positive and finite
    for &scale in scale_values {
        assert!(scale > 0.0 && scale.is_finite());
    }
}
