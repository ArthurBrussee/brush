use burn::tensor::{Tensor, backend::Backend, module::conv2d, ops::ConvOptions};

pub(crate) struct Ssim<B: Backend> {
    weights_1d_v: Tensor<B, 4>,
}

impl<B: Backend> Ssim<B> {
    // Workaround for a burn-fusion bug on wasm/WebGPU.
    //
    // Building the normalized gaussian kernel via the natural lazy chain
    //     Tensor::from_floats(vals) / vals.sum()
    //         .reshape([n, 1])
    //         .unsqueeze()
    //         .repeat_dim(0, channels)
    // gives a tensor that is later read back as uninitialized memory inside
    // conv2d (values like `sum=inf, min≈f32::MAX, max=inf`, varying between
    // runs), which makes the loss NaN roughly half of the time. Only on
    // wasm; native works fine. Triggered by `brush-bench-test::test_multi_
    // step_training` running two trainers in a row with SSIM enabled.
    //
    // Diagnostics showed pred_rgb and gt_rgb were finite and l1 was correct;
    // only the SSIM weights tensor was garbage. Adding any sync point
    // (even a trivial `Tensor::zeros([4], &device).into_data_async().await`)
    // between the forward render and the loss computation also hid it. The
    // deterministic signature when it fires is that exactly the visible
    // splat indices with non-zero gradients get NaN in their means/opacities.
    //
    // We couldn't reduce this to a pure-burn reproducer — it needs state
    // that brush's custom `launch_unchecked` render kernels leave behind,
    // which plain burn ops don't replicate. It may be related to the
    // separately-reported `sum_dim` handle-lookup bug (see
    // repros/burn_sum_dim_handle/), since both involve burn-fusion's
    // handling of `sum` + scalar-divide fusion, but that's unconfirmed.
    //
    // The workaround is to pre-compute the normalized gaussian on the CPU
    // and upload it directly at its final 4D shape, bypassing the lazy op
    // chain entirely.
    pub fn new(window_size: usize, channels: usize, device: &B::Device) -> Self {
        let sigma = 1.5f32;
        let window_extent = (window_size / 2) as f32;
        let raw: Vec<f32> = (0..window_size)
            .map(|x| f32::exp(-(x as f32 - window_extent).powf(2.0) / (2.0 * sigma.powf(2.0))))
            .collect();
        let sum: f32 = raw.iter().sum();
        let normalized: Vec<f32> = raw.into_iter().map(|v| v / sum).collect();
        let mut data = Vec::with_capacity(channels * window_size);
        for _ in 0..channels {
            data.extend_from_slice(&normalized);
        }
        let weights_1d_v = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
            channels,
            1,
            window_size,
            1,
        ]);
        Self { weights_1d_v }
    }

    fn gaussian_blur(&self, img: Tensor<B, 4>) -> Tensor<B, 4> {
        let [channels, _, window_size, _] = self.weights_1d_v.dims();
        let padding = window_size / 2;

        let conv_options_v = ConvOptions::new([1, 1], [padding, 0], [1, 1], channels);
        let conv_options_h = ConvOptions::new([1, 1], [0, padding], [1, 1], channels);
        let kernel_v = self.weights_1d_v.clone();
        let kernel_h = self
            .weights_1d_v
            .clone()
            .reshape([channels, 1, 1, window_size]);

        let v_blur = conv2d(img, kernel_v, None, conv_options_v);
        conv2d(v_blur, kernel_h, None, conv_options_h)
    }

    pub fn ssim(&self, img1: Tensor<B, 3>, img2: Tensor<B, 3>) -> Tensor<B, 3> {
        // Images are [H, W, C], need them as [N, C, H, W].
        let img1 = img1.permute([2, 0, 1]).unsqueeze();
        let img2 = img2.permute([2, 0, 1]).unsqueeze();

        let mu_x = self.gaussian_blur(img1.clone());
        let mu_y = self.gaussian_blur(img2.clone());
        let mu_xx = mu_x.clone() * mu_x.clone();
        let mu_yy = mu_y.clone() * mu_y.clone();
        let mu_xy = mu_x * mu_y;

        let sigma_xx = self.gaussian_blur(img1.clone() * img1.clone()) - mu_xx.clone();
        let sigma_yy = self.gaussian_blur(img2.clone() * img2.clone()) - mu_yy.clone();
        let sigma_xy = self.gaussian_blur(img1 * img2) - mu_xy.clone();

        let c1 = 0.01f32.powf(2.0);
        let c2 = 0.03f32.powf(2.0);

        let ssim = ((mu_xy * 2.0 + c1) * (sigma_xy * 2.0 + c2))
            / ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2));

        let ssim = ssim.squeeze_dim(0);
        ssim.permute([1, 2, 0])
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Wgpu,
        tensor::{Float, Tensor},
    };
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    type Backend = Wgpu;

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ssim() {
        use super::Ssim;

        let device = brush_kernel::test_helpers::test_device().await;
        let img_shape = [30, 50, 3];
        let pixels = img_shape.iter().product::<usize>();

        let create_test_img = |s: f32, o: f32| -> Tensor<Backend, 3, Float> {
            Tensor::<Backend, 1, Float>::from_floats(
                (0..pixels)
                    .map(|i| ((i as f32 * s + o).sin() + 1.0) / 2.0)
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &device,
            )
            .reshape(img_shape)
        };
        let img1 = create_test_img(0.12, 0.5);
        let img2 = create_test_img(0.53, 2.0);

        let ssim = Ssim::new(11, 3, &device);
        let ssim_val = ssim.ssim(img1, img2).mean();

        // You get 0.078679755 when using  a naive 2d conv.
        // The separable approach results in 0.078679785
        let val = ssim_val
            .into_data_async()
            .await
            .expect("readback")
            .as_slice::<f32>()
            .expect("Wrong type")[0];
        assert!((val - 0.078679755).abs() < 1e-7);
    }
}
