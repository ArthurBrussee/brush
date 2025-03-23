use burn::tensor::{Tensor, backend::Backend, module::conv2d, ops::ConvOptions};

pub struct Ssim<B: Backend> {
    weights_1d_v: Tensor<B, 4>,
}

fn gaussian<B: Backend>(window_size: usize, sigma: f32, device: &B::Device) -> Tensor<B, 1> {
    let window_extent = (window_size / 2) as f32;
    let vals: Vec<_> = (0..window_size)
        .map(|x| f32::exp(-(x as f32 - window_extent).powf(2.0) / (2.0 * sigma.powf(2.0))))
        .collect();
    let gauss = Tensor::from_floats(vals.as_slice(), device);
    gauss.clone() / gauss.sum()
}

impl<B: Backend> Ssim<B> {
    pub fn new(window_size: usize, channels: usize, device: &B::Device) -> Self {
        // Channels out, in, h, w.
        let weights_1d_v = gaussian(window_size, 1.5, device)
            .reshape([window_size, 1])
            .unsqueeze()
            .repeat_dim(0, channels);
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

        let ssim = ssim.squeeze(0);
        ssim.permute([1, 2, 0])
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use burn::{
        backend::{Wgpu, wgpu::WgpuDevice},
        tensor::{Float, Tensor},
    };
    type Backend = Wgpu;

    #[test]
    fn test_ssim() {
        use super::Ssim;

        let device = WgpuDevice::DefaultDevice;
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
        let ssim_val = ssim.ssim(img1.clone(), img2.clone()).mean();

        // You get 0.078679755 when using  a naive 2d conv.
        // The separable approach results in 0.078679785
        assert_approx_eq!(ssim_val.into_scalar(), 0.078679755, 1e-7);
    }
}
