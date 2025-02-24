use burn::tensor::{Tensor, backend::Backend, module::conv2d, ops::ConvOptions};

pub struct Ssim<B: Backend> {
    weights: Tensor<B, 4>,
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
    // TODO: Try a separable convolution.
    // fn gaussian_blur<B: Backend>(img: Tensor<B, 4>, window: Tensor<B, 4>) -> Tensor<B, 4> {
    //     let [_, channel, _, _] = img.dims();
    //     let window_size = window.dims()[2];
    //     let padding = window_size / 2;
    //     let conv_options = ConvOptions::new([1, 1], [padding, 1], [1, 1], channel);
    //     let xx = conv2d(img.clone(), window.clone(), None, conv_options.clone());
    //     println!("Window shape {:?}", window.dims());
    //     let conv_options = ConvOptions::new([1, 1], [1, padding], [1, 1], channel);
    //     conv2d(
    //         xx.clone(),
    //         window.clone().permute([0, 1, 3, 2]),
    //         None,
    //         conv_options.clone(),
    //     )
    // }

    pub fn new(window_size: usize, channels: usize, device: &B::Device) -> Self {
        let window1d = gaussian(window_size, 1.5, device).reshape([window_size, 1]);
        let window2d = window1d.clone().matmul(window1d.transpose());
        // Channels out, in, h, w.
        let weights = window2d.unsqueeze().repeat_dim(0, channels);
        Self { weights }
    }

    pub fn ssim(&self, img1: Tensor<B, 3>, img2: Tensor<B, 3>) -> Tensor<B, 3> {
        // Images are [H, W, C], need them as [N, C, H, W].
        let img1 = img1.permute([2, 0, 1]).unsqueeze();
        let img2 = img2.permute([2, 0, 1]).unsqueeze();

        let [channels, _, _, window_size] = self.weights.dims();
        let padding = window_size / 2;
        let conv_options = ConvOptions::new([1, 1], [padding, padding], [1, 1], channels);
        let mu_x = conv2d(
            img1.clone(),
            self.weights.clone(),
            None,
            conv_options.clone(),
        );

        let mu_y = conv2d(
            img2.clone(),
            self.weights.clone(),
            None,
            conv_options.clone(),
        );

        let mu_xx = mu_x.clone() * mu_x.clone();
        let mu_yy = mu_y.clone() * mu_y.clone();
        let mu_xy = mu_x * mu_y;

        let sigma_xx = conv2d(
            img1.clone() * img1.clone(),
            self.weights.clone(),
            None,
            conv_options.clone(),
        ) - mu_xx.clone();

        let sigma_yy = conv2d(
            img2.clone() * img2.clone(),
            self.weights.clone(),
            None,
            conv_options.clone(),
        ) - mu_yy.clone();

        let sigma_xy =
            conv2d(img1 * img2, self.weights.clone(), None, conv_options) - mu_xy.clone();

        let c1 = 0.01f32.powf(2.0);
        let c2 = 0.03f32.powf(2.0);

        let ssim = ((mu_xy * 2.0 + c1) * (sigma_xy * 2.0 + c2))
            / ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2));

        let ssim = ssim.squeeze(0);
        ssim.permute([1, 2, 0])
    }
}
