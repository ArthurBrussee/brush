use std::future::Future;

use burn::{
    prelude::*,
    tensor::{BasicOps, Tensor},
};
use rerun::{ChannelDatatype, ColorModel};

trait BurnToRerunData {
    fn into_rerun_data(self) -> impl Future<Output = rerun::TensorData>;
}

fn tensor_dims<B: Backend, const D: usize, K: BasicOps<B>>(
    tensor: &Tensor<B, D, K>,
) -> Vec<rerun::TensorDimension> {
    tensor
        .dims()
        .map(|x| rerun::TensorDimension::unnamed(x as u64))
        .to_vec()
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D> {
    async fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::F32(
                self.into_data_async()
                    .await
                    .to_vec::<f32>()
                    .expect("Wrong type")
                    .into(),
            ),
        )
    }
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D, Int> {
    async fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::I32(
                self.into_data_async()
                    .await
                    .to_vec::<i32>()
                    .expect("Wrong type")
                    .into(),
            ),
        )
    }
}

impl<B: Backend, const D: usize> BurnToRerunData for Tensor<B, D, Bool> {
    async fn into_rerun_data(self) -> rerun::TensorData {
        rerun::TensorData::new(
            tensor_dims(&self),
            rerun::TensorBuffer::U8(
                self.into_data_async()
                    .await
                    .convert::<u8>()
                    .to_vec()
                    .expect("Wrong type")
                    .into(),
            ),
        )
    }
}

pub trait BurnToRerun {
    fn into_rerun(self) -> impl Future<Output = rerun::Tensor>;
}

impl<T: BurnToRerunData> BurnToRerun for T {
    async fn into_rerun(self) -> rerun::Tensor {
        rerun::Tensor::new(self.into_rerun_data().await)
    }
}

pub trait BurnToImage {
    fn into_rerun_image(self) -> impl Future<Output = rerun::Image>;
}

impl<B: Backend> BurnToImage for Tensor<B, 3> {
    async fn into_rerun_image(self) -> rerun::Image {
        let [h, w, c] = self.dims();
        let color_model = if c == 3 {
            ColorModel::RGB
        } else {
            ColorModel::RGBA
        };
        rerun::Image::from_color_model_and_bytes(
            self.into_data_async().await.as_bytes().to_vec(),
            [w as u32, h as u32],
            color_model,
            ChannelDatatype::F32,
        )
    }
}

// Assume int images encode u8 data.
impl<B: Backend> BurnToImage for Tensor<B, 3, Int> {
    async fn into_rerun_image(self) -> rerun::Image {
        let [h, w, c] = self.dims();
        let color_model = if c == 3 {
            ColorModel::RGB
        } else {
            ColorModel::RGBA
        };
        rerun::Image::from_color_model_and_bytes(
            self.into_data_async().await.as_bytes().to_vec(),
            [w as u32, h as u32],
            color_model,
            ChannelDatatype::U8,
        )
    }
}
