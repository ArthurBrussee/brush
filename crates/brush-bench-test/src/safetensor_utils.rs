use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::{
    prelude::Backend,
    tensor::{Float, Tensor, TensorData},
};
use safetensors::{SafeTensors, tensor::TensorView};

fn float_from_u8(data: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(data).to_vec()
}

pub(crate) fn safetensor_to_burn<B: Backend, const D: usize>(
    t: &TensorView,
    device: &B::Device,
) -> Tensor<B, D, Float> {
    let data = TensorData::new::<f32, _>(float_from_u8(t.data()), t.shape());
    Tensor::from_data(data, device)
}

pub fn splats_from_safetensors<B: Backend>(
    tensors: &SafeTensors,
    device: &B::Device,
) -> anyhow::Result<Splats<B>> {
    Ok(Splats::from_tensor_data(
        safetensor_to_burn::<B, 2>(&tensors.tensor("means")?, device),
        safetensor_to_burn::<B, 2>(&tensors.tensor("quats")?, device),
        safetensor_to_burn::<B, 2>(&tensors.tensor("scales")?, device),
        safetensor_to_burn::<B, 3>(&tensors.tensor("coeffs")?, device),
        safetensor_to_burn::<B, 1>(&tensors.tensor("opacities")?, device),
        SplatRenderMode::Default,
    ))
}
