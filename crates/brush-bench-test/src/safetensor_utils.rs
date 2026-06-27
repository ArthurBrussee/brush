use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use burn::tensor::{Device, Float, Tensor, TensorData};
use safetensors::{SafeTensors, tensor::TensorView};

fn float_from_u8(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn safetensor_to_burn<const D: usize>(
    t: &TensorView,
    device: &Device,
) -> anyhow::Result<Tensor<D, Float>> {
    if t.dtype() != safetensors::Dtype::F32 {
        anyhow::bail!("Expected F32 tensor, got {:?}", t.dtype());
    }
    let data = TensorData::new::<f32, _>(float_from_u8(t.data()), t.shape());
    Ok(Tensor::from_data(data, device))
}

pub fn splats_from_safetensors(tensors: &SafeTensors, device: &Device) -> anyhow::Result<Splats> {
    Ok(Splats::from_tensor_data(
        safetensor_to_burn::<2>(&tensors.tensor("means")?, device)?,
        safetensor_to_burn::<2>(&tensors.tensor("quats")?, device)?,
        safetensor_to_burn::<2>(&tensors.tensor("scales")?, device)?,
        safetensor_to_burn::<3>(&tensors.tensor("coeffs")?, device)?,
        safetensor_to_burn::<1>(&tensors.tensor("opacities")?, device)?,
        SplatRenderMode::Default,
    ))
}
