//! Handing the render's finished tensors back to burn's fusion stream.

use burn::backend::TensorMetadata;
use burn::backend::ir::{BackendIr, InitOperationIr, OperationIr, OperationOutput};
use burn_cubecl::CubeBackend;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::stream::StreamId;
use burn_fusion::{Client, FusionTensor, NoOp};

/// Seed an already-computed tensor into the stream, the way `float_from_data`
/// seeds an upload. The render sizes its outputs from a mid-pipeline readback,
/// so it can't be registered as a lazy op like the rest.
pub(crate) fn bind(
    client: &Client<FusionCubeRuntime>,
    tensor: CubeTensor,
) -> FusionTensor<FusionCubeRuntime> {
    let (shape, dtype) = (tensor.shape(), tensor.dtype);
    let handle = CubeBackend::float_tensor_handle(tensor);
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));
    client
        .register(
            StreamId::current(),
            OperationIr::Init(desc),
            NoOp::<CubeBackend>::new(),
        )
        .output()
}
