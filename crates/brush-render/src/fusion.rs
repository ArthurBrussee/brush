//! Running the render under burn's `Fusion` backend.
//!
//! `#[backend_extension(.., Fusion)]` generates this plumbing for any op whose
//! output shapes are known before it runs. The render is the one that isn't: it
//! sizes its outputs from a mid-pipeline readback, so it runs eagerly and hands
//! its finished tensors back to the stream through a custom op. Binding them as
//! `Init` tensors instead would work, but an `Init` makes burn execute the
//! pending queue, which costs ~640 unfused launches in each refine step.

use burn::tensor::{DType, Shape};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::FusionHandle;
use burn_fusion::custom::{
    CustomOpIr, HandleContainer, OperationFn, OperationIr, OperationOutput, StreamId, TensorIr,
};

/// Handle container a fusion custom op executes against.
type FusionHandles = HandleContainer<FusionHandle<FusionCubeRuntime>>;
/// A tensor living in the fusion stream.
type FusionTensor = burn_fusion::FusionTensor<FusionCubeRuntime>;
/// The fusion client that owns the stream.
type FusionClient = burn_fusion::Client<FusionCubeRuntime>;

/// Register a concrete-backend function as a custom op on the fusion stream.
/// `inputs` reach the op once the stream gets to it; each `(shape, dtype)`
/// becomes a handle the op fills in. `desc.as_fixed()` looks both up by id.
pub(crate) fn register_custom<const N: usize, const M: usize, F>(
    client: &FusionClient,
    name: &'static str,
    inputs: [FusionTensor; N],
    outputs: [(Shape, DType); M],
    op: F,
) -> [FusionTensor; M]
where
    F: Fn(&CustomOpIr, &mut FusionHandles) + Send + Sync + 'static,
{
    let outputs =
        outputs.map(|(shape, dtype)| TensorIr::uninit(client.create_empty_handle(), shape, dtype));
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &outputs);
    let run = {
        let desc = desc.clone();
        OperationFn(move |handles: &mut FusionHandles| {
            op(&desc, handles);
            Ok(())
        })
    };
    client
        .register(StreamId::current(), OperationIr::Custom(desc), run)
        .outputs()
}
