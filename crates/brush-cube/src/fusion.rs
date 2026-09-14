//! Running brush's own kernels under burn's `Fusion` backend.
//!
//! A custom op takes concrete `CubeTensor`s once the fusion stream reaches
//! it, so the elementwise ops on either side still fuse among themselves.

use burn::tensor::{DType, Shape};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::custom::{CustomOpIr, HandleContainer, OperationIr, TensorIr};
use burn_fusion::{
    ExecutionError, FusionHandle,
    stream::{Operation, StreamId},
};

/// Handle container a fusion custom op executes against.
pub type FusionHandles = HandleContainer<FusionHandle<FusionCubeRuntime>>;
/// A tensor living in the fusion stream.
pub type FusionTensor = burn_fusion::FusionTensor<FusionCubeRuntime>;
/// The fusion client that owns the stream.
pub type FusionClient = burn_fusion::Client<FusionCubeRuntime>;

struct ClosureOp<F> {
    desc: CustomOpIr,
    op: F,
}

impl<F> std::fmt::Debug for ClosureOp<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ClosureOp({:?})", self.desc)
    }
}

impl<F> Operation<FusionCubeRuntime> for ClosureOp<F>
where
    F: Fn(&CustomOpIr, &mut FusionHandles) + Send + Sync + 'static,
{
    fn execute(&self, h: &mut FusionHandles) -> Result<(), ExecutionError> {
        (self.op)(&self.desc, h);
        Ok(())
    }
}

/// Register a concrete-backend function as a custom op on the fusion stream.
///
/// `inputs` are handed to the op once the stream reaches it, and each
/// `(shape, dtype)` in `outputs` becomes a fresh handle the op must fill in
/// through the handle container. The op gets the description so it can look
/// both up by id with `desc.as_fixed()`.
pub fn register_custom<const N: usize, const M: usize, F>(
    client: &FusionClient,
    name: &'static str,
    inputs: [FusionTensor; N],
    outputs: [(Shape, DType); M],
    op: F,
) -> [FusionTensor; M]
where
    F: Fn(&CustomOpIr, &mut FusionHandles) + Send + Sync + 'static,
{
    use burn_fusion::custom::OperationOutput;

    let outputs =
        outputs.map(|(shape, dtype)| TensorIr::uninit(client.create_empty_handle(), shape, dtype));
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &outputs);
    let op = ClosureOp {
        desc: desc.clone(),
        op,
    };
    client
        .register(StreamId::current(), OperationIr::Custom(desc), op)
        .outputs()
}
