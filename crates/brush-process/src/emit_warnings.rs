use crate::message::ProcessMessage;
use async_fn_stream::TryStreamEmitter;

pub(crate) struct WarningEmitter<'a> {
    emitter: &'a TryStreamEmitter<ProcessMessage, anyhow::Error>,
}

impl<'a> WarningEmitter<'a> {
    pub(crate) fn new(emitter: &'a TryStreamEmitter<ProcessMessage, anyhow::Error>) -> Self {
        Self { emitter }
    }

    pub(crate) async fn warn_if_err(&self, res: Result<(), anyhow::Error>) {
        if let Err(error) = res {
            self.emitter.emit(ProcessMessage::Warning { error }).await;
        }
    }
}
