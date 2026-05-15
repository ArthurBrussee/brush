//! Latest-value request/response worker.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering::SeqCst},
};

use tokio::sync::watch;

use crate::Actor;

pub struct AsyncMap<Req, Out> {
    req: watch::Sender<Option<Req>>,
    out: watch::Receiver<Option<Out>>,
    running: Arc<AtomicBool>,
    _actor: Actor,
}

impl<Req, Out> AsyncMap<Req, Out>
where
    Req: Clone + Send + Sync + 'static,
    Out: Clone + Send + Sync + 'static,
{
    /// Spawn a worker on `actor` that calls `work(req)` for each new request.
    /// Returning `None` skips publishing — the previous output stays visible.
    pub fn new(
        actor: Actor,
        mut map: impl AsyncFnMut(&Req) -> Out + Send + 'static,
        mut on_done: impl FnMut(&Req) + Send + 'static,
    ) -> Self {
        let (req, mut req_rx) = watch::channel::<Option<Req>>(None);
        let (out_tx, out) = watch::channel::<Option<Out>>(None);

        let running = Arc::new(AtomicBool::new(false));
        let running_task = running.clone();

        actor
            .run(move || async move {
                while req_rx.changed().await.is_ok() {
                    let Some(r) = req_rx.borrow_and_update().clone() else {
                        continue;
                    };
                    running_task.store(true, SeqCst);
                    let output = map(&r).await;
                    running_task.store(false, SeqCst);
                    if out_tx.send(Some(output)).is_err() {
                        break;
                    }
                    on_done(&r);
                }
            })
            .detach();

        Self {
            req,
            out,
            running,
            _actor: actor,
        }
    }

    /// Queue `req` for processing, superseding any older request.
    pub fn request(&self, req: Req) {
        let _ = self.req.send(Some(req));
    }

    /// The most recent successful output, if any.
    pub fn latest(&self) -> Option<Out> {
        self.out.borrow().clone()
    }

    /// The most recently submitted request, if any.
    pub fn last_request(&self) -> Option<Req> {
        self.req.borrow().clone()
    }

    /// Whether the worker is currently processing a request.
    pub fn is_running(&self) -> bool {
        self.running.load(SeqCst)
    }
}
