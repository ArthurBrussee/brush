//! Latest-value request/response worker.
use tokio::sync::watch;

use crate::Actor;

pub struct AsyncMap<Req, Out> {
    req: watch::Sender<Option<Req>>,
    out: watch::Receiver<Option<Out>>,
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

        actor
            .run(move || async move {
                while req_rx.changed().await.is_ok() {
                    let Some(r) = req_rx.borrow_and_update().clone() else {
                        continue;
                    };
                    let output = map(&r).await;
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
}
