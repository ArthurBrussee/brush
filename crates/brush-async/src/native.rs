//! Native [`Actor`] backed by a dedicated `std::thread` running a
//! `tokio` current-thread runtime with a `LocalSet`. Futures spawned
//! on the actor live entirely on that one thread and are not required
//! to be `Send`.

use std::future::Future;

use tokio::sync::{mpsc, oneshot};
use tokio::task::LocalSet;

/// A task to be set up on the actor's thread. When invoked there it
/// builds the (possibly !Send) future and spawns it on the `LocalSet`.
type Setup = Box<dyn FnOnce() + Send + 'static>;

/// Single-threaded pinned async executor. See crate docs for rationale.
pub struct Actor {
    tx: mpsc::UnboundedSender<Setup>,
    // Held to keep the thread name diagnostic alive; the actor thread
    // exits on Drop when `tx` is dropped (channel closes).
    _join: Option<std::thread::JoinHandle<()>>,
}

impl Actor {
    /// Spin up an actor on its own OS thread named `name`.
    ///
    /// The thread runs a `tokio` current-thread runtime + `LocalSet`.
    /// It exits cleanly when this `Actor` (and all clones of its
    /// internal sender) is dropped.
    pub fn new(name: &str) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Setup>();
        let name_owned = name.to_owned();
        let join = std::thread::Builder::new()
            .name(name_owned.clone())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .thread_name(&name_owned)
                    .build()
                    .expect("brush-async: build current_thread runtime");
                let local = LocalSet::new();
                local.block_on(&rt, async move {
                    while let Some(setup) = rx.recv().await {
                        setup();
                    }
                });
            })
            .expect("brush-async: spawn actor thread");
        Self {
            tx,
            _join: Some(join),
        }
    }

    /// Run a closure on the actor that produces a (possibly !Send)
    /// future, and await its result.
    ///
    /// The closure must be `Send + 'static` because it travels from
    /// the caller's thread to the actor's thread. The future it
    /// returns lives on the actor's thread and is not required to be
    /// `Send`. The return value `R` must be `Send` because it travels
    /// back to the caller across thread boundaries.
    pub fn run<F, Fut, R>(&self, f: F) -> impl Future<Output = R>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        // The actor spawns the future locally and ships back the
        // `JoinHandle` over a oneshot. The caller then awaits the
        // handle like any other tokio task — gets panic propagation
        // for free.
        let (jh_tx, jh_rx) = oneshot::channel::<tokio::task::JoinHandle<R>>();
        let setup: Setup = Box::new(move || {
            let jh = tokio::task::spawn_local(f());
            // If the caller dropped before we got here, no one's
            // waiting — that's fine.
            let _ = jh_tx.send(jh);
        });
        let send_result = self.tx.send(setup);
        async move {
            send_result.expect("brush-async: actor receiver dropped");
            let jh = jh_rx
                .await
                .expect("brush-async: actor dropped before scheduling task");
            jh.await.expect("brush-async: task panicked")
        }
    }

    /// Fire-and-forget spawn on the actor.
    pub fn spawn<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        let setup: Setup = Box::new(move || {
            tokio::task::spawn_local(f());
        });
        let _ = self.tx.send(setup);
    }
}

/// Cooperatively yield to the executor.
pub async fn yield_now() {
    tokio::task::yield_now().await;
}

/// A re-export of [`tokio::task::JoinHandle`] for the migration shim
/// (see [`crate::task::spawn`]).
pub type JoinHandle<T> = tokio::task::JoinHandle<T>;

/// Migration shim for `tokio_with_wasm::task::spawn`. Same semantics as
/// `tokio::task::spawn` — requires `Send` and runs on whatever runtime
/// is current. Prefer [`Actor::run`] / [`Actor::spawn`] for new code.
pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    tokio::task::spawn(future)
}
