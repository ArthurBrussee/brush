//! Native [`Actor`] backed by a dedicated `std::thread` running a
//! `tokio` current-thread runtime with a `LocalSet`. Futures spawned
//! on the actor live entirely on that one thread and are not required
//! to be `Send`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use tokio::sync::{mpsc, oneshot};
use tokio::task::{AbortHandle, LocalSet};

/// A task to be set up on the actor's thread. When invoked there it
/// builds the (possibly !Send) future and spawns it on the `LocalSet`.
type Setup = Box<dyn FnOnce() + Send + 'static>;

/// Single-threaded pinned async executor. See crate docs for rationale.
pub struct Actor {
    tx: mpsc::UnboundedSender<Setup>,
    // Held to keep the thread name diagnostic alive; the actor thread
    // exits on Drop when `tx` is dropped (channel closes).
    _join: std::thread::JoinHandle<()>,
}

impl Actor {
    /// Spin up an actor on its own OS thread named `name`.
    ///
    /// The thread runs a `tokio` current-thread runtime + `LocalSet`.
    /// It exits cleanly when this `Actor` is dropped.
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
        Self { tx, _join: join }
    }

    /// Run a closure on the actor that produces a (possibly !Send)
    /// future. Returns a [`JoinHandle`] for the result.
    ///
    /// The closure must be `Send + 'static` (it crosses to the actor's
    /// thread). The future does NOT need to be `Send`. `R` must be
    /// `Send` (it crosses back via the join channel). Drop the
    /// [`JoinHandle`] (or `.detach()`) to fire-and-forget — the task
    /// keeps running until it completes or [`JoinHandle::abort`] is
    /// called.
    pub fn run<F, Fut, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel::<R>();
        let state = Arc::new(HandleState::default());
        let state_setup = state.clone();
        let state_task = state.clone();
        let setup: Setup = Box::new(move || {
            let jh = tokio::task::spawn_local(async move {
                let r = f().await;
                state_task.finished.store(true, Ordering::SeqCst);
                let _ = tx.send(r);
            });
            // If abort() was called before the task got scheduled,
            // honor it now; otherwise stash the AbortHandle so a later
            // abort() can cancel. `finished` is set by abort() itself
            // — the cancelled future won't run its completion block.
            if state_setup.aborted.load(Ordering::SeqCst) {
                jh.abort();
            } else {
                *state_setup.abort.lock().expect("poisoned") = Some(jh.abort_handle());
            }
        });
        let _ = self.tx.send(setup);
        JoinHandle { rx, state }
    }
}

#[derive(Default)]
struct HandleState {
    abort: Mutex<Option<AbortHandle>>,
    aborted: AtomicBool,
    finished: AtomicBool,
}

/// Awaitable handle to the result of [`Actor::run`].
///
/// `await` resolves to the task's return value, or panics if the task
/// panicked / was aborted / the actor was dropped before completion.
/// Tokio's default panic hook prints the original panic to stderr
/// from the actor's thread, so the real cause is visible regardless.
pub struct JoinHandle<R> {
    rx: oneshot::Receiver<R>,
    state: Arc<HandleState>,
}

impl<R> JoinHandle<R> {
    /// Drop the handle without awaiting. Cleaner than `let _ =
    /// actor.run(...)` (which clippy flags).
    pub fn detach(self) {}

    /// Cancel the underlying task. Real cancellation via
    /// `tokio::task::AbortHandle` once the task has been scheduled;
    /// otherwise records the intent so the actor cancels it as soon as
    /// it schedules the task. `is_finished()` returns `true` after.
    pub fn abort(&self) {
        self.state.aborted.store(true, Ordering::SeqCst);
        if let Some(h) = self.state.abort.lock().expect("poisoned").as_ref() {
            h.abort();
        }
        // The aborted future won't run its completion block, so mark
        // finished here. Idempotent — natural completion would have
        // set it too.
        self.state.finished.store(true, Ordering::SeqCst);
    }

    /// `true` once the task has finished (completed or been aborted).
    pub fn is_finished(&self) -> bool {
        self.state.finished.load(Ordering::SeqCst)
    }
}

impl<R> Future for JoinHandle<R> {
    type Output = R;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(r)) => Poll::Ready(r),
            Poll::Ready(Err(_)) => {
                panic!("brush-async: actor task panicked, was aborted, or actor dropped")
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Cooperatively yield to the executor.
pub async fn yield_now() {
    tokio::task::yield_now().await;
}
