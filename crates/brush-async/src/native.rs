//! Native [`Actor`] backed by a dedicated `std::thread` running a
//! `tokio` current-thread runtime with a `LocalSet`. Futures spawned
//! on the actor live entirely on that one thread.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use tokio::runtime::LocalRuntime;
use tokio::sync::{mpsc, oneshot};
use tokio::task::AbortHandle;

/// A task to be set up on the actor's thread. When invoked there it
/// builds the (possibly !Send) future and spawns it on the `LocalSet`.
type Setup = Box<dyn FnOnce() + Send + 'static>;

/// Single-threaded pinned async executor. See crate docs for rationale.
///
/// `Actor` is itself a cheap handle: cloning it shares the underlying thread
/// and runtime. The thread exits when the last clone is dropped.
#[derive(Clone)]
pub struct Actor {
    tx: mpsc::UnboundedSender<Setup>,
}

impl Actor {
    /// Spin up an actor on its own OS thread named `name`.
    ///
    /// The thread runs a `tokio` current-thread runtime + `LocalSet`.
    /// It exits cleanly when this `Actor` is dropped.
    pub fn new(name: &str) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<Setup>();
        let name_owned = name.to_owned();
        std::thread::Builder::new()
            .name(name_owned)
            .spawn(move || {
                let rt = LocalRuntime::new().expect("brush-async: build current_thread runtime");

                rt.block_on(async move {
                    while let Some(setup) = rx.recv().await {
                        setup();
                    }
                });
            })
            .expect("brush-async: spawn actor thread");
        Self { tx }
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
        // Result delivery is `Result<R, JoinError>` so panic payloads
        // ride back to the caller with their original location + msg
        // (via `resume_unwind`), not flattened into a generic string.
        let (tx, rx) = oneshot::channel::<Result<R, tokio::task::JoinError>>();
        let state = Arc::new(HandleState::default());
        let state_setup = state.clone();
        let state_task = state.clone();
        let setup: Setup = Box::new(move || {
            let user_task = tokio::task::spawn_local(f());
            let abort = user_task.abort_handle();
            // Waiter forwards the user task's join result to the caller.
            tokio::task::spawn_local(async move {
                let result = user_task.await;
                state_task.finished.store(true, Ordering::SeqCst);
                let _ = tx.send(result);
            });
            // If abort() was called before scheduling, honor it now;
            // otherwise stash the AbortHandle for a later abort().
            if state_setup.aborted.load(Ordering::SeqCst) {
                abort.abort();
            } else {
                *state_setup.abort.lock().expect("poisoned") = Some(abort);
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
/// `.await` resolves to the task's return value. If the task panicked,
/// the original panic is re-raised on the awaiter via
/// [`std::panic::resume_unwind`] so the panic message, location, and
/// payload type are preserved.
pub struct JoinHandle<R> {
    rx: oneshot::Receiver<Result<R, tokio::task::JoinError>>,
    state: Arc<HandleState>,
}

impl<R> JoinHandle<R> {
    /// Drop the handle without awaiting.
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
            Poll::Ready(Ok(Ok(r))) => Poll::Ready(r),
            Poll::Ready(Ok(Err(join_err))) => {
                if join_err.is_panic() {
                    // Re-raise the original panic with full info.
                    std::panic::resume_unwind(join_err.into_panic());
                } else {
                    panic!("brush-async: actor task was cancelled");
                }
            }
            Poll::Ready(Err(_)) => panic!("brush-async: actor dropped before task completed"),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Cooperatively yield to the executor.
pub async fn yield_now() {
    tokio::task::yield_now().await;
}
