//! WASM [`Actor`] — single-threaded.
//!
//! Wasm only has one thread by default — the JS event loop — so every
//! `Actor` here just shares the main-thread `wasm_bindgen_futures`
//! executor. `Actor::run` still works because single-thread is
//! trivially `!Send`-friendly.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use tokio::sync::oneshot;
use wasm_bindgen_futures::spawn_local;

/// Single-threaded `Actor`: shares the main-thread executor.
///
/// On wasm everything lives on the main thread anyway; `Actor` is just a
/// trivial handle that's cheap to clone.
#[derive(Clone)]
pub struct Actor;

impl Actor {
    pub fn new(_name: &str) -> Self {
        Self
    }

    /// Run a closure that produces a (possibly !Send) future. Returns
    /// a [`JoinHandle`] for the result. Drop the handle (or
    /// `.detach()`) to fire-and-forget.
    pub fn run<F, Fut, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> Fut + 'static,
        Fut: Future<Output = R> + 'static,
        R: 'static,
    {
        let (tx, rx) = oneshot::channel::<R>();
        let state = Arc::new(HandleState::default());
        let state_task = state.clone();
        spawn_local(async move {
            let r = f().await;
            state_task.finished.store(true, Ordering::SeqCst);
            let _ = tx.send(r);
        });
        JoinHandle { rx, state }
    }
}

#[derive(Default)]
struct HandleState {
    /// Set by `abort()`. Cooperative: tasks that care should check it
    /// at await points (wasm-bindgen-futures has no real cancellation).
    aborted: AtomicBool,
    /// Set by the spawned task when it returns.
    finished: AtomicBool,
}

/// Awaitable handle to the result of [`Actor::run`].
///
/// `await` resolves to the task's return value, or panics if the task
/// panicked. `abort()` on wasm is cooperative: the task continues
/// running but tasks that check the flag can early-exit.
pub struct JoinHandle<R> {
    rx: oneshot::Receiver<R>,
    state: Arc<HandleState>,
}

impl<R> JoinHandle<R> {
    /// Drop the handle without awaiting. Cleaner than `let _ =
    /// actor.run(...)` (which clippy flags).
    pub fn detach(self) {}

    /// Mark the task as aborted. Cooperative on wasm — the task keeps
    /// running unless it checks the flag at a yield point.
    pub fn abort(&self) {
        self.state.aborted.store(true, Ordering::SeqCst);
    }

    /// `true` once the task has finished.
    pub fn is_finished(&self) -> bool {
        self.state.finished.load(Ordering::SeqCst)
    }
}

impl<R> Future for JoinHandle<R> {
    type Output = R;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(r)) => Poll::Ready(r),
            Poll::Ready(Err(_)) => panic!("brush-async: actor task panicked"),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Yield to the browser event loop.
///
/// Schedules a `setTimeout(_, 0)`-resolved Promise and awaits it. That's
/// a real macrotask yield, so the browser gets a chance to paint, run
/// requestAnimationFrame, handle input, and run GC between iterations
/// of a long-running async task. A plain
/// `cx.waker().wake_by_ref(); Poll::Pending` only yields to the
/// `wasm_bindgen_futures` microtask queue — the browser stays starved.
/// Matches `tokio_with_wasm::task::yield_now`.
pub async fn yield_now() {
    #[wasm_bindgen::prelude::wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = globalThis, js_name = setTimeout)]
        fn set_timeout(cb: &js_sys::Function, ms: f64);
    }

    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        set_timeout(&resolve, 0.0);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}
