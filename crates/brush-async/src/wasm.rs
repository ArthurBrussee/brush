//! WASM [`Actor`] — single-threaded.
//!
//! Wasm only has one thread by default — the JS event loop — so every
//! `Actor` here just shares the main-thread `wasm_bindgen_futures`
//! executor. The `Actor::run` / `Actor::spawn` surface still works
//! because single-thread is trivially `!Send`-friendly.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use tokio::sync::oneshot;
use wasm_bindgen_futures::spawn_local;

/// Single-threaded `Actor`: shares the main-thread executor. No `Send`
/// bound since nothing actually crosses threads.
pub struct Actor {
    _name: String,
}

impl Actor {
    pub fn new(name: &str) -> Self {
        Self {
            _name: name.to_owned(),
        }
    }

    pub fn run<F, Fut, R>(&self, f: F) -> impl Future<Output = R>
    where
        F: FnOnce() -> Fut + 'static,
        Fut: Future<Output = R> + 'static,
        R: 'static,
    {
        let (tx, rx) = oneshot::channel();
        spawn_local(async move {
            let r = f().await;
            let _ = tx.send(r);
        });
        async move { rx.await.expect("brush-async: spawned task dropped") }
    }

    pub fn spawn<F, Fut>(&self, f: F)
    where
        F: FnOnce() -> Fut + 'static,
        Fut: Future<Output = ()> + 'static,
    {
        spawn_local(async move { f().await });
    }
}

/// Mirror of [`tokio::task::JoinError`] for the wasm shim. Only the
/// `is_cancelled` / `is_panic` query methods are provided — the
/// `into_panic` family is unsupported on wasm.
#[derive(Debug)]
pub struct JoinError {
    cancelled: bool,
}

impl JoinError {
    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }
    pub fn is_panic(&self) -> bool {
        !self.cancelled
    }
}

impl std::fmt::Display for JoinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.cancelled {
            f.write_str("task was cancelled")
        } else {
            f.write_str("task panicked")
        }
    }
}

impl std::error::Error for JoinError {}

/// Wasm shim mirroring [`tokio::task::JoinHandle`]. Backed by a
/// `oneshot` channel plus shared `AtomicBool` flags for abort/finished
/// state. Awaiting yields `Result<T, JoinError>`.
///
/// `abort` is cooperative on wasm: the spawned future is not literally
/// cancelled, but on completion the handle reports `JoinError::is_cancelled`
/// instead of returning the result. Tasks that need promptness should
/// `task::yield_now().await` and exit when the handle's caller is gone.
pub struct JoinHandle<T> {
    rx: oneshot::Receiver<T>,
    aborted: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

impl<T> JoinHandle<T> {
    pub fn abort(&self) {
        self.aborted.store(true, Ordering::SeqCst);
    }
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = Result<T, JoinError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.aborted.load(Ordering::SeqCst) && self.finished.load(Ordering::SeqCst) {
            return Poll::Ready(Err(JoinError { cancelled: true }));
        }
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(v)) => Poll::Ready(Ok(v)),
            Poll::Ready(Err(_)) => Poll::Ready(Err(JoinError { cancelled: true })),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Migration shim for `tokio_with_wasm::task::spawn`. Schedules `future`
/// on `wasm_bindgen_futures::spawn_local` and returns a `JoinHandle`.
pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
where
    F: Future + 'static,
    F::Output: 'static,
{
    let (tx, rx) = oneshot::channel();
    let aborted = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(AtomicBool::new(false));
    let finished_c = finished.clone();
    spawn_local(async move {
        let r = future.await;
        finished_c.store(true, Ordering::SeqCst);
        let _ = tx.send(r);
    });
    JoinHandle {
        rx,
        aborted,
        finished,
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
