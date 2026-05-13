//! WASM [`Actor`] — single-threaded.
//!
//! Wasm only has one thread by default — the JS event loop — so every
//! `Actor` here just shares the main-thread `wasm_bindgen_futures`
//! executor. `Actor::run` still works because single-thread is
//! trivially `!Send`-friendly.

use std::future::Future;
use std::pin::Pin;
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

    /// Run a closure that produces a (possibly !Send) future. Returns
    /// a [`JoinHandle`] for the result. Drop the handle to
    /// fire-and-forget.
    pub fn run<F, Fut, R>(&self, f: F) -> JoinHandle<R>
    where
        F: FnOnce() -> Fut + 'static,
        Fut: Future<Output = R> + 'static,
        R: 'static,
    {
        let (tx, rx) = oneshot::channel::<R>();
        spawn_local(async move {
            let r = f().await;
            let _ = tx.send(r);
        });
        JoinHandle { rx }
    }
}

/// Awaitable handle to the result of [`Actor::run`].
///
/// Backed by a `oneshot`. Drop to fire-and-forget; awaiting panics if
/// the task panicked or was dropped before completing.
pub struct JoinHandle<R> {
    rx: oneshot::Receiver<R>,
}

impl<R> JoinHandle<R> {
    /// Drop the handle without awaiting. The underlying task keeps
    /// running until it completes. Cleaner than `let _ =
    /// actor.run(...)` (which clippy flags).
    pub fn detach(self) {}
}

impl<R> Future for JoinHandle<R> {
    type Output = R;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<R> {
        match Pin::new(&mut self.rx).poll(cx) {
            Poll::Ready(Ok(r)) => Poll::Ready(r),
            Poll::Ready(Err(_)) => panic!("brush-async: actor task panicked or actor dropped"),
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
