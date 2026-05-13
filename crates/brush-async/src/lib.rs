//! Pinned single-threaded async executors that hide the native/wasm
//! split behind one API.
//!
//! Brush's GPU layer (cubecl/burn-fusion) keys ordering on the current
//! OS thread (`StreamId::current()` is thread-local). tokio's
//! work-stealing scheduler moves async tasks across threads at every
//! `.await`, so a single logical render that issues GPU work before
//! and after an await ends up registering ops against two different
//! `StreamId`s. The resulting cross-stream dispatch produces visible
//! corruption (duplicate IDs, NaNs, stale buffers) even when the
//! underlying handle bookkeeping stays consistent.
//!
//! [`Actor`] sidesteps that: it owns one OS thread (native) or runs
//! against one JS event loop (wasm) and pins every future it executes
//! to that single context. Futures spawned on an `Actor` therefore do
//! NOT need to be `Send`, and `StreamId::current()` is invariant for
//! their entire lifetime.
//!
//! ## Usage
//!
//! ```no_run
//! # use brush_async::Actor;
//! # async fn doc() {
//! let renderer = Actor::new("renderer");
//! // Closure is Send + 'static (must cross threads to reach the
//! // actor). The future it produces does NOT need to be Send. The
//! // returned `JoinHandle` resolves to the closure's output; drop it
//! // to fire-and-forget.
//! let pixels: Vec<u8> = renderer
//!     .run(|| async move {
//!         // ... work that needs !Send types ...
//!         vec![0u8; 4]
//!     })
//!     .await;
//! # }
//! ```

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::*;

#[cfg(target_family = "wasm")]
mod wasm;
#[cfg(target_family = "wasm")]
pub use wasm::*;

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use std::rc::Rc;

    /// Holding an `Rc` across an `.await` makes the future `!Send`. If
    /// our Actor required `Send` futures (or tokio's multi-thread
    /// scheduler tried to move it across threads) this wouldn't
    /// compile.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_accepts_non_send_future() {
        let actor = Actor::new("test-actor");
        let result = actor
            .run(|| async move {
                let rc = Rc::new(42);
                tokio::task::yield_now().await;
                *rc
            })
            .await;
        assert_eq!(result, 42);
    }

    /// Each call to `run` produces a fresh future, so we can issue many
    /// in flight and they should serialise on the actor thread.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn actor_serialises_tasks() {
        let actor = Actor::new("test-actor");
        let mut handles = Vec::new();
        for i in 0..16u32 {
            handles.push(actor.run(move || async move {
                tokio::task::yield_now().await;
                i
            }));
        }
        let mut results: Vec<u32> = futures::future::join_all(handles).await;
        results.sort();
        assert_eq!(results, (0..16).collect::<Vec<_>>());
    }

    /// Panic in a spawned task should propagate to the caller with
    /// the original message preserved (via `resume_unwind`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[should_panic(expected = "boom")]
    async fn actor_propagates_panic() {
        let actor = Actor::new("test-actor");
        actor
            .run(|| async move {
                panic!("boom");
            })
            .await;
    }
}
