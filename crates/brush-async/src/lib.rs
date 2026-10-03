//! Local async executors with a shared native/wasm API.
//!
//! [`Actor`] owns one OS thread on native and uses the JS event loop on
//! wasm, so its futures need not be `Send`. This also keeps `CubeCL`'s
//! default per-thread stream identity stable.
//!
//! `CubeCL`'s explicit stream scopes can instead preserve GPU stream identity
//! across Tokio worker hops. Migrating GPU callers to those scopes does not
//! remove the general Actor API's support for `!Send` jobs.

#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(not(target_family = "wasm"))]
pub use native::*;

#[cfg(target_family = "wasm")]
mod wasm;
#[cfg(target_family = "wasm")]
pub use wasm::*;

mod latest;
pub use latest::AsyncMap;

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
