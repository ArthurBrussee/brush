use std::sync::Arc;
use tokio::sync::Mutex;

/// A thread-safe async slot for sharing data between the process and UI.
/// Uses tokio's async Mutex so locks can be held across await points.
#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Vec<T>>>);

impl<T: Clone> Slot<T> {
    /// Temporarily take ownership of a value at a specific index, do something with it,
    /// and put the result back. The lock is held across the async operation.
    ///
    /// Uses swap + push to avoid needing a placeholder value.
    /// The value is moved to the end during the operation, then swapped back.
    pub async fn act<F, R>(&self, index: usize, f: F) -> Option<R>
    where
        F: AsyncFnOnce(T) -> (T, R),
    {
        let mut guard = self.0.lock().await;
        let len = guard.len();
        if index >= len {
            return None;
        }
        // Swap the target element to the end, then pop it
        guard.swap(index, len - 1);
        let value = guard.pop().unwrap();
        let (new_value, result) = f(value).await;

        // Push it back and swap to original position
        guard.push(new_value);
        let new_len = guard.len();
        guard.swap(index, new_len - 1);
        Some(result)
    }

    pub async fn map<F, R>(&self, index: usize, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        self.act(index, async move |value| {
            let ret = f(&value);
            (value, ret)
        })
        .await
    }

    /// Get a clone of the main (last) value.
    pub async fn clone_main(&self) -> Option<T> {
        self.0.lock().await.last().cloned()
    }

    /// Set the slot to contain a single value, clearing any previous contents.
    pub async fn set(&self, value: T) {
        let mut guard = self.0.lock().await;
        guard.clear();
        guard.push(value);
    }

    /// Set the value at the given index, or push if index equals current length.
    /// Panics if index > len (gaps not allowed).
    pub async fn set_at(&self, index: usize, value: T) {
        let mut guard = self.0.lock().await;
        if index == guard.len() {
            guard.push(value);
        } else {
            guard[index] = value;
        }
    }

    /// Clear the slot.
    pub async fn clear(&self) {
        self.0.lock().await.clear();
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
