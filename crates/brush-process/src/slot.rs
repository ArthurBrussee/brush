use std::sync::Arc;
use tokio::sync::Mutex;

/// Async slot for sharing data between the process and UI.
#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Vec<T>>>);

impl<T: Clone> Slot<T> {
    /// Take ownership of value at index, apply async function, put result back.
    pub async fn act<F, R>(&self, index: usize, f: F) -> Option<R>
    where
        F: AsyncFnOnce(T) -> (T, R),
    {
        let mut guard = self.0.lock().await;
        let len = guard.len();
        if index >= len {
            return None;
        }
        guard.swap(index, len - 1);
        let value = guard.pop().unwrap();
        let (new_value, result) = f(value).await;

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

    pub async fn clone_main(&self) -> Option<T> {
        self.0.lock().await.last().cloned()
    }

    /// Replace all contents with a single value.
    pub async fn set(&self, value: T) {
        let mut guard = self.0.lock().await;
        guard.clear();
        guard.push(value);
    }

    /// Set value at index, or push if index == len. Panics if index > len.
    pub async fn set_at(&self, index: usize, value: T) {
        let mut guard = self.0.lock().await;
        if index == guard.len() {
            guard.push(value);
        } else {
            guard[index] = value;
        }
    }

    pub async fn clear(&self) {
        self.0.lock().await.clear();
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
