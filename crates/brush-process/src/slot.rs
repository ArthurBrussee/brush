use std::sync::Arc;

use tokio::sync::{Mutex, MutexGuard};

#[derive(Clone)]
pub struct Slot<T> {
    inner: Arc<Mutex<Option<T>>>,
}

impl<T: Clone> Slot<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(value))),
        }
    }

    pub async fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.inner.lock().await
    }

    pub async fn put(&self, value: T) {
        *self.inner.lock().await = Some(value);
    }

    pub fn block_cloned(&self) -> Option<T> {
        self.inner.blocking_lock().as_ref().cloned()
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }
}
