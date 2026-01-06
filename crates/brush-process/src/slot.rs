use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Option<T>>>);

impl<T: Clone> Slot<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(Mutex::new(Some(value))))
    }

    pub fn lock(&self) -> MutexGuard<'_, Option<T>> {
        self.0.lock().unwrap()
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }
}
