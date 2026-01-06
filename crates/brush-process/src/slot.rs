use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Vec<T>>>);

impl<T: Clone> Slot<T> {
    pub fn lock(&self) -> MutexGuard<'_, Vec<T>> {
        self.0.lock().unwrap()
    }

    /// Push a new frame to the sequence.
    pub fn push(&self, value: T) {
        self.0.lock().unwrap().push(value);
    }

    /// Clear all frames.
    pub fn clear(&self) {
        self.0.lock().unwrap().clear();
    }

    /// Get the number of frames.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
