use std::sync::{Arc, Mutex, MutexGuard};

/// A thread-safe slot for sharing data between the process and UI.
/// For sequences (like animated splats), stores a Vec of frames.
#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Vec<T>>>);

impl<T> Slot<T> {
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

impl<T: Clone> Slot<T> {
    /// Get a clone of the frame at the given index.
    pub fn get(&self, index: usize) -> Option<T> {
        self.0.lock().unwrap().get(index).cloned()
    }

    /// Get the last frame (for single-splat usage during training).
    pub fn last(&self) -> Option<T> {
        self.0.lock().unwrap().last().cloned()
    }

    /// Set a single value, clearing any existing frames.
    pub fn set(&self, value: T) {
        let mut guard = self.0.lock().unwrap();
        guard.clear();
        guard.push(value);
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
