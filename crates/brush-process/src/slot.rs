use std::sync::{Arc, Mutex, MutexGuard};

/// A thread-safe slot for sharing data between the process and UI.
/// Uses Mutex because the inner type (Splats) is not Sync.
#[derive(Clone)]
pub struct Slot<T>(Arc<Mutex<Vec<T>>>);

impl<T: Clone> Slot<T> {
    pub fn write(&self) -> MutexGuard<'_, Vec<T>> {
        self.0.lock().unwrap()
    }

    pub fn push(&self, value: T) {
        self.0.lock().unwrap().push(value);
    }

    pub fn clear(&self) {
        self.0.lock().unwrap().clear();
    }

    pub fn len(&self) -> usize {
        self.0.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.lock().unwrap().is_empty()
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.0.lock().unwrap().get(index).cloned()
    }

    pub fn get_main(&self) -> Option<T> {
        self.0.lock().unwrap().last().cloned()
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(Mutex::new(Vec::new())))
    }
}
