use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone)]
pub struct Slot<T> {
    inner: Arc<Mutex<Option<T>>>,
}

pub struct SlotGuard<'a, T> {
    pub value: Option<T>,
    slot: &'a Slot<T>,
}

impl<T> Slot<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Some(value))),
        }
    }

    pub fn take(&self) -> SlotGuard<'_, T> {
        let mut guard = self.inner.lock().unwrap();
        let value = guard.take();
        SlotGuard { value, slot: self }
    }

    pub fn put(&self, value: T) {
        *self.inner.lock().unwrap() = Some(value);
    }

    pub fn borrow(&self) -> MutexGuard<'_, Option<T>> {
        self.inner.lock().unwrap()
    }
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }
}

impl<T> SlotGuard<'_, T> {
    pub fn put(self, value: T) {
        *self.slot.inner.lock().unwrap() = Some(value);
    }
}
