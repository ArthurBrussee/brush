use std::sync::Arc;
use tokio::sync::watch;

/// Frame-indexed view of splat snapshots shared between producer
/// (train / load stream) and consumers (UI viewer, export, eval,
/// rerun).
///
/// Backed by `tokio::sync::watch::Sender<Vec<T>>`. Producers hold
/// their working state locally and call `set()` to publish snapshots;
/// consumers clone snapshots out via `get()` / `latest()` and use
/// them outside any lock. No `act`-style transform-in-place. Cloning
/// a `Slot` shares the same underlying view (every handle can read
/// and write).
#[derive(Clone)]
pub struct Slot<T>(Arc<watch::Sender<Vec<T>>>);

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self(Arc::new(watch::Sender::new(Vec::new())))
    }
}

impl<T: Clone + Send + Sync + 'static> Slot<T> {
    /// Replace value at `index`. If `index == len()` the value is
    /// appended. Panics if `index > len()`.
    pub fn set(&self, index: usize, value: T) {
        self.0.send_modify(|vec| match index.cmp(&vec.len()) {
            std::cmp::Ordering::Less => vec[index] = value,
            std::cmp::Ordering::Equal => vec.push(value),
            std::cmp::Ordering::Greater => {
                panic!("Slot::set index {index} past end (len = {})", vec.len())
            }
        });
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.0.borrow().get(index).cloned()
    }

    pub fn latest(&self) -> Option<T> {
        self.0.borrow().last().cloned()
    }

    pub fn len(&self) -> usize {
        self.0.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }

    pub fn clear(&self) {
        self.0.send_modify(|vec| vec.clear());
    }
}
