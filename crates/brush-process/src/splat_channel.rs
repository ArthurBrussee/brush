use std::sync::Arc;
use tokio::sync::watch;

/// Frame-indexed broadcast channel for splat snapshots.
///
/// Producers (the train / load stream) push frames via `set()`;
/// consumers (UI viewer, export, eval, rerun) clone snapshots out via
/// `get()` / `latest()`. Backed by `tokio::sync::watch::Sender<Vec<T>>`,
/// so a consumer could also `subscribe()` to a `Receiver` for change
/// notifications — brush currently signals via the process message bus
/// so the simple read APIs are enough.
///
/// Producers hold their working state locally; no `act`-style
/// transform-in-place. Cloning a `SplatChannel` shares the same
/// underlying channel (every handle can read and write).
#[derive(Clone)]
pub struct SplatChannel<T>(Arc<watch::Sender<Vec<T>>>);

impl<T> Default for SplatChannel<T> {
    fn default() -> Self {
        Self(Arc::new(watch::Sender::new(Vec::new())))
    }
}

impl<T: Clone + Send + Sync + 'static> SplatChannel<T> {
    /// Replace value at `index`. If `index == len()` the value is
    /// appended. Panics if `index > len()`.
    pub fn set(&self, index: usize, value: T) {
        self.0.send_modify(|vec| match index.cmp(&vec.len()) {
            std::cmp::Ordering::Less => vec[index] = value,
            std::cmp::Ordering::Equal => vec.push(value),
            std::cmp::Ordering::Greater => panic!(
                "SplatChannel::set index {index} past end (len = {})",
                vec.len()
            ),
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
