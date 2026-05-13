use tokio::sync::watch;

/// Read-only frame-indexed view of splat snapshots published by a
/// single producer (the train / load stream) into a [`SlotSender`].
///
/// Backed by `tokio::sync::watch::Receiver<Vec<T>>`, so cloning a
/// `Slot` is cheap and every consumer sees the same shared state.
/// Reads (`get` / `latest` / `len`) clone snapshots out without
/// blocking the producer.
#[derive(Clone)]
pub struct Slot<T> {
    rx: watch::Receiver<Vec<T>>,
}

impl<T: Clone + Send + Sync + 'static> Slot<T> {
    /// An empty, never-updated `Slot`. Useful as a placeholder before
    /// a process has been wired up.
    pub fn empty() -> Self {
        // Drop the sender immediately — the receiver still works for
        // reads but observes the initial empty Vec forever.
        let (_, rx) = watch::channel(Vec::new());
        Self { rx }
    }

    pub fn get(&self, index: usize) -> Option<T> {
        self.rx.borrow().get(index).cloned()
    }

    pub fn latest(&self) -> Option<T> {
        self.rx.borrow().last().cloned()
    }

    pub fn len(&self) -> usize {
        self.rx.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.rx.borrow().is_empty()
    }
}

impl<T: Clone + Send + Sync + 'static> Default for Slot<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Write side of a [`Slot`]. Single producer; consumers hold cloneable
/// [`Slot`] receivers.
pub struct SlotSender<T> {
    tx: watch::Sender<Vec<T>>,
}

impl<T: Send + Sync + 'static> SlotSender<T> {
    /// Replace value at `index`. If `index == len()` the value is
    /// appended. Panics if `index > len()`.
    pub fn set(&self, index: usize, value: T) {
        self.tx.send_modify(|vec| match index.cmp(&vec.len()) {
            std::cmp::Ordering::Less => vec[index] = value,
            std::cmp::Ordering::Equal => vec.push(value),
            std::cmp::Ordering::Greater => {
                panic!(
                    "SlotSender::set index {index} past end (len = {})",
                    vec.len()
                )
            }
        });
    }

    pub fn clear(&self) {
        self.tx.send_modify(|vec| vec.clear());
    }
}

/// Build a fresh `(SlotSender, Slot)` pair backed by a single
/// `watch::channel`. The sender goes to the producer; the receiver
/// (and any clones) goes to consumers.
pub fn channel<T: Send + Sync + 'static>() -> (SlotSender<T>, Slot<T>) {
    let (tx, rx) = watch::channel(Vec::new());
    (SlotSender { tx }, Slot { rx })
}
