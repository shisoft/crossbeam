use std::sync::atomic::Ordering::{Acquire, Release, Relaxed};

use mem::epoch::{self, Atomic, Owned, StaticDrop};

// FIXME: add Drop impl to drop pending items

/// Treiber's lock-free stack.
///
/// Usable with any number of producers and consumers.
pub struct TreiberStack<T> {
    head: NodePtr<T>,
}
type NodePtr<T> = Atomic<SharedData<T>, T>;

struct SharedData<T> {
    next: NodePtr<T>,
}

unsafe impl<T> StaticDrop for SharedData<T> {}

impl<T> TreiberStack<T> {
    /// Create a new, empty stack.
    pub fn new() -> TreiberStack<T> {
        TreiberStack {
            head: Atomic::null()
        }
    }

    /// Push `t` on top of the stack.
    pub fn push(&self, t: T) {
        let mut n = Owned::new(SharedData { next: Atomic::null() }, t);
        let guard = epoch::pin();
        loop {
            let head = self.head.load(Relaxed, &guard);
            n.shared_mut().next.store_shared(head, Relaxed);
            match self.head.cas_and_ref(head, n, Release, &guard) {
                Ok(_) => break,
                Err(owned) => n = owned,
            }
        }
    }

    /// Attempt to pop the top element of the stack.
    ///
    /// Returns `None` if the stack is observed to be empty.
    pub fn pop(&self) -> Option<T> {
        let guard = epoch::pin();
        loop {
            match self.head.load(Acquire, &guard) {
                Some(head) => {
                    let next = head.next.load(Relaxed, &guard);
                    if self.head.cas_shared(Some(head), next, Release) {
                        unsafe {
                            return Some(guard.unlinked(head));
                        }
                    }
                }
                None => return None
            }
        }
    }

    /// Check if this queue is empty.
    pub fn is_empty(&self) -> bool {
        let guard = epoch::pin();
        self.head.load(Acquire, &guard).is_none()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn is_empty() {
        let q: TreiberStack<i64> = TreiberStack::new();
        assert!(q.is_empty());
        q.push(1);
        q.push(2);
        assert!(!q.is_empty());
        assert!(q.pop() == Some(1));
        assert!(q.pop() == Some(2));
        assert!(q.is_empty());
        q.push(25);
        assert!(!q.is_empty());
    }
}
