use std::sync::atomic::Ordering::{Acquire, Release, Relaxed};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::{ptr, mem};
use std::cmp;
use std::cell::UnsafeCell;

use mem::epoch::{self, Atomic, Owned, StaticDrop};

const SEG_SIZE: usize = 32;

// FIXME: add Drop impl to drop pending items

/// A Michael-Scott queue that allocates "segments" (arrays of nodes)
/// for efficiency.
///
/// Usable with any number of producers and consumers.
pub struct SegQueue<T> {
    head: AtomicSegment<T>,
    tail: AtomicSegment<T>,
}

type AtomicSegment<T> = Atomic<Segment<T>, ()>;

struct Segment<T> {
    low: AtomicUsize,
    items: [Item<T>; SEG_SIZE],
    high: AtomicUsize,
    next: AtomicSegment<T>,
}

unsafe impl<T> StaticDrop for Segment<T> {}

struct Item<T> {
    // Uninitialized until `ready` is `true`
    data: UnsafeCell<Option<T>>,
    ready: AtomicBool,
}

impl<T> Item<T> {
    // Insert data into an item. Can only be invoked once per item,
    // and only when the `ready` flag is false.
    fn put(&self, data: T) {
        debug_assert!(!self.ready.load(Relaxed));

        // the existing `data` contents are uninitialized
        unsafe {
            ptr::write(self.data.get(), Some(data));
        }

        // store `ready` flag in `Release` so that readers will see the data
        self.ready.store(true, Release);
    }

    // Spin until data is ready, and then extract that data. Can only be invoked
    // once per item.
    fn take(&self) -> T {
        loop {
            if self.ready.load(Acquire) { break; }
        }
        unsafe {
            (*self.data.get()).take().unwrap()

        }
    }
}

impl<T> Drop for Item<T> {
    fn drop(&mut self) {
        // If the item was never written, go ahead and initialize the data
        // component so that the interio drop will work correctly.
        if !self.ready.load(Relaxed) {
            unsafe {
                ptr::write(self.data.get(), None);
            }
        }
    }
}

unsafe impl<T: Send> Sync for Segment<T> {}

impl<T> Segment<T> {
    fn new() -> Segment<T> {
        let mut seg = Segment {
            items: unsafe { mem::uninitialized() },
            low: AtomicUsize::new(0),
            high: AtomicUsize::new(0),
            next: Atomic::null(),
        };
        for item in &mut seg.items {
            item.ready = AtomicBool::new(false);
        }
        seg
    }
}

impl<T> SegQueue<T> {
    /// Create a new, empty queue.
    pub fn new() -> SegQueue<T> {
        let q = SegQueue {
            head: Atomic::null(),
            tail: Atomic::null(),
        };
        let sentinel = Owned::new(Segment::new(), ());
        let guard = epoch::pin();
        let sentinel = q.head.store_and_ref(sentinel, Relaxed, &guard);
        q.tail.store_shared(Some(sentinel), Relaxed);
        q
    }

    /// Add `t` to the back of the queue.
    pub fn push(&self, t: T) {
        let guard = epoch::pin();
        loop {
            let tail = self.tail.load(Acquire, &guard).unwrap();
            if tail.high.load(Relaxed) >= SEG_SIZE { continue }
            let i = tail.high.fetch_add(1, Relaxed);
            if i < SEG_SIZE {
                unsafe {
                    (*tail).items.get_unchecked(i).put(t);
                }
                if i + 1 == SEG_SIZE {
                    let tail_seg = Owned::new(Segment::new(), ());
                    let tail = tail.next.store_and_ref(tail_seg, Release, &guard);
                    self.tail.store_shared(Some(tail), Release);
                }

                return
            }
        }
    }

    /// Attempt to dequeue from the front.
    ///
    /// Returns `None` if the queue is observed to be empty.
    pub fn try_pop(&self) -> Option<T> {
        let guard = epoch::pin();
        loop {
            let head = self.head.load(Acquire, &guard).unwrap();
            loop {
                let low = head.low.load(Relaxed);
                if low >= cmp::min(head.high.load(Relaxed), SEG_SIZE) { break }
                if head.low.compare_and_swap(low, low+1, Relaxed) == low {
                    let data = unsafe {
                        (*head).items.get_unchecked(low).take()
                    };
                    if low + 1 == SEG_SIZE {
                        loop {
                            if let Some(next) = head.next.load(Acquire, &guard) {
                                self.head.store_shared(Some(next), Release);
                                unsafe { guard.unlinked(head); }
                                break;
                            }
                        }
                    }
                    return Some(data);
                }
            }
            if head.next.load(Relaxed, &guard).is_none() { return None }
        }
    }
}

#[cfg(test)]
mod test {
    const CONC_COUNT: i64 = 1000000;

    use scope;
    use super::*;

    #[test]
    fn push_pop_1() {
        let q: SegQueue<i64> = SegQueue::new();
        q.push(37);
        assert_eq!(q.try_pop(), Some(37));
    }

    #[test]
    fn push_pop_2() {
        let q: SegQueue<i64> = SegQueue::new();
        q.push(37);
        q.push(48);
        assert_eq!(q.try_pop(), Some(37));
        assert_eq!(q.try_pop(), Some(48));
    }

    #[test]
    fn push_pop_many_seq() {
        let q: SegQueue<i64> = SegQueue::new();
        for i in 0..200 {
            q.push(i)
        }
        for i in 0..200 {
            assert_eq!(q.try_pop(), Some(i));
        }
    }

    #[test]
    fn push_pop_many_spsc() {
        let q: SegQueue<i64> = SegQueue::new();

        scope(|scope| {
            scope.spawn(|| {
                let mut next = 0;

                while next < CONC_COUNT {
                    if let Some(elem) = q.try_pop() {
                        assert_eq!(elem, next);
                        next += 1;
                    }
                }
            });

            for i in 0..CONC_COUNT {
                q.push(i)
            }
        });
    }

    #[test]
    fn push_pop_many_spmc() {
        fn recv(_t: i32, q: &SegQueue<i64>) {
            let mut cur = -1;
            for _i in 0..CONC_COUNT {
                if let Some(elem) = q.try_pop() {
                    assert!(elem > cur);
                    cur = elem;

                    if cur == CONC_COUNT - 1 { break }
                }
            }
        }

        let q: SegQueue<i64> = SegQueue::new();
        let qr = &q;
        scope(|scope| {
            for i in 0..3 {
                scope.spawn(move || recv(i, qr));
            }

            scope.spawn(|| {
                for i in 0..CONC_COUNT {
                    q.push(i);
                }
            })
        });
    }

    #[test]
    fn push_pop_many_mpmc() {
        enum LR { Left(i64), Right(i64) }

        let q: SegQueue<LR> = SegQueue::new();

        scope(|scope| {
            for _t in 0..2 {
                scope.spawn(|| {
                    for i in CONC_COUNT-1..CONC_COUNT {
                        q.push(LR::Left(i))
                    }
                });
                scope.spawn(|| {
                    for i in CONC_COUNT-1..CONC_COUNT {
                        q.push(LR::Right(i))
                    }
                });
                scope.spawn(|| {
                    let mut vl = vec![];
                    let mut vr = vec![];
                    for _i in 0..CONC_COUNT {
                        match q.try_pop() {
                            Some(LR::Left(x)) => vl.push(x),
                            Some(LR::Right(x)) => vr.push(x),
                            _ => {}
                        }
                    }

                    let mut vl2 = vl.clone();
                    let mut vr2 = vr.clone();
                    vl2.sort();
                    vr2.sort();

                    assert_eq!(vl, vl2);
                    assert_eq!(vr, vr2);
                });
            }
        });
    }
}
