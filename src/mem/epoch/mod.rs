//! Epoch-based memory management
//!
//! This module provides fast, easy to use memory management for lock free data
//! structures. It's inspired by [Keir Fraser's *epoch-based
//! reclamation*](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-579.pdf).
//!
//! The basic problem this is solving is the fact that when one thread has
//! removed a node from a data structure, other threads may still have pointers
//! to that node (in the form of snapshots that will be validated through things
//! like compare-and-swap), so the memory cannot be immediately freed. Put differently:
//!
//! 1. There are two sources of reachability at play -- the data structure, and
//! the snapshots in threads accessing it. Before we delete a node, we need to know
//! that it cannot be reached in either of these ways.
//!
//! 2. Once a node has been unliked from the data structure, no *new* snapshots
//! reaching it will be created.
//!
//! Using the epoch scheme is fairly straightforward, and does not require
//! understanding any of the implementation details:
//!
//! - When operating on a shared data structure, a thread must "pin the current
//! epoch", which is done by calling `pin()`. This function returns a `Guard`
//! which unpins the epoch when destroyed.
//!
//! - When the thread subsequently reads from a lock-free data structure, the
//! pointers it extracts act like references with lifetime tied to the
//! `Guard`. This allows threads to safely read from snapshotted data, being
//! guaranteed that the data will remain allocated until they exit the epoch.
//!
//! To put the `Guard` to use, Crossbeam provides a set of three pointer types meant to work together:
//!
//! - `Owned<T>`, akin to `Box<T>`, which points to uniquely-owned data that has
//!   not yet been published in a concurrent data structure.
//!
//! - `Shared<'a, T>`, akin to `&'a T`, which points to shared data that may or may
//!   not be reachable from a data structure, but it guaranteed not to be freed
//!   during lifetime `'a`.
//!
//! - `Atomic<T>`, akin to `std::sync::atomic::AtomicPtr`, which provides atomic
//!   updates to a pointer using the `Owned` and `Shared` types, and connects them
//!   to a `Guard`.
//!
//! Each of these types provides further documentation on usage.
//!
//! # Example
//!
//! ```
//! use std::sync::atomic::Ordering::{Acquire, Release, Relaxed};
//! use std::ptr;
//!
//! use crossbeam::mem::epoch::{self, Atomic, Owned};
//!
//! struct TreiberStack<T> {
//!     head: Atomic<Node<T>>,
//! }
//!
//! struct Node<T> {
//!     data: T,
//!     next: Atomic<Node<T>>,
//! }
//!
//! impl<T> TreiberStack<T> {
//!     fn new() -> TreiberStack<T> {
//!         TreiberStack {
//!             head: Atomic::null()
//!         }
//!     }
//!
//!     fn push(&self, t: T) {
//!         // allocate the node via Owned
//!         let mut n = Owned::new(Node {
//!             data: t,
//!             next: Atomic::null(),
//!         });
//!
//!         // become active
//!         let guard = epoch::pin();
//!
//!         loop {
//!             // snapshot current head
//!             let head = self.head.load(Relaxed, &guard);
//!
//!             // update `next` pointer with snapshot
//!             n.next.store_shared(head, Relaxed);
//!
//!             // if snapshot is still good, link in the new node
//!             match self.head.cas_and_ref(head, n, Release, &guard) {
//!                 Ok(_) => return,
//!                 Err(owned) => n = owned,
//!             }
//!         }
//!     }
//!
//!     fn pop(&self) -> Option<T> {
//!         // become active
//!         let guard = epoch::pin();
//!
//!         loop {
//!             // take a snapshot
//!             match self.head.load(Acquire, &guard) {
//!                 // the stack is non-empty
//!                 Some(head) => {
//!                     // read through the snapshot, *safely*!
//!                     let next = head.next.load(Relaxed, &guard);
//!
//!                     // if snapshot is still good, update from `head` to `next`
//!                     if self.head.cas_shared(Some(head), next, Release) {
//!                         unsafe {
//!                             // mark the node as unlinked
//!                             guard.unlinked(head);
//!
//!                             // extract out the data from the now-unlinked node
//!                             return Some(ptr::read(&(*head).data))
//!                         }
//!                     }
//!                 }
//!
//!                 // we observed the stack empty
//!                 None => return None
//!             }
//!         }
//!     }
//! }
//! ```

// FIXME: document implementation details

mod atomic;
mod garbage;
mod global;
mod guard;
mod local;
mod participant;
mod participants;

pub use self::atomic::Atomic;
pub use self::guard::{pin, Guard};

use std::ops::Deref;
use std::ptr;
use std::mem;

pub unsafe trait StaticDrop {}

// FIXME: want to do the following, but that rules out any other impls due to coherence
// pub unsafe impl<T: 'static> StaticDrop for T {}

/// Like `Box`, represents a fully-owned heap allocation, but with two pieces of
/// data of type `S` and `E` respectively.
///
/// Ownership of the allocated data is intended to pass into a concurrent data
/// structure by use of an atomic pointer like `Atomic`. After ownership is
/// transferred, the two pieces of data play different roles:
///
/// - `S`: this data is *shared* freely between threads, and the atomic pointer
/// retains ownership of it until the memory is reclaimed by crossbeam, at which
/// point the destructor is run.
///
/// - `E`: this data is held in *escrow* by crossbeam, and will be transferred
/// back when `unlinked` is called.
pub struct Owned<S, E> {
    data: Box<Managed<S, E>>,
}

struct Managed<S, E> {
    shared: S,
    escrow: E,
}

impl<S, E> Owned<S, E> {
    /// Move the given data to a new heap allocation.
    pub fn new(shared: S, escrow: E) -> Owned<S, E> {
        Owned {
            data: Box::new(Managed {
                shared: shared,
                escrow: escrow
            })
        }
    }

    fn as_raw(&self) -> *mut Managed<S, E> {
        self.deref() as *const _ as *mut _
    }

    /// Move data out of the owned box, deallocating the box.
    pub fn into_inner(self) -> (S, E) {
        let managed = *self.data;
        (managed.shared, managed.escrow)
    }

    pub fn shared(&self) -> &S {
        &self.data.shared
    }

    pub fn shared_mut(&mut self) -> &mut S {
        &mut self.data.shared
    }

    pub fn escrow(&self) -> &E {
        &self.data.escrow
    }

    pub fn escrow_mut(&mut self) -> &mut E {
        &mut self.data.escrow
    }
}

/// Like `&'a S`: a shared reference valid for lifetime `'a`.
pub struct Shared<'a, S: 'a, E: 'a> {
    managed: &'a Managed<S, E>,
    shared: &'a S, // FIXME: get rid of this hack
}

impl<'a, S, E> Copy for Shared<'a, S, E> {}
impl<'a, S, E> Clone for Shared<'a, S, E> {
    fn clone(&self) -> Shared<'a, S, E> {
        *self
    }
}

impl<'a, S, E> Deref for Shared<'a, S, E> {
    type Target = &'a S;
    fn deref(&self) -> &&'a S {
        &&self.shared
    }
}

impl<'a, S, E> Shared<'a, S, E> {
    unsafe fn from_raw(raw: *mut Managed<S, E>) -> Option<Shared<'a, S, E>> {
        if raw == ptr::null_mut() { None }
        else {
            let managed: &'a Managed<S, E> = mem::transmute(raw);

            Some(Shared {
                managed: managed,
                shared: &managed.shared
            })
        }
    }

    unsafe fn from_owned(owned: Owned<S, E>) -> Shared<'a, S, E> {
        let managed: *mut Managed<S, E> = Box::into_raw(owned.data);
        let managed: &'a Managed<S, E> = mem::transmute(managed);
        Shared {
            managed: managed,
            shared: &managed.shared
        }
    }

    fn as_raw(&self) -> *mut Managed<S, E> {
        self.managed as *const _ as *mut _
    }

    pub fn eq_ptr(&self, other: Shared<S, E>) -> bool {
        self.as_raw() == other.as_raw()
    }
}


#[cfg(test)]
mod test {
    use std::sync::atomic::Ordering;
    use super::*;

    #[test]
    fn test_no_drop() {
        static mut DROPS: i32 = 0;
        struct Test;
        unsafe impl StaticDrop for Test {}
        impl Drop for Test {
            fn drop(&mut self) {
                unsafe {
                    DROPS += 1;
                }
            }
        }
        let g = pin();

        let x = Atomic::null();
        x.store(Some(Owned::new(Test, ())), Ordering::Relaxed);
        x.store_and_ref(Owned::new(Test, ()), Ordering::Relaxed, &g);
        let y = x.load(Ordering::Relaxed, &g);
        let z = x.cas_and_ref(y, Owned::new(Test, ()), Ordering::Relaxed, &g).ok();
        let _ = x.cas(z, Some(Owned::new(Test, ())), Ordering::Relaxed);
        x.swap(Some(Owned::new(Test, ())), Ordering::Relaxed, &g);

        unsafe {
            assert_eq!(DROPS, 0);
        }
    }
}
