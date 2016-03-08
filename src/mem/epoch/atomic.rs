use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::sync::atomic::{self, Ordering};

use super::{Owned, Shared, Guard, Managed, StaticDrop};

/// Like `std::sync::atomic::AtomicPtr`.
///
/// Provides atomic access to a (nullable) pointer of type `T`, interfacing with
/// the `Owned` and `Shared` types.
pub struct Atomic<S, E> {
    ptr: atomic::AtomicPtr<Managed<S, E>>,
    _marker: PhantomData<*const ()>,
}

// TODO: rationale
unsafe impl<S: Send + Sync, E: Send> Send for Atomic<S, E> {}

// TODO: rationale
unsafe impl<S: Send + Sync, E: Send> Sync for Atomic<S, E> {}

unsafe impl<S, E> StaticDrop for Atomic<S, E> {}

fn opt_shared_into_raw<S, E>(val: Option<Shared<S, E>>) -> *mut Managed<S, E> {
    val.map(|p| p.as_raw()).unwrap_or(ptr::null_mut())
}

fn opt_owned_as_raw<S, E>(val: &Option<Owned<S, E>>) -> *mut Managed<S, E> {
    val.as_ref().map(Owned::as_raw).unwrap_or(ptr::null_mut())
}

fn opt_owned_into_raw<S, E>(val: Option<Owned<S, E>>) -> *mut Managed<S, E> {
    let ptr = val.as_ref().map(Owned::as_raw).unwrap_or(ptr::null_mut());
    mem::forget(val);
    ptr
}

impl<S, E> From<Owned<S, E>> for Atomic<S, E> {
    fn from(owned: Owned<S, E>) -> Self {
        Atomic {
            ptr: atomic::AtomicPtr::new(opt_owned_into_raw(Some(owned))),
            _marker: PhantomData
        }
    }
}

impl<S: StaticDrop, E> Atomic<S, E> {
    /// Create a new, null atomic pointer.
    #[cfg(feature = "nightly")]
    pub const fn null() -> Atomic<S, E> {
        Atomic {
            ptr: atomic::AtomicPtr::new(0 as *mut _),
            _marker: PhantomData
        }
    }

    /// Create a new, null atomic pointer.
    #[cfg(not(feature = "nightly"))]
    pub fn null() -> Atomic<S, E> {
        Atomic {
            ptr: atomic::AtomicPtr::new(0 as *mut _),
            _marker: PhantomData
        }
    }

    /// Do an atomic load with the given memory ordering.
    ///
    /// In order to perform the load, we must pass in a borrow of a
    /// `Guard`. This is a way of guaranteeing that the thread has pinned the
    /// epoch for the entire lifetime `'a`. In return, you get an optional
    /// `Shared` pointer back (`None` if the `Atomic` is currently null), with
    /// lifetime tied to the guard.
    ///
    /// # Panics
    ///
    /// Panics if `ord` is `Release` or `AcqRel`.
    pub fn load<'a>(&self, ord: Ordering, _: &'a Guard) -> Option<Shared<'a, S, E>> {
        unsafe { Shared::from_raw(self.ptr.load(ord)) }
    }

    /// Do an atomic store with the given memory ordering.
    ///
    /// Transfers ownership of the given `Owned` pointer, if any. Since no
    /// lifetime information is acquired, no `Guard` value is needed.
    ///
    /// # Panics
    ///
    /// Panics if `ord` is `Acquire` or `AcqRel`.
    pub fn store(&self, val: Option<Owned<S, E>>, ord: Ordering) {
        self.ptr.store(opt_owned_into_raw(val), ord)
    }

    /// Do an atomic store with the given memory ordering, immediately yielding
    /// a shared reference to the pointer that was stored.
    ///
    /// Transfers ownership of the given `Owned` pointer, yielding a `Shared`
    /// reference to it. Since the reference is valid only for the curent epoch,
    /// it's lifetime is tied to a `Guard` value.
    ///
    /// # Panics
    ///
    /// Panics if `ord` is `Acquire` or `AcqRel`.
    pub fn store_and_ref<'a>(&self, val: Owned<S, E>, ord: Ordering, _: &'a Guard)
                             -> Shared<'a, S, E>
    {
        unsafe {
            let shared = Shared::from_owned(val);
            self.store_shared(Some(shared), ord);
            shared
        }
    }

    /// Do an atomic store of a `Shared` pointer with the given memory ordering.
    ///
    /// This operation does not require a guard, because it does not yield any
    /// new information about the lifetime of a pointer.
    ///
    /// # Panics
    ///
    /// Panics if `ord` is `Acquire` or `AcqRel`.
    pub fn store_shared(&self, val: Option<Shared<S, E>>, ord: Ordering) {
        self.ptr.store(opt_shared_into_raw(val), ord)
    }

    /// Do a compare-and-set from a `Shared` to an `Owned` pointer with the
    /// given memory ordering.
    ///
    /// As with `store`, this operation does not require a guard; it produces no new
    /// lifetime information. The `Result` indicates whether the CAS succeeded; if
    /// not, ownership of the `new` pointer is returned to the caller.
    pub fn cas(&self, old: Option<Shared<S, E>>, new: Option<Owned<S, E>>, ord: Ordering)
               -> Result<(), Option<Owned<S, E>>>
    {
        if self.ptr.compare_and_swap(opt_shared_into_raw(old),
                                     opt_owned_as_raw(&new),
                                     ord) == opt_shared_into_raw(old)
        {
            mem::forget(new);
            Ok(())
        } else {
            Err(new)
        }
    }

    /// Do a compare-and-set from a `Shared` to an `Owned` pointer with the
    /// given memory ordering, immediatley acquiring a new `Shared` reference to
    /// the previously-owned pointer if successful.
    ///
    /// This operation is analogous to `store_and_ref`.
    pub fn cas_and_ref<'a>(&self, old: Option<Shared<S, E>>, new: Owned<S, E>,
                           ord: Ordering, _: &'a Guard)
                           -> Result<Shared<'a, S, E>, Owned<S, E>>
    {
        if self.ptr.compare_and_swap(opt_shared_into_raw(old), new.as_raw(), ord)
            == opt_shared_into_raw(old)
        {
            Ok(unsafe { Shared::from_owned(new) })
        } else {
            Err(new)
        }
    }

    /// Do a compare-and-set from a `Shared` to another `Shared` pointer with
    /// the given memory ordering.
    ///
    /// The boolean return value is `true` when the CAS is successful.
    pub fn cas_shared(&self, old: Option<Shared<S, E>>, new: Option<Shared<S, E>>, ord: Ordering)
                      -> bool
    {
        self.ptr.compare_and_swap(opt_shared_into_raw(old),
                                  opt_shared_into_raw(new),
                                  ord) == opt_shared_into_raw(old)
    }

    /// Do an atomic swap with an `Owned` pointer with the given memory ordering.
    pub fn swap<'a>(&self, new: Option<Owned<S, E>>, ord: Ordering, _: &'a Guard)
                    -> Option<Shared<'a, S, E>> {
        unsafe { Shared::from_raw(self.ptr.swap(opt_owned_into_raw(new), ord)) }
    }

    /// Do an atomic swap with a `Shared` pointer with the given memory ordering.
    pub fn swap_shared<'a>(&self, new: Option<Shared<S, E>>, ord: Ordering, _: &'a Guard)
                           -> Option<Shared<'a, S, E>> {
        unsafe { Shared::from_raw(self.ptr.swap(opt_shared_into_raw(new), ord)) }
    }
}
