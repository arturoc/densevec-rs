#![cfg_attr(feature = "unstable", feature(test))]

#[cfg(feature="rayon")]
extern crate rayon;

#[cfg(feature="serialize")]
extern crate serde;

#[cfg(feature="serialize")]
#[macro_use] extern crate serde_derive;

#[cfg(feature="rayon")]
use rayon::prelude::*;

#[cfg(feature="serialize")]
use serde::{
    Serialize, Serializer, ser::SerializeStruct,
    Deserialize, Deserializer
};

#[cfg(feature="rayon")]
use rayon::iter::{FromParallelIterator, plumbing::UnindexedConsumer};

use std::mem;
use std::ptr;
use std::slice;
use std::usize;
use std::iter::{Enumerate, FromIterator};
use std::ops::{Index, IndexMut};
use std::marker::PhantomData;

pub trait Key: Copy + PartialEq{
    fn to_usize(self) -> usize;
    fn from_usize(k: usize) -> Self;
}

impl Key for usize{
    #[inline(always)]
    fn to_usize(self) -> usize{
        self
    }

    #[inline(always)]
    fn from_usize(k: usize) -> Self{
        k
    }
}

impl Key for u64{
    #[inline(always)]
    fn to_usize(self) -> usize{
        self as usize
    }

    #[inline(always)]
    fn from_usize(k: usize) -> Self{
        k as u64
    }
}

impl Key for u32{
    #[inline(always)]
    fn to_usize(self) -> usize{
        self as usize
    }

    #[inline(always)]
    fn from_usize(k: usize) -> Self{
        k as u32
    }
}

impl Key for u16{
    #[inline(always)]
    fn to_usize(self) -> usize{
        self as usize
    }

    #[inline(always)]
    fn from_usize(k: usize) -> Self{
        k as u16
    }
}

impl Key for u8{
    #[inline(always)]
    fn to_usize(self) -> usize{
        self as usize
    }

    #[inline(always)]
    fn from_usize(k: usize) -> Self{
        k as u8
    }
}

pub type DenseVec<T> = KeyedDenseVec<usize, T>;
pub type FastIndex = usize;

pub struct KeyedDenseVec<K,T>{
    storage: Vec<T>,
    packed: Vec<usize>,
    sparse: Vec<usize>,
    marker: PhantomData<K>,
}

#[cfg(feature="serialize")]
impl<K,T: Serialize> serde::Serialize for KeyedDenseVec<K,T>{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("KeyedDenseVec", 2)?;
        state.serialize_field("storage", &self.storage)?;
        state.serialize_field("packed", &self.packed)?;
        state.serialize_field("sparse", &self.sparse)?;
        state.end()
    }
}

#[cfg(feature="serialize")]
impl<'de, K, T: Deserialize<'de>> Deserialize<'de> for KeyedDenseVec<K,T> {
    fn deserialize<D>(deserializer: D) -> Result<KeyedDenseVec<K,T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        mod deserializer{
            #[derive(Deserialize)]
            pub struct KeyedDenseVec<T>{
                pub storage: Vec<T>,
                pub packed: Vec<usize>,
                pub sparse: Vec<usize>,
            }
        }

        let densevec = deserializer::KeyedDenseVec::deserialize(deserializer)?;
        Ok(KeyedDenseVec{
            storage: densevec.storage,
            packed: packed.sparse,
            sparse: densevec.sparse,
            marker: PhantomData
        })
    }
}

impl<K, T: Clone> Clone for KeyedDenseVec<K,T>{
    fn clone(&self) -> KeyedDenseVec<K,T>{
        KeyedDenseVec{
            storage: self.storage.clone(),
            packed: self.packed.clone(),
            sparse: self.sparse.clone(),
            marker: PhantomData
        }
    }
}

impl<K: Key, T> KeyedDenseVec<K,T>{
    pub fn new() -> KeyedDenseVec<K,T>{
        KeyedDenseVec{
            storage: Vec::new(),
            packed: Vec::new(),
            sparse: Vec::new(),
            marker: PhantomData,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self{
        KeyedDenseVec{
            storage: Vec::with_capacity(capacity),
            packed: Vec::with_capacity(capacity),
            sparse: Vec::with_capacity(capacity),
            marker: PhantomData,
        }
    }

    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    pub fn reserve(&mut self, additional: usize){
        self.storage.reserve(additional);
        self.packed.reserve(additional);
        self.sparse.reserve(additional);
    }

    pub fn reserve_exact(&mut self, additional: usize){
        self.storage.reserve_exact(additional);
        self.packed.reserve_exact(additional);
        self.sparse.reserve_exact(additional);
    }

    pub fn shrink_to_fit(&mut self){
        self.storage.shrink_to_fit();
        self.packed.shrink_to_fit();
        self.sparse.shrink_to_fit();
    }

    pub fn keys_unordered(&self) -> KeysUnordered<K> {
        let packed_range = self.packed.as_ptr_range();
        KeysUnordered{
            ptr: packed_range.start,
            end: packed_range.end,
            marker: PhantomData,
        }
    }

    pub fn keys(&self) -> Keys<K> {
        let range = self.sparse.as_ptr_range();
        Keys{
            ptr: range.start,
            end: range.end,
            next: 0,
            marker: PhantomData,
        }
    }

    pub fn values(&self) -> Values<T> {
        Values{ iter: self.storage.iter() }
    }

    pub fn values_mut(&mut self) -> ValuesMut<T> {
        ValuesMut{ iter: self.storage.iter_mut() }
    }

    pub fn iter_unordered(&self) -> IterUnordered<K,T>{
        let range_guid = self.packed.as_ptr_range();
        let ptr_value = self.storage.as_ptr();
        IterUnordered {
            ptr_guid: range_guid.start,
            end_guid: range_guid.end,
            ptr_value,
            marker: PhantomData
        }
    }

    pub fn iter(&self) -> Iter<K,T>{
        let range_guid = self.sparse.as_ptr_range();
        Iter {
            ptr: range_guid.start,
            end: range_guid.end,
            next: 0,
            storage: &self.storage,
            marker: PhantomData
        }
    }

    pub fn iter_unordered_mut(&mut self) -> IterUnorderedMut<K,T>{
        let range_guid = self.packed.as_ptr_range();
        let ptr_value = self.storage.as_mut_ptr();
        IterUnorderedMut {
            ptr_guid: range_guid.start,
            end_guid: range_guid.end,
            ptr_value,
            marker: PhantomData
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K,T>{
        let range_guid = self.sparse.as_ptr_range();
        IterMut {
            ptr: range_guid.start,
            end: range_guid.end,
            next: 0,
            storage: &mut self.storage,
            marker: PhantomData
        }
    }

    pub fn fast_ids_iter(&self) -> FastIter<T>{
        FastIter{
            iter: self.storage.iter().enumerate()
        }
    }

    pub fn fast_ids_iter_mut(&mut self) -> FastIterMut<T>{
        FastIterMut{
            iter: self.storage.iter_mut().enumerate()
        }
    }

    pub fn entry(&mut self, guid: K) -> Entry<K,T>{
        let u_guid = guid.to_usize();
        if u_guid >= self.sparse.len() {
            Entry::Vacant(VacantEntry{storage: self, guid})
        }else{
            let idx = unsafe{ *self.sparse.get_unchecked(u_guid) };
            if idx == usize::MAX {
                Entry::Vacant(VacantEntry{storage: self, guid})
            }else{
                Entry::Occupied(OccupiedEntry{
                    storage: self,
                    idx,
                    guid,
                })
            }
        }
    }

    pub fn len(&self) -> usize{
        self.storage.len()
    }

    pub fn is_empty(&self) -> bool{
        self.storage.is_empty()
    }

    pub fn clear(&mut self){
        self.storage.clear();
        self.sparse.clear();
    }

    pub fn get(&self, guid: K) -> Option<&T>{
        let storage = &self.storage;
        self.sparse.get(guid.to_usize()).and_then(move |idx| {
            if *idx != usize::MAX {
                Some(unsafe{ storage.get_unchecked(*idx) })
            }else{
                None
            }
        })
    }

    pub fn get_mut(&mut self, guid: K) -> Option<&mut T>{
        let storage = &mut self.storage;
        self.sparse.get(guid.to_usize()).and_then(move |idx|
            if *idx != usize::MAX {
                Some(unsafe{ storage.get_unchecked_mut(*idx) } )
            }else{
                None
            }
        )
    }

    pub unsafe fn get_unchecked(&self, guid: K) -> &T{
        let idx = *self.sparse.get_unchecked(guid.to_usize());
        self.storage.get_unchecked(idx)
    }

    pub unsafe fn get_unchecked_mut(&mut self, guid: K) -> &mut T{
        let idx = *self.sparse.get_unchecked(guid.to_usize());
        self.storage.get_unchecked_mut(idx)
    }

    pub fn get_fast(&self, idx: FastIndex) -> Option<&T>{
        self.storage.get(idx)
    }

    pub fn get_fast_mut(&mut self, idx: FastIndex) -> Option<&mut T>{
        self.storage.get_mut(idx)
    }

    pub unsafe fn get_fast_unchecked(&self, idx: FastIndex) -> &T{
        self.storage.get_unchecked(idx)
    }

    pub unsafe fn get_fast_unchecked_mut(&mut self, idx: FastIndex) -> &mut T{
        self.storage.get_unchecked_mut(idx)
    }

    pub fn contains_key(&self, guid: K) -> bool{
        let guid = guid.to_usize();
        guid < self.sparse.len() && unsafe{ *self.sparse.get_unchecked(guid) } < usize::MAX
    }

    pub fn insert(&mut self, guid: K, t: T) -> Option<T> {
        let guid = guid.to_usize();
        if self.sparse.len() < guid + 1 {
            let id = self.storage.len();
            self.storage.push(t);
            self.packed.push(guid);
            self.sparse.resize(guid + 1, usize::MAX);
            unsafe{ ptr::write(self.sparse.get_unchecked_mut(guid), id) };
            None
        }else{
            let current_idx = unsafe{ self.sparse.get_unchecked_mut(guid) };
            if *current_idx == usize::MAX {
                let id = self.storage.len();
                self.storage.push(t);
                self.packed.push(guid);
                unsafe{ ptr::write(current_idx, id) };
                None
            } else {
                // unsafe{ ptr::write(
                //     self.packed.get_unchecked_mut(*current_idx),
                //     guid
                // ) };
                Some(mem::replace(
                    unsafe{ self.storage.get_unchecked_mut(*current_idx) },
                    t
                ))
            }
        }
    }

    pub fn insert_fast_index(&mut self, guid: K, t: T) -> (Option<T>, FastIndex){
        let guid = guid.to_usize();
        if self.sparse.len() < guid + 1 {
            let id = self.storage.len();
            self.storage.push(t);
            self.packed.push(guid);
            self.sparse.resize(guid + 1, usize::MAX);
            unsafe{ ptr::write(self.sparse.get_unchecked_mut(guid), id) };
            (None, id)
        }else{
            let current_idx = unsafe{ self.sparse.get_unchecked_mut(guid) };
            if *current_idx == usize::MAX {
                let id = self.storage.len();
                self.storage.push(t);
                self.packed.push(guid);
                unsafe{ ptr::write(current_idx, id) };
                (None, id)
            } else {
                // unsafe{ ptr::write(
                //     self.packed.get_unchecked_mut(*current_idx),
                //     guid
                // ) };
                (Some(mem::replace(
                    unsafe{ self.storage.get_unchecked_mut(*current_idx) },
                    t
                )), *current_idx)
            }
        }
    }

    pub fn remove(&mut self, guid: K) -> Option<T> {
        let guid = guid.to_usize();
        let idx = self.sparse.get_mut(guid)?;
        if *idx != usize::MAX {
            let idx = mem::replace(idx, usize::MAX);
            let back = *self.packed.last().unwrap();
            if back != guid {
                unsafe{ *self.sparse.get_unchecked_mut(back) = idx };
            }
            self.packed.swap_remove(idx);
            Some(self.storage.swap_remove(idx))
        }else{
            None
        }
    }

    pub unsafe fn remove_unchecked(&mut self, guid: K) -> T {
        let guid = guid.to_usize();
        let idx = self.sparse.get_unchecked_mut(guid);
        let idx = mem::replace(idx, usize::MAX);
        let back = *self.packed.last().unwrap();
        if back != guid {
            *self.sparse.get_unchecked_mut(back) = idx;
        }
        self.packed.swap_remove(idx);
        self.storage.swap_remove(idx)
    }

    pub fn insert_key_gen(&mut self, value: T) -> K {
        let key = K::from_usize(self.sparse.len());
        let ret = self.insert(key, value);
        debug_assert!(ret.is_none());
        key
    }

    pub fn insert_fast_index_key_gen(&mut self, value: T) -> (K, FastIndex) {
        let key = K::from_usize(self.sparse.len());
        let (ret, idx) = self.insert_fast_index(key, value);
        debug_assert!(ret.is_none());
        (key, idx)
    }

    pub fn swap(&mut self, guid1: K, guid2: K){
        let i1 = self.sparse[guid1.to_usize()];
        let i2 = self.sparse[guid2.to_usize()];
        self.storage.swap(i1, i2)
    }

    pub fn fast_index_for(&self, guid: K) -> Option<FastIndex> {
        self.sparse.get(guid.to_usize()).and_then(|fast|
            (fast.to_usize() != usize::MAX).then_some(fast)
        ).copied()
    }

    pub unsafe fn fast_index_unchecked_for(&self, guid: K) -> FastIndex {
        *self.sparse.get_unchecked(guid.to_usize())
    }

    pub fn guid_from_fast_index(&self, idx: usize) -> Option<K> {
        self.packed.get(idx).map(|key| K::from_usize(*key))
    }

    pub unsafe fn unchecked_guid_from_fast_index(&self, idx: usize) -> K {
        let key = self.packed.get_unchecked(idx);
        K::from_usize(*key)
    }

    pub fn drain(&mut self) -> Drain<K,T> {
        Drain {
            drain: self.packed.drain(..).zip(self.storage.drain(..)),
            sparse: &mut self.sparse,
            marker: PhantomData
        }
    }
}

pub struct Drain<'a, K, T> {
    drain: std::iter::Zip<std::vec::Drain<'a, usize>, std::vec::Drain<'a,T>>,
    sparse: &'a mut Vec<usize>,
    marker: PhantomData<K>,
}

impl<'a, K: Key, T> Iterator for Drain<'a, K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        let (key, value) = self.drain.next()?;
        unsafe{ *self.sparse.get_unchecked_mut(key) = usize::MAX };
        Some((K::from_usize(key), value))
    }
}

pub struct Values<'a, T> {
    iter: slice::Iter<'a, T>
}

impl<'a, T> Iterator for Values<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for Values<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

pub struct ValuesMut<'a, T> {
    iter: slice::IterMut<'a, T>
}

impl<'a, T> Iterator for ValuesMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for ValuesMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[cfg(feature="rayon")]
pub struct ParIter<'a, K: Sync, T> {
    iter: rayon::iter::Enumerate<rayon::slice::Iter<'a, usize>>,
    storage: &'a [T],
    marker_key: PhantomData<K>,
}

#[cfg(feature="rayon")]
impl<'a, K, T> ParallelIterator for ParIter<'a, K, T>
where
    K: Send + Sync + Key,
    T: Send + Sync + 'a
{
    type Item = (K, &'a T);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>
    {
        let storage = self.storage;
        self.iter.filter_map(|(guid, id)| if *id == usize::MAX {
            None
        }else{
            Some((K::from_usize(guid), unsafe{ storage.get_unchecked(*id) }))
        }).drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.storage.len())
    }
}

#[cfg(feature="rayon")]
impl<K: Key + Sync, T: Sync> KeyedDenseVec<K,T>{
    pub fn par_iter(&self) -> ParIter<K, T>{
        ParIter {
            iter: self.sparse.par_iter().enumerate(),
            storage: &self.storage,
            marker_key: PhantomData,
        }
    }
}

#[cfg(feature="rayon")]
pub struct ParIterMut<'a, K: Sync, T> {
    iter: rayon::iter::Enumerate<rayon::slice::Iter<'a, usize>>,
    storage: &'a mut [T],
    marker_key: PhantomData<K>,
}

#[cfg(feature="rayon")]
impl<'a, K, T> ParallelIterator for ParIterMut<'a, K, T>
where
    K: Send + Sync + Key,
    T: Send + Sync + 'a
{
    type Item = (K, &'a mut T);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>
    {
        let storage = self.storage;
        self.iter.filter_map(|(guid, id)| if *id == usize::MAX {
            None
        }else{
            let storage = unsafe{ &mut *(storage as *const [T] as *mut [T]) };
            Some((K::from_usize(guid), unsafe{ storage.get_unchecked_mut(*id) }))
        }).drive_unindexed(consumer)
    }

    fn opt_len(&self) -> Option<usize> {
        Some(self.storage.len())
    }
}

#[cfg(feature="rayon")]
impl<K: Key + Send + Sync, T: Send + Sync> KeyedDenseVec<K,T>{
    pub fn par_iter_mut(&mut self) -> ParIterMut<K, T>{
        ParIterMut {
            iter: self.sparse.par_iter().enumerate(),
            storage: &mut self.storage,
            marker_key: PhantomData,
        }
    }
}

#[cfg(feature="rayon")]
pub struct ParValues<'a, T: Sync>{
    iter: rayon::slice::Iter<'a, T>
}

#[cfg(feature="rayon")]
impl<'a, T: Sync> ParallelIterator for ParValues<'a, T>{
    type Item = &'a T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where C: UnindexedConsumer<Self::Item>
    {
        self.iter.drive_unindexed(consumer)
    }
}

#[cfg(feature="rayon")]
impl<K, T: Sync> KeyedDenseVec<K,T>{
    pub fn par_values(&self) -> ParValues<T> {
        ParValues {
            iter: self.storage.par_iter()
        }
    }
}


#[cfg(feature="rayon")]
pub struct ParValuesMut<'a, T: Send>{
    iter: rayon::slice::IterMut<'a, T>
}

#[cfg(feature="rayon")]
impl<'a, T: Send> ParallelIterator for ParValuesMut<'a, T>{
    type Item = &'a mut T;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where C: UnindexedConsumer<Self::Item>
    {
        self.iter.drive_unindexed(consumer)
    }
}


#[cfg(feature="rayon")]
impl<K, T: Send> KeyedDenseVec<K,T>{
    pub fn par_values_mut(&mut self) -> ParValuesMut<T> {
        ParValuesMut {
            iter: self.storage.par_iter_mut()
        }
    }
}

use std::fmt::{self, Debug};
impl<K: Key + Debug, T: Debug> Debug for KeyedDenseVec<K,T>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        f.write_str("{").and_then(|_|{
            let mut iter = self.iter();
            let err = match iter.next(){
                Some((k,v)) => f.write_str(&format!("{:?}: {:?}", k, v) ),
                None => Ok(()),
            };
            if let Ok(()) = err {
                iter.map(|(k,v)| f.write_str(&format!(", {:?}: {:?}", k, v) ))
                    .find(|result| match result{
                        &Err(_) => true,
                        &Ok(()) => false,
                    }).unwrap_or(Ok(()))
            }else{
                err
            }
        }).and_then(|_| f.write_str("}"))
    }
}

impl<K: Key, T: PartialEq> PartialEq for KeyedDenseVec<K,T>{
    fn eq(&self, other: &KeyedDenseVec<K,T>) -> bool{
        self.len() == other.len()
            && self.iter()
                .zip(other.iter())
                .all(|((k1,v1), (k2,v2))| k1 == k2 && v1 == v2 )
    }
}

impl<K: Key, T: Eq> Eq for KeyedDenseVec<K,T>{}

//TODO: impl OccupiedEntry api
pub struct OccupiedEntry<'a, K: 'a, T: 'a>{
    storage: &'a mut KeyedDenseVec<K,T>,
    guid: K,
    idx: usize,
}

impl<'a, K: Key, T:'a> OccupiedEntry<'a, K, T> {
    pub fn get(&self) -> &T{
        unsafe{ self.storage.storage.get_unchecked(self.idx) }
    }

    pub fn get_mut(&mut self) -> &mut T{
        unsafe{ self.storage.storage.get_unchecked_mut(self.idx) }
    }

    pub fn insert(&mut self, t: T) -> T{
        mem::replace(self.get_mut(), t)
    }

    pub fn key(&self) -> K{
        self.guid
    }

    pub fn remove_entry(self) -> (K, T) {
        (self.guid, self.storage.remove(self.guid).unwrap())
    }

    pub fn into_mut(self) -> &'a mut T{
        unsafe{ self.storage.storage.get_unchecked_mut(self.idx) }
    }

    pub fn remove(&mut self) -> T {
        self.storage.remove(self.guid).unwrap()
    }
}


pub struct VacantEntry<'a, K: 'a, T: 'a>{
    storage: &'a mut KeyedDenseVec<K, T>,
    guid: K,
}

impl<'a, K: Key, T:'a> VacantEntry<'a, K, T> {
    pub fn insert(&mut self, t: T) -> &mut T{
        self.storage.insert(self.guid, t);
        unsafe{ self.storage.get_unchecked_mut(self.guid) }
    }

    pub fn key(&self) -> K {
        self.guid
    }
}

pub enum Entry<'a, K:Key + 'a, T: 'a>{
    Occupied(OccupiedEntry<'a, K, T>),
    Vacant(VacantEntry<'a, K, T>),
}

impl<'a, K: Key, T: 'a> Entry<'a, K, T>{
    pub fn or_insert(self, default: T) -> &'a mut T{
        match self{
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(VacantEntry{storage, guid}) => {
                storage.insert(guid, default);
                unsafe{ storage.get_unchecked_mut(guid) }
            }
        }
    }

    pub fn or_insert_with<F>(self, default: F) -> &'a mut T
        where F: FnOnce() -> T
    {
        match self{
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(VacantEntry{storage, guid}) => {
                storage.insert(guid, default());
                unsafe{ storage.get_unchecked_mut(guid) }
            }
        }
    }
}

pub struct IntoIter<K, T> {
    storage: KeyedDenseVec<K, T>,
    next: usize,
    ptr: *mut usize,
    end: *mut usize,
}

impl<K:Key,T> IntoIterator for KeyedDenseVec<K, T>{
    type Item = (K, T);
    type IntoIter = IntoIter<K,T>;
    fn into_iter(mut self) -> Self::IntoIter{
        let range_guid = self.sparse.as_mut_ptr_range();
        IntoIter {
            ptr: range_guid.start,
            end: range_guid.end,
            next: 0,
            storage: self
        }
    }
}

impl<K:Key, T> IntoIter<K,T>{
    pub fn iter(&self) -> Iter<K,T>{
        self.storage.iter()
    }
}

impl<K: Key,T> Iterator for IntoIter<K,T>{
    type Item = (K, T);
    fn next(&mut self) -> Option<(K, T)> {
        unsafe {
            while self.ptr < self.end {
                if *self.ptr != usize::MAX {
                    let guid = self.next;
                    let idx = mem::replace(&mut *self.ptr, usize::MAX);
                    let back = *self.storage.packed.last().unwrap();
                    if back != guid {
                        *self.storage.sparse.get_unchecked_mut(back) = idx;
                    }
                    self.storage.packed.swap_remove(idx);
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                    let guid = K::from_usize(guid);
                    return Some((guid, self.storage.storage.swap_remove(idx)))
                }else{
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<K:Key,T> ExactSizeIterator for IntoIter<K,T> {
    fn len(&self) -> usize {
        unsafe{ self.end.offset_from(self.ptr) as usize }
    }
}

#[derive(Clone)]
pub struct KeysUnordered<'a, K>{
    ptr: *const usize,
    end: *const usize,
    marker: PhantomData<&'a K>,
}

impl<'a, K: Key> Iterator for KeysUnordered<'a, K>{
    type Item = K;

    fn next(&mut self) -> Option<K>{
        if self.ptr == self.end {
            return None
        }else{
            unsafe {
                let next = *self.ptr;
                self.ptr = self.ptr.add(1);
                Some(K::from_usize(next))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K: Key> ExactSizeIterator for KeysUnordered<'a, K> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end.offset_from(self.ptr) as usize }
    }
}

#[derive(Clone)]
pub struct Keys<'a, K>{
    ptr: *const usize,
    end: *const usize,
    next: usize,
    marker: PhantomData<&'a K>,
}

impl<'a, K: Key> Iterator for Keys<'a, K>{
    type Item = K;
    fn next(&mut self) -> Option<K>{
        unsafe {
            while self.ptr < self.end {
                if *self.ptr != usize::MAX {
                    let next = self.next;
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                    return Some(K::from_usize(next))
                }else{
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                }
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K: Key> ExactSizeIterator for Keys<'a, K> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end.offset_from(self.ptr) as usize }
    }
}


pub struct FastIter<'a, T: 'a>{
    iter: Enumerate<slice::Iter<'a, T>>
}

impl<'a, T: 'a> Iterator for FastIter<'a, T>{
    type Item = (usize, &'a T);

    #[inline]
    fn next(&mut self) -> Option<(usize, &'a T)>{
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for FastIter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}


pub struct FastIterMut<'a, T: 'a>{
    iter: Enumerate<slice::IterMut<'a, T>>
}

impl<'a, T: 'a> Iterator for FastIterMut<'a, T>{
    type Item = (usize, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<(usize, &'a mut T)>{
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a> ExactSizeIterator for FastIterMut<'a, T>{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[derive(Clone)]
pub struct IterUnordered<'a, K: 'a, T: 'a>{
    ptr_guid: *const usize,
    end_guid: *const usize,
    ptr_value: *const T,
    marker: PhantomData<&'a K>
}

impl<'a, K: Key, T: 'a> Iterator for IterUnordered<'a, K, T>{
    type Item = (K, &'a T);
    fn next(&mut self) -> Option<(K, &'a T)>{
        unsafe {
            if self.ptr_guid < self.end_guid {
                let guid = *self.ptr_guid;
                let value = &*self.ptr_value;
                self.ptr_guid = self.ptr_guid.add(1);
                self.ptr_value = self.ptr_value.add(1);
                Some((K::from_usize(guid), value))
            }else{
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K: Key, T> ExactSizeIterator for IterUnordered<'a, K, T> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end_guid.offset_from(self.ptr_guid) as usize }
    }
}

pub struct IterUnorderedMut<'a, K: 'a, T: 'a>{
    ptr_guid: *const usize,
    end_guid: *const usize,
    ptr_value: *mut T,
    marker: PhantomData<&'a K>
}

impl<'a, K: Key, T: 'a> Iterator for IterUnorderedMut<'a, K, T>{
    type Item = (K, &'a mut T);
    fn next(&mut self) -> Option<(K, &'a mut T)>{
        unsafe {
            if self.ptr_guid < self.end_guid {
                let guid = *self.ptr_guid;
                let value = &mut *self.ptr_value;
                self.ptr_guid = self.ptr_guid.add(1);
                self.ptr_value = self.ptr_value.add(1);
                Some((K::from_usize(guid), value))
            }else{
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K:Key, T> ExactSizeIterator for IterUnorderedMut<'a, K, T> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end_guid.offset_from(self.ptr_guid) as usize }
    }
}

#[derive(Clone)]
pub struct Iter<'a, K: 'a, T: 'a>{
    next: usize,
    ptr: *const usize,
    end: *const usize,
    storage: &'a [T],
    marker: PhantomData<K>
}

impl<'a, K: Key, T: 'a> Iterator for Iter<'a, K, T>{
    type Item = (K, &'a T);
    fn next(&mut self) -> Option<(K, &'a T)>{
        unsafe {
            while self.ptr < self.end {
                if *self.ptr != usize::MAX {
                    let id = self.next;
                    let idx = *self.ptr;
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                    return Some((K::from_usize(id), self.storage.get_unchecked(idx)))
                }else{
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K:Key, T> ExactSizeIterator for Iter<'a, K, T> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end.offset_from(self.ptr) as usize }
    }
}

pub struct IterMut<'a, K: 'a, T: 'a>{
    next: usize,
    ptr: *const usize,
    end: *const usize,
    storage: &'a mut [T],
    marker: PhantomData<K>
}

impl<'a, K: Key, T: 'a> Iterator for IterMut<'a, K, T>{
    type Item = (K, &'a mut T);
    fn next(&mut self) -> Option<(K, &'a mut T)>{
        unsafe {
            while self.ptr < self.end {
                if *self.ptr != usize::MAX {
                    let id = self.next;
                    let idx = *self.ptr;
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                    let storage: &mut [T] = &mut *(self.storage as *mut _);
                    return Some((K::from_usize(id), storage.get_unchecked_mut(idx)))
                }else{
                    self.ptr = self.ptr.add(1);
                    self.next += 1;
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        let len = self.len();
        (len, Some(len))
    }
}

impl<'a, K:Key, T> ExactSizeIterator for IterMut<'a, K, T> {
    #[inline]
    fn len(&self) -> usize {
        unsafe{ self.end.offset_from(self.ptr) as usize }
    }
}


impl<K: Key, T> FromIterator<(K,T)> for KeyedDenseVec<K, T>{
    fn from_iter<I>(iter: I) -> KeyedDenseVec<K, T>
    where I: IntoIterator<Item = (K,T)>
    {
        let iter = iter.into_iter();
        let mut dense_vec = match iter.size_hint(){
            (_lower, Some(upper)) => KeyedDenseVec::with_capacity(upper),
            (lower, None) => KeyedDenseVec::with_capacity(lower),
        };
        for (id, t) in iter {
            dense_vec.insert(id, t);
        }
        dense_vec
    }
}

#[cfg(feature="rayon")]
impl<K: Key + Send, T: Send> FromParallelIterator<(K,T)> for KeyedDenseVec<K, T> {
    fn from_par_iter<I>(par_iter: I) -> Self
        where I: IntoParallelIterator<Item = (K,T)>
    {
        let par_iter = par_iter.into_par_iter();
        // par_iter.fold(|| KeyedDenseVec::new(), |mut v, (k, t)|{
        //     v.insert(k, t);
        //     v
        // }).reduce(|| KeyedDenseVec::new(), |mut v1, v2| {
        //     v1.extend(v2);
        //     v1
        // })
        par_iter.collect::<std::collections::LinkedList<_>>().into_iter().collect()
    }
}

impl<'a, K:Key, T> IntoIterator for &'a KeyedDenseVec<K,T>{
    type Item = (K, &'a T);
    type IntoIter = Iter<'a, K,T>;
    fn into_iter(self) -> Self::IntoIter{
        self.iter()
    }
}

impl<'a, K:Key, T> IntoIterator for &'a mut KeyedDenseVec<K,T>{
    type Item = (K, &'a mut T);
    type IntoIter = IterMut<'a, K,T>;
    fn into_iter(self) -> Self::IntoIter{
        self.iter_mut()
    }
}

impl<K:Key, T> Default for KeyedDenseVec<K,T>{
    fn default() -> KeyedDenseVec<K,T>{
        KeyedDenseVec::new()
    }
}

impl<K:Key, T> Index<K> for KeyedDenseVec<K,T>{
    type Output = T;
    fn index(&self, i: K) -> &T {
        self.get(i).expect("no entry found for key")
    }
}

impl<K:Key, T> IndexMut<K> for KeyedDenseVec<K,T>{
    fn index_mut(&mut self, i: K) -> &mut T {
        self.get_mut(i).expect("no entry found for key")
    }
}

impl<K:Key,T> Extend<(K, T)> for KeyedDenseVec<K,T>{
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = (K, T)>{
        for (guid, t) in iter {
            self.insert(guid, t);
        }
    }
}

impl<'a, K:Key, T: 'a + Copy> Extend<(K, &'a T)> for KeyedDenseVec<K,T>{
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = (K, &'a T)>{
        for (guid, t) in iter {
            self.insert(guid, *t);
        }
    }
}

#[cfg(test)]
mod test_map {
    use super::DenseVec;
    use super::Entry::{Occupied, Vacant};

    #[test]
    fn count(){
        let mut v = DenseVec::new();
        for i in 0..100 {
            v.insert(i, i);
        }

        for i in 0..100 {
            v.insert(i + 100, i);
        }

        assert_eq!(v.values().count(), 200);
    }

    #[test]
    fn test_zero_capacities() {
        type HM = DenseVec<i32>;

        let m = HM::new();
        assert_eq!(m.capacity(), 0);

        let m = HM::default();
        assert_eq!(m.capacity(), 0);

        let m = HM::with_capacity(0);
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.insert(1, 1);
        m.insert(2, 2);
        m.remove(1);
        m.remove(2);
        m.shrink_to_fit();
        assert_eq!(m.capacity(), 0);

        let mut m = HM::new();
        m.reserve(0);
        assert_eq!(m.capacity(), 0);
    }

    #[test]
    fn test_create_capacity_zero() {
        let mut m = DenseVec::with_capacity(0);

        assert!(m.insert(1, 1).is_none());

        assert!(m.contains_key(1));
        assert!(!m.contains_key(0));
    }

    #[test]
    fn test_insert() {
        let mut m = DenseVec::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        assert_eq!(*m.get(1).unwrap(), 2);
        assert_eq!(*m.get(2).unwrap(), 4);
    }

    #[test]
    fn test_clone() {
        let mut m = DenseVec::new();
        assert_eq!(m.len(), 0);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(m.len(), 1);
        assert!(m.insert(2, 4).is_none());
        assert_eq!(m.len(), 2);
        let m2 = m.clone();
        assert_eq!(*m2.get(1).unwrap(), 2);
        assert_eq!(*m2.get(2).unwrap(), 4);
        assert_eq!(m2.len(), 2);
    }


    #[test]
    fn test_empty_remove() {
        let mut m: DenseVec<bool> = DenseVec::new();
        assert_eq!(m.remove(0), None);
    }

    #[test]
    fn test_empty_entry() {
        let mut m: DenseVec<bool> = DenseVec::new();
        match m.entry(0) {
            Occupied(_) => panic!(),
            Vacant(_) => {}
        }
        assert!(*m.entry(0).or_insert(true));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_empty_iter() {
        let mut m: DenseVec<bool> = DenseVec::new();
        assert_eq!(m.keys().next(), None);
        assert_eq!(m.values().next(), None);
        assert_eq!(m.values_mut().next(), None);
        assert_eq!(m.iter().next(), None);
        assert_eq!(m.iter_mut().next(), None);
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
        assert_eq!(m.into_iter().next(), None);
    }

    #[test]
    fn test_lots_of_insertions() {
        let mut m = DenseVec::new();

        // Try this a few times to make sure we never screw up the densevec's
        // internal state.
        for _ in 0..10 {
            assert!(m.is_empty());

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());

                for j in 1..i + 1 {
                    let r = m.get(j);
                    assert_eq!(r, Some(&j));
                }

                for j in i + 1..1001 {
                    let r = m.get(j);
                    assert_eq!(r, None);
                }
            }

            for i in 1001..2001 {
                assert!(!m.contains_key(i));
            }

            // remove forwards
            for i in 1..1001 {
                assert!(m.remove(i).is_some());

                for j in 1..i + 1 {
                    assert!(!m.contains_key(j));
                }

                for j in i + 1..1001 {
                    assert!(m.contains_key(j));
                }
            }

            for i in 1..1001 {
                assert!(!m.contains_key(i));
            }

            for i in 1..1001 {
                assert!(m.insert(i, i).is_none());
            }

            // remove backwards
            for i in (1..1001).rev() {
                assert!(m.remove(i).is_some());

                for j in i..1001 {
                    assert!(!m.contains_key(j));
                }

                for j in 1..i {
                    assert!(m.contains_key(j));
                }
            }
        }
    }

    #[test]
    fn test_find_mut() {
        let mut m = DenseVec::new();
        assert!(m.insert(1, 12).is_none());
        assert!(m.insert(2, 8).is_none());
        assert!(m.insert(5, 14).is_none());
        let new = 100;
        match m.get_mut(5) {
            None => panic!(),
            Some(x) => *x = new,
        }
        assert_eq!(m.get(5), Some(&new));
    }

    #[test]
    fn test_insert_overwrite() {
        let mut m = DenseVec::new();
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(1).unwrap(), 2);
        assert!(!m.insert(1, 3).is_none());
        assert_eq!(*m.get(1).unwrap(), 3);
    }

    #[test]
    fn test_insert_conflicts() {
        let mut m = DenseVec::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(m.insert(5, 3).is_none());
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(9).unwrap(), 4);
        assert_eq!(*m.get(5).unwrap(), 3);
        assert_eq!(*m.get(1).unwrap(), 2);
    }

    #[test]
    fn test_conflict_remove() {
        let mut m = DenseVec::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert_eq!(*m.get(1).unwrap(), 2);
        assert!(m.insert(5, 3).is_none());
        assert_eq!(*m.get(1).unwrap(), 2);
        assert_eq!(*m.get(5).unwrap(), 3);
        assert!(m.insert(9, 4).is_none());
        assert_eq!(*m.get(1).unwrap(), 2);
        assert_eq!(*m.get(5).unwrap(), 3);
        assert_eq!(*m.get(9).unwrap(), 4);
        assert!(m.remove(1).is_some());
        assert_eq!(*m.get(9).unwrap(), 4);
        assert_eq!(*m.get(5).unwrap(), 3);
    }

    #[test]
    fn test_is_empty() {
        let mut m = DenseVec::with_capacity(4);
        assert!(m.insert(1, 2).is_none());
        assert!(!m.is_empty());
        assert!(m.remove(1).is_some());
        assert!(m.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut m = DenseVec::new();
        m.insert(1, 2);
        assert_eq!(m.remove(1), Some(2));
        assert_eq!(m.remove(1), None);
    }

    // #[test]
    // fn test_remove_entry() {
    //     let mut m = DenseVec::new();
    //     m.insert(1, 2);
    //     assert_eq!(m.remove_entry(1), Some((1, 2)));
    //     assert_eq!(m.remove(1), None);
    // }

    #[test]
    fn test_iterate() {
        let mut m = DenseVec::with_capacity(4);
        for i in 0..32 {
            assert!(m.insert(i, i*2).is_none());
        }
        assert_eq!(m.len(), 32);

        let mut observed: u32 = 0;

        for (k, v) in &m {
            assert_eq!(*v, k * 2);
            observed |= 1 << k;
        }
        assert_eq!(observed, 0xFFFF_FFFF);
    }

    #[test]
    fn test_keys() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: DenseVec<_> = vec.into_iter().collect();
        let keys: Vec<_> = map.keys().collect();
        println!("{:?}", keys);
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&1));
        assert!(keys.contains(&2));
        assert!(keys.contains(&3));
    }

    #[test]
    fn test_values() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c')];
        let map: DenseVec<_> = vec.into_iter().collect();
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&'a'));
        assert!(values.contains(&'b'));
        assert!(values.contains(&'c'));
    }

    #[test]
    fn test_values_mut() {
        let vec = vec![(1, 1), (2, 2), (3, 3)];
        let mut map: DenseVec<_> = vec.into_iter().collect();
        for value in map.values_mut() {
            *value = (*value) * 2
        }
        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&2));
        assert!(values.contains(&4));
        assert!(values.contains(&6));
    }

    #[test]
    fn test_into_iter() {
        let vec = vec![(1, 'a'), (2, 'b'), (3, 'c'), (5, 'e')];
        let map: DenseVec<_> = vec.clone().into_iter().collect();
        let vec2: Vec<_> = map.into_iter().collect();
        assert_eq!(vec, vec2);
    }

    #[test]
    fn test_find() {
        let mut m = DenseVec::new();
        assert!(m.get(1).is_none());
        m.insert(1, 2);
        match m.get(1) {
            None => panic!(),
            Some(v) => assert_eq!(*v, 2),
        }
    }

    #[test]
    fn test_eq() {
        let mut m1 = DenseVec::new();
        m1.insert(1, 2);
        m1.insert(2, 3);
        m1.insert(3, 4);

        let mut m2 = DenseVec::new();
        m2.insert(1, 2);
        m2.insert(2, 3);

        assert!(m1 != m2);

        m2.insert(3, 4);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_show() {
        let mut map = DenseVec::new();
        let empty: DenseVec<i32> = DenseVec::new();

        map.insert(1, 2);
        map.insert(3, 4);

        let map_str = format!("{:?}", map);
        println!("{}", map_str);
        assert!(map_str == "{1: 2, 3: 4}" ||
                map_str == "{3: 4, 1: 2}");
        assert_eq!(format!("{:?}", empty), "{}");
    }

    #[test]
    fn test_expand() {
        let mut m = DenseVec::new();

        assert_eq!(m.len(), 0);
        assert!(m.is_empty());

        let mut i = 0;
        let old_raw_cap = m.capacity();
        while old_raw_cap == m.capacity() {
            m.insert(i, i);
            i += 1;
        }

        assert_eq!(m.len(), i);
        assert!(!m.is_empty());
    }



    #[test]
    fn test_reserve_shrink_to_fit() {
        let mut m = DenseVec::new();
        m.insert(0, 0);
        m.remove(0);
        assert!(m.capacity() >= m.len());
        for i in 0..128 {
            m.insert(i, i);
        }
        m.reserve(256);

        let usable_cap = m.capacity();
        for i in 128..(128 + 256) {
            m.insert(i, i);
            assert_eq!(m.capacity(), usable_cap);
        }

        for i in 100..(128 + 256) {
            assert_eq!(m.remove(i), Some(i));
        }
        m.shrink_to_fit();

        assert_eq!(m.len(), 100);
        assert!(!m.is_empty());
        assert!(m.capacity() >= m.len());

        for i in 0..100 {
            assert_eq!(m.remove(i), Some(i));
        }
        m.shrink_to_fit();
        m.insert(0, 0);

        assert_eq!(m.len(), 1);
        assert!(m.capacity() >= m.len());
        assert_eq!(m.remove(0), Some(0));
    }

    #[test]
    fn test_from_iter() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: DenseVec<_> = xs.iter().cloned().collect();

        for &(k, v) in &xs {
            assert_eq!(map.get(k), Some(&v));
        }
    }

    #[test]
    fn test_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: DenseVec<_> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let map: DenseVec<_> = xs.iter().cloned().collect();

        let mut iter = map.iter();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_mut_size_hint() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: DenseVec<_> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_iter_mut_len() {
        let xs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];

        let mut map: DenseVec<_> = xs.iter().cloned().collect();

        let mut iter = map.iter_mut();

        for _ in iter.by_ref().take(3) {}

        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_index() {
        let mut map = DenseVec::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        assert_eq!(map[2], 1);
    }

    #[test]
    #[should_panic]
    fn test_index_nonexistent() {
        let mut map = DenseVec::new();

        map.insert(1, 2);
        map.insert(2, 1);
        map.insert(3, 4);

        map[4];
    }

    #[test]
    fn test_entry() {
        let xs = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)];

        let mut map: DenseVec<_> = xs.iter().cloned().collect();

        // Existing key (insert)
        match map.entry(1) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                assert_eq!(view.get(), &10);
                assert_eq!(view.insert(100), 10);
            }
        }
        assert_eq!(map.get(1).unwrap(), &100);
        assert_eq!(map.len(), 6);


        // Existing key (update)
        match map.entry(2) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => {
                let v = view.get_mut();
                let new_v = (*v) * 10;
                *v = new_v;
            }
        }
        assert_eq!(map.get(2).unwrap(), &200);
        assert_eq!(map.len(), 6);

        // Existing key (take)
        match map.entry(3) {
            Vacant(_) => unreachable!(),
            Occupied(mut view) => { //TODO: we shouldn't need mut here
                assert_eq!(view.remove(), 30);
            }
        }
        assert_eq!(map.get(3), None);
        assert_eq!(map.len(), 5);


        // Inexistent key (insert)
        match map.entry(10) {
            Occupied(_) => unreachable!(),
            Vacant(mut view) => { //TODO: we shouldn't need mut here
                assert_eq!(*view.insert(1000), 1000);
            }
        }
        assert_eq!(map.get(10).unwrap(), &1000);
        assert_eq!(map.len(), 6);
    }

    // // #[test]
    // // fn test_entry_take_doesnt_corrupt() {
    // //     #![allow(deprecated)] //rand
    // //     // Test for #19292
    // //     fn check(m: &DenseVec<()>) {
    // //         for k in m.keys() {
    // //             assert!(m.contains_key(k),
    // //                     "{} is in keys() but not in the map?", k);
    // //         }
    // //     }

    // //     let mut m = DenseVec::new();
    // //     let mut rng = thread_rng();

    // //     // Populate the map with some items.
    // //     for _ in 0..50 {
    // //         let x = rng.gen_range(-10, 10);
    // //         m.insert(x, ());
    // //     }

    // //     for i in 0..1000 {
    // //         let x = rng.gen_range(-10, 10);
    // //         match m.entry(x) {
    // //             Vacant(_) => {}
    // //             Occupied(e) => {
    // //                 println!("{}: remove {}", i, x);
    // //                 e.remove();
    // //             }
    // //         }

    // //         check(&m);
    // //     }
    // // }

    #[test]
    fn test_extend_ref() {
        let mut a = DenseVec::new();
        a.insert(1, "one");
        let mut b = DenseVec::new();
        b.insert(2, "two");
        b.insert(3, "three");

        a.extend(&b);

        assert_eq!(a.len(), 3);
        assert_eq!(a[1], "one");
        assert_eq!(a[2], "two");
        assert_eq!(a[3], "three");
    }

    #[test]
    fn test_capacity_not_less_than_len() {
        let mut a = DenseVec::new();
        let mut item = 0;

        for _ in 0..116 {
            a.insert(item, 0);
            item += 1;
        }

        assert!(a.capacity() > a.len());

        let free = a.capacity() - a.len();
        for _ in 0..free {
            a.insert(item, 0);
            item += 1;
        }

        assert_eq!(a.len(), a.capacity());

        // Insert at capacity should cause allocation.
        a.insert(item, 0);
        assert!(a.capacity() > a.len());
    }

    #[test]
    fn test_occupied_entry_key() {
        let mut a = DenseVec::new();
        let key = 1;
        let value = "value goes here";
        assert!(a.is_empty());
        a.insert(key.clone(), value.clone());
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);

        match a.entry(key.clone()) {
            Vacant(_) => panic!(),
            Occupied(e) => assert_eq!(key, e.key()),
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_vacant_entry_key() {
        let mut a = DenseVec::new();
        let key = 1;
        let value = "value goes here";

        assert!(a.is_empty());
        match a.entry(key) {
            Occupied(_) => panic!(),
            Vacant(mut e) => {
                assert_eq!(key, e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
    }

    #[test]
    fn test_key_gen() {
        let mut a = DenseVec::new();
        let keys = (0..1000)
            .map(|_| a.insert_key_gen(0))
            .collect::<Vec<_>>();
        for k in keys {
            assert!(a.get(k).is_some());
        }
    }

    #[cfg(feature="rayon")]
    #[test]
    fn par_iter() {
        use rayon::iter::ParallelIterator;

        let xs = vec![(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)];
        let map: DenseVec<_> = xs.clone().into_iter().collect();
        let vec = map.par_iter().map(|(guid, v)| (guid, *v)).collect::<Vec<_>>();
        assert_eq!(xs, vec)
    }

    // #[test]
    // fn test_retain() {
    //     let mut map: DenseVec<i32> = (0..100).map(|x|(x, x*10)).collect();

    //     map.retain(|&k, _| k % 2 == 0);
    //     assert_eq!(map.len(), 50);
    //     assert_eq!(map[&2], 20);
    //     assert_eq!(map[&4], 40);
    //     assert_eq!(map[&6], 60);
    // }

    // #[test]
    // fn test_adaptive() {
    //     const TEST_LEN: usize = 5000;
    //     // by cloning we get maps with the same hasher seed
    //     let mut first = DenseVec::new();
    //     let mut second = first.clone();
    //     first.extend((0..TEST_LEN).map(|i| (i, i)));
    //     second.extend((TEST_LEN..TEST_LEN * 2).map(|i| (i, i)));

    //     for (k, &v) in &second {
    //         let prev_cap = first.capacity();
    //         let expect_grow = first.len() == prev_cap;
    //         first.insert(k, v);
    //         if !expect_grow && first.capacity() != prev_cap {
    //             return;
    //         }
    //     }
    //     panic!("Adaptive early resize failed");
    // }

    // #[test]
    // fn test_placement_in() {
    //     let mut map = DenseVec::new();
    //     map.extend((0..10).map(|i| (i, i)));

    //     map.entry(100) <- 100;
    //     assert_eq!(map[100], 100);

    //     map.entry(0) <- 10;
    //     assert_eq!(map[0], 10);

    //     assert_eq!(map.len(), 11);
    // }

    // #[test]
    // fn test_placement_panic() {
    //     let mut map = DenseVec::new();
    //     map.extend((0..10).map(|i| (i, i)));

    //     fn mkpanic() -> usize { panic!() }

    //     // modify existing key
    //     // when panic happens, previous key is removed.
    //     let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| { map.entry(0) <- mkpanic(); }));
    //     assert_eq!(map.len(), 9);
    //     assert!(!map.contains_key(&0));

    //     // add new key
    //     let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| { map.entry(100) <- mkpanic(); }));
    //     assert_eq!(map.len(), 9);
    //     assert!(!map.contains_key(&100));
    // }

    // #[test]
    // fn test_placement_drop() {
    //     // correctly drop
    //     struct TestV<'a>(&'a mut bool);
    //     impl<'a> Drop for TestV<'a> {
    //         fn drop(&mut self) {
    //             if !*self.0 { panic!("value double drop!"); } // no double drop
    //             *self.0 = false;
    //         }
    //     }

    //     fn makepanic<'a>() -> TestV<'a> { panic!() }

    //     let mut can_drop = true;
    //     let mut hm = DenseVec::new();
    //     hm.insert(0, TestV(&mut can_drop));
    //     let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| { hm.entry(0) <- makepanic(); }));
    //     assert_eq!(hm.len(), 0);
    // }

    // #[test]
    // fn test_try_reserve() {

    //     let mut empty_bytes: DenseVec<u8> = DenseVec::new();

    //     const MAX_USIZE: usize = usize::MAX;

    //     // HashMap and RawTables use complicated size calculations
    //     // hashes_size is sizeof(HashUint) * capacity;
    //     // pairs_size is sizeof((K. V)) * capacity;
    //     // alignment_hashes_size is 8
    //     // alignment_pairs size is 4
    //     let size_of_multiplier = (size_of::<usize>() + size_of::<(u8, u8)>()).next_power_of_two();
    //     // The following formula is used to calculate the new capacity
    //     let max_no_ovf = ((MAX_USIZE / 11) * 10) / size_of_multiplier - 1;

    //     if let Err(CapacityOverflow) = empty_bytes.try_reserve(MAX_USIZE) {
    //     } else { panic!("usize::MAX should trigger an overflow!"); }

    //     if size_of::<usize>() < 8 {
    //         if let Err(CapacityOverflow) = empty_bytes.try_reserve(max_no_ovf) {
    //         } else { panic!("isize::MAX + 1 should trigger a CapacityOverflow!") }
    //     } else {
    //         if let Err(AllocErr(_)) = empty_bytes.try_reserve(max_no_ovf) {
    //         } else { panic!("isize::MAX + 1 should trigger an OOM!") }
    //     }
    // }

}

#[cfg(all(test, feature = "unstable"))]
mod benches{
    extern crate test;
    use self::test::Bencher;
    use super::DenseVec;
    use std::collections::HashMap;


    #[bench]
    fn bench_densevec(b: &mut Bencher) {
        b.iter(||{
            let mut m = DenseVec::new();

            // Try this a few times to make sure we never screw up the densevec's
            // internal state.
            for _ in 0..10 {
                assert!(m.is_empty());

                for i in 1..1001 {
                    assert!(m.insert(i, i).is_none());

                    for j in 1..i + 1 {
                        let r = m.get(j);
                        assert_eq!(r, Some(&j));
                    }

                    for j in i + 1..1001 {
                        let r = m.get(j);
                        assert_eq!(r, None);
                    }
                }

                for i in 1001..2001 {
                    assert!(!m.contains_key(i));
                }

                // remove forwards
                for i in 1..1001 {
                    assert!(m.remove(i).is_some());

                    for j in 1..i + 1 {
                        assert!(!m.contains_key(j));
                    }

                    for j in i + 1..1001 {
                        assert!(m.contains_key(j));
                    }
                }

                for i in 1..1001 {
                    assert!(!m.contains_key(i));
                }

                for i in 1..1001 {
                    assert!(m.insert(i, i).is_none());
                }

                // remove backwards
                for i in (1..1001).rev() {
                    assert!(m.remove(i).is_some());

                    for j in i..1001 {
                        assert!(!m.contains_key(j));
                    }

                    for j in 1..i {
                        assert!(m.contains_key(j));
                    }
                }
            }
        });
    }



    #[bench]
    fn bench_hashmap(b: &mut Bencher) {
        b.iter(||{
            let mut m = HashMap::new();

            // Try this a few times to make sure we never screw up the densevec's
            // internal state.
            for _ in 0..10 {
                assert!(m.is_empty());

                for i in 1..1001 {
                    assert!(m.insert(i, i).is_none());

                    for j in 1..i + 1 {
                        let r = m.get(&j);
                        assert_eq!(r, Some(&j));
                    }

                    for j in i + 1..1001 {
                        let r = m.get(&j);
                        assert_eq!(r, None);
                    }
                }

                for i in 1001..2001 {
                    assert!(!m.contains_key(&i));
                }

                // remove forwards
                for i in 1..1001 {
                    assert!(m.remove(&i).is_some());

                    for j in 1..i + 1 {
                        assert!(!m.contains_key(&j));
                    }

                    for j in i + 1..1001 {
                        assert!(m.contains_key(&j));
                    }
                }

                for i in 1..1001 {
                    assert!(!m.contains_key(&i));
                }

                for i in 1..1001 {
                    assert!(m.insert(i, i).is_none());
                }

                // remove backwards
                for i in (1..1001).rev() {
                    assert!(m.remove(&i).is_some());

                    for j in i..1001 {
                        assert!(!m.contains_key(&j));
                    }

                    for j in 1..i {
                        assert!(m.contains_key(&j));
                    }
                }
            }
        });
    }
}
