#![cfg_attr(feature = "unstable", feature(test))]


use std::mem;
use std::ptr;
use std::slice;
use std::usize;
use std::iter::FromIterator;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct DenseVec<T>{
    storage: Vec<T>,
    index: Vec<usize>,
}

impl<T> DenseVec<T>{
    pub fn new() -> DenseVec<T>{
        DenseVec{
            storage: vec![],
            index: vec![],
        }
    }

    pub fn with_capacity(capacity: usize) -> Self{
        DenseVec{
            storage: Vec::with_capacity(capacity),
            index: Vec::with_capacity(capacity),
        }
    }

    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    pub fn reserve(&mut self, additional: usize){
        self.storage.reserve(additional);
        self.index.reserve(additional);
    }

    pub fn reserve_exact(&mut self, additional: usize){
        self.storage.reserve_exact(additional);
        self.index.reserve_exact(additional);
    }

    pub fn shrink_to_fit(&mut self){
        self.storage.shrink_to_fit();
        self.index.shrink_to_fit();
    }

    pub fn keys(&self) -> Keys {
        Keys{
            next: 0,
            indices: &self.index,
            len: self.len(),
        }
    }

    pub fn values(&self) -> slice::Iter<T> {
        self.storage.iter()
    }

    pub fn values_mut(&mut self) -> slice::IterMut<T> {
        self.storage.iter_mut()
    }

    pub fn iter(&self) -> Iter<T>{
        Iter{
            next: 0,
            storage: self,
            len: self.len(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T>{
        IterMut{
            next: 0,
            len: self.len(),
            storage: self,
        }
    }

    pub fn entry(&mut self, guid: usize) -> Entry<T>{
        if guid >= self.index.len() {
            Entry::Vacant(VacantEntry{storage: self, guid})
        }else{
            let idx = unsafe{ *self.index.get_unchecked(guid) };
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
        self.index.clear();
    }

    pub fn get(&self, guid: usize) -> Option<&T>{
        self.index.get(guid).and_then(|idx|
            if *idx != usize::MAX {
                Some(unsafe{ self.storage.get_unchecked(*idx) })
            }else{
                None
            }
        )
    }

    pub fn get_mut(&mut self, guid: usize) -> Option<&mut T>{
        let storage = unsafe{
            mem::transmute::<&mut Vec<T>, &mut Vec<T>>(&mut self.storage)
        };
        self.index.get(guid).and_then(move |idx|
            if *idx != usize::MAX {
                Some(unsafe{ storage.get_unchecked_mut(*idx) })
            }else{
                None
            }
        )
    }

    pub unsafe fn get_unchecked(&self, guid: usize) -> &T{
        let idx = *self.index.get_unchecked(guid);
        self.storage.get_unchecked(idx)
    }

    pub unsafe fn get_unchecked_mut(&mut self, guid: usize) -> &mut T{
        let idx = *self.index.get_unchecked(guid);
        self.storage.get_unchecked_mut(idx)
    }

    pub fn contains_key(&self, guid: usize) -> bool{
        guid < self.index.len() && unsafe{ *self.index.get_unchecked(guid) } < usize::MAX
    }

    pub fn insert(&mut self, guid: usize, t: T) -> Option<T> {
        if !self.contains_key(guid) {
            let id = self.storage.len();
            self.storage.push(t);
            if self.index.len() < guid + 1{
                self.index.resize(guid + 1, usize::MAX)
            }
            unsafe{ ptr::write(self.index.get_unchecked_mut(guid), id) };
            None
        } else {
            let idx = unsafe{ self.index.get_unchecked(guid) };
            if *idx != usize::MAX {
                Some(mem::replace(unsafe{ self.storage.get_unchecked_mut(*idx) }, t))
            }else{
                None
            }
        }
    }

    pub fn remove(&mut self, guid: usize) -> Option<T> {
        let storage = unsafe{
            mem::transmute::<&mut Vec<T>, &mut Vec<T>>(&mut self.storage)
        };
        let index = unsafe{
            mem::transmute::<&mut Vec<usize>, &mut Vec<usize>>(&mut self.index)
        };
        self.index.get_mut(guid).and_then(move |idx|
            if *idx != usize::MAX {
                let ret = storage.remove(*idx);
                for i in index.iter_mut().filter(|i| **i > *idx && **i < usize::MAX){
                    *i -= 1;
                }
                *idx = usize::MAX;
                Some(ret)
            }else{
                None
            }
        )
    }
}

use std::fmt::{self, Debug};
impl<T: Debug> Debug for DenseVec<T>{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
        f.write_str("{").and_then(|_|{
            let mut iter = self.iter();
            let err = match iter.next(){
                Some((k,v)) => f.write_str(&format!("{}: {:?}", k, v) ),
                None => Ok(()),
            };
            if let Ok(()) = err {
                iter.map(|(k,v)| f.write_str(&format!(", {}: {:?}", k, v) ))
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

impl<T: PartialEq> PartialEq for DenseVec<T>{
    fn eq(&self, other: &DenseVec<T>) -> bool{
        self.len() == other.len() && self.iter().zip(other.iter()).all(|((k1,v1), (k2,v2))| k1 == k2 && v1 == v2 )
    }
}

//TODO: impl OccupiedEntry api
pub struct OccupiedEntry<'a, T: 'a>{
    storage: &'a mut DenseVec<T>,
    guid: usize,
    idx: usize,
}

impl<'a, T:'a> OccupiedEntry<'a, T> {
    pub fn get(&self) -> &T{
        unsafe{ self.storage.storage.get_unchecked(self.idx) }
    }

    pub fn get_mut(&mut self) -> &mut T{
        unsafe{ self.storage.storage.get_unchecked_mut(self.idx) }
    }

    pub fn insert(&mut self, t: T) -> T{
        mem::replace(self.get_mut(), t)
    }

    pub fn key(&self) -> usize{
        self.guid
    }

    pub fn remove_entry(self) -> (usize, T) {
        (self.guid, self.storage.remove(self.guid).unwrap())
    }

    pub fn into_mut(self) -> &'a mut T{
        unsafe{ self.storage.storage.get_unchecked_mut(self.idx) }
    }

    pub fn remove(&mut self) -> T {
        self.storage.remove(self.guid).unwrap()
    }
}


pub struct VacantEntry<'a, T: 'a>{
    storage: &'a mut DenseVec<T>,
    guid: usize,
}

impl<'a, T:'a> VacantEntry<'a, T> {
    pub fn insert(&mut self, t: T) -> &mut T{
        self.storage.insert(self.guid, t);
        unsafe{ self.storage.get_unchecked_mut(self.guid) }
    }

    pub fn key(&self) -> usize {
        self.guid
    }
}

pub enum Entry<'a, T: 'a>{
    Occupied(OccupiedEntry<'a, T>),
    Vacant(VacantEntry<'a, T>),
}

impl<'a, T: 'a> Entry<'a, T>{
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

pub struct IntoIter<T> {
    storage: DenseVec<T>,
    next: usize,
    len: usize,
}

impl<T> IntoIterator for DenseVec<T>{
    type Item = (usize, T);
    type IntoIter = IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter{
        IntoIter{
            next: 0,
            len: self.len(),
            storage: self
        }
    }
}

impl<T> Iterator for IntoIter<T>{
    type Item = (usize, T);
    fn next(&mut self) -> Option<(usize, T)> {
        unsafe {
            while self.next < self.storage.index.len()  && *self.storage.index.get_unchecked(self.next) == usize::MAX {
                self.next += 1;
            }
            if self.next == self.storage.index.len() {
                None
            }else{
                let id = self.next;
                self.next += 1;
                self.len -= 1;
                let t = mem::replace(self.storage.storage.get_unchecked_mut(id), mem::uninitialized());
                Some((id, t))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (self.len, Some(self.len))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {
    fn len(&self) -> usize {
        self.len
    }
}

pub struct Keys<'a>{
    next: usize,
    len: usize,
    indices: &'a [usize],
}

impl<'a> Iterator for Keys<'a>{
    type Item = usize;
    fn next(&mut self) -> Option<usize>{
        unsafe {
            while self.next < self.indices.len() && *self.indices.get_unchecked(self.next) == usize::MAX {
                self.next += 1;
            }
            if self.next == self.indices.len() {
                None
            }else{
                let id = self.next;
                self.next += 1;
                self.len -= 1;
                Some(id)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (self.len, Some(self.len))
    }
}


impl<'a> ExactSizeIterator for Keys<'a> {
    fn len(&self) -> usize {
        self.len
    }
}


pub struct Iter<'a, T: 'a>{
    storage: &'a DenseVec<T>,
    next: usize,
    len: usize,
}

impl<'a, T: 'a> Iterator for Iter<'a, T>{
    type Item = (usize, &'a T);
    fn next(&mut self) -> Option<(usize, &'a T)>{
        unsafe {
            while self.next < self.storage.index.len()  && *self.storage.index.get_unchecked(self.next) == usize::MAX {
                self.next += 1;
            }
            if self.next == self.storage.index.len() {
                None
            }else{
                let id = self.next;
                self.next += 1;
                self.len -= 1;
                Some((id, self.storage.storage.get_unchecked(*self.storage.index.get_unchecked(id))))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (self.len, Some(self.len))
    }
}


impl<'a, T> ExactSizeIterator for Iter<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}

pub struct IterMut<'a, T: 'a>{
    storage: &'a mut DenseVec<T>,
    next: usize,
    len: usize,
}

impl<'a, T: 'a> Iterator for IterMut<'a, T>{
    type Item = (usize, &'a mut T);
    fn next(&mut self) -> Option<(usize, &'a mut T)>{
        unsafe {
            while self.next < self.storage.index.len()  && *self.storage.index.get_unchecked(self.next) == usize::MAX {
                self.next += 1;
            }
            if self.next == self.storage.index.len() {
                None
            }else{
                let id = self.next;
                self.next += 1;
                self.len -= 1;
                let storage = mem::transmute::<&mut Vec<T>, &mut Vec<T>>(&mut self.storage.storage);
                Some((id, storage.get_unchecked_mut(*self.storage.index.get_unchecked(id))))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>){
        (self.len, Some(self.len))
    }
}

impl<'a, T> ExactSizeIterator for IterMut<'a, T> {
    fn len(&self) -> usize {
        self.len
    }
}


impl<T> FromIterator<(usize,T)> for DenseVec<T>{
    fn from_iter<I>(iter: I) -> DenseVec<T>
    where I: IntoIterator<Item = (usize,T)>
    {
        let iter = iter.into_iter();
        let mut dense_vec = match iter.size_hint(){
            (_lower, Some(upper)) => DenseVec::with_capacity(upper),
            (lower, None) => DenseVec::with_capacity(lower),
        };
        for (id, t) in iter {
            dense_vec.insert(id, t);
        }
        dense_vec
    }
}

impl<'a, T> IntoIterator for &'a DenseVec<T>{
    type Item = (usize, &'a T);
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter{
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut DenseVec<T>{
    type Item = (usize, &'a mut T);
    type IntoIter = IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter{
        self.iter_mut()
    }
}

impl<T> Default for DenseVec<T>{
    fn default() -> DenseVec<T>{
        DenseVec::new()
    }
}

impl<T> Index<usize> for DenseVec<T>{
    type Output = T;
    fn index(&self, i: usize) -> &T {
        self.get(i).expect("no entry found for key")
    }
}

impl<T> IndexMut<usize> for DenseVec<T>{
    fn index_mut(&mut self, i: usize) -> &mut T {
        self.get_mut(i).expect("no entry found for key")
    }
}

impl<T> Extend<(usize, T)> for DenseVec<T>{
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = (usize, T)>{
        for (guid, t) in iter {
            self.insert(guid, t);
        }
    }
}

impl<'a, T: 'a + Copy> Extend<(usize, &'a T)> for DenseVec<T>{
    fn extend<I>(&mut self, iter: I) where I: IntoIterator<Item = (usize, &'a T)>{
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
        match a.entry(key.clone()) {
            Occupied(_) => panic!(),
            Vacant(mut e) => {
                assert_eq!(key, e.key());
                e.insert(value.clone());
            }
        }
        assert_eq!(a.len(), 1);
        assert_eq!(a[key], value);
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
