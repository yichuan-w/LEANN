/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
use crate::model::Neighbor;

/// Neighbor priority Queue based on the distance to the query node
#[derive(Debug)]
pub struct NeighborPriorityQueue {
    /// The size of the priority queue
    size: usize,

    /// The capacity of the priority queue
    capacity: usize,

    /// The current notvisited neighbor whose distance is smallest among all notvisited neighbor
    cur: usize,

    /// The neighbor collection
    data: Vec<Neighbor>,
}

impl Default for NeighborPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborPriorityQueue {
    /// Create NeighborPriorityQueue without capacity
    pub fn new() -> Self {
        Self {
            size: 0,
            capacity: 0,
            cur: 0,
            data: Vec::new(),
        }
    }

    /// Create NeighborPriorityQueue with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            size: 0,
            capacity,
            cur: 0,
            data: vec![Neighbor::default(); capacity + 1],
        }
    }

    /// Inserts item with order.
    /// The item will be dropped if queue is full / already exist in queue / it has a greater distance than the last item.
    /// The set cursor that is used to pop() the next item will be set to the lowest index of an uncheck item.
    pub fn insert(&mut self, nbr: Neighbor) {
        if self.size == self.capacity && self.get_at(self.size - 1) < &nbr {
            return;
        }

        let mut lo = 0;
        let mut hi = self.size;
        while lo < hi {
            let mid = (lo + hi) >> 1;
            if &nbr < self.get_at(mid) {
                hi = mid;
            } else if self.get_at(mid).id == nbr.id {
                // Make sure the same neighbor isn't inserted into the set
                return;
            } else {
                lo = mid + 1;
            }
        }

        if lo < self.capacity {
            self.data.copy_within(lo..self.size, lo + 1);
        }
        self.data[lo] = Neighbor::new(nbr.id, nbr.distance);
        if self.size < self.capacity {
            self.size += 1;
        }
        if lo < self.cur {
            self.cur = lo;
        }
    }

    /// Get the neighbor at index - SAFETY: index must be less than size
    fn get_at(&self, index: usize) -> &Neighbor {
        unsafe { self.data.get_unchecked(index) }
    }

    /// Get the closest and notvisited neighbor
    pub fn closest_notvisited(&mut self) -> Neighbor {
        self.data[self.cur].visited = true;
        let pre = self.cur;
        while self.cur < self.size && self.get_at(self.cur).visited {
            self.cur += 1;
        }
        self.data[pre]
    }

    /// Whether there is notvisited node or not
    pub fn has_notvisited_node(&self) -> bool {
        self.cur < self.size
    }

    /// Get the size of the NeighborPriorityQueue
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the capacity of the NeighborPriorityQueue
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sets an artificial capacity of the NeighborPriorityQueue. For benchmarking purposes only.
    pub fn set_capacity(&mut self, capacity: usize) {
        if capacity < self.data.len() {
            self.capacity = capacity;
        }
    }

    /// Reserve capacity
    pub fn reserve(&mut self, capacity: usize) {
        if capacity > self.capacity {
            self.data.resize(capacity + 1, Neighbor::default());
            self.capacity = capacity;
        }
    }

    /// Set size and cur to 0
    pub fn clear(&mut self) {
        self.size = 0;
        self.cur = 0;
    }
}

impl std::ops::Index<usize> for NeighborPriorityQueue {
    type Output = Neighbor;

    fn index(&self, i: usize) -> &Self::Output {
        &self.data[i]
    }
}

#[cfg(test)]
mod neighbor_priority_queue_test {
    use super::*;

    #[test]
    fn test_reserve_capacity() {
        let mut queue = NeighborPriorityQueue::with_capacity(10);
        assert_eq!(queue.capacity(), 10);
        queue.reserve(20);
        assert_eq!(queue.capacity(), 20);
    }

    #[test]
    fn test_insert() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        assert_eq!(queue.size(), 0);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        queue.insert(Neighbor::new(2, 0.5)); // should be ignored as the same neighbor
        assert_eq!(queue.size(), 2);
        queue.insert(Neighbor::new(3, 0.9));
        assert_eq!(queue.size(), 3);
        assert_eq!(queue[2].id, 1);
        queue.insert(Neighbor::new(4, 2.0)); // should be dropped as queue is full and distance is greater than last item
        assert_eq!(queue.size(), 3);
        assert_eq!(queue[0].id, 2); // node id in queue should be [2,3,1]
        assert_eq!(queue[1].id, 3);
        assert_eq!(queue[2].id, 1);
        println!("{:?}", queue);
    }

    #[test]
    fn test_index() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5));
        assert_eq!(queue[0].id, 2);
        assert_eq!(queue[0].distance, 0.5);
    }

    #[test]
    fn test_visit() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        queue.insert(Neighbor::new(3, 1.5)); // node id in queue should be [2,1,3]
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 2);
        assert_eq!(nbr.distance, 0.5);
        assert!(nbr.visited);
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 1);
        assert_eq!(nbr.distance, 1.0);
        assert!(nbr.visited);
        assert!(queue.has_notvisited_node());
        let nbr = queue.closest_notvisited();
        assert_eq!(nbr.id, 3);
        assert_eq!(nbr.distance, 1.5);
        assert!(nbr.visited);
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_clear_queue() {
        let mut queue = NeighborPriorityQueue::with_capacity(3);
        queue.insert(Neighbor::new(1, 1.0));
        queue.insert(Neighbor::new(2, 0.5));
        assert_eq!(queue.size(), 2);
        assert!(queue.has_notvisited_node());
        queue.clear();
        assert_eq!(queue.size(), 0);
        assert!(!queue.has_notvisited_node());
    }

    #[test]
    fn test_reserve() {
        let mut queue = NeighborPriorityQueue::new();
        queue.reserve(10);
        assert_eq!(queue.data.len(), 11);
        assert_eq!(queue.capacity, 10);
    }

    #[test]
    fn test_set_capacity() {
        let mut queue = NeighborPriorityQueue::with_capacity(10);
        queue.set_capacity(5);
        assert_eq!(queue.capacity, 5);
        assert_eq!(queue.data.len(), 11);

        queue.set_capacity(11);
        assert_eq!(queue.capacity, 5);
    }
}

