/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 */
#![warn(missing_debug_implementations, missing_docs)]

//! Aligned allocator

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::common::{ANNError, ANNResult};

#[derive(Debug)]
/// Query scratch data structures
pub struct ConcurrentQueue<T> {
    q: Mutex<VecDeque<T>>,
    c: Mutex<bool>,
    push_cv: Condvar,
}

impl Default for ConcurrentQueue<usize> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ConcurrentQueue<T> {
    /// Create a concurrent queue
    pub fn new() -> Self {
        Self {
            q: Mutex::new(VecDeque::new()),
            c: Mutex::new(false),
            push_cv: Condvar::new(),
        }
    }

    /// Block the current thread until it is able to acquire the mutex
    pub fn reserve(&self, size: usize) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        guard.reserve(size);
        Ok(())
    }

    /// queue stats
    pub fn size(&self) -> ANNResult<usize> {
        let guard = lock(&self.q)?;

        Ok(guard.len())
    }

    /// empty the queue
    pub fn is_empty(&self) -> ANNResult<bool> {
        Ok(self.size()? == 0)
    }

    /// push back
    pub fn push(&self, new_val: T) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        self.push_internal(&mut guard, new_val);
        self.push_cv.notify_all();
        Ok(())
    }

    /// push back
    fn push_internal(&self, guard: &mut MutexGuard<VecDeque<T>>, new_val: T) {
        guard.push_back(new_val);
    }

    /// insert into queue
    pub fn insert<I>(&self, iter: I) -> ANNResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        let mut guard = lock(&self.q)?;
        for item in iter {
            self.push_internal(&mut guard, item);
        }

        self.push_cv.notify_all();
        Ok(())
    }

    /// pop front
    pub fn pop(&self) -> ANNResult<Option<T>> {
        let mut guard = lock(&self.q)?;
        Ok(guard.pop_front())
    }

    /// Empty - is this necessary?
    pub fn empty_queue(&self) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        while !guard.is_empty() {
            let _ = guard.pop_front();
        }
        Ok(())
    }

    /// register for push notifications
    pub fn wait_for_push_notify(&self, wait_time: Duration) -> ANNResult<()> {
        let guard_lock = lock(&self.c)?;
        let _ = self
            .push_cv
            .wait_timeout(guard_lock, wait_time)
            .map_err(|err| {
                ANNError::log_lock_poison_error(format!(
                    "ConcurrentQueue Lock is poisoned, err={}",
                    err
                ))
            })?;
        Ok(())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> ANNResult<MutexGuard<T>> {
    let guard = mutex.lock().map_err(|err| {
        ANNError::log_lock_poison_error(format!("ConcurrentQueue lock is poisoned, err={}", err))
    })?;
    Ok(guard)
}

/// A thread-safe queue that holds instances of `T`.
/// Each instance is stored in a `Box` to keep the size of the queue node constant.
#[derive(Debug)]
pub struct ArcConcurrentBoxedQueue<T> {
    internal_queue: Arc<ConcurrentQueue<Box<T>>>,
}

impl<T> ArcConcurrentBoxedQueue<T> {
    /// Create a new `ArcConcurrentBoxedQueue`.
    pub fn new() -> Self {
        Self {
            internal_queue: Arc::new(ConcurrentQueue::new()),
        }
    }
}

impl<T> Default for ArcConcurrentBoxedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for ArcConcurrentBoxedQueue<T> {
    /// Create a new `ArcConcurrentBoxedQueue` that shares the same internal queue
    /// with the existing one. This allows multiple `ArcConcurrentBoxedQueue` to
    /// operate on the same underlying queue.
    fn clone(&self) -> Self {
        Self {
            internal_queue: Arc::clone(&self.internal_queue),
        }
    }
}

/// Deref to the ConcurrentQueue.
impl<T> Deref for ArcConcurrentBoxedQueue<T> {
    type Target = ConcurrentQueue<Box<T>>;

    fn deref(&self) -> &Self::Target {
        &self.internal_queue
    }
}

#[cfg(test)]
mod tests {
    use crate::model::ConcurrentQueue;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_push_pop() {
        let queue = ConcurrentQueue::<i32>::new();

        queue.push(1).unwrap();
        queue.push(2).unwrap();
        queue.push(3).unwrap();

        assert_eq!(queue.pop().unwrap(), Some(1));
        assert_eq!(queue.pop().unwrap(), Some(2));
        assert_eq!(queue.pop().unwrap(), Some(3));
        assert_eq!(queue.pop().unwrap(), None);
    }

    #[test]
    fn test_size_empty() {
        let queue = ConcurrentQueue::new();

        assert_eq!(queue.size().unwrap(), 0);
        assert!(queue.is_empty().unwrap());

        queue.push(1).unwrap();
        queue.push(2).unwrap();

        assert_eq!(queue.size().unwrap(), 2);
        assert!(!queue.is_empty().unwrap());

        queue.pop().unwrap();
        queue.pop().unwrap();

        assert_eq!(queue.size().unwrap(), 0);
        assert!(queue.is_empty().unwrap());
    }

    #[test]
    fn test_insert() {
        let queue = ConcurrentQueue::new();

        let data = vec![1, 2, 3];
        queue.insert(data.into_iter()).unwrap();

        assert_eq!(queue.pop().unwrap(), Some(1));
        assert_eq!(queue.pop().unwrap(), Some(2));
        assert_eq!(queue.pop().unwrap(), Some(3));
        assert_eq!(queue.pop().unwrap(), None);
    }

    #[test]
    fn test_notifications() {
        let queue = Arc::new(ConcurrentQueue::new());
        let queue_clone = Arc::clone(&queue);

        let producer = thread::spawn(move || {
            for i in 0..3 {
                thread::sleep(Duration::from_millis(50));
                queue_clone.push(i).unwrap();
            }
        });

        let consumer = thread::spawn(move || {
            let mut values = vec![];

            for _ in 0..3 {
                let mut val = -1;
                while val == -1 {
                    queue
                        .wait_for_push_notify(Duration::from_millis(10))
                        .unwrap();
                    val = queue.pop().unwrap().unwrap_or(-1);
                }

                values.push(val);
            }

            values
        });

        producer.join().unwrap();
        let consumer_results = consumer.join().unwrap();

        assert_eq!(consumer_results, vec![0, 1, 2]);
    }

    #[test]
    fn test_multithreaded_push_pop() {
        let queue = Arc::new(ConcurrentQueue::new());
        let queue_clone = Arc::clone(&queue);

        let producer = thread::spawn(move || {
            for i in 0..10 {
                queue_clone.push(i).unwrap();
                thread::sleep(Duration::from_millis(50));
            }
        });

        let consumer = thread::spawn(move || {
            let mut values = vec![];

            for _ in 0..10 {
                let mut val = -1;
                while val == -1 {
                    val = queue.pop().unwrap().unwrap_or(-1);
                    thread::sleep(Duration::from_millis(10));
                }

                values.push(val);
            }

            values
        });

        producer.join().unwrap();
        let consumer_results = consumer.join().unwrap();

        assert_eq!(consumer_results, (0..10).collect::<Vec<_>>());
    }

    /// This is a single value test. It avoids the unlimited wait until the collectin got empty on the previous test.
    /// It will make sure the signal mutex is matching the waiting mutex.  
    #[test]
    fn test_wait_for_push_notify() {
        let queue = Arc::new(ConcurrentQueue::<usize>::new());
        let queue_clone = Arc::clone(&queue);

        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100));
            queue_clone.push(1).unwrap();
        });

        let consumer = thread::spawn(move || {
            queue
                .wait_for_push_notify(Duration::from_millis(200))
                .unwrap();
            assert_eq!(queue.pop().unwrap(), Some(1));
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }
}
