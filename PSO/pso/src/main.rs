extern crate pso;

use threadpool::ThreadPool;
use std::sync::{Arc, Barrier};
use pso::pso::StateVector;

fn main() {
    let n_particles = 200;
    let n_workers = 16;
    let pool = ThreadPool::new(n_workers);
    let s = StateVector{0 as f32, 0 as f32, 0 as f32, 0 as f32};

    let barrier = Arc::new(Barrier::new(n_workers + 1));

    for _ in 0..n_particles {
        let cloned_barrier = barrier.clone();

        pool.execute(move|| {
            // get system state
            // update each particle state
            // Compute new speed and position
            // check from stop criteria
            cloned_barrier.wait();

        });
    }

    barrier.wait();
}
