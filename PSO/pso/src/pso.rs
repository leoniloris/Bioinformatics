extern crate rand;
extern crate rayon;
use rayon::prelude::*;
use cost_function::Model;

use rand::Rng;

#[derive(Debug)]
pub struct StateVector {
    velocity: f32,
    position: f32,
    local_best: f32,
    global_best: &'static f32,
}

#[derive(Debug)]
struct Particle (Vec<StateVector>);

pub struct Swarm<m: Model> {
    particles: Vec<Particle>,
    model: m,
}

// impl Particle {
//     fn new(n_dimensions: i32, min_bound: f32, max_bound: f32) -> Particle {
//         let mut position = Vec::new();
//         let mut velocity = Vec::new();
//         let mut state_vector = Vec<StateVector>::new();
//         let mut global_best = Vec::new();

//         for _ in 0..n_dimensions {
//             let v = rand::thread_rng().gen_range(min_bound.abs(), -(max_bound - 1).abs());
//             let p = rand::thread_rng().gen_range(min_bound.abs(), (max_bound - 1).abs());

//             state_vector.push(StateVector {v, p, p, p});
//         }
//         Particle {state_vector.clone()}
//     }

//     fn update_velocity(&mut self, parent_best: &Vec<f32>) {
//         let momentum = 1f32;
//         let social_component = 1.25_f32;
//         let cognitive_component = 1.5_f32;
//         let g_rand: f32 = rand::thread_rng().gen();
//         let p_rand: f32 = rand::thread_rng().gen();

//         arr.par_iter_mut().for_each(|state| {
//             (momentum * *state.velocity) +
//             (cognitive_component * p_rand * (state.best - state.pos)) +
//             ((social_component * g_rand * ([i] - state.pos)));
//         }
//         *p -= 1);
//         for i in 0..self.vel.len() {
//             self.vel[i] = (momentum * self.vel[i]) +
//                           (cognitive_component * p_rand * (self.best[i] - self.pos[i])) +
//                           ((social_component * g_rand * (parent_best[i] - self.pos[i])));
//         }
//     }
// }
