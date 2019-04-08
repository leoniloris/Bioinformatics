use std::ops;

pub trait Particle {
    type Pos: Copy + ops::Add<Output = Self::Pos> + ops::Sub<Output = Self::Pos> + ops::Mul<f64, Output = Self::Pos>;
    type Eval: Copy + PartialOrd;

    fn new_random() -> Self;
    fn eval(&self) -> Self::Eval;

    fn pos(&self) -> Self::Pos;
    fn vel(&self) -> Self::Pos;
    fn best(&self) -> (Self::Pos, Self::Eval);
    fn pos_mut(&mut self) -> &mut Self::Pos;
    fn vel_mut(&mut self) -> &mut Self::Pos;
    fn best_mut(&mut self) -> &mut (Self::Pos, Self::Eval);
}

pub struct PSO<T: Particle> {
    particles: Vec<T>,
    inetia: f64,
    c_local: f64,
    c_global: f64,
    best: (T, T::Eval),
}

impl<T> PSO<T>
    where T: Particle + Ord + Copy
{
    pub fn new(particles_num: usize, inetia: f64, c_local: f64, c_global: f64) -> Self {
        let mut particles = Vec::with_capacity(particles_num);
        for _ in 0..particles_num {
            particles.push(T::new_random());
        }

        let best = Self::calc_best(&particles);

        Self {
            particles,
            inetia,
            c_local,
            c_global,
            best,
        }
    }

    fn calc_best(particles: &[T]) -> (T, T::Eval) {
        let best = particles.iter().max().unwrap();
        (*best, best.eval())
    }

    pub fn update(&mut self) {
        for p in &mut self.particles {
            let new_pos = p.pos() + p.vel();
            *p.pos_mut() = new_pos;
        }

        for mut p in &mut self.particles {
            let new_vel = p.vel() * self.inetia +
                          (p.best().0 - p.pos()) * self.c_local * Self::rand_01() +
                          (self.best.0.pos() - p.pos()) * self.c_global * Self::rand_01();
            *p.vel_mut() = new_vel;
        }

        for mut p in &mut self.particles {
            let e = p.eval();
            if e > p.best().1 {
                *p.best_mut() = (p.pos(), e);
            }
        }

        self.best = Self::calc_best(&self.particles);
    }

    pub fn best(&self) -> (T, T::Eval) {
        self.best
    }

    fn rand_01() -> f64 {
        use rand::{random, Closed01};

        let Closed01(val) = random::<Closed01<_>>();
        val
    }
