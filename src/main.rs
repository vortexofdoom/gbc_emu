#![feature(slice_pattern)]

use cpu::CPU;

mod cpu;
mod bootstrap;

#[macro_use]
extern crate bitflags;

fn main() {
    let mut cpu = CPU::new();
    cpu.run();
}
