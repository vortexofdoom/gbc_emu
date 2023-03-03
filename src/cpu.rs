use core::slice::SlicePattern;
use std::result;

use crate::bootstrap::BOOTSTRAP_ROM;

#[derive(Debug)]
pub enum Operand {
    D8,
    A8,
    A16,
    A,
    B,
    C,
    D,
    E,
    H,
    L,
    AF,
    BC,
    DE,
    HL,
    SP,
    PC,
}

bitflags! {
    #[derive(Default)]
    pub struct Flags: u8 {
        const ZERO          = 0b10000000;
        const SUBTRACTION   = 0b01000000;
        const HALF_CARRY    = 0b00100000;
        const CARRY         = 0b00010000;
    }
}

impl Flags {
    pub fn z(mut self, cond: bool) -> Self {
        self.set(Flags::ZERO, cond);
        self
    }

    pub fn s(mut self, cond: bool) -> Self {
        self.set(Flags::SUBTRACTION, cond);
        self
    }

    pub fn h(mut self, cond: bool) -> Self {
        self.set(Flags::HALF_CARRY, cond);
        self
    }

    pub fn c(mut self, cond: bool) -> Self {
        self.set(Flags::CARRY, cond);
        self
    }
}

pub trait HalfCarry {
    fn half_carry(&self, result: &Self) -> bool;
}

impl HalfCarry for u8 {
    fn half_carry(&self, result: &Self) -> bool {
        (self & 0xF) + (result & 0xF) > 0xF
    }
}

impl HalfCarry for u16 {
    fn half_carry(&self, result: &Self) -> bool {
        (self & 0xF) + (result & 0xF) > 0xF
    }
}

pub trait Address {
    fn calculate_offset(&self, offset: u8) -> (Self, bool) where Self: Sized;
}

impl Address for u16 {
    fn calculate_offset(&self, offset: u8) -> (Self, bool) {
        if offset > 127 {
            self.overflowing_sub(offset.wrapping_neg() as u16)
        } else {
            self.overflowing_add(offset as u16)
        }
    }
}

#[derive(Debug, Default)]
#[repr(C)]
struct Registers {
    a: u8,
    f: Flags,
    b: u8,
    c: u8,
    d: u8,
    e: u8,
    h: u8,
    l: u8,
}

impl Registers {
    pub fn af(&self) -> u16 {
        u16::from_be_bytes([self.a, self.f.bits])
    }

    pub fn set_af(&mut self, data: u16) {
        let data = data.to_be_bytes();
        self.a = data[0];
        self.f = Flags::from_bits(data[1]).expect("4 lowest bits of F should never be 1");
    }

    pub fn bc(&self) -> u16 {
        u16::from_be_bytes([self.b, self.c])
    }

    pub fn set_bc(&mut self, data: u16) {
        let data = data.to_be_bytes();
        self.b = data[0];
        self.c = data[1];
    }

    pub fn de(&self) -> u16 {
        u16::from_be_bytes([self.d, self.e])
    }

    pub fn set_de(&mut self, data: u16) {
        let data = data.to_be_bytes();
        self.d = data[0];
        self.e = data[1];
    }

    pub fn hl(&self) -> u16 {
        u16::from_be_bytes([self.h, self.l])
    }

    pub fn set_hl(&mut self, data: u16) {
        let data = data.to_be_bytes();
        self.h = data[0];
        self.l = data[1];
    }
}

pub struct CPU {
    registers: Registers,
    sp: u16,
    pc: u16,
    memory: [u8; 0x10000],
    // might break these into flags
    ime: bool,
    interrupt: bool,
    // temporary clock counter for testing
    clock: u64,
}

impl CPU {
    pub fn new() -> Self {
        let mut memory = [0; 0x10000];
        let (left, _) = memory.split_at_mut(256);
        left.copy_from_slice(&BOOTSTRAP_ROM);
        Self {
            registers: Registers::default(), 
            sp: 0, 
            pc: 0, 
            memory,
            clock: 0,
            ime: false,
            interrupt: false,
        }
    }

    pub fn run(&mut self) -> ! {
        loop {
            let opcode = self.fetch_opcode();
            self.execute(opcode);
        }
    }
    
    // Takes a u8 because I don't want to ever conceivably overflow multiple times with a single call
    fn tick(&mut self, ticks: u8) {
        // Right now just a dummy function
        self.clock += ticks as u64;
    }

    // Might go back to the Option variant, it's a bit safer
    fn set_flags(&mut self/*, zero: Option<bool>, sub: Option<bool>, half_carry: Option<bool>, carry: Option<bool>*/) -> &mut Flags {
        // if let Some(z) = zero {
        //     self.registers.f.set(Flags::ZERO, z);
        // }
        // if let Some(s) = sub {
        //     self.registers.f.set(Flags::SUBTRACTION, s);
        // }
        // if let Some(h) = half_carry {
        //     self.registers.f.set(Flags::HALF_CARRY, h);
        // }
        // if let Some(c) = carry {
        //     self.registers.f.set(Flags::CARRY, c);
        // }
        &mut self.registers.f
    }

    /// Tells whether the Zero flag is set
    fn z_flag(&self) -> bool {
        self.registers.f.contains(Flags::ZERO)
    }

    fn set_z(&mut self, condition: bool) {
        self.registers.f.set(Flags::ZERO, condition);
    }

    /// Tells whether the Subtraction flag is set
    pub fn s_flag(&self) -> bool {
        self.registers.f.contains(Flags::SUBTRACTION)
    }

    fn set_s(&mut self, condition: bool) {
        self.registers.f.set(Flags::SUBTRACTION, condition);
    }
    
    /// Tells whether the Half Carry flag is set
    pub fn h_flag(&self) -> bool {
        self.registers.f.contains(Flags::HALF_CARRY)
    }
    
    fn set_h(&mut self, condition: bool) {
        self.registers.f.set(Flags::HALF_CARRY, condition);
    }

    /// Tells whether the Half Carry flag is set
    pub fn c_flag(&self) -> bool {
        self.registers.f.contains(Flags::CARRY)
    }
    
    fn set_c(&mut self, condition: bool) {
        self.registers.f.set(Flags::CARRY, condition);
    }

    fn read_bytes(&mut self, address: u16, n: u8) -> &[u8] {
        let address = address as usize;
        self.tick(n);
        &self.memory[address..address.wrapping_add(n  as usize)]
    }

    fn fetch_opcode(&mut self) -> u8 {
        //println!("{:04x}", self.pc);
        self.pc = self.pc.wrapping_add(1);
        self.read_bytes(self.pc - 1, 1)[0]
    }

    fn get_addr_at_pc(&mut self) -> u16 {
        u16::from_le_bytes(self.read_bytes(self.pc, 2).try_into().unwrap())
    }

    // some of the more important 16 bit stuff consolidated
    fn get_operand_16(&mut self, operand: Operand) -> u16 {
        match operand {
            Operand::D8 => self.fetch_opcode() as u16,
            Operand::A16 => self.get_addr_at_pc(),
            Operand::HL => self.registers.hl(),
            Operand::AF => self.registers.af(),
            Operand::BC => self.registers.bc(),
            Operand::DE => self.registers.de(),
            Operand::SP => self.sp,
            _ => panic!("Only 16 bit operands")
        }
    }

    
    // functions to increment and decrement HL without adding a cycle or being too verbose
    fn inc_hl(&mut self) {
        self.registers.set_hl(self.registers.hl().wrapping_add(1));
    }
    fn dec_hl(&mut self) {
        self.registers.set_hl(self.registers.hl().wrapping_sub(1));
    }

    // INC BC, DE, HL, SP
    fn inc_16(&mut self, operand: Operand) {
        match operand {
            Operand::BC => self.registers.set_bc(self.registers.bc().wrapping_add(1)),
            Operand::DE => self.registers.set_de(self.registers.de().wrapping_add(1)),
            Operand::HL => self.inc_hl(),
            Operand::SP => self.sp = self.sp.wrapping_add(1),
            _ => panic!("Invalid INC operand"),
        }
        self.tick(1);
    }

    // DEC BC, DE, HL, SP
    fn dec_16(&mut self, operand: Operand) {
        match operand {
            Operand::BC => self.registers.set_bc(self.registers.bc().wrapping_sub(1)),
            Operand::DE => self.registers.set_de(self.registers.de().wrapping_sub(1)),
            Operand::HL => self.registers.set_hl(self.registers.hl().wrapping_sub(1)),
            Operand::SP => self.sp = self.sp.wrapping_sub(1),
            _ => panic!("Invalid INC operand"),
        }
        self.tick(1);
    }

    // ADD HL xx, ADD SP i8
    // Should probably break this up
    fn add_16(&mut self, sp: bool, value: Operand) {
        let value = self.get_operand_16(value);
        let (result, carry);
        let half_carry;
        if sp {
            (result, carry) = self.sp.calculate_offset(value as u8);
            half_carry = self.sp.half_carry(&result);
            self.tick(2);
            self.sp = result;
        } else {
            (result, carry) = self.registers.hl().overflowing_add(value);
            half_carry = self.registers.hl().half_carry(&result);
            self.registers.set_hl(result);
        }

        self.set_flags()
            .z(!sp && result == 0)
            .s(false)
            .h(half_carry)
            .c(carry);
    }


    // Helper function for setting full 16 bit registers
    fn ld_16(&mut self, dest: Operand) {
        let value = self.get_addr_at_pc();
        match dest {
            Operand::AF => self.registers.set_af(value),
            Operand::BC => self.registers.set_bc(value),
            Operand::DE => self.registers.set_de(value),
            Operand::HL => self.registers.set_hl(value),
            Operand::SP => self.sp = value,
            Operand::PC => self.pc = value,
            _ => panic!("Invalid 16 bit register to assign to"),
        }
    }

    
    fn get_operand(&mut self, operand: Operand) -> u8 {
        match operand {
            Operand::D8 => self.fetch_opcode(),
            Operand::A8 => self.memory[0xFF00 + self.fetch_opcode() as usize],
            Operand::A16 => self.memory[self.get_addr_at_pc() as usize],
            Operand::A => self.registers.a,
            Operand::B => self.registers.b,
            Operand::C => self.registers.c,
            Operand::D => self.registers.d,
            Operand::E => self.registers.e,
            Operand::H => self.registers.h,
            Operand::L => self.registers.l,
            Operand::HL => self.memory[self.registers.hl() as usize], // 1 tick extra?
            Operand::AF => self.memory[self.registers.af() as usize], // 1 tick extra?
            Operand::BC => self.memory[self.registers.bc() as usize], // 1 tick extra?
            Operand::DE => self.memory[self.registers.de() as usize], // 1 tick extra?
            _ => panic!("Invalid 8 bit operand"),
        }
    }

    fn get_operand_mut(&mut self, operand: Operand) -> &mut u8 {
        match operand {
            Operand::A16 => {
                self.tick(1);   // 1 extra cycle to read
                &mut self.memory[self.get_addr_at_pc() as usize]
            },
            Operand::A8 => {
                let opcode = self.fetch_opcode();  // 1 extra cycle to read
                &mut self.memory[0xFF00 + opcode as usize]
            },
            Operand::A => &mut self.registers.a,
            Operand::B => &mut self.registers.b,
            Operand::C => &mut self.registers.c,
            Operand::D => &mut self.registers.d,
            Operand::E => &mut self.registers.e,
            Operand::H => &mut self.registers.h,
            Operand::L => &mut self.registers.l,
            Operand::HL => {
                self.tick(1);   // 1 extra cycle to read
                &mut self.memory[self.registers.hl() as usize]
            },
            Operand::BC => {
                self.tick(1);   // 1 extra cycle to read
                &mut self.memory[self.registers.bc() as usize]
            },
            Operand::DE => {
                self.tick(1);   // 1 extra cycle to read
                &mut self.memory[self.registers.de() as usize]
            },
            _ => panic!("invalid 8 bit operand")
        }
    }

    fn ld(&mut self, dest: Operand, value: Operand) {
        *self.get_operand_mut(dest) = self.get_operand(value);
    }

    // INC
    fn inc(&mut self, operand: Operand) {
        let operand = self.get_operand_mut(operand);
        let result = operand.wrapping_add(1);
        let half_carry = operand.half_carry(&result);
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .s(half_carry);
    }

    // DEC
    fn dec(&mut self, operand: Operand) {
        let operand = self.get_operand_mut(operand);
        let result = operand.wrapping_sub(1);
        let half_carry = operand.half_carry(&result);
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(true)
            .h(half_carry);
    }

    // JR, JP
    fn jump(&mut self, addr: Operand, condition: bool) {
        let addr = match addr {
            Operand::A8 => {
                let opcode = self.fetch_opcode();   // JP
                self.pc.calculate_offset(opcode).0  // JR
            }, 
            Operand::A16 => self.get_addr_at_pc(),
            _ => panic!("invalid jump operand"),
        };

        if condition {
            self.pc = addr;
            self.tick(1);
        }
    }

    // Sets A to be binary coded decimal, used after an arithmetic operation with BCDs
    fn daa(&mut self) {
        let a = self.registers.a;
        let mut correction = 0;

        if self.h_flag() || (!self.s_flag() && (a & 0xF) > 9) {
            correction |= 0x06;
        }

        if self.c_flag() || (!self.s_flag() && (a >> 4) > 9) || self.h_flag() {
            correction |= 0x60;
        }

        let (a, carry) = if self.s_flag() {
            a.overflowing_add(correction)
        } else {
            a.overflowing_sub(correction)
        };

        self.registers.a = a;

        self.set_flags()
            .z(self.registers.a == 0)
            .h(false)
            .c(carry);
    }

    // ADD, ADC
    fn add(&mut self, value: Operand, with_carry: bool) {
        let value = self.get_operand(value);
        let (mut result, mut carry) = self.registers.a.overflowing_add(value);
        if with_carry {
            (result, carry) = result.overflowing_add(self.c_flag() as u8);
        }
        let half_carry = self.registers.a.half_carry(&result);

        self.registers.a = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(half_carry)
            .c(carry);
    }

    // SUB, SBC
    fn sub(&mut self, value: Operand, with_carry: bool) {
        let value = self.get_operand(value);
        let (mut result, mut carry) = self.registers.a.overflowing_sub(value);
        if with_carry {
            (result, carry) = result.overflowing_sub(self.c_flag() as u8);
        }
        let half_carry = self.registers.a.half_carry(&result);

        self.registers.a = result;
        self.set_flags()
            .z(result == 0)
            .s(true)
            .h(half_carry)
            .c(carry);
    }

    pub fn and(&mut self, operand: Operand) {
        self.registers.a &= self.get_operand(operand);
        self.set_flags()
            .z(self.registers.a == 0)
            .s(false)
            .h(true)
            .c(false);
    }

    pub fn or(&mut self, operand: Operand) {
        self.registers.a |= self.get_operand(operand);
        self.set_flags()
            .z(self.registers.a == 0)
            .s(false)
            .h(false)
            .c(false);
    }

    pub fn xor(&mut self, operand: Operand) {
        self.registers.a ^= self.get_operand(operand);
        self.set_flags()
            .z(self.registers.a == 0)
            .s(false)
            .h(false)
            .c(false);
    }

    // COMPARE
    fn cp(&mut self, operand: Operand) {
        let operand = self.get_operand(operand);
        let (result, carry) = self.registers.a.overflowing_sub(operand);
        self.set_flags()
            .z(result == 0)
            .s(true)
            .h(self.registers.a.half_carry(&result))
            .c(carry);
    }
    
    // PUSH
    fn push(&mut self, value: Operand) {
        let [hi, lo] = match value {
            Operand::AF => [self.registers.a, self.registers.f.bits],
            Operand::BC => [self.registers.b, self.registers.c],
            Operand::DE => [self.registers.d, self.registers.e],
            Operand::HL => [self.registers.h, self.registers.l],
            Operand::SP => self.sp.to_le_bytes(),
            Operand::PC => self.pc.to_le_bytes(),
            _ => panic!("Invalid operand for push"),
        };
        
        self.sp -= 1;
        self.memory[self.sp as usize] = hi;
        self.sp -= 1;
        self.memory[self.sp as usize] = lo;
        self.tick(3);
    }
    
    // POP
    fn pop(&mut self, dest: Operand) {
        let result = u16::from_le_bytes(self.read_bytes(self.sp, 2).try_into().unwrap());
        self.sp = self.sp.wrapping_add(2);
        match dest {
            Operand::AF => self.registers.set_af(result),
            Operand::BC => self.registers.set_bc(result),
            Operand::DE => self.registers.set_de(result),
            Operand::HL => self.registers.set_hl(result),
            Operand::SP => self.sp = result,
            Operand::PC => self.pc = result,
            _ => panic!("Invalid operand for pop"),
        }
        self.tick(2);
    }

    fn call(&mut self, condition: bool) {
        let addr = self.get_addr_at_pc();
        if condition {
            self.push(Operand::PC);
            self.pc = addr;
        }
    }

    fn ret(&mut self, condition: bool) {
        if condition {
            self.tick(2);
            self.pop(Operand::PC);
        } else {
            self.tick(1);
        }
    }

    fn rst(&mut self, val: u8) {
        self.push(Operand::SP);
        self.pc = self.memory[val as usize] as u16;
    }

    // *************************//
   //  CB-PREFIXED FUNCTIONS   //
  // *************************//

    // RLCA (mostly) RLA, RLC, RL
    fn rot_l(&mut self, operand: Operand, carry_to_0: bool) {
        let c_flag = carry_to_0 && self.c_flag();
        let operand = self.get_operand_mut(operand);
        let bit_7 = *operand >> 7 == 1;
        let result = if c_flag {
            (*operand << 1) | c_flag as u8
        } else {
            operand.rotate_left(1)
        };
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(false)
            .c(bit_7);
    }

    // RRCA (mostly) RRA, RRC, RR
    fn rot_r(&mut self, operand: Operand, carry_to_7: bool) {
        let c_flag = carry_to_7 && self.c_flag();
        let operand = self.get_operand_mut(operand);
        let bit_0 = *operand & 1 == 1;
        let result = if c_flag {
            (*operand >> 1) | (c_flag as u8) << 7
        } else {
            operand.rotate_right(1)
        };
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(false)
            .c(bit_0);
    }

    // SLA
    fn shift_l(&mut self, operand: Operand) {
        let operand = self.get_operand_mut(operand);
        let bit_7 = *operand >> 7 == 1;
        let result = *operand << 1;
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(false)
            .c(bit_7);
    }

    // SRA, SRL
    fn shift_r(&mut self, operand: Operand, keep_bit_7: bool) {
        let operand = self.get_operand_mut(operand);
        let bit_0 = *operand & 1 == 1;
        let bit_7 = *operand & 1 << 7;
        let result = if keep_bit_7 {
            (*operand >> 1) | bit_7
        } else {
            *operand >> 1
        };
        *operand = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(false)
            .c(bit_0);
    }

    // SWAP
    pub fn swap(&mut self, dest: Operand) {
        let dest = self.get_operand_mut(dest);
        let result = *dest << 4 | *dest >> 4;
        *dest = result;
        self.set_flags()
            .z(result == 0)
            .s(false)
            .h(false)
            .c(false);
    }

    // BIT
    fn bit(&mut self, bit: u8, operand: Operand) {
        self.set_flags()
            .z(self.get_operand(operand) & (1 << bit) == 0)
            .s(false)
            .h(true);
    }

    // RES
    fn res(&mut self, bit: u8, operand: Operand) {
        let operand = self.get_operand_mut(operand);
        *operand |= 1 << bit;
    }

    // SET
    fn set(&mut self, bit: u8, operand: Operand) {
        let operand = self.get_operand_mut(operand);
        *operand &= !(1 << bit);
    }

    // ***************************//
   //   THE MAGIC HAPPENS HERE   //
  // ***************************//

    pub fn execute(&mut self, opcode: u8) {
        use Operand::*;
        let and7 = |x: u8| match x & 7 {
            0 => B,
            1 => C,
            2 => D,
            3 => E,
            4 => H,
            5 => L,
            6 => HL,
            7 => A,
            _ => panic!("categorically impossible"),
        };
        match opcode {
            // Opcodes with no operands
            0x00 => { /*NOP*/ }
            0x10 => { self.fetch_opcode(); loop {} } // VERY crude STOP
            0x27 => self.daa(),
            0x2F => self.registers.a = !self.registers.a,
            0x37 => self.registers.f.insert(Flags::CARRY),
            0x3F => self.registers.f.remove(Flags::CARRY),
            0x76 => { // HALT
                if self.ime {
                    if self.interrupt {
                        self.pc = self.pc.wrapping_sub(1); // crude HALT
                    }
                } else {
                    self.fetch_opcode(); // discard an instruction when interrupts disabled
                }
            }
            0xF3 => self.ime = false,
            0xFB => self.ime = true,

            // CB Prefix
            0xCB => {
                let opcode = self.fetch_opcode();
                // might need to tweak on opcode & 7 == 6 to add logic for cycle timing
                let bit_4 = opcode & 1 << 4 != 0;
                match (opcode >> 4, bit_4) {
                    (0x0, true) => self.rot_l(and7(opcode), false),
                    (0x0, false) => self.rot_r(and7(opcode), false),
                    (0x1, true) => self.rot_l(and7(opcode), true),
                    (0x1, false) => self.rot_r(and7(opcode), true),
                    (0x2, true) => self.shift_l(and7(opcode)),
                    (0x2, false) => self.shift_r(and7(opcode), true),
                    (0x3, true) => self.swap(and7(opcode)),
                    (0x3, false) => self.shift_r(and7(opcode), false),
                    (0x4, _) => self.bit(0 + bit_4 as u8, and7(opcode)),
                    (0x5, _) => self.bit(2 + bit_4 as u8, and7(opcode)),
                    (0x6, _) => self.bit(4 + bit_4 as u8, and7(opcode)),
                    (0x7, _) => self.bit(6 + bit_4 as u8, and7(opcode)),
                    (0x8, _) => self.res(0 + bit_4 as u8, and7(opcode)),
                    (0x9, _) => self.res(2 + bit_4 as u8, and7(opcode)),
                    (0xA, _) => self.res(4 + bit_4 as u8, and7(opcode)),
                    (0xB, _) => self.res(6 + bit_4 as u8, and7(opcode)),
                    (0xC, _) => self.set(0 + bit_4 as u8, and7(opcode)),
                    (0xD, _) => self.set(2 + bit_4 as u8, and7(opcode)),
                    (0xE, _) => self.set(4 + bit_4 as u8, and7(opcode)),
                    (0xF, _) => self.set(6 + bit_4 as u8, and7(opcode)),
                    _ => panic!("categorically impossible"),
                }
            }

            // LD
            val @ 0x40..=0x7F => self.ld(and7(val), and7(val >> 3)),

            // ALU
            val @ 0x80..=0xBF => {
                match (val >> 3) & 7 {
                    0 => self.add(and7(val), false),    // ADD
                    1 => self.add(and7(val), true),     // ADC
                    2 => self.sub(and7(val), false),    // SUB
                    3 => self.sub(and7(val), true),     // SBC
                    4 => self.and(and7(val)),           // AND
                    5 => self.xor(and7(val)),           // XOR
                    6 => self.or(and7(val)),            // OR
                    7 => self.cp(and7(val)),            // CP
                    _ => panic!("categorically impossible"),
                }
            }
            
            // Jump to $00, $08, $10, $18, $20, $28, $30, $38
            val @ (0xC7 | 0xCF | 0xD7 | 0xDF | 0xE7 | 0xEF | 0xF7 | 0xFF) => self.rst(val - 0xC7),

            // TODO: think of clever ways to consolidate the rest of this
            0x01 => self.ld_16(BC),
            0x02 => self.ld(BC, A),
            0x03 => self.inc_16(BC), // INC BC
            0x04 => self.inc(B), // INC B
            0x05 => self.dec(B), // DE
            0x06 => self.ld(B, D8),
            0x07 => {
                self.rot_l(A, false);
                self.set_flags().z(false);
            },
            0x08 => {
                let a16 = self.get_addr_at_pc() as usize;
                let bytes = self.sp.to_le_bytes();
                self.memory[a16] = bytes[0];
                self.memory[a16 + 1] = bytes[1];
                self.tick(2);
            },
            0x09 => self.add_16(false, BC),
            0x0A => self.ld(A, BC),
            0x0B => self.dec_16(BC),
            0x0C => self.inc(C),
            0x0D => self.dec(C),
            0x0E => self.ld(C, D8),
            0x0F => {
                self.rot_r(A, false);
                self.set_flags().z(false);
            },
            0x11 => self.ld_16(DE),
            0x12 => self.ld(DE, A),
            0x13 => self.inc_16(DE),
            0x14 => self.inc(D),
            0x15 => self.dec(D),
            0x16 => self.ld(D, D8),
            0x17 => {
                self.rot_l(A, true);
                self.set_flags().z(false);
            },
            0x18 => self.jump(A8, true),
            0x19 => self.add_16(false, DE),
            0x1A => self.ld(A, DE),
            0x1B => self.dec_16(DE),
            0x1C => self.inc(E),
            0x1D => self.dec(E),
            0x1E => self.ld(E, D8),
            0x1F => {
                self.rot_r(A, true);
                self.set_flags().z(false);
            },
            0x20 => self.jump(A8, !self.z_flag()),
            0x21 => self.ld_16(HL),
            0x22 => {
                self.ld(HL, A);
                self.inc_hl();
            }
            0x23 => self.inc_16(HL),
            0x24 => self.inc(H),
            0x25 => self.dec(H),
            0x26 => self.ld(H, D8),
            0x28 => { /*DAA */}
            0x29 => self.jump(A8, self.z_flag()),
            0x2A => {
                self.ld(A, HL);
                self.inc_hl();
            }
            0x2B => self.dec_16(SP),
            0x2C => self.inc(L),
            0x2D => self.dec(L),
            0x2E => self.ld(L, D8),
            0x30 => self.jump(A8, !self.c_flag()),
            0x31 => self.sp = self.get_addr_at_pc(),
            0x32 => {
                self.ld(HL, A);
                self.dec_hl();
            }
            0x33 => self.inc_16(SP),
            0x34 => self.inc(HL),
            0x35 => self.dec(HL),
            0x36 => self.ld(HL, D8),
            0x38 => self.jump(A8, self.c_flag()),
            0x39 => self.add_16(false, SP),
            0x3A => {
                self.ld(A, HL);
                self.dec_hl();
            }
            0x3B => self.dec_16(SP),
            0x3C => self.inc(A),
            0x3D => self.dec(A),
            0x3E => self.ld(A, D8),

            // These too      
            0xC0 => self.ret(!self.z_flag()),
            0xC1 => self.pop(BC),
            0xC2 => self.jump(A16, !self.z_flag()),
            0xC3 => self.jump(A16, true),
            0xC4 => self.call(!self.z_flag()),
            0xC5 => self.push(BC),
            0xC6 => self.add(D8, false),
            0xC8 => self.ret(self.z_flag()),
            0xC9 => { // RET
                self.tick(1);
                self.pop(PC)
            },
            0xCA => self.jump(A16, self.z_flag()),
            0xCC => self.call(self.z_flag()),
            0xCD => self.call(true),
            0xCE => self.add(D8, true),
            0xD0 => self.ret(!self.c_flag()),
            0xD1 => self.pop(DE),
            0xD2 => self.jump(A16, !self.c_flag()),
            0xD4 => self.call(!self.c_flag()),
            0xD5 => self.push(DE),
            0xD6 => self.sub(D8, false),
            0xDC => self.call(self.c_flag()),
            0xDE => self.sub(D8, true),
            0xE0 => self.ld(A8, A),
            0xE1 => self.pop(HL),
            0xE2 => self.memory[0xFF00 + self.registers.c as usize] = self.registers.a,
            0xE5 => self.push(HL),
            0xE6 => self.and(D8),
            0xE8 => self.add_16(true, D8),
            0xE9 => self.pc = self.registers.hl(),
            0xEA => self.ld(A16, A),
            0xEE => self.xor(D8),
            0xF0 => self.ld(A, A8),
            0xF1 => self.pop(AF),
            0xF2 => self.registers.a = self.memory[0xFF00 + self.registers.c as usize],
            0xF5 => self.push(AF),
            0xF6 => self.or(D8),
            0xF8 => { // LD HL, SP+s8
                let offset = self.fetch_opcode();
                let (s8_offset, carry) = self.sp.calculate_offset(offset);
                self.registers.set_hl(s8_offset);
                self.tick(1);
                self.set_flags()
                    .h(self.pc.half_carry(&s8_offset))
                    .c(carry);
            }
            0xF9 => {
                self.sp = self.registers.hl();
                self.tick(1);
            }
            0xFA => self.ld(A, A16),
            0xFE => self.cp(D8),
            _ => { /*Invalid*/ }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_combined_registers() {
        let mut reg = Registers::default();
        reg.b = 8;
        reg.c = 7;

        let bc = (reg.b as u16) << 8 | reg.c as u16;

        assert_eq!(reg.bc(), bc);

        reg.set_bc(0xFF00);

        assert_eq!((reg.b, reg.c), (0xFF, 0x00));
    }

    #[test]
    fn test_s8_offset() {
        assert_eq!(128u16.calculate_offset(128), (0, false));
    }
}