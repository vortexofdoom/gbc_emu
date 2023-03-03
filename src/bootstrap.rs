pub const BOOTSTRAP_ROM: [u8; 256] = [
    // Set up stack
    0x31, 0xFE, 0xFF,   // LD SP, $FFFE
    //Zero the memory from $8000 - $9FFF (VRAM)
    0xAF,               // XOR A
    0x21, 0xFF, 0x9F,   // LD HL, $9FFF
// addr $0007
    0x32,               // LD (HL-), A
    0xCB, 0x7C,         // BIT 7, H
    0x20, 0xFB,         // JR NZ, +$FB ($0007)
    // Set up audio
    0x21, 0x26, 0xFF,   // LD HL, $FF26
    0x0E, 0x11,         // LD C, $11
    0x3E, 0x80,         // LD A, $80
    0x32,               // LD (HL-), A
    0xE2,               // LD ($FF00 + C), A
    0x0C,               // INC C
    0x3E, 0xF3,         // LD A, $F3
    0xE2,               // LD ($FF00 + C), A
    0x32,               // LD (HL-), A
    0x3E, 0x77,         // LD A, $77
    0x77,               // LD (HL), A
    // Set up BG palette
    0x3E, 0xFC,         // LD A, $FC
    0xE0, 0x47,         // LD ($FF00+$47), A
    // Load logo from cart into VRAM
    0x11, 0x04, 0x01,   // LD DE $0104
    0x21, 0x10, 0x80,   // LD HL $8010
// addr $0027
    0x1A,               // LD A, (DE)
    0xCD, 0x95, 0x00,   // CALL $0095
    0xCD, 0x96, 0x00,   // CALL $0096
    0x13,               // INC DE
    0x7B,               // LD A, E
    0xFE, 0x34,         // CP $34
    0x20, 0xF3,         // JR NZ, +0xF3 ($0027)
    // Load 8 additional bytes into VRAM
    0x11, 0xD8, 0x00,   // LD DE, $00D8
    0x06, 0x08,         // LD B, $08
// addr $0039
    0x1A,               // LD A, DE
    0x13,               // INC DE
    0x22,               // LD (HL+), A
    0x23,               // INC HL
    0x05,               // DEC B
    0x20, 0xF9,         // JR NZ +$F9 ($0039)
    // Set up BG tilemap
    0x3E, 0x19,         // LD A, $19
    0xEA, 0x10, 0x99,   // LD ($9910), A
    0x21, 0x2F, 0x99,   // LD HL, $992F
// addr $0048
    0x0E, 0x0c,         // LD C, $0C
// addr $004A
    0x3D,               // DEC A
    0x28, 0x08,         // JR Z +$08 ($0055)
    0x32,               // LD (HL-), A
    0x0D,               // DEC C
    0x20, 0xF9,         // JR NZ +$F9 ($004A)
    0x2E, 0x0F,         // LD L, $0F
    0x18, 0xF3,         // JR +$F3 ($0048)
    // Logo and sound
// addr $0055
    0x67,               // LD H, A
    0x3E, 0x64,         // LD A, $64
    0x57,               // LD D, A
    // Set vertical scroll register
    0xE0, 0x42,         // LD ($FF00 + $42), A
    0x3E, 0x91,         // LD A, $91
    // Turn on LCD
    0xE0, 0x40,         // LD ($FF00 + $40), A
    0x04,               // INC B
// addr $0060
    0x1E, 0x02,         // LD E, $02
// addr $0062
    0x0E, 0x0C,         // LD C, $0C
// addr $0064
    0xF0, 0x44,         // LD A, ($FF00 + $44)
    0xFE, 0x90,         // CP $90
    0x20, 0xFA,         // JR NZ +$FA ($0064)
    0x0D,               // DEC C
    0x20, 0xF7,         // JR NZ +$F7 ($0064)
    0x1D,               // DEC E
    0x20, 0xF2,         // JR NZ +$F2 ($0062)
    
    0x0E, 0x13,         // LD C, $13
    0x24,               // INC H
    0x7C,               // LD A, H
    0x1E, 0x83,         // LD E, $83
    // when count is 62, play 1st sound
    0xFE, 0x62,         // CP $62
    0x28, 0x06,         // JR Z +$06 ($0080)
    0x1E, 0xC1,         // LD E $C1
    0xFE, 0x64,         // CP $64
    // when count is 64, play 2nd sound
    0x20, 0x06,         // JR NZ +$06 ($0080)
// addr $0080
    0x7B,               // LD A, E
    0xE2,               // LD ($FF00+C), A
    0x0C,               // INC C
    0x3E, 0x87,         // LD A, $87
    0xE2,               // LD ($FF00+C), A
// addr $0086
    0xF0, 0x42,         // LD A, ($FF00 + $42)
    0x90,               // SUB B
    // Scroll logo up if B = 1
    0xE0, 0x42,         // LD ($FF00 + $42), A
    0x15,               // DEC D
    0x20, 0xD2,         // JR NZ +$D2 ($0060)
    // Set B to zero first time, second time jumps to Logo check
    0x05,               // DEC B
    0x20, 0x4F,         // JR NZ +$4F ($00E0)

    0x16, 0x20,         // LD D, $20
    0x18, 0xCB,         // JR +$CB ($0060)

    // Graphic routine
    // Double up all the bits of graphics data and store in VRAM
    0x4F,               // LD C, A
    0x06, 0x04,         // LD B, $04
// addr $0098
    0xC5,               // PUSH BC
    0xCB, 0x11,         // RL C
    0x17,               // RLA
    0xC1,               // POP BC
    0xCB, 0x11,         // RL C
    0x17,               // RLA
    0x05,               // DEC B
    0x20, 0xF5,         // JR NZ +$F5 ($0098)
    0x22,               // LD (HL+), A
    0x23,               // INC HL
    0x22,               // LD (HL+), A
    0x23,               // INC HL
    0xC9,               // RET

// addr $00A8 
    // Nintendo Logo
    0xce, 0xed, 0x66, 0x66, 0xcc, 0x0d, 0x00, 0x0b, 0x03, 0x73, 0x00, 0x83, 0x00, 0x0c, 0x00, 0x0d, 
    0x00, 0x08, 0x11, 0x1f, 0x88, 0x89, 0x00, 0x0e, 0xdc, 0xcc, 0x6e, 0xe6, 0xdd, 0xdd, 0xd9, 0x99, 
    0xbb, 0xbb, 0x67, 0x63, 0x6e, 0x0e, 0xec, 0xcc, 0xdd, 0xdc, 0x99, 0x9f, 0xbb, 0xb9, 0x33, 0x3e,
// addr $00D8
    // copyright symbol
    0x3C, 0x42, 0xb9, 0xa5, 0xb9, 0xa5, 0x42, 0x3c,

    // Logo check
// addr $00E0
    // point HL to logo in cart
    0x21, 0x04, 0x01,   // LD HL, $0104
    // Point DE to logo in rom
    0x11, 0xA8, 0x00,   // LD DE, $00A8

// addr $00E6
    0x1A,               // LD A, (DE)
    0x13,               // INC DE
    // Compare cart/rom
    0xBE,               // CP (HL)
    // Lock up if it doesn't match
    0x20, 0xFE,         // JR NZ $FE
    0x23,               // INC HL
    0x7D,               // LD A, L
    0xFE, 0x34,         // CP $34
    0x20, 0xF5,         // JR NZ $F5 ($00F4)
    
    0x06, 0x19,         // LD B, $19
    0x78,               // LD A, B
// addr $00F4
    0x86,               // ADD A, (HL)
    0x23,               // INC HL
    0x05,               // DEC B
    0x20, 0xFB,         // JR NZ $FB ($00F4)
    0x86,               // ADD A, (HL)
    // lock up if checksum fails
    0x20, 0xFE,         // JR NZ $FE ($00F4)

    0x3E, 0x01,         // LD A, $01
    // Turn off DMG rom
    0xE0, 0x50,         // LD ($FF00+$50), A
];