#include <catch2/catch_test_macros.hpp>

#include <optimizer/translation/ir_to_sala.hpp>
#include <optimizer/translation/sala_to_ir.hpp>

#include <sala/streaming.hpp>

using namespace optimizer;

static const char* const SALA_json_0 = R"JSON({
"magic": "sala",
"version": "0.1",
"system": "x86_64-pc-linux-gnu",
"num_cpu_bits": 64,
"name": "test",
"entry_function": 1,
"constants": [
  "00000000"
],
"static": [
],
"functions": [
  {
    "name": [ "__sala_static_initializer__", [0,0] ],
    "parameters": [
    ],
    "locals": [
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "RET", "n", [0,0] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "main", [1,1] ],
    "parameters": [
      [ 8, [1,1] ]
    ],
    "locals": [
      [ 8, [0,0] ],
      [ 4, [0,0] ]
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ll", 0, 1, [0,0] ],
          [ "STORE", "n", "lc", 0, 0, [0,0] ],
          [ "STORE", "n", "pc", 0, 0, [2,3] ],
          [ "RET", "n", [2,3] ]
        ],
        "successors": [ ]
      }
    ]
  }
],
"external_variables": [
],
"external_functions": [
]
}
)JSON";

static const char* const SALA_json_1 = R"JSON({
"magic": "sala",
"version": "0.1",
"system": "x86_64-pc-linux-gnu",
"num_cpu_bits": 64,
"name": "test2",
"entry_function": 1,
"constants": [
  "00000000",
  "01000000",
  "00",
  "0800000000000000",
  "11000000",
  "0800000000000000",
  "25000000",
  "0b000000",
  "65000000",
  "03000000",
  "0000000000408f40",
  "ff00000000000000",
  "02000000",
  "ffffffff"
],
"static": [
],
"functions": [
  {
    "name": [ "__sala_static_initializer__", [0,0] ],
    "parameters": [
    ],
    "locals": [
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "RET", "n", [0,0] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "main", [21,1] ],
    "parameters": [
      [ 8, [21,1] ]
    ],
    "locals": [
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 4, [24,19] ],
      [ 1, [24,20] ],
      [ 4, [24,20] ],
      [ 4, [24,16] ],
      [ 4, [25,16] ],
      [ 8, [25,5] ],
      [ 8, [25,5] ],
      [ 8, [25,5] ],
      [ 4, [26,21] ],
      [ 4, [26,25] ],
      [ 1, [26,23] ],
      [ 4, [26,52] ],
      [ 4, [26,53] ],
      [ 4, [26,57] ],
      [ 4, [26,63] ],
      [ 8, [26,42] ],
      [ 4, [26,81] ],
      [ 4, [26,82] ],
      [ 8, [26,72] ],
      [ 8, [26,70] ],
      [ 4, [26,37] ],
      [ 8, [26,33] ],
      [ 8, [26,33] ],
      [ 4, [26,28] ],
      [ 4, [26,28] ],
      [ 4, [27,16] ],
      [ 4, [27,17] ],
      [ 4, [28,21] ],
      [ 4, [28,25] ],
      [ 1, [28,23] ],
      [ 4, [28,51] ],
      [ 8, [28,47] ],
      [ 8, [28,47] ],
      [ 8, [28,47] ],
      [ 8, [28,54] ],
      [ 8, [28,40] ],
      [ 8, [28,37] ],
      [ 8, [28,37] ],
      [ 4, [28,28] ],
      [ 4, [28,28] ],
      [ 8, [29,20] ],
      [ 8, [29,22] ],
      [ 4, [29,12] ],
      [ 8, [30,1] ],
      [ 4, [30,1] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ],
      [ 8, [0,0] ],
      [ 4, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ]
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ll", 0, 53, [0,0] ],
          [ "ADDRESS", "n", "ll", 1, 54, [0,0] ],
          [ "ADDRESS", "n", "ll", 2, 55, [0,0] ],
          [ "ADDRESS", "n", "ll", 3, 56, [0,0] ],
          [ "ADDRESS", "n", "ll", 4, 57, [0,0] ],
          [ "ADDRESS", "n", "ll", 5, 58, [0,0] ],
          [ "ADDRESS", "n", "ll", 6, 59, [0,0] ],
          [ "ADDRESS", "n", "ll", 7, 60, [0,0] ],
          [ "STORE", "n", "lc", 0, 0, [0,0] ],
          [ "STORE", "n", "lc", 1, 1, [22,26] ],
          [ "MEMSET", "n", "lcc", 2, 2, 3, [23,35] ],
          [ "LOAD", "n", "ll", 8, 1, [24,19] ],
          [ "EQUAL", "u", "llc", 9, 8, 1, [24,20] ],
          [ "EXTEND", "u", "ll", 10, 9, [24,20] ],
          [ "ADD", "s", "lcl", 11, 4, 10, [24,16] ],
          [ "STORE", "n", "ll", 3, 11, [24,9] ],
          [ "LOAD", "n", "ll", 12, 3, [25,16] ],
          [ "EXTEND", "u", "ll", 13, 12, [25,5] ],
          [ "STACKSAVE", "n", "l", 14, [25,5] ],
          [ "STORE", "n", "ll", 4, 14, [25,5] ],
          [ "ALLOCA", "n", "llc", 15, 13, 5, [25,5] ],
          [ "STORE", "n", "ll", 5, 13, [25,5] ],
          [ "STORE", "n", "lc", 6, 0, [26,14] ],
          [ "JUMP", "n", [26,10] ]
        ],
        "successors": [ 1 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 16, 6, [26,21] ],
          [ "LOAD", "n", "ll", 17, 3, [26,25] ],
          [ "LESS", "s", "lll", 18, 16, 17, [26,23] ],
          [ "BRANCH", "n", "l", 18, [26,5] ]
        ],
        "successors": [ 4, 2 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 19, 6, [26,52] ],
          [ "MUL", "s", "llc", 20, 19, 6, [26,53] ],
          [ "ADD", "s", "llc", 21, 20, 7, [26,57] ],
          [ "REM", "s", "llc", 22, 21, 8, [26,63] ],
          [ "I2F", "s", "ll", 23, 22, [26,42] ],
          [ "LOAD", "n", "ll", 24, 6, [26,81] ],
          [ "REM", "s", "llc", 25, 24, 9, [26,82] ],
          [ "I2F", "s", "ll", 26, 25, [26,72] ],
          [ "SUB", "f", "lll", 27, 23, 26, [26,70] ],
          [ "LOAD", "n", "ll", 28, 6, [26,37] ],
          [ "EXTEND", "s", "ll", 29, 28, [26,33] ],
          [ "MOVEPTR", "n", "lllc", 30, 15, 29, 5, [26,33] ],
          [ "STORE", "n", "ll", 30, 27, [26,40] ],
          [ "JUMP", "n", [26,33] ]
        ],
        "successors": [ 3 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 31, 6, [26,28] ],
          [ "ADD", "s", "llc", 32, 31, 1, [26,28] ],
          [ "STORE", "n", "ll", 6, 32, [26,28] ],
          [ "JUMP", "n", [26,5] ]
        ],
        "successors": [ 1 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 33, 3, [27,16] ],
          [ "SUB", "s", "llc", 34, 33, 1, [27,17] ],
          [ "CALL", "n", "flcl", 2, 15, 0, 34, [27,5] ],
          [ "STORE", "n", "lc", 7, 0, [28,14] ],
          [ "JUMP", "n", [28,10] ]
        ],
        "successors": [ 5 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 35, 7, [28,21] ],
          [ "LOAD", "n", "ll", 36, 3, [28,25] ],
          [ "LESS", "s", "lll", 37, 35, 36, [28,23] ],
          [ "BRANCH", "n", "l", 37, [28,5] ]
        ],
        "successors": [ 8, 6 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 38, 7, [28,51] ],
          [ "EXTEND", "s", "ll", 39, 38, [28,47] ],
          [ "MOVEPTR", "n", "lllc", 40, 15, 39, 5, [28,47] ],
          [ "LOAD", "n", "ll", 41, 40, [28,47] ],
          [ "ADD", "f", "llc", 42, 41, 10, [28,54] ],
          [ "F2I", "s", "ll", 43, 42, [28,40] ],
          [ "LOAD", "n", "ll", 44, 2, [28,37] ],
          [ "XOR", "n", "lll", 45, 44, 43, [28,37] ],
          [ "STORE", "n", "ll", 2, 45, [28,37] ],
          [ "JUMP", "n", [28,33] ]
        ],
        "successors": [ 7 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 46, 7, [28,28] ],
          [ "ADD", "s", "llc", 47, 46, 1, [28,28] ],
          [ "STORE", "n", "ll", 7, 47, [28,28] ],
          [ "JUMP", "n", [28,5] ]
        ],
        "successors": [ 5 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 48, 2, [29,20] ],
          [ "AND", "n", "llc", 49, 48, 11, [29,22] ],
          [ "TRUNCATE", "u", "ll", 50, 49, [29,12] ],
          [ "STORE", "n", "ll", 0, 50, [29,5] ],
          [ "LOAD", "n", "ll", 51, 4, [30,1] ],
          [ "STACKRESTORE", "n", "l", 51, [30,1] ],
          [ "LOAD", "n", "ll", 52, 0, [30,1] ],
          [ "STORE", "n", "pl", 0, 52, [30,1] ],
          [ "RET", "n", [30,1] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "qs", [4,1] ],
    "parameters": [
      [ 8, [4,1] ],
      [ 4, [4,1] ],
      [ 4, [4,1] ]
    ],
    "locals": [
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 4, [5,12] ],
      [ 4, [5,17] ],
      [ 1, [5,15] ],
      [ 4, [6,17] ],
      [ 4, [6,25] ],
      [ 8, [7,17] ],
      [ 4, [7,20] ],
      [ 4, [7,25] ],
      [ 4, [7,23] ],
      [ 4, [7,29] ],
      [ 8, [7,17] ],
      [ 8, [7,17] ],
      [ 8, [7,17] ],
      [ 4, [8,16] ],
      [ 4, [8,21] ],
      [ 1, [8,18] ],
      [ 8, [9,20] ],
      [ 4, [9,22] ],
      [ 8, [9,20] ],
      [ 8, [9,20] ],
      [ 8, [9,20] ],
      [ 8, [9,27] ],
      [ 1, [9,25] ],
      [ 4, [9,30] ],
      [ 4, [9,30] ],
      [ 8, [10,20] ],
      [ 4, [10,22] ],
      [ 8, [10,20] ],
      [ 8, [10,20] ],
      [ 8, [10,20] ],
      [ 8, [10,27] ],
      [ 1, [10,25] ],
      [ 4, [10,30] ],
      [ 4, [10,30] ],
      [ 4, [11,17] ],
      [ 4, [11,22] ],
      [ 1, [11,19] ],
      [ 8, [12,25] ],
      [ 4, [12,27] ],
      [ 8, [12,25] ],
      [ 8, [12,25] ],
      [ 8, [12,25] ],
      [ 8, [12,38] ],
      [ 4, [12,40] ],
      [ 8, [12,38] ],
      [ 8, [12,38] ],
      [ 8, [12,38] ],
      [ 8, [12,31] ],
      [ 4, [12,33] ],
      [ 8, [12,31] ],
      [ 8, [12,31] ],
      [ 8, [12,51] ],
      [ 8, [12,44] ],
      [ 4, [12,46] ],
      [ 8, [12,44] ],
      [ 8, [12,44] ],
      [ 4, [13,17] ],
      [ 4, [13,17] ],
      [ 4, [13,22] ],
      [ 4, [13,22] ],
      [ 4, [16,13] ],
      [ 4, [16,17] ],
      [ 4, [16,15] ],
      [ 4, [16,22] ],
      [ 4, [16,27] ],
      [ 4, [16,25] ],
      [ 1, [16,20] ],
      [ 4, [16,36] ],
      [ 4, [16,41] ],
      [ 1, [16,39] ],
      [ 8, [16,47] ],
      [ 4, [16,50] ],
      [ 4, [16,54] ],
      [ 4, [16,63] ],
      [ 4, [17,20] ],
      [ 4, [17,24] ],
      [ 1, [17,22] ],
      [ 8, [17,31] ],
      [ 4, [17,34] ],
      [ 4, [17,37] ],
      [ 4, [17,47] ],
      [ 8, [0,0] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ],
      [ 8, [0,0] ],
      [ 8, [0,0] ]
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ll", 0, 88, [0,0] ],
          [ "ADDRESS", "n", "ll", 1, 89, [0,0] ],
          [ "ADDRESS", "n", "ll", 2, 90, [0,0] ],
          [ "ADDRESS", "n", "ll", 3, 91, [0,0] ],
          [ "ADDRESS", "n", "ll", 4, 92, [0,0] ],
          [ "ADDRESS", "n", "ll", 5, 93, [0,0] ],
          [ "ADDRESS", "n", "ll", 6, 94, [0,0] ],
          [ "STORE", "n", "lp", 0, 0, [0,0] ],
          [ "STORE", "n", "lp", 1, 1, [0,0] ],
          [ "STORE", "n", "lp", 2, 2, [0,0] ],
          [ "JUMP", "n", [5,5] ]
        ],
        "successors": [ 1 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 7, 1, [5,12] ],
          [ "LOAD", "n", "ll", 8, 2, [5,17] ],
          [ "LESS", "s", "lll", 9, 7, 8, [5,15] ],
          [ "BRANCH", "n", "l", 9, [5,5] ]
        ],
        "successors": [ 21, 2 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 10, 1, [6,17] ],
          [ "STORE", "n", "ll", 3, 10, [6,13] ],
          [ "LOAD", "n", "ll", 11, 2, [6,25] ],
          [ "STORE", "n", "ll", 4, 11, [6,21] ],
          [ "LOAD", "n", "ll", 12, 0, [7,17] ],
          [ "LOAD", "n", "ll", 13, 1, [7,20] ],
          [ "LOAD", "n", "ll", 14, 2, [7,25] ],
          [ "ADD", "s", "lll", 15, 13, 14, [7,23] ],
          [ "DIV", "s", "llc", 16, 15, 12, [7,29] ],
          [ "EXTEND", "s", "ll", 17, 16, [7,17] ],
          [ "MOVEPTR", "n", "lllc", 18, 12, 17, 5, [7,17] ],
          [ "LOAD", "n", "ll", 19, 18, [7,17] ],
          [ "STORE", "n", "ll", 5, 19, [7,13] ],
          [ "JUMP", "n", [8,9] ]
        ],
        "successors": [ 3 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 20, 3, [8,16] ],
          [ "LOAD", "n", "ll", 21, 4, [8,21] ],
          [ "LESS_EQUAL", "s", "lll", 22, 20, 21, [8,18] ],
          [ "BRANCH", "n", "l", 22, [8,9] ]
        ],
        "successors": [ 13, 4 ]
      },
      {
        "instructions": [
          [ "JUMP", "n", [9,13] ]
        ],
        "successors": [ 5 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 23, 0, [9,20] ],
          [ "LOAD", "n", "ll", 24, 3, [9,22] ],
          [ "EXTEND", "s", "ll", 25, 24, [9,20] ],
          [ "MOVEPTR", "n", "lllc", 26, 23, 25, 5, [9,20] ],
          [ "LOAD", "n", "ll", 27, 26, [9,20] ],
          [ "LOAD", "n", "ll", 28, 5, [9,27] ],
          [ "LESS", "f", "lll", 29, 27, 28, [9,25] ],
          [ "BRANCH", "n", "l", 29, [9,13] ]
        ],
        "successors": [ 7, 6 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 30, 3, [9,30] ],
          [ "ADD", "s", "llc", 31, 30, 1, [9,30] ],
          [ "STORE", "n", "ll", 3, 31, [9,30] ],
          [ "JUMP", "n", [9,13] ]
        ],
        "successors": [ 5 ]
      },
      {
        "instructions": [
          [ "JUMP", "n", [10,13] ]
        ],
        "successors": [ 8 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 32, 0, [10,20] ],
          [ "LOAD", "n", "ll", 33, 4, [10,22] ],
          [ "EXTEND", "s", "ll", 34, 33, [10,20] ],
          [ "MOVEPTR", "n", "lllc", 35, 32, 34, 5, [10,20] ],
          [ "LOAD", "n", "ll", 36, 35, [10,20] ],
          [ "LOAD", "n", "ll", 37, 5, [10,27] ],
          [ "GREATER", "f", "lll", 38, 36, 37, [10,25] ],
          [ "BRANCH", "n", "l", 38, [10,13] ]
        ],
        "successors": [ 10, 9 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 39, 4, [10,30] ],
          [ "ADD", "s", "llc", 40, 39, 13, [10,30] ],
          [ "STORE", "n", "ll", 4, 40, [10,30] ],
          [ "JUMP", "n", [10,13] ]
        ],
        "successors": [ 8 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 41, 3, [11,17] ],
          [ "LOAD", "n", "ll", 42, 4, [11,22] ],
          [ "LESS_EQUAL", "s", "lll", 43, 41, 42, [11,19] ],
          [ "BRANCH", "n", "l", 43, [11,17] ]
        ],
        "successors": [ 12, 11 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 44, 0, [12,25] ],
          [ "LOAD", "n", "ll", 45, 3, [12,27] ],
          [ "EXTEND", "s", "ll", 46, 45, [12,25] ],
          [ "MOVEPTR", "n", "lllc", 47, 44, 46, 5, [12,25] ],
          [ "LOAD", "n", "ll", 48, 47, [12,25] ],
          [ "STORE", "n", "ll", 6, 48, [12,21] ],
          [ "LOAD", "n", "ll", 49, 0, [12,38] ],
          [ "LOAD", "n", "ll", 50, 4, [12,40] ],
          [ "EXTEND", "s", "ll", 51, 50, [12,38] ],
          [ "MOVEPTR", "n", "lllc", 52, 49, 51, 5, [12,38] ],
          [ "LOAD", "n", "ll", 53, 52, [12,38] ],
          [ "LOAD", "n", "ll", 54, 0, [12,31] ],
          [ "LOAD", "n", "ll", 55, 3, [12,33] ],
          [ "EXTEND", "s", "ll", 56, 55, [12,31] ],
          [ "MOVEPTR", "n", "lllc", 57, 54, 56, 5, [12,31] ],
          [ "STORE", "n", "ll", 57, 53, [12,36] ],
          [ "LOAD", "n", "ll", 58, 6, [12,51] ],
          [ "LOAD", "n", "ll", 59, 0, [12,44] ],
          [ "LOAD", "n", "ll", 60, 4, [12,46] ],
          [ "EXTEND", "s", "ll", 61, 60, [12,44] ],
          [ "MOVEPTR", "n", "lllc", 62, 59, 61, 5, [12,44] ],
          [ "STORE", "n", "ll", 62, 58, [12,49] ],
          [ "LOAD", "n", "ll", 63, 3, [13,17] ],
          [ "ADD", "s", "llc", 64, 63, 1, [13,17] ],
          [ "STORE", "n", "ll", 3, 64, [13,17] ],
          [ "LOAD", "n", "ll", 65, 4, [13,22] ],
          [ "ADD", "s", "llc", 66, 65, 13, [13,22] ],
          [ "STORE", "n", "ll", 4, 66, [13,22] ],
          [ "JUMP", "n", [14,13] ]
        ],
        "successors": [ 12 ]
      },
      {
        "instructions": [
          [ "JUMP", "n", [8,9] ]
        ],
        "successors": [ 3 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 67, 4, [16,13] ],
          [ "LOAD", "n", "ll", 68, 1, [16,17] ],
          [ "SUB", "s", "lll", 69, 67, 68, [16,15] ],
          [ "LOAD", "n", "ll", 70, 2, [16,22] ],
          [ "LOAD", "n", "ll", 71, 3, [16,27] ],
          [ "SUB", "s", "lll", 72, 70, 71, [16,25] ],
          [ "LESS", "s", "lll", 73, 69, 72, [16,20] ],
          [ "BRANCH", "n", "l", 73, [16,13] ]
        ],
        "successors": [ 17, 14 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 74, 1, [16,36] ],
          [ "LOAD", "n", "ll", 75, 4, [16,41] ],
          [ "LESS", "s", "lll", 76, 74, 75, [16,39] ],
          [ "BRANCH", "n", "l", 76, [16,36] ]
        ],
        "successors": [ 16, 15 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 77, 0, [16,47] ],
          [ "LOAD", "n", "ll", 78, 1, [16,50] ],
          [ "LOAD", "n", "ll", 79, 4, [16,54] ],
          [ "CALL", "n", "flll", 2, 77, 78, 79, [16,44] ],
          [ "JUMP", "n", [16,44] ]
        ],
        "successors": [ 16 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 80, 3, [16,63] ],
          [ "STORE", "n", "ll", 1, 80, [16,61] ],
          [ "JUMP", "n", [16,66] ]
        ],
        "successors": [ 20 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 81, 3, [17,20] ],
          [ "LOAD", "n", "ll", 82, 2, [17,24] ],
          [ "LESS", "s", "lll", 83, 81, 82, [17,22] ],
          [ "BRANCH", "n", "l", 83, [17,20] ]
        ],
        "successors": [ 19, 18 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 84, 0, [17,31] ],
          [ "LOAD", "n", "ll", 85, 3, [17,34] ],
          [ "LOAD", "n", "ll", 86, 2, [17,37] ],
          [ "CALL", "n", "flll", 2, 84, 85, 86, [17,28] ],
          [ "JUMP", "n", [17,28] ]
        ],
        "successors": [ 19 ]
      },
      {
        "instructions": [
          [ "LOAD", "n", "ll", 87, 4, [17,47] ],
          [ "STORE", "n", "ll", 2, 87, [17,45] ],
          [ "JUMP", "n", [0,0] ]
        ],
        "successors": [ 20 ]
      },
      {
        "instructions": [
          [ "JUMP", "n", [5,5] ]
        ],
        "successors": [ 1 ]
      },
      {
        "instructions": [
          [ "RET", "n", [19,1] ]
        ],
        "successors": [ ]
      }
    ]
  }
],
"external_variables": [
],
"external_functions": [
]
}
)JSON";

static const char* const SALA_json_2 = R"JSON({
"magic": "sala",
"version": "0.1",
"system": "x86_64-pc-linux-gnu",
"num_cpu_bits": 64,
"name": "test3",
"entry_function": 2,
"constants": [
  "00000000"
],
"static": [
  [ 8, [0,0] ],
  [ 4, [0,0] ]
],
"functions": [
  {
    "name": [ "__sala_static_initializer__", [0,0] ],
    "parameters": [
    ],
    "locals": [
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ss", 0, 1, [0,0] ],
          [ "RET", "n", [0,0] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "using_extern_var", [5,1] ],
    "parameters": [
      [ 8, [5,1] ],
      [ 4, [5,1] ]
    ],
    "locals": [
      [ 8, [0,0] ],
      [ 8, [0,0] ],
      [ 4, [6,11] ],
      [ 4, [6,15] ],
      [ 4, [6,13] ],
      [ 4, [7,10] ],
      [ 4, [0,0] ],
      [ 4, [0,0] ]
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ll", 0, 6, [0,0] ],
          [ "ADDRESS", "n", "ll", 1, 7, [0,0] ],
          [ "STORE", "n", "lp", 0, 1, [0,0] ],
          [ "LOAD", "n", "ll", 2, 0, [6,11] ],
          [ "LOAD", "n", "ls", 3, 0, [6,15] ],
          [ "XOR", "n", "lll", 4, 2, 3, [6,13] ],
          [ "STORE", "n", "ll", 1, 4, [6,7] ],
          [ "LOAD", "n", "ll", 5, 1, [7,10] ],
          [ "STORE", "n", "pl", 0, 5, [7,3] ],
          [ "RET", "n", [7,3] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "main", [10,1] ],
    "parameters": [
      [ 8, [10,1] ]
    ],
    "locals": [
      [ 8, [0,0] ],
      [ 4, [10,38] ],
      [ 4, [10,21] ],
      [ 4, [0,0] ],
      [ 8, [10,38] ],
      [ 8, [10,21] ]
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "ADDRESS", "n", "ll", 0, 3, [0,0] ],
          [ "STORE", "n", "lc", 0, 0, [0,0] ],
          [ "ADDRESS", "n", "ll", 4, 1, [10,38] ],
          [ "CALL", "n", "fl", 3, 4, [10,38] ],
          [ "ADDRESS", "n", "ll", 5, 2, [10,21] ],
          [ "CALL", "n", "fll", 1, 5, 1, [10,21] ],
          [ "STORE", "n", "pl", 0, 2, [10,14] ],
          [ "RET", "n", [10,14] ]
        ],
        "successors": [ ]
      }
    ]
  },
  {
    "name": [ "external_func_test", [0,0] ],
    "parameters": [
      [ 8, [0,0] ]
    ],
    "locals": [
    ],
    "basic_blocks": [
      {
        "instructions": [
          [ "RET", "n", [0,0] ]
        ],
        "successors": [ ]
      }
    ]
  }
],
"external_variables": [
  [ 1, "extern_var" ]
],
"external_functions": [
  3
]
}
)JSON";

const auto get_sala_program = [](const char* const program) -> std::shared_ptr<sala::Program>
{
    auto               P = std::make_shared<sala::Program>();
    std::istringstream iss(program);
    iss >> *P;
    return P;
};

const auto get_sala_json = [](const sala::Program& program) -> std::string
{
    std::ostringstream oss;
    oss << program;
    return oss.str();
};

TEST_CASE("SALA -> IR -> SALA", "[translation chain]")
{
    const auto translation_sala_to_ir = std::make_unique<translation::SalaToIR>();
    const auto translation_ir_to_sala = std::make_unique<translation::IRToSala>();
    SECTION("SALA program 0")
    {
        const auto sala_expected_ptr = get_sala_program(SALA_json_0);
        REQUIRE(sala_expected_ptr != nullptr);
        program::ProgramIR_sptr ir_got;
        REQUIRE_NOTHROW(ir_got = translation_sala_to_ir->translate(sala_expected_ptr));
        REQUIRE(ir_got != nullptr);
        std::shared_ptr<sala::Program> sala_got = nullptr;
        REQUIRE_NOTHROW(sala_got = translation_ir_to_sala->translate(ir_got));
        REQUIRE(sala_got != nullptr);
        CHECK(get_sala_json(*sala_got) == SALA_json_0);
    }

    SECTION("SALA program 1")
    {
        const auto sala_expected_ptr = get_sala_program(SALA_json_1);
        REQUIRE(sala_expected_ptr != nullptr);
        program::ProgramIR_sptr ir_got;
        REQUIRE_NOTHROW(ir_got = translation_sala_to_ir->translate(sala_expected_ptr));
        REQUIRE(ir_got != nullptr);
        std::shared_ptr<sala::Program> sala_got = nullptr;
        REQUIRE_NOTHROW(sala_got = translation_ir_to_sala->translate(ir_got));
        REQUIRE(sala_got != nullptr);
        CHECK(get_sala_json(*sala_got) == SALA_json_1);
    }

    SECTION("SALA program 2")
    {
        const auto sala_expected_ptr = get_sala_program(SALA_json_2);
        REQUIRE(sala_expected_ptr != nullptr);
        program::ProgramIR_sptr ir_got;
        REQUIRE_NOTHROW(ir_got = translation_sala_to_ir->translate(sala_expected_ptr));
        REQUIRE(ir_got != nullptr);
        std::shared_ptr<sala::Program> sala_got = nullptr;
        REQUIRE_NOTHROW(sala_got = translation_ir_to_sala->translate(ir_got));
        REQUIRE(sala_got != nullptr);
        CHECK(get_sala_json(*sala_got) == SALA_json_2);
    }
}
