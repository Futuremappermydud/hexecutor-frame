/* exact legacy V69 provider sha256=9b687661f8a2b5c1511a21b324803aa96692a2c4f4a0f73ad18151562eb94d22 */
/* worker 0x20c960+0x1f0 sha256=970c0b0305f3d750d50e50933bc8a7461271420043d45623b6f34e844cf75894 */
/* helper 0x20cb50+0x32c sha256=a498809d76f6a97258b32167d434a5702ad356c4aceccd6572b285d7078e6b0c */
/* leaf 0x3c8a80+0x2c8 sha256=93314c110fc5cb6e6e41c39db692b17ea8f4ae210f20540e1a0c93fba962a52d */
/* 1xN leaf 0x3c7d00+0x198 sha256=03977cdce58ef5223ee6a30a6a028d5bee099f978a960f159ebcf1378a55a2de */
/* Nx1 leaf 0x3c7f40+0x1b0 sha256=e62d10b915cd2851d8268760105064b4d345a2ef29b2aecc18e24af502245956 */
/* wrapper 0x21bfd4+0x18 sha256=3c8565d86b8696b98e4a1dd69940870db325e8fb8c86bc42c72dc01384ff4908 */
/* setter span 0x3ffaa0+0xe80 sha256=35317b1c47e3b051e9ddcb4096eae846c1c26a8e75a957b28b2a7af9563b48e2 */
.section .text.v69s2,"ax",@progbits
.org 0x12cfe0
.globl qd_v69_s2_zero_stub
qd_v69_s2_zero_stub:
  r0 = #0
  jumpr r31
.org 0x12d1d0
.globl qd_v69_s2_identity_stub
qd_v69_s2_identity_stub:
  jumpr r31
.org 0x20c960
.globl qd_v69_s2_worker
qd_v69_s2_worker:
  .word 0xf5044206
  .word 0xf5014010
  .word 0xebf41da0
  .word 0x70634004
  .word 0xb01d4300
  .word 0x91824048
  .word 0x9180c042
  .word 0x70704001
  .word 0xeac50abe
  .word 0xa1dd5616
  .word 0xa1ddd815
  .word 0xf5065102
  .word 0x91824035
  .word 0x9188c032
  .word 0x70674005
  .word 0x91884056
  .word 0xa19dc500
  .word 0x5a0040d6
  .word 0xa1ddda14
  .word 0x100dc0c6
  .word 0xb01d4304
  .word 0x78004017
  .word 0x02830292
  .word 0xb0044413
  .word 0xb0044814
  .word 0x3c681c79
  .word 0x08250726
  .word 0x8c054525
  .word 0x91834107
  .word 0x9183c0e8
  .word 0x8c064326
  .word 0x8c074527
  .word 0x06220633
  .word 0xed064505
  .word 0x8c08c328
  .word 0xed084706
  .word 0x8c02c322
  .word 0x8c03c323
  .word 0xed064318
  .word 0xed054219
  .word 0x919dc102
  .word 0x8058425a
  .word 0x70724003
  .word 0xb004c112
  .word 0x7503c020
  .word 0x89404003
  .word 0x78004018
  .word 0x755b4000
  .word 0xa1bdd605
  .word 0x89404003
  .word 0x755a4000
  .word 0xa1bdd404
  .word 0x89404003
  .word 0xa1bdd203
  .word 0x919dc083
  .word 0x85434000
  .word 0x74176805
  .word 0x741b67e3
  .word 0x5c20c828
  .word 0x8c034624
  .word 0xf302d703
  .word 0xbfe47fc6
  .word 0x1144e11a
  .word 0x60064108
  .word 0x75844040
  .word 0xf3024504
  .word 0xb005c805
  .word 0x5c20c00a
  .word 0x70648003
  .word 0xf3024504
  .word 0xb0054805
  .word 0x9403c000
  .word 0x70644003
  .word 0x9403c000
  .word 0x9403c000
  .word 0x919dc062
  .word 0x85424000
  .word 0x741a67e3
  .word 0x5c20c82a
  .word 0x8c034624
  .word 0x480500b2
  .word 0xf3024503
  .word 0xb0054805
  .word 0x1144e11e
  .word 0xbfe47fc6
  .word 0x75844040
  .word 0xf3024504
  .word 0xb005c805
  .word 0x60064010
  .word 0x5c20c00c
  .word 0x70648003
  .word 0xf3024504
  .word 0xb0054805
  .word 0x9403c000
  .word 0x70644003
  .word 0x9403c000
  .word 0x9403c000
  .word 0xf5125300
  .word 0x919dc0a2
  .word 0x85424000
  .word 0xfd117082
  .word 0x74146004
  .word 0x5c20d80e
{
  call qd_v69_s2_capture_1xn
  r3:2 = combine(r17,r16)
}
  .word 0x5800c00c
  .word 0x70744004
  .word 0x104ee108
  call qd_v69_s2_capture_nx1
  .word 0x5800c004
  call qd_v69_s2_capture_generic
  .word 0xb0184038
  .word 0x919d4102
  .word 0x3e5dc71a
  .word 0xf218d500
  .word 0xc4194242
  .word 0x5cff6086
  .word 0xa1bdd408
  .word 0x78004000
  .word 0x3ecc1ec5
  .word 0x3ebe1eb7
  .word 0x91dd42b8
  .word 0x91ddc29a
  .word 0x961ec01e
.org 0x20cb50
.globl qd_v69_s2_helper
qd_v69_s2_helper:
  .word 0x8c055826
  .word 0x70604010
  .word 0xebf41ca0
  .word 0x59600237
  .word 0xa1dd5a04
  .word 0xa1ddd805
  .word 0xf5054216
  .word 0x70644005
  .word 0x9181405a
  .word 0xa1ddd606
  .word 0x89404002
  .word 0x91824059
  .word 0xa1ddd208
  .word 0x9187404b
  .word 0xa1ddd407
  .word 0x748b4015
  .word 0x919a4058
  .word 0xa19dc602
  .word 0x91994051
  .word 0xa19dc205
  .word 0x91994074
  .word 0xa19dc504
  .word 0x919a4073
  .word 0xa19dc303
  .word 0x919a408e
  .word 0x9199c092
  .word 0x5c004018
  .word 0x9187403b
  .word 0x9187c064
  .word 0x8c044740
  .word 0xb0064221
  .word 0xe811080a
  .word 0x5be44300
  .word 0x706ec012
  .word 0x7072400e
  .word 0x7075400b
  .word 0x50043c11
  .word 0x919dc012
  .word 0x76174606
  .word 0xbfeb7fe5
  .word 0x919a40e8
  .word 0x9199c0e3
  .word 0xde054116
  .word 0x75064200
  .word 0x919a4107
  .word 0x9199c10c
  .word 0x8c034323
  .word 0x8c084328
  .word 0xb0104800
  .word 0x9199c0c6
  .word 0x8c0c452d
  .word 0x8c074527
  .word 0x760540e5
  .word 0x919ac0c9
  .word 0x8c064326
  .word 0x8c094329
  .word 0x9196406a
  .word 0xa190d207
  .word 0xed034d0f
  .word 0xed084703
  .word 0xa1904a02
  .word 0xa190d405
  .word 0xed094315
  .word 0xa1905106
  .word 0xa190cd03
  .word 0x919d42d2
  .word 0xa190cf04
  .word 0xed064f13
  .word 0x83120b8b
  .word 0x78036001
  .word 0xf50b5b02
  .word 0xa8820a83
  .word 0xa190580c
  .word 0xa190ce0d
  .word 0x5c20400a
  .word 0xa190c709
  .word 0x5a3edc8a
  .word 0x58004008
  .word 0x3c451c34
  .word 0x5a3ed97c
  .word 0x3c451c34
  .word 0x000759b5
  .word 0x6a494202
  .word 0x91844043
  .word 0xa190d21e
  .word 0x03440355
  .word 0x912344e6
  .word 0x9183c0e7
  .word 0x8c06452d
  .word 0x8c074527
  .word 0x912344c8
  .word 0x9183c0a9
  .word 0xed0d4707
  .word 0x7068401c
  .word 0x7068400f
  .word 0x9123c48c
  .word 0x0000400e
  .word 0xde1c4500
  .word 0x912344ae
  .word 0x9183c0c3
  .word 0x8e0845a7
  .word 0x8c064246
  .word 0xa190d30e
  .word 0xaf8d0185
  .word 0xda4fc07c
  .word 0xda46cf8f
  .word 0xef07c90c
  .word 0xef0cc30e
  .word 0x8e4ecac6
  .word 0xf3044611
  .word 0xa1b0d200
  .word 0x0fff7fd5
  .word 0x91824283
  .word 0x0fff7fd5
  .word 0x9182c306
  .word 0x6be14101
  .word 0x91834021
  .word 0x2002c618
  .word 0x5be4415c
  .word 0x0fff7fd5
  .word 0x9182c300
  .word 0x75004001
  .word 0x919dc0a2
  .word 0x85424003
  .word 0x5c20cb0c
  .word 0x1040e092
  .word 0x5800c006
  .word 0x919dc0a2
  .word 0x8542c003
  .word 0x89414002
  .word 0x919042c4
  .word 0x9190c306
  .word 0x7c004002
  .word 0x91904327
  .word 0xa19dc204
  .word 0x75044000
  .word 0x75064001
  .word 0x74846023
  .word 0x9190c2e4
  .word 0x75074002
  .word 0x4490c3b0
  .word 0xb0044025
  .word 0xb0064024
  .word 0x31763940
  .word 0x449045b8
  .word 0x4490c4c1
  .word 0x5c004344
  .word 0x4490c6ca
  .word 0x74024005
  .word 0x74224004
  .word 0x74c64002
  .word 0x919dc049
  .word 0x8c034327
  .word 0x8c054328
  .word 0x75034001
  .word 0x7504c002
  .word 0x8c044323
  .word 0x8c024324
  .word 0xb0094026
  .word 0x7505c000
  .word 0xde06441c
  .word 0x8cd14400
  .word 0x75024003
  .word 0xa1b0d400
  .word 0x00054487
  .word 0x6a494f01
  .word 0x7800c080
  .word 0xed074602
  .word 0xed08c605
  .word 0xed034603
  .word 0xed04c604
  .word 0xda627009
  .word 0xda657009
  .word 0x44b0d4b1
  .word 0xda637009
  .word 0xda647009
  .word 0x4490c5b8
  .word 0x5a3e614a
  .word 0x449043c2
  .word 0x4490c4cb
  .word 0x69004138
  .word 0x5c013c44
  .word 0x85444001
  .word 0x78004024
  .word 0x4790cb83
  .word 0xb0104b03
  .word 0xa190c31c
  .word 0x91834005
  .word 0x2402c012
  .word 0xbfe57005
  .word 0xa1a3d200
  .word 0x9190c385
  .word 0x7065c006
  .word 0xcc04c286
  .word 0x74a5c006
  .word 0xa190c61c
  .word 0x7f008000
  .word 0x20433122
  .word 0x3e4c1e45
  .word 0x3e3e1e37
  .word 0x91dd40b8
  .word 0x91ddc09a
  .word 0x961ec01e
.org 0x21bfd4
.globl qd_v69_s2_wrapper
qd_v69_s2_wrapper:
  .word 0xa09dc000
  .word 0x02020404
  .word 0x01010303
  .word 0x5bfe44c0
  .word 0x30403c02
  .word 0x961ec01e
.org 0x3c7d00
.globl qd_v69_s2_leaf_1xn
qd_v69_s2_leaf_1xn:
  .word 0xbffd771d
  .word 0x78037fed
  .word 0x91c44026
  .word 0x91c4c048
  .word 0x91c4400a
  .word 0x91c4c06e
  .word 0xf1064d10
  .word 0xf1084d11
  .word 0xea040a0d
  .word 0xf10a4d13
  .word 0xea160a1f
  .word 0x78005c0d
  .word 0x01140315
  .word 0x8c044124
  .word 0xc78a4d02
  .word 0xa1dd5804
  .word 0xa1ddda05
  .word 0xbfe47fe4
  .word 0x021d0011
  .word 0x75444001
  .word 0x91804039
  .word 0x9180c077
  .word 0x8c194259
  .word 0xf4174545
  .word 0x918040b8
  .word 0x9180c054
  .word 0x80544254
  .word 0xb00540e5
  .word 0x040e0000
  .word 0x8c054325
  .word 0xa19d5e0c
  .word 0xa19dc111
  .word 0xb0036003
  .word 0xbff87c18
  .word 0xb000409a
  .word 0x9203c3fe
  .word 0x7800401c
  .word 0x7f004000
  .word 0x7f004000
  .word 0xa19dda10
  .word 0x60254200
  .word 0xb01c411c
  .word 0x7800401e
  .word 0xa6e0c011
  .word 0x6b634303
  .word 0x7f004000
  .word 0x7f004000
  .word 0xa1ddc007
  .word 0x60044200
  .word 0x7062400c
  .word 0x9b81405a
  .word 0x9181c03b
  .word 0xf1305a06
  .word 0xf1315b08
  .word 0x5c20411c
  .word 0x9180c00d
  .word 0x9b81405a
  .word 0x9181c03b
  .word 0xe20e402c
  .word 0xf1305a06
  .word 0x920647ed
  .word 0x920ceee0
  .word 0xe20f802c
  .word 0xf1315b08
  .word 0x920849ed
  .word 0x920cefe0
  .word 0xe20e402c
  .word 0xf1334d0a
  .word 0x920647ed
  .word 0x920ceee0
  .word 0xe20f402c
  .word 0x5c00421c
  .word 0x920849ed
  .word 0x920cefe0
  .word 0x5c20c304
  .word 0xa6f2cb0d
  .word 0xf2575e03
  .word 0x5c204b16
  .word 0xb01ec11e
  .word 0x706a4012
  .word 0xf2579e00
  .word 0xfb007900
  .word 0xa6eacb01
  .word 0x5800c008
  .word 0xf3005900
  .word 0x7f008000
  .word 0xa6eacb05
  .word 0xf2565c00
  .word 0x91ddc0e0
  .word 0xd30054a0
  .word 0x5cdfe09e
  .word 0x706c4002
  .word 0x75584000
  .word 0x5cdf788a
  .word 0x91ddc100
  .word 0x919dc19e
  .word 0x91dd40ba
  .word 0x91ddc098
  .word 0x3e1f1e16
  .word 0xb01d491d
  .word 0x529f4000
  .word 0x3e0d1e04
.org 0x3c7f40
.globl qd_v69_s2_leaf_nx1
qd_v69_s2_leaf_nx1:
  .word 0xbffd781d
  .word 0x78037fef
  .word 0x91c44026
  .word 0x9184c148
  .word 0xf1064f10
  .word 0xf1084f11
  .word 0xea040a0d
  .word 0xf1074f0e
  .word 0xea160a1f
  .word 0x918440ca
  .word 0x9184c10b
  .word 0x91c4400c
  .word 0xa1ddd804
  .word 0x7f004000
  .word 0x7f004000
  .word 0x0114021b
  .word 0x8c044124
  .word 0x04150011
  .word 0xf10c4f0f
  .word 0xbfe47fe4
  .word 0x918040bc
  .word 0x9180c036
  .word 0x8c164256
  .word 0x7f004000
  .word 0x020a030d
  .word 0x80524252
  .word 0xb01540f5
  .word 0x040c0000
  .word 0x62364007
  .word 0x8c154315
  .word 0xa1dd5a05
  .word 0xa19dc10f
  .word 0xb0036003
  .word 0xbffc7c1c
  .word 0xb000409b
  .word 0x9203c3fe
  .word 0x7800411a
  .word 0x7f004000
  .word 0x7f004000
  .word 0xa19ddb0e
  .word 0x60354300
  .word 0x7800401b
  .word 0x78036017
  .word 0xa6e0c011
  .word 0xc7a65701
  .word 0xc7a85702
  .word 0xf2455a01
  .word 0xf245da02
  .word 0x7c004018
  .word 0x7f004000
  .word 0x7f004000
  .word 0xa1ddc006
  .word 0x60044300
  .word 0x75444000
  .word 0x9b814037
  .word 0x3081d338
  .word 0xfb375838
  .word 0xf1305706
  .word 0x9b814037
  .word 0x3081d359
  .word 0xf12e5807
  .word 0x70624009
  .word 0x5c204028
  .word 0x7f00c000
  .word 0xfb375959
  .word 0xf1315708
  .word 0x9b814037
  .word 0x3081d338
  .word 0xe20a4029
  .word 0xf12e5907
  .word 0x920647ed
  .word 0x9209eae0
  .word 0xfb375838
  .word 0xf1305706
  .word 0x9b814037
  .word 0x3081d359
  .word 0xe20b8029
  .word 0xf12e5807
  .word 0x920847ed
  .word 0x9209ebe0
  .word 0xfb375959
  .word 0xf1315708
  .word 0x9d80e017
  .word 0xe20a4029
  .word 0xf12e5907
  .word 0x920647ed
  .word 0x9209eae0
  .word 0xf12f570c
  .word 0x920847ed
  .word 0x9209ebe0
  .word 0xe20b4029
  .word 0x7f008000
  .word 0xa6eccd05
  .word 0xf2545a00
  .word 0xb01a411a
  .word 0x91ddc0c0
  .word 0xd30052a0
  .word 0x5cdfe090
  .word 0x70694002
  .word 0x755c4000
  .word 0x5cdf787c
  .word 0x91ddc0e0
  .word 0x91dd40ba
  .word 0x91ddc098
  .word 0x3e1f1e16
  .word 0xb01d481d
  .word 0x529f4000
  .word 0x3e0d1e04
.org 0x3c8a80
.globl qd_v69_s2_generic_leaf
qd_v69_s2_generic_leaf:
  .word 0xbffd761d
  .word 0x78037fef
  .word 0x91c44026
  .word 0x91c4c048
  .word 0xf1064f06
  .word 0xea040a0d
  .word 0x7f004000
  .word 0x7f004000
  .word 0xea160a1f
  .word 0x7f004000
  .word 0x7f004000
  .word 0xa1dd5804
  .word 0xa1ddda05
  .word 0x78005c1c
  .word 0x91c4400a
  .word 0x91c4c06c
  .word 0xc78a5c03
  .word 0x91c44092
  .word 0x91c4c0b0
  .word 0x7f004000
  .word 0x7f004000
  .word 0x0114021f
  .word 0xbfe47f84
  .word 0x0315041c
  .word 0x8c044224
  .word 0x91814001
  .word 0xa1ddde06
  .word 0x91804079
  .word 0x9180c035
  .word 0x8c154255
  .word 0xf4194565
  .word 0x918040ba
  .word 0x9180c056
  .word 0x80564256
  .word 0xb00540e5
  .word 0x91804098
  .word 0x9180c000
  .word 0x60044818
  .word 0x8c054325
  .word 0xe911092e
  .word 0xb0036003
  .word 0xbffa7c1a
  .word 0xb000409f
  .word 0x9203c3fe
  .word 0x7c04401e
  .word 0xa19d5f10
  .word 0xa19dda13
  .word 0x60254208
  .word 0x7803601c
  .word 0xa6e0c011
  .word 0xc7a65c01
  .word 0xc7b05c02
  .word 0xf2545e01
  .word 0xf254de02
  .word 0x7c00401a
  .word 0xa1ddc007
  .word 0xf1074f07
  .word 0xf1084f08
  .word 0x9b81403c
  .word 0x3081d73a
  .word 0xfb3c5a3a
  .word 0xf1094f09
  .word 0x7062400e
  .word 0x7544c000
  .word 0xf1265c06
  .word 0xf1275a07
  .word 0x9b81403c
  .word 0x3081d73a
  .word 0xfb3c5a3a
  .word 0x5c204058
  .word 0x920647ed
  .word 0x920eece0
  .word 0xe20c402e
  .word 0xf1285c08
  .word 0xf1295a09
  .word 0xf110cf10
  .word 0xf1074f07
  .word 0xf1114f11
  .word 0x9b81403c
  .word 0x3081d75b
  .word 0xe20d402e
  .word 0xfb3c5b5b
  .word 0x920849ed
  .word 0x920eede0
  .word 0xf1305c10
  .word 0xf1275b07
  .word 0x9b81403c
  .word 0x3081d75b
  .word 0xf1094f09
  .word 0xfb3c5b5b
  .word 0x921047ed
  .word 0x920ef2e0
  .word 0xe212402e
  .word 0xf1315c11
  .word 0xf1295b09
  .word 0xf106cf06
  .word 0xf1074f07
  .word 0xf1084f08
  .word 0x9b81403c
  .word 0x3081d73a
  .word 0xe213402e
  .word 0xfb3c5a3a
  .word 0x921149ed
  .word 0x920ef3e0
  .word 0xf1265c06
  .word 0xf1275a07
  .word 0x9b81403c
  .word 0x3081d73a
  .word 0xfb3c9a3a
  .word 0xf1094f09
  .word 0x920647ed
  .word 0x920eece0
  .word 0xe20c402e
  .word 0xf1285c08
  .word 0xf1295a09
  .word 0xf110cf10
  .word 0xf1074f07
  .word 0xf1114f11
  .word 0x9b81403c
  .word 0x3081d75b
  .word 0xe20d402e
  .word 0xfb3c5b5b
  .word 0x920849ed
  .word 0x920eede0
  .word 0xf1305c10
  .word 0xf1275b07
  .word 0x9b81403c
  .word 0x3081d75b
  .word 0xf1094f09
  .word 0xfb3c5b5b
  .word 0x921047ed
  .word 0x920ef2e0
  .word 0xe212402e
  .word 0xf1315c11
  .word 0xf1295b09
  .word 0xf106cf06
  .word 0x60045000
  .word 0xf10a4f0a
  .word 0x5c004322
  .word 0x9180c01c
  .word 0x619f400a
  .word 0xf12a5c0a
  .word 0x921149ed
  .word 0x920ef3e0
  .word 0xa6f6cb0d
  .word 0xf2595f00
  .word 0x5c20481c
  .word 0xb01fc11f
  .word 0x706a4016
  .word 0xf2599f00
  .word 0xfb007500
  .word 0xa6eacb01
  .word 0x5800c00e
  .word 0xf12a5c0a
  .word 0x921149ed
  .word 0x920ef3e0
  .word 0xf3005500
  .word 0x7f008000
  .word 0xa6eacb05
  .word 0xe213402e
  .word 0xf2585e00
  .word 0x3d2e1e38
  .word 0xd30056a0
  .word 0xb01e411e
  .word 0x7e00401f
  .word 0x5cdfe01a
  .word 0x91dd4100
  .word 0x919dc27a
  .word 0x706e4002
  .word 0x755a4000
  .word 0x5cdff800
  .word 0x91ddc0de
  .word 0x91dd40ba
  .word 0x91ddc098
  .word 0x3e1f1e16
  .word 0xb01d4a1d
  .word 0x529f4000
  .word 0x3e0d1e04
.org 0x3ffaa0
.globl qd_v69_s2_setter_span
qd_v69_s2_setter_span:
  .word 0x8d014147
  .word 0xbfe47fe9
  .word 0x758443e1
  .word 0xa09dc005
  .word 0x8d0146a4
  .word 0x8c094229
  .word 0x7601418f
  .word 0xa1ddd004
  .word 0x85a14c00
  .word 0x8c03415c
  .word 0x7e2040e9
  .word 0x919dc1a6
  .word 0x8d264508
  .word 0xb007402c
  .word 0x919d418e
  .word 0xa1ddd203
  .word 0xf32e420d
  .word 0xa1ddd402
  .word 0x70694002
  .word 0x75884043
  .word 0x7ee063e7
  .word 0x5c00db12
  .word 0x750cc041
  .word 0xde824704
  .word 0x5c20c10c
  .word 0x75084041
  .word 0x5c20c908
  .word 0x8c084542
  .word 0xbfe8ffe7
  .word 0xed024c0a
  .word 0x750f4001
  .word 0x7483401c
  .word 0x7583c082
  .word 0x8c044542
  .word 0xbfed7ff2
  .word 0x78df7fef
  .word 0x750ec000
  .word 0x89414010
  .word 0x89424014
  .word 0xa1bdd403
  .word 0x5c00416e
  .word 0x5c20426e
  .word 0xa19dd402
  .word 0xed0a4d10
  .word 0x75044601
  .word 0x75044402
  .word 0x7302e0f4
  .word 0x8e4942d5
  .word 0x70644011
  .word 0x78df7fed
  .word 0x7a81e08b
  .word 0x8e1042cd
  .word 0x7e4040ab
  .word 0xa1a0d406
  .word 0xcc524b91
  .word 0xc69f4bf3
  .word 0x764ec112
  .word 0xed0a4e0e
  .word 0x7613c0c9
  .word 0xf1695414
  .word 0x76954069
  .word 0xa1a0d203
  .word 0xf131540d
  .word 0x4480c928
  .word 0x8c0d454d
  .word 0x5c004014
  .word 0xa1a0d402
  .word 0xc652cb8a
  .word 0xef4a4474
  .word 0x78dfffea
  .word 0x8e0e42ca
  .word 0xa1a0d207
  .word 0x8c14454a
  .word 0xa1a0d204
  .word 0xb01c4efc
  .word 0x007f7fff
  .word 0x780047ea
  .word 0xbfe3ff83
  .word 0xef53ca5c
  .word 0xe303500f
  .word 0xf1315c0a
  .word 0xa1a0d408
  .word 0x8c0a454a
  .word 0x5c004060
  .word 0xa1a0d40a
  .word 0xe3034e0f
  .word 0xc6524b8a
  .word 0xa1a0d409
  .word 0xef4ac47c
  .word 0x8c1c4543
  .word 0xa1a0d20b
  .word 0x919dc063
  .word 0x85434002
  .word 0x5c00c362
  .word 0x5800c056
  .word 0x75044602
  .word 0x75044401
  .word 0x007f7fff
  .word 0x7302e7f0
  .word 0xed0a4d0d
  .word 0x8e4942d1
  .word 0xb01c4ffc
  .word 0x7e40c08b
  .word 0x7ec0c06b
  .word 0x7e20c0ab
  .word 0xc69f4bf3
  .word 0xc652cb89
  .word 0xef53505c
  .word 0xe3034d0f
  .word 0xa1a0d206
  .word 0x707cc010
  .word 0xef494470
  .word 0x76914069
  .word 0xa1a0d203
  .word 0x8c10454d
  .word 0x5c004028
  .word 0xa1a0d402
  .word 0xed0a4e0a
  .word 0x764e410e
  .word 0xa180c905
  .word 0xc64ecb8e
  .word 0xef4e447c
  .word 0xe3034a0f
  .word 0xa1a0d207
  .word 0x8c1c4543
  .word 0xa1a0d204
  .word 0x919dc063
  .word 0x85434002
  .word 0x5c00c31c
  .word 0x5800c010
  .word 0x919dc063
  .word 0x85434002
  .word 0x5c00c314
  .word 0x5800c008
  .word 0x919dc063
  .word 0x85434002
  .word 0x5c00c30c
  .word 0x750c4041
  .word 0x5c20c93c
  .word 0x75084041
  .word 0x5c20c938
  .word 0xda697c07
  .word 0x780043e3
  .word 0x45804088
  .word 0xa1a0d603
  .word 0xf1674303
  .word 0x4580c0ac
  .word 0xf12d430d
  .word 0x5c00400e
  .word 0xa1a0d402
  .word 0xda687c03
  .word 0xda6c7c07
  .word 0xa1a0d404
  .word 0xa180cc05
  .word 0x5c004216
  .word 0x459dd047
  .word 0x85474001
  .word 0x5c204912
  .word 0x4380c947
  .word 0xda677c03
  .word 0x5c00400c
  .word 0xa1a0d40a
  .word 0x9180c167
  .word 0xda677c03
  .word 0xa1a0d20b
  .word 0x0fff7f83
  .word 0x7606c7e6
  .word 0x8c054543
  .word 0x85014001
  .word 0xac060102
  .word 0x8d034b06
  .word 0x5c204164
  .word 0xa1a0d400
  .word 0x8d0d42a6
  .word 0xde6d62c0
  .word 0x762d7007
  .word 0x7629f00c
  .word 0x8d0942a8
  .word 0xde69e2c0
  .word 0xefcd4706
  .word 0xa1a0d202
  .word 0xefc94c08
  .word 0x5c00401c
  .word 0xa1a0d403
  .word 0x04060507
  .word 0x8d0642a9
  .word 0xde6662c0
  .word 0x7627700d
  .word 0x7626f00c
  .word 0x8d0742a8
  .word 0xde67e2c0
  .word 0xefc64c09
  .word 0xa1a0d204
  .word 0xefc74d08
  .word 0xa1a0d205
  .word 0x5c004222
  .word 0x459dd046
  .word 0x85464001
  .word 0x5c20491e
  .word 0x4380c946
  .word 0x8d0642a7
  .word 0xde6662c0
  .word 0x7626f008
  .word 0xefc64807
  .word 0x5c004012
  .word 0x45804166
  .word 0xa1a0d60a
  .word 0x8d0642a7
  .word 0xde6662c0
  .word 0x7626f008
  .word 0xefc64807
  .word 0xa1a0d20b
  .word 0x0000401e
  .word 0x76034003
  .word 0x7604c064
  .word 0xda454063
  .word 0x0000401e
  .word 0xda424004
  .word 0xa1a0d400
  .word 0xa180c201
  .word 0x3e241e1d
  .word 0x3e161f40
.globl qd_v69_s2_3ffe18
qd_v69_s2_3ffe18:
  .word 0x8d014148
  .word 0x8c054526
  .word 0xb00243e9
  .word 0x7582c402
  .word 0x8d064567
  .word 0x85054501
  .word 0x7605440c
  .word 0xa09dc001
  .word 0x8d094342
  .word 0x8d0146a6
  .word 0xb0084029
  .word 0x7587c040
  .word 0x6b014203
  .word 0x5c00c026
  .word 0x89434008
  .word 0x75094043
  .word 0xa1bdd401
  .word 0x70624008
  .word 0x750c4001
  .word 0x7800c3ec
  .word 0xde88c704
  .word 0xed08c908
  .word 0x8c08c14d
  .word 0x00004020
  .word 0x7e40c00d
  .word 0x74ad4008
  .word 0x5c20437e
  .word 0x419dd82d
  .word 0x854d4003
  .word 0x7507c041
  .word 0x5c20c108
  .word 0xed094708
  .word 0xbfe7ffec
  .word 0x8c08c548
  .word 0x8c02424d
  .word 0x75064601
  .word 0x76444004
  .word 0x7506c402
  .word 0x8c064542
  .word 0x7066400e
  .word 0x760440e4
  .word 0x7a81e08f
  .word 0x7e60438d
  .word 0x7e40c0af
  .word 0xcc444f8e
  .word 0xbfe87fe4
  .word 0xa1a0d206
  .word 0xf122cd04
  .word 0x8c0e4548
  .word 0x76844064
  .word 0xa1a0d203
  .word 0x5c00400a
  .word 0xa180c802
  .word 0x75094040
  .word 0x5c20c816
  .word 0x1047c212
  .word 0x780043e7
  .word 0xf12c4204
  .word 0x0fff7f83
  .word 0x7605c7e5
  .word 0xefcc4867
  .word 0xa180c403
  .word 0xa180c702
  .word 0x7067c008
  .word 0x8c034547
  .word 0x85014000
  .word 0xac050102
  .word 0x8d074b05
  .word 0x5c204028
  .word 0xa1a0d400
  .word 0x8d0842a5
  .word 0xde6862c0
  .word 0x76287009
  .word 0x7624f00d
  .word 0x8d0442ac
  .word 0xde6462c0
  .word 0x0000401e
  .word 0x7607c007
  .word 0xda434067
  .word 0xefc84905
  .word 0x7606c066
  .word 0x0000401e
  .word 0xda424006
  .word 0xa0030205
  .word 0xefc44d0c
  .word 0xa180c201
  .word 0xa180cc03
  .word 0x961ec01e
  .word 0x919dc02d
  .word 0x854d4003
  .word 0x59ffff92
  .word 0x8d014147
  .word 0xbfe47fe6
  .word 0x758443e0
  .word 0xb002c028
  .word 0x8c064229
  .word 0x8c034126
  .word 0xb003402e
  .word 0x7800c184
  .word 0x8d0146a4
  .word 0xc7414400
  .word 0x7e0040e9
  .word 0xb007c0ef
  .word 0x8c0e412c
  .word 0x8c084128
  .word 0xb009c02d
  .word 0x8c024127
  .word 0xc64d4f8d
  .word 0x7486400f
  .word 0x5c00c014
  .word 0x7601c182
  .word 0x706c4002
  .word 0x1042c812
  .word 0x8cce4022
  .word 0x8cc3402f
  .word 0x5800c00e
  .word 0x8cce4022
  .word 0x8cc3c023
  .word 0xbfe27fe2
  .word 0xbfe3ffef
  .word 0x75044600
  .word 0xb0024ffc
  .word 0xebf41c30
  .word 0x8c044542
  .word 0xbfe87fee
  .word 0x007f7fff
  .word 0x7304e7ea
  .word 0x7a016090
  .word 0x75044400
  .word 0x70232a0d
  .word 0x8e4942c3
  .word 0x7e0040b0
  .word 0x304a3379
  .word 0xcc4e508b
  .word 0xc69f50e9
  .word 0xb00f4fef
  .word 0xa1ddd400
  .word 0xcc515092
  .word 0xed0d4c13
  .word 0xf1694a09
  .word 0x78dffff4
  .word 0xed0d4810
  .word 0x85014000
  .word 0xf10f490f
  .word 0xf11cc90e
  .word 0xf12e4b09
  .word 0xf12f4b0c
  .word 0xf12e520e
  .word 0xf12fd20f
  .word 0xed0d4612
  .word 0xe3075314
  .word 0x70734011
  .word 0x7683c06b
  .word 0xe3065014
  .word 0x8c054546
  .word 0xa1804b03
  .word 0xa180cb05
  .word 0xe3085114
  .word 0x8c09455c
  .word 0xa1a0d202
  .word 0xe3075214
  .word 0x8c0c454a
  .word 0xa6090708
  .word 0x8c0e454d
  .word 0x8c0f4548
  .word 0xa180ca04
  .word 0x8d064b07
  .word 0xa1804d0a
  .word 0xa180c80b
  .word 0xa80b090a
  .word 0xa0070102
  .word 0xde14300c
  .word 0x3e0d1e06
  .word 0x5c204038
  .word 0x901ec01e
  .word 0x8d0b42a7
  .word 0x8c0b4241
  .word 0x763c701c
  .word 0x762af00a
  .word 0x762d700d
  .word 0x76287008
  .word 0x0000401e
  .word 0x7606c006
  .word 0xda637007
  .word 0xda49407c
  .word 0x76044064
  .word 0xa1a0d402
  .word 0xda4c406a
  .word 0xda4e406d
  .word 0xa1a0d404
  .word 0xda414f83
  .word 0xda4f4068
  .word 0xa180ce0a
  .word 0xda454066
  .word 0xa1804f0b
  .word 0xa180c103
  .word 0x0000401e
  .word 0xda424004
  .word 0xa180c500
  .word 0x529f4000
  .word 0xa1020501
  .word 0x529fc000
  .word 0x8d014148
  .word 0xbfe47fe7
  .word 0x0000401f
  .word 0x7601c406
  .word 0xed034209
  .word 0x8c074227
  .word 0x758443e0
  .word 0x7066c004
  .word 0x8c054543
  .word 0x7e0040e7
  .word 0xb00840e8
  .word 0x3c40c100
  .word 0x8e4742c4
  .word 0x85014000
  .word 0xb007c022
  .word 0x8d034b0e
  .word 0xc6424887
  .word 0x919d4002
  .word 0xa1a0d600
  .word 0x76844062
  .word 0xbfe77fef
  .word 0xa180420c
  .word 0x3840c100
  .word 0x0fff7fff
  .word 0xd76967e7
  .word 0xa3020106
  .word 0x5c204022
  .word 0xa1804f07
  .word 0xa180c706
  .word 0x8d0242a7
  .word 0x8d0642a6
  .word 0x0000401e
  .word 0x7603c003
  .word 0xda454063
  .word 0x8c02c242
  .word 0xda647007
  .word 0x0000401e
  .word 0xda414006
  .word 0xa180c500
  .word 0xa180c101
  .word 0xda424f84
  .word 0x529f4000
  .word 0xa1a0d403
  .word 0x529fc000
  .word 0xed024304
  .word 0x8c054543
  .word 0x0000401f
  .word 0x7601c402
  .word 0x85014000
  .word 0x768243e7
  .word 0x919d4006
  .word 0xa1a0d403
  .word 0x8d034b07
  .word 0x0fff7fff
  .word 0xde2469ec
  .word 0x3c40c100
  .word 0xa1804700
  .word 0x00004007
  .word 0x3c40c3bf
  .word 0xa1020c06
  .word 0x5c20401e
  .word 0xa1804406
  .word 0x3840c100
  .word 0x8d0242a4
  .word 0x0000401e
  .word 0x7603c003
  .word 0xda454063
  .word 0xa1a0d200
  .word 0x0000401e
  .word 0xda414004
  .word 0xf1224402
  .word 0xa1a0d401
  .word 0x76824f82
  .word 0x529f4000
  .word 0xa1a0d403
  .word 0x529fc000
.globl qd_v69_s2_400270
qd_v69_s2_400270:
  .word 0x8d0146a6
  .word 0x85a14c01
  .word 0x78df7fec
  .word 0xa09dc002
  .word 0x008041ff
  .word 0x780047ce
  .word 0x75834080
  .word 0xa1ddd001
  .word 0x8c064547
  .word 0x7426400a
  .word 0x919d40c8
  .word 0xa1ddd200
  .word 0x74a7401c
  .word 0x919dc0e9
  .word 0xf328420d
  .word 0x74264002
  .word 0x5c20c154
  .word 0x8c03415c
  .word 0x75064402
  .word 0xbfed7fe4
  .word 0x7e00c0f0
  .word 0x74a3401c
  .word 0x75064601
  .word 0x5c00c0a2
  .word 0xb01c4fef
  .word 0xb80e403c
  .word 0x75064402
  .word 0x7a81e08e
  .word 0xde0265ea
  .word 0x8c034943
  .word 0x75084001
  .word 0x7e40c0ae
  .word 0xc69f4eea
  .word 0xc6444e84
  .word 0xa1804203
  .word 0x4480c229
  .word 0xef4adc4f
  .word 0x706fc01c
  .word 0xef44467c
  .word 0x7063c004
  .word 0xe30d440c
  .word 0xa1a0d206
  .word 0x8c1c4544
  .word 0x5c004114
  .word 0xa1a0d402
  .word 0xe308430c
  .word 0x7648c10d
  .word 0xc64d4e8d
  .word 0xa180c307
  .word 0xef4dc66f
  .word 0x8c0f4543
  .word 0xa1a0d204
  .word 0x0000401f
  .word 0xde254500
  .word 0x580040c0
  .word 0xa180c90c
  .word 0x8d014142
  .word 0x75064402
  .word 0x758443e1
  .word 0xbfe4ffe4
  .word 0x8c044224
  .word 0xb002402b
  .word 0x7648400a
  .word 0xb80ec02e
  .word 0x70664010
  .word 0x7e2040e4
  .word 0x75064601
  .word 0xbfedffed
  .word 0x8e4442dc
  .word 0xde844704
  .word 0x760a40ea
  .word 0x7a81e092
  .word 0x00004002
  .word 0xdb857123
  .word 0x7e4040b2
  .word 0xb005c02f
  .word 0xed044b0b
  .word 0x769c4062
  .word 0x76657ffc
  .word 0xf263cf01
  .word 0xc69f52e4
  .word 0xf31c431c
  .word 0xa3020502
  .word 0xcc4d5290
  .word 0xc64a528e
  .word 0xf164ce04
  .word 0xef445170
  .word 0x8cc94603
  .word 0xf104c505
  .word 0xef4e4665
  .word 0x74a3c009
  .word 0xa180c90c
  .word 0x8c054544
  .word 0x8c104545
  .word 0xa1a0d204
  .word 0xe31c4b0c
  .word 0x706b4005
  .word 0xa180c402
  .word 0xa180cb07
  .word 0xe30f450c
  .word 0xa1a0d206
  .word 0x1605c05e
  .word 0x7a81e08f
  .word 0xde0265ea
  .word 0x75084001
  .word 0x7e4040af
  .word 0xa1a0d603
  .word 0xcc444f8a
  .word 0xc69f4feb
  .word 0x78df7fe4
  .word 0x4480c229
  .word 0x8e0d4bc4
  .word 0x760b40d1
  .word 0xa1a0d406
  .word 0xf1715011
  .word 0x7648c110
  .word 0xf12ad112
  .word 0x8c124544
  .word 0x5c004114
  .word 0xa1a0d402
  .word 0xc650cf92
  .word 0xef524671
  .word 0x78dffff2
  .word 0x8e084bd2
  .word 0xa1a0d207
  .word 0x8c114551
  .word 0xa1a0d204
  .word 0x0fff7fe0
  .word 0xde034904
  .word 0xb01c4efc
  .word 0xb80ec02e
  .word 0xef4bce5c
  .word 0xf12a5c0e
  .word 0x7063c00a
  .word 0x8c0e454e
  .word 0xa1a0d20a
  .word 0xe30d4a0c
  .word 0x5c004112
  .word 0xa1a0d408
  .word 0xe308430c
  .word 0xc6504f8d
  .word 0xa1a0d409
  .word 0xef4dc67c
  .word 0x8c1c4543
  .word 0xa1a0d20b
  .word 0x0000401f
  .word 0xde254500
  .word 0xa180c90c
  .word 0x85014001
  .word 0xa0050107
  .word 0x3e0c1e05
  .word 0x5c204162
  .word 0x901ec01e
  .word 0x8d0442ab
  .word 0x8d0242ac
  .word 0x7622700e
  .word 0x9180c089
  .word 0x8d0942ad
  .word 0xde6262c0
  .word 0x76014183
  .word 0x7508c001
  .word 0xefc24e0c
  .word 0xde6962c0
  .word 0x7629700f
  .word 0x4580c8a2
  .word 0xda64700b
  .word 0xefc94f0d
  .word 0xa1a0d402
  .word 0xa1804c03
  .word 0xa180cd04
  .word 0x5c00c10c
  .word 0x8d0242a4
  .word 0xde6262c0
  .word 0x7622f008
  .word 0xefc24804
  .word 0xa1a0d205
  .word 0x75034002
  .word 0x5c00ca20
  .word 0x5c20401c
  .word 0x4180c142
  .word 0x8d0242a3
  .word 0xde6262c0
  .word 0x7622f004
  .word 0xefc24403
  .word 0x5c004112
  .word 0x45804962
  .word 0xa1a0d60a
  .word 0x8d0242a3
  .word 0xde6262c0
  .word 0x7622f004
  .word 0xefc24403
  .word 0xa1a0d20b
  .word 0x8d0542a2
  .word 0x7606c063
  .word 0x0000401e
  .word 0xda474003
  .word 0xa1a0d201
  .word 0x0000401e
  .word 0xda454002
  .word 0xa1a0d200
  .word 0x529fc000
  .word 0x8d0146a6
  .word 0x85a14c00
  .word 0x7a073c20
  .word 0x74034028
  .word 0x7402402c
  .word 0xa1ddd001
  .word 0x5c20401a
  .word 0xa1ddd200
  .word 0x8c034124
  .word 0x8cc3c02d
  .word 0x8c024122
  .word 0x8c08c123
  .word 0x8cc8402e
  .word 0x8c0c4128
  .word 0x76014189
  .word 0x2442ec82
  .word 0xbfee7fec
  .word 0xbfed7fe9
  .word 0x5800c084
  .word 0x758443e0
  .word 0xbfe47fe4
  .word 0x75064601
  .word 0x7603c028
  .word 0x8d014145
  .word 0x8c044224
  .word 0x7e20408e
  .word 0x7068c00a
  .word 0x8c02412d
  .word 0x7e0040e4
  .word 0xb002402c
  .word 0x7ea0c06e
  .word 0x8c0c413c
  .word 0x75064400
  .word 0xb00540e2
  .word 0xb004c02f
  .word 0x8e0341aa
  .word 0x70664005
  .word 0x7e0040ae
  .word 0xbfedffec
  .word 0xcc4c4e85
  .word 0xc64f428f
  .word 0x70664009
  .word 0xbffcffec
  .word 0xcc4c4e89
  .word 0x8c064542
  .word 0xf125480b
  .word 0xbfeaffec
  .word 0xc69f4ee8
  .word 0x007f7fff
  .word 0x2bf83023
  .word 0xed0f4c0e
  .word 0x8e4442c3
  .word 0x764a4124
  .word 0x706fc00a
  .word 0xef485044
  .word 0xe31c4a07
  .word 0x7683c068
  .word 0xe30d4f07
  .word 0xa1a0d209
  .word 0xa1804803
  .word 0xa180ca08
  .word 0xf1244903
  .word 0xf1244504
  .word 0x706e4005
  .word 0xa180c805
  .word 0xe30d4e07
  .word 0x8c03c54c
  .word 0xe31c4507
  .word 0x8c094543
  .word 0xa1804c02
  .word 0xa180ce07
  .word 0x8c044544
  .word 0x8c0b4549
  .word 0x68050605
  .word 0xa4040a03
  .word 0x58004070
  .word 0xa180c90b
  .word 0x75094100
  .word 0x7063400c
  .word 0x7064c009
  .word 0x740e400c
  .word 0x740dc009
  .word 0x8d085708
  .word 0x8d02571c
  .word 0xbfe87fed
  .word 0x7506c600
  .word 0x7e00408f
  .word 0x75064401
  .word 0xbfe27ff0
  .word 0x7066c00e
  .word 0x8c084951
  .word 0x8c1c495c
  .word 0xb00c4fec
  .word 0x7e80c06f
  .word 0x007f7fff
  .word 0x730667ea
  .word 0x70714012
  .word 0x7e20c0af
  .word 0xe3045107
  .word 0xcc4d4f8e
  .word 0xb0094fe9
  .word 0xa1a0d607
  .word 0xe3045c07
  .word 0xcc504f8b
  .word 0x707cc011
  .word 0xe3035207
  .word 0xc69f4fe4
  .word 0xa180dc09
  .word 0xe3035107
  .word 0x8c064542
  .word 0xf164ca04
  .word 0xf10c440c
  .word 0xf109c404
  .word 0x0000401f
  .word 0xde254500
  .word 0xf12c4e03
  .word 0xf12ccb09
  .word 0x8c03454c
  .word 0xf1244e07
  .word 0xf1244b0d
  .word 0x7682c3e8
  .word 0x8c094543
  .word 0x8c074544
  .word 0xa1804803
  .word 0xa180c805
  .word 0x8c0d4549
  .word 0xa1804c02
  .word 0xa180c404
  .word 0xa60a0809
  .word 0xa180430a
  .word 0xa180c90b
  .word 0x85014000
  .word 0xa0050102
  .word 0xde0c300c
  .word 0x5c204036
  .word 0x3e051f00
  .word 0x8d0c42ae
  .word 0x8d0842a7
  .word 0x7628700d
  .word 0x7606c066
  .word 0xde6862c0
  .word 0x8d04c2af
  .word 0x8d0342bc
  .word 0x8d09c2a1
  .word 0xda6c700e
  .word 0x8d0542ae
  .word 0xa1a0d402
  .word 0xefc84d07
  .word 0xda64700f
  .word 0xa1a0d204
  .word 0xda63701c
  .word 0xda697001
  .word 0xa180c703
  .word 0x0000401e
  .word 0xda45400e
  .word 0xa180430a
  .word 0xa180c90b
  .word 0xda627006
  .word 0xa180c500
  .word 0x529f4000
  .word 0xa1020507
  .word 0x529fc000
.globl qd_v69_s2_40081c
qd_v69_s2_40081c:
  .word 0x8d0146a6
  .word 0x85a14900
  .word 0x76054408
  .word 0x5c20c814
  .word 0x75084000
  .word 0x78014002
  .word 0x7800c0e7
  .word 0x00004010
  .word 0x73024008
  .word 0x5800c026
  .word 0x8d014149
  .word 0xbfe27fe7
  .word 0x7582c3e0
  .word 0x8c07c227
  .word 0x7e0040e7
  .word 0xb0094029
  .word 0x7508c000
  .word 0x7067c00c
  .word 0xde8cc704
  .word 0xed0c4908
  .word 0x5c00c00e
  .word 0x8c084148
  .word 0x75824400
  .word 0x7e00e0e7
  .word 0x00004020
  .word 0x7e00c008
  .word 0x8c064542
  .word 0x75064600
  .word 0x76444004
  .word 0x7506c401
  .word 0x760440ed
  .word 0x70624004
  .word 0x007f7fff
  .word 0x7800c7e9
  .word 0x8e4742c4
  .word 0x85014000
  .word 0x7a01608c
  .word 0xa180c50c
  .word 0x7e2040ac
  .word 0xbfe87fe8
  .word 0xa1023000
  .word 0xc69f4ce7
  .word 0x76844065
  .word 0xa1804806
  .word 0x3840c000
  .word 0xef474943
  .word 0xc64d4c87
  .word 0xa180c503
  .word 0xef47c663
  .word 0x8c034547
  .word 0x5c20401e
  .word 0xa1a0d402
  .word 0x8d0542a8
  .word 0x76277007
  .word 0x7606c066
  .word 0xda434067
  .word 0x0000401e
  .word 0xda424006
  .word 0xa1a0d402
  .word 0xda647008
  .word 0xa180c201
  .word 0x8f054544
  .word 0x529f4000
  .word 0xa1a0d403
  .word 0x529fc000
.org 0x4010a8
.globl qd_v69_s2_noop_stub
qd_v69_s2_noop_stub:
  jumpr r31
.org 0x402000
.globl qd_v69_s2_capture_1xn
qd_v69_s2_capture_1xn:
  r14 = ##qd_v69_s2_leaf_capture_args
  memw(r14+#0)=r0
  memw(r14+#4)=r1
  memw(r14+#8)=r2
  memw(r14+#12)=r3
  memw(r14+#16)=r4
  memw(r14+#20)=r5
  r14 = ##qd_v69_s2_leaf_request_words
  r15 = memw(r0+#0)
  memw(r14+#0)=r15
  r15 = memw(r0+#4)
  memw(r14+#4)=r15
  r15 = memw(r0+#8)
  memw(r14+#8)=r15
  r15 = memw(r0+#12)
  memw(r14+#12)=r15
  r15 = memw(r0+#16)
  memw(r14+#16)=r15
  r15 = memw(r0+#20)
  memw(r14+#20)=r15
  r15 = memw(r0+#24)
  memw(r14+#24)=r15
  r15 = memw(r0+#28)
  memw(r14+#28)=r15
  r15 = memw(r0+#32)
  memw(r14+#32)=r15
  r15 = memw(r0+#36)
  memw(r14+#36)=r15
  r15 = memw(r0+#40)
  memw(r14+#40)=r15
  r15 = memw(r0+#44)
  memw(r14+#44)=r15
  r15 = memw(r0+#48)
  memw(r14+#48)=r15
  r15 = memw(r0+#52)
  memw(r14+#52)=r15
  r15 = memw(r0+#56)
  memw(r14+#56)=r15
  r15 = memw(r0+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_outdesc_words
  r15 = memw(r1+#0)
  memw(r14+#0)=r15
  r15 = memw(r1+#4)
  memw(r14+#4)=r15
  r15 = memw(r1+#8)
  memw(r14+#8)=r15
  r15 = memw(r1+#12)
  memw(r14+#12)=r15
  r15 = memw(r1+#16)
  memw(r14+#16)=r15
  r15 = memw(r1+#20)
  memw(r14+#20)=r15
  r15 = memw(r1+#24)
  memw(r14+#24)=r15
  r15 = memw(r1+#28)
  memw(r14+#28)=r15
  r15 = memw(r1+#32)
  memw(r14+#32)=r15
  r15 = memw(r1+#36)
  memw(r14+#36)=r15
  r15 = memw(r1+#40)
  memw(r14+#40)=r15
  r15 = memw(r1+#44)
  memw(r14+#44)=r15
  r15 = memw(r1+#48)
  memw(r14+#48)=r15
  r15 = memw(r1+#52)
  memw(r14+#52)=r15
  r15 = memw(r1+#56)
  memw(r14+#56)=r15
  r15 = memw(r1+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_config_words
  r15 = memw(r4+#0)
  memw(r14+#0)=r15
  r15 = memw(r4+#4)
  memw(r14+#4)=r15
  r15 = memw(r4+#8)
  memw(r14+#8)=r15
  r15 = memw(r4+#12)
  memw(r14+#12)=r15
  r15 = memw(r4+#16)
  memw(r14+#16)=r15
  r15 = memw(r4+#20)
  memw(r14+#20)=r15
  r15 = memw(r4+#24)
  memw(r14+#24)=r15
  r15 = memw(r4+#28)
  memw(r14+#28)=r15
  r15 = memw(r4+#32)
  memw(r14+#32)=r15
  r15 = memw(r4+#36)
  memw(r14+#36)=r15
  r15 = memw(r4+#40)
  memw(r14+#40)=r15
  r15 = memw(r4+#44)
  memw(r14+#44)=r15
  r15 = memw(r4+#48)
  memw(r14+#48)=r15
  r15 = memw(r4+#52)
  memw(r14+#52)=r15
  r15 = memw(r4+#56)
  memw(r14+#56)=r15
  r15 = memw(r4+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_selected
  r15 = #1
  memw(r14+#0)=r15
#ifdef S2_CAPTURE_ONLY
  jumpr r31
#else
  jump qd_v69_s2_leaf_1xn
#endif
.globl qd_v69_s2_capture_nx1
qd_v69_s2_capture_nx1:
  r14 = ##qd_v69_s2_leaf_capture_args
  memw(r14+#0)=r0
  memw(r14+#4)=r1
  memw(r14+#8)=r2
  memw(r14+#12)=r3
  memw(r14+#16)=r4
  memw(r14+#20)=r5
  r14 = ##qd_v69_s2_leaf_request_words
  r15 = memw(r0+#0)
  memw(r14+#0)=r15
  r15 = memw(r0+#4)
  memw(r14+#4)=r15
  r15 = memw(r0+#8)
  memw(r14+#8)=r15
  r15 = memw(r0+#12)
  memw(r14+#12)=r15
  r15 = memw(r0+#16)
  memw(r14+#16)=r15
  r15 = memw(r0+#20)
  memw(r14+#20)=r15
  r15 = memw(r0+#24)
  memw(r14+#24)=r15
  r15 = memw(r0+#28)
  memw(r14+#28)=r15
  r15 = memw(r0+#32)
  memw(r14+#32)=r15
  r15 = memw(r0+#36)
  memw(r14+#36)=r15
  r15 = memw(r0+#40)
  memw(r14+#40)=r15
  r15 = memw(r0+#44)
  memw(r14+#44)=r15
  r15 = memw(r0+#48)
  memw(r14+#48)=r15
  r15 = memw(r0+#52)
  memw(r14+#52)=r15
  r15 = memw(r0+#56)
  memw(r14+#56)=r15
  r15 = memw(r0+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_outdesc_words
  r15 = memw(r1+#0)
  memw(r14+#0)=r15
  r15 = memw(r1+#4)
  memw(r14+#4)=r15
  r15 = memw(r1+#8)
  memw(r14+#8)=r15
  r15 = memw(r1+#12)
  memw(r14+#12)=r15
  r15 = memw(r1+#16)
  memw(r14+#16)=r15
  r15 = memw(r1+#20)
  memw(r14+#20)=r15
  r15 = memw(r1+#24)
  memw(r14+#24)=r15
  r15 = memw(r1+#28)
  memw(r14+#28)=r15
  r15 = memw(r1+#32)
  memw(r14+#32)=r15
  r15 = memw(r1+#36)
  memw(r14+#36)=r15
  r15 = memw(r1+#40)
  memw(r14+#40)=r15
  r15 = memw(r1+#44)
  memw(r14+#44)=r15
  r15 = memw(r1+#48)
  memw(r14+#48)=r15
  r15 = memw(r1+#52)
  memw(r14+#52)=r15
  r15 = memw(r1+#56)
  memw(r14+#56)=r15
  r15 = memw(r1+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_config_words
  r15 = memw(r4+#0)
  memw(r14+#0)=r15
  r15 = memw(r4+#4)
  memw(r14+#4)=r15
  r15 = memw(r4+#8)
  memw(r14+#8)=r15
  r15 = memw(r4+#12)
  memw(r14+#12)=r15
  r15 = memw(r4+#16)
  memw(r14+#16)=r15
  r15 = memw(r4+#20)
  memw(r14+#20)=r15
  r15 = memw(r4+#24)
  memw(r14+#24)=r15
  r15 = memw(r4+#28)
  memw(r14+#28)=r15
  r15 = memw(r4+#32)
  memw(r14+#32)=r15
  r15 = memw(r4+#36)
  memw(r14+#36)=r15
  r15 = memw(r4+#40)
  memw(r14+#40)=r15
  r15 = memw(r4+#44)
  memw(r14+#44)=r15
  r15 = memw(r4+#48)
  memw(r14+#48)=r15
  r15 = memw(r4+#52)
  memw(r14+#52)=r15
  r15 = memw(r4+#56)
  memw(r14+#56)=r15
  r15 = memw(r4+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_selected
  r15 = #2
  memw(r14+#0)=r15
#ifdef S2_CAPTURE_ONLY
  jumpr r31
#else
  jump qd_v69_s2_leaf_nx1
#endif
.globl qd_v69_s2_capture_generic
qd_v69_s2_capture_generic:
  r14 = ##qd_v69_s2_leaf_capture_args
  memw(r14+#0)=r0
  memw(r14+#4)=r1
  memw(r14+#8)=r2
  memw(r14+#12)=r3
  memw(r14+#16)=r4
  memw(r14+#20)=r5
  r14 = ##qd_v69_s2_leaf_request_words
  r15 = memw(r0+#0)
  memw(r14+#0)=r15
  r15 = memw(r0+#4)
  memw(r14+#4)=r15
  r15 = memw(r0+#8)
  memw(r14+#8)=r15
  r15 = memw(r0+#12)
  memw(r14+#12)=r15
  r15 = memw(r0+#16)
  memw(r14+#16)=r15
  r15 = memw(r0+#20)
  memw(r14+#20)=r15
  r15 = memw(r0+#24)
  memw(r14+#24)=r15
  r15 = memw(r0+#28)
  memw(r14+#28)=r15
  r15 = memw(r0+#32)
  memw(r14+#32)=r15
  r15 = memw(r0+#36)
  memw(r14+#36)=r15
  r15 = memw(r0+#40)
  memw(r14+#40)=r15
  r15 = memw(r0+#44)
  memw(r14+#44)=r15
  r15 = memw(r0+#48)
  memw(r14+#48)=r15
  r15 = memw(r0+#52)
  memw(r14+#52)=r15
  r15 = memw(r0+#56)
  memw(r14+#56)=r15
  r15 = memw(r0+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_outdesc_words
  r15 = memw(r1+#0)
  memw(r14+#0)=r15
  r15 = memw(r1+#4)
  memw(r14+#4)=r15
  r15 = memw(r1+#8)
  memw(r14+#8)=r15
  r15 = memw(r1+#12)
  memw(r14+#12)=r15
  r15 = memw(r1+#16)
  memw(r14+#16)=r15
  r15 = memw(r1+#20)
  memw(r14+#20)=r15
  r15 = memw(r1+#24)
  memw(r14+#24)=r15
  r15 = memw(r1+#28)
  memw(r14+#28)=r15
  r15 = memw(r1+#32)
  memw(r14+#32)=r15
  r15 = memw(r1+#36)
  memw(r14+#36)=r15
  r15 = memw(r1+#40)
  memw(r14+#40)=r15
  r15 = memw(r1+#44)
  memw(r14+#44)=r15
  r15 = memw(r1+#48)
  memw(r14+#48)=r15
  r15 = memw(r1+#52)
  memw(r14+#52)=r15
  r15 = memw(r1+#56)
  memw(r14+#56)=r15
  r15 = memw(r1+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_config_words
  r15 = memw(r4+#0)
  memw(r14+#0)=r15
  r15 = memw(r4+#4)
  memw(r14+#4)=r15
  r15 = memw(r4+#8)
  memw(r14+#8)=r15
  r15 = memw(r4+#12)
  memw(r14+#12)=r15
  r15 = memw(r4+#16)
  memw(r14+#16)=r15
  r15 = memw(r4+#20)
  memw(r14+#20)=r15
  r15 = memw(r4+#24)
  memw(r14+#24)=r15
  r15 = memw(r4+#28)
  memw(r14+#28)=r15
  r15 = memw(r4+#32)
  memw(r14+#32)=r15
  r15 = memw(r4+#36)
  memw(r14+#36)=r15
  r15 = memw(r4+#40)
  memw(r14+#40)=r15
  r15 = memw(r4+#44)
  memw(r14+#44)=r15
  r15 = memw(r4+#48)
  memw(r14+#48)=r15
  r15 = memw(r4+#52)
  memw(r14+#52)=r15
  r15 = memw(r4+#56)
  memw(r14+#56)=r15
  r15 = memw(r4+#60)
  memw(r14+#60)=r15
  r14 = ##qd_v69_s2_leaf_selected
  r15 = #3
  memw(r14+#0)=r15
#ifdef S2_CAPTURE_ONLY
  jumpr r31
#else
  jump qd_v69_s2_generic_leaf
#endif
.org 0x972f30
.globl qd_v69_s2_global_pair
qd_v69_s2_global_pair:
  .word 0,0
