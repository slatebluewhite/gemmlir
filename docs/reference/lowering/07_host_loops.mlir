module attributes {torch.debug_module_name = "Block"} {
  memref.global "private" @__gemmlir_arena_forward_0 : memref<106176xi8> = uninitialized {alignment = 64 : i64}
  memref.global "private" constant @__constant_16xf32 : memref<16xf32> = dense<[-0.0932260155, -0.0050244038, -0.0149413012, 0.292073935, 0.0149224252, 0.24778448, 0.249534339, -0.10057988, -0.0420088544, 0.459543973, -0.305139065, -0.245570511, 0.427169263, -0.234537929, -0.161344975, -0.167135388]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_3x3x8x16xi8 : memref<3x3x8x16xi8> = dense<"0xFF29AAB1ECB4493FE7AF44764A55E22F1A5D3432BDEEACE4DA5ED4CB04ECF7B4BE26AA3E0F51471A22F219C6C94CBAFFA6C4D9003C46E749484FD43843831699C7FA0A543E50BDD93F2EDC1D3F7F3A0FCE05343A39AA28494861C5CED78358EF41E6C7D049B0352F30A43436210B33E01DF5511D4636F7F321CF0594C5E8F4C23457D3CBFC9106D8BAE7C221E095DFA7E34F34B825B7EF130FCFF063FE87BAD4D635ADC026572ECA214FC4D5474131F4C3625634D1D9ABC8DFF1F77B1481A3255410ECE251F3E511DF463F054A5312A93D0F4C36C26CCFB91209EED6271C95B2C7CDFED032DA30041417CA5C0A6374C535032303F038B83AD3A7D20239F53C3BB05B4152CAA5C7181F38F3D3FA5C152CED98A4E3E24E3A1A1D2543A90B88234F234BF12E236456D4BEE0BD222C11D7F5E746239E0136DEC3DDFCE6DA3510E3022BD544C435BF26EF1BF0DD4F45B854963919414E235629B8B6A049201BBBF5AC12DB04A0CAE103B718BF54062518CDF5F4F70CBC2399D1260BA2DDC7189DB38CB85CFFAE1E59E11F4AD6CECDDF8814F2A3143CDED5ABAEDC3CE6ED003A3145CF512701B8F5E32CDA465205AA29BA54E8DAD2AA49250D13C72846BD43E0609BC72F3E1BDA0C5755DC42ABC6E7330B33D1D529BD2544D7C34AEA48C0EA0C126430B4AA3E1650EAE5D407F5CE18B4D7B2010415FFAE10EC4C3F1864E7524BFF771DDA15D0F91C3D13EAEA56D0B1F61C3D31BFEEB5F110C948073BEA08704ACAB44BEC99D5D5B732C745489D43142A2A606054E63DAC07E5BC00DDE400DC23B34A8B05C02E9B2A48320A26330D0C35418BECFC0604DE593FD5D1CCB4E75112AD2FA3BCEAB8C72B0BEDE71C0D5405BA56F25D1742ED5126AC4F2B05E6AE8CB59687A21AA5B764FE6ABE19C1BA14AA10ADAC56D8EFEC01D293DE2BF54BC57BD3B3BAD049BC241EF26448EC44F1E836C406125AC19CF50107C01BEDDB5A3FEEBD1CB3D1CE5F222ED408F63D0B1F32D2F6704D923ECDBDE44EC0B90E42181C0742A066A4CE16BEA531680B26EE57AF562E078E873C630C11E2D0B0E9CF05044BB7E58C3CFE0AA725EC5A2BEBBBAA4FEC0ACDF73504D0D95415C019F9BDB147B8FBCE36E4F034AA221F250500C0EE28CF08A5F6AED3FED5EF0BC12ECFF9BC47EBFB596F291048E9CFB6CB1ECE4129B498C581E9AF6131CB1E45EE2DFAF30FB62F1CF5D6452CBCC859C8062BD0E2CFB5CC301830F75E42EA542C293DCE491C0F7B334AE3984DBFAF03C99BB22E13D2BDEAF21744F62728EE0D125D0417BB0BF516E298F8030A16FA4CDF7129DCE3214867461BDA38BCB14CCBBAA90CE5F9F508FF1927E190A5A8ECBCB368AB030AC14C32E445CA08279CC853EB60F7F0BBD02E36BED5BD1727FB43BE1ABAC446B94234BEBCD75936B563FFDEC2B02BFAD446001943A7634BF79FEDA83EB3ADEBFA113337B4CE69F93AC2ED5BBE5CB2EE4760C30FBA9A8E9158AE54DF223ABB4BCC2838F8D674FAA1A5AFB30D074CC60EC5FB13D1171CBB8ABAE6B1C5592EB6E4DD421E10F8DB4D8D0DC2211CE10542EFFB242CE2C94A686FC64BDEF4E10648EFC7F309361D001DBDDCF80C1E142A333E3E571BF82EAA0C8D"> {alignment = 64 : i64}
  memref.global "private" constant @__constant_16xi32 : memref<16xi32> = dense<[-1550, -2474, 1465, -2521, 2753, 5854, -5913, 1243, -4467, -7415, -5871, -7936, -1411, 13854, -5680, -558]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_144x16xi8 : memref<144x16xi8> = dense<"0x9720D01F1DE1F345A1BCAFB2DE16BF21D9259CFAD9AB043949EFCDE796E709AFD71ADD2C050EE2D9560CAEBB5B06EF2AB4C53605E5307D13C7C952B9AD1D3DCF6D3FA60049A70CF8119FEDD39249C7B1D5155037281B3146C76C4E4CB21CE61DB0DAC84004FF194E28B3102FCE1237EC8512DA1C4A04C5F20624FBD928E3FD05F3BDC6070EF2B71C0CEA010643D4FFDEDD1B382849E322B5ED171BBCC1E332152DFE561620BD11FDECFCD3D2501FCCB677C9FCC650EDE7503A42000A4BBCCE2E8C14C92548B7DCF2019132E4340BDEDE6B4AACB8B65C23C9AC214406C959BC1150DAEF3E45E392A81A1ED8DDF4F94CC46802E4BE592962F8F516B8C864AED4ED02DCEEF6B52051FDA6C9D5AE19D02C52C1D949524FC86AB9216ADEEC1DCB1E3562362AF2D3573BF1F50F0D5B24E10145EAB15EE1CE2A3E1FC268CC0D04FEE7499323D0B9D240D4C0E49A06276CD5F81041052BEDEB26EFA303A505E8C923D9EE78513C0ED61D4E393C502409A1D90BCD362FD6E14A28CE105957EED7E2C037FB3F44A4D70025CCEFA8284D16A118D31E3EF7E7CDBFDB9EAE142CAF0317B83C408CE3F1EDB74DD35B9DB557ED43CCFCA7391960F7AC269405EBCBCDFA5143BBFD0D0725D254C943CBDAEF00D26DA80DD3F04AB52825A3AC249CBBD2FFD14E1AD555D8AC133447AA231203C113281D30A7DE3A2B42D5D2400E1EE1AFA8773E0BC76244E5BC0F1AA4D53A199D54334DCF0369C351002FA4C9D1BD18A6E34423B059A7ACD204AA583122A7BB9ED866DFDD04CE38DE15F01B16EAA6C815EEDA3513E8FBD554C6FAF6F3F2F230DFD8A707F4F3E14AC9F64E4720FB17F2D32275320F18F54BEB393403A052EACF3EB26212B9013E51EBCB3128ABB3F41ED9BDC853E259ABD0E328BD169CA254A603026CDC26399E15D84F2FC472BF9FBB19F28A3CE8EFBACC554FE7E05F4B61A29FF897F7250490DDBBC92CB8BCC1070B1A1840C20C576B3AD540061B684B28011DA94512DA126904ACE11BCF3719200AB1209C2AD156A3ACC7E14D34C1CDFBA0CC1B73A6B941A9DEBEDACF58132229474610DFF0434D95BCA8214DC9AA10C52F434009D9F7B434B444BEBC18DD5B180FF62CEDCD0516A7C4F4F7D0F9BCF4FA531B361EE9ECA965F605BDE0DA53DAFF9B3335084C04ED49E3224C59AEFDCB61F3DB3FE9DDB63535E14F3BCDE320C52D3DE8F05A5516C9AD4FA9B4CA0353E4FABB16A3392CCEE4B2CBDB2025AE964CABA33C0F62E521DE3F36A30E1658954238B8E713514637FE31543F3B2F4681E8CE0F3DD13919D00E8E3D00E008D45048319D0AD10DF0E5313F2B26D5A8005C4FCCFF34FCEB26DFCAA5DD18504C16502CA7DBB13EEF57FDBF8C3215E25B3C67242F42A225E3DE2AE5F2082ADC3011FAF8352C5FD5E3ACFD4D55D20AEA06BF2040F76DE1EE2FD8D7C6C8F4EE363F0993B249D4630820150BE7A504DFFA50B75413E4F2073242B628EFD93FE313EEA35437073BA3CC404017A786E1141934FFCB1B1F31205BA11BC8B13726613E204B6D31FDBDA9483EBA11D770EA0900E3CB7CAFA366F5E1094512B8C8F85BC5E8A66AC650162ED3481D17381F2D370FE8AD66BBD2DE5CF32844D9F346AFEDB8BC55715D5733F9DA00AE1C311DC9570F42E50D3725300C31513CB31DA7FAF5B60BBE17F7C4B29FEAD753CBF430F85A2AA7DDFC10AB3400F495DF0022291E6314D6335F1BCDB7B1FC101AF651BDAB14BFDAF193F6ECBF24C3F4ABD34B123AA32E030E171A6144ED3A4900B7DF55D329001D2C9EA7BF469EB2771D08A96E021FF250B8B2E71FD73AB178E6D11B14523429FF1FE0F1D8461B140423CC320214A20346E1F8B343A9BA014B1BB2ECB0EA642DD20CBDE7F9BCF315491EE1AF66E14A2ED1CD72C258FD0639DA302C1B744942CA22F9860905B101D40AB130B6BB4DFEB1B2ADC1C2AF5F35560C421CB65C1CEDC531B7F05C5C16AE05CF39FF17B4F102CB2DC49E2F2C92204CCFE8B14A4FFBA42517DDF5D9FB60F7F85ACA1FFEC850C6D9DFD3DD02B7D0FDFDC7DFC7516107EFC2C62EF832BDB356208D5AF33937D1224BDBF28E1626B2FACEB4004AF8E5CED52221CE6BF4A720A628C9CC40F92022C70F5031DB0D54D753A8F9E23ED4D9F9F9F12FBC34D9B4DC5C0024EE4ADCEB2CD72732E4031A195820E8D72D2C30FFF0A2E5FCBE7A31F5C9BAF45BC914E522FBEFCE44DCE0D20398EB2014D43E319AD25CF70ED0DBD961EEA1FD9235E306FD160139C8219E4B0D9AF5016B4944248EBB2AC04A4AD0D2E62FEF3C9ADC2BC733DB5142A708164EBBB65EBEAFF54CDF122D4CCA2D02FCDBEAA236280BFC304B6601FA12C900EE5CDE4D19F6180EC6D094E0EA1D3340E2FCD0AC4D210ACB2A466730DFB4A4B6053361B2252A545849F085C55027F45232AEF505C0B4260DB9C82112F7AEF431D302AB4AF3DA38EDDBE3DB0F9EC4DB3A3BF9C6FC9C4655584F2D40D84B45C93B873D5B9102DDD23FF10FE7AF1210261E81AC614E02F02F2B49123445FC1FF8DBC6FA50C618A85201C7A6CDFCE7D5E7C3CD38D14CF4F703DE00BD040FAB0FC8ECFD481DE3A0AAE4C13AC4E9E7CCC0CCF8AD0CEDDD49D233E1EC0FAC17C220DF5E2F25DB1E582BCBCF442DAD06352E5908AA44412E1ECD5A30D4DA93E3B2FF342FBFD1073CA433313129C6E628172720C4C1D6D4134527E6B525447B0FCF33E210DB1ADFAAFBF4B10936EC984AF7CF495757FFA9C8A84F47DBF20DE74A2044E8C972E1009EE956EFC81D124DB4023C1AF3AC11B7682BB1FFB1B7AB7CED19F70AAC0F3632DC4B19A3184C4B730036EF2E06AC190B3DCD0645AD053FCEB2CD1F58D9F10CDE5BF3C9491AFF31A4B5B4494F42CBDBE537BD262C17F021883900DF18531CF6A3AD612067321446B9C2EC44E7D0FFF0581AADFEB3E945C0E2F33A02FC24590C1B4A512B1902221861B7AB00E3F43717BFF7A8234F10B83153CB622EE30431B04A35DFABF022C7E1CC52A8440F42C2B6224353F0EECC380D8C2023DA063F1BF4D8F322430E1D1D0B09E0F13FFDE259F32DC412E3F607EE0ACF0746E6D4522FF931FE20AC4E4BF4AF3849DCFF12FC6CC34021D2D52906B1A748432CDB24AA0EDAC409614513DBBE516620B500B9D0F73963503900B410C11D663BC610BF1994EB08F949BC53D8C2193BBDC725FB047AB8044AAB2F96A911BDDABCD6425BF66117DEE6E3C995B1CE1698532DCFE90BDE39F20BB1B7CFD94C23"> {alignment = 64 : i64}
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @forward(%arg0: memref<1x8x16x16xf32>) -> memref<1x16x8x8xf32> {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c127_i32 = arith.constant 127 : i32
    %c256_i32 = arith.constant 256 : i32
    %c-128_i32 = arith.constant -128 : i32
    %cst = arith.constant 3.488010e+01 : f32
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c102080 = arith.constant 102080 : index
    %c85696 = arith.constant 85696 : index
    %c48768 = arith.constant 48768 : index
    %c32384 = arith.constant 32384 : index
    %c27200 = arith.constant 27200 : index
    %c23104 = arith.constant 23104 : index
    %c2048 = arith.constant 2048 : index
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c0_i8 = arith.constant 0 : i8
    %cst_2 = arith.constant 0.024088297 : f32
    %cst_3 = arith.constant 2.12572231E-5 : f32
    %0 = memref.get_global @__constant_144x16xi8 : memref<144x16xi8>
    %1 = memref.get_global @__constant_16xi32 : memref<16xi32>
    %2 = memref.get_global @__constant_3x3x8x16xi8 : memref<3x3x8x16xi8>
    %3 = memref.get_global @__constant_16xf32 : memref<16xf32>
    %4 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view = memref.view %4[%c2048][] : memref<106176xi8> to memref<1x16x16x8xi8>
    cf.br ^bb1(%c0 : index)
  ^bb1(%5: index):  // 2 preds: ^bb0, ^bb43
    %6 = arith.cmpi slt, %5, %c1 : index
    cf.cond_br %6, ^bb2, ^bb44
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%7: index):  // 2 preds: ^bb2, ^bb42
    %8 = arith.cmpi slt, %7, %c8 : index
    cf.cond_br %8, ^bb4, ^bb43
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%9: index):  // 2 preds: ^bb4, ^bb41
    %10 = arith.cmpi slt, %9, %c16 : index
    cf.cond_br %10, ^bb6, ^bb42
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%c0 : index)
  ^bb7(%11: index):  // 2 preds: ^bb6, ^bb40
    %12 = arith.cmpi slt, %11, %c16 : index
    cf.cond_br %12, ^bb8, ^bb41
  ^bb8:  // pred: ^bb7
    %13 = memref.load %arg0[%5, %7, %9, %11] : memref<1x8x16x16xf32>
    %14 = arith.mulf %13, %cst : f32
    %15 = math.roundeven %14 : f32
    %16 = arith.fptosi %15 : f32 to i32
    %17 = arith.subi %16, %c-128_i32 : i32
    %18 = arith.cmpi ult, %17, %c256_i32 : i32
    cf.cond_br %18, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    cf.br ^bb11(%16 : i32)
  ^bb10:  // pred: ^bb8
    %19 = arith.cmpi slt, %16, %c-128_i32 : i32
    %20 = arith.select %19, %c-128_i32, %c127_i32 : i32
    cf.br ^bb11(%20 : i32)
  ^bb11(%21: i32):  // 2 preds: ^bb9, ^bb10
    cf.br ^bb12
  ^bb12:  // pred: ^bb11
    %22 = arith.trunci %21 : i32 to i8
    memref.store %22, %view[%5, %9, %11, %7] : memref<1x16x16x8xi8>
    %23 = arith.addi %11, %c1 : index
    %24 = memref.load %arg0[%5, %7, %9, %23] : memref<1x8x16x16xf32>
    %25 = arith.mulf %24, %cst : f32
    %26 = math.roundeven %25 : f32
    %27 = arith.fptosi %26 : f32 to i32
    %28 = arith.subi %27, %c-128_i32 : i32
    %29 = arith.cmpi ult, %28, %c256_i32 : i32
    cf.cond_br %29, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    cf.br ^bb15(%27 : i32)
  ^bb14:  // pred: ^bb12
    %30 = arith.cmpi slt, %27, %c-128_i32 : i32
    %31 = arith.select %30, %c-128_i32, %c127_i32 : i32
    cf.br ^bb15(%31 : i32)
  ^bb15(%32: i32):  // 2 preds: ^bb13, ^bb14
    cf.br ^bb16
  ^bb16:  // pred: ^bb15
    %33 = arith.trunci %32 : i32 to i8
    memref.store %33, %view[%5, %9, %23, %7] : memref<1x16x16x8xi8>
    %34 = arith.addi %11, %c2 : index
    %35 = memref.load %arg0[%5, %7, %9, %34] : memref<1x8x16x16xf32>
    %36 = arith.mulf %35, %cst : f32
    %37 = math.roundeven %36 : f32
    %38 = arith.fptosi %37 : f32 to i32
    %39 = arith.subi %38, %c-128_i32 : i32
    %40 = arith.cmpi ult, %39, %c256_i32 : i32
    cf.cond_br %40, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    cf.br ^bb19(%38 : i32)
  ^bb18:  // pred: ^bb16
    %41 = arith.cmpi slt, %38, %c-128_i32 : i32
    %42 = arith.select %41, %c-128_i32, %c127_i32 : i32
    cf.br ^bb19(%42 : i32)
  ^bb19(%43: i32):  // 2 preds: ^bb17, ^bb18
    cf.br ^bb20
  ^bb20:  // pred: ^bb19
    %44 = arith.trunci %43 : i32 to i8
    memref.store %44, %view[%5, %9, %34, %7] : memref<1x16x16x8xi8>
    %45 = arith.addi %11, %c3 : index
    %46 = memref.load %arg0[%5, %7, %9, %45] : memref<1x8x16x16xf32>
    %47 = arith.mulf %46, %cst : f32
    %48 = math.roundeven %47 : f32
    %49 = arith.fptosi %48 : f32 to i32
    %50 = arith.subi %49, %c-128_i32 : i32
    %51 = arith.cmpi ult, %50, %c256_i32 : i32
    cf.cond_br %51, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    cf.br ^bb23(%49 : i32)
  ^bb22:  // pred: ^bb20
    %52 = arith.cmpi slt, %49, %c-128_i32 : i32
    %53 = arith.select %52, %c-128_i32, %c127_i32 : i32
    cf.br ^bb23(%53 : i32)
  ^bb23(%54: i32):  // 2 preds: ^bb21, ^bb22
    cf.br ^bb24
  ^bb24:  // pred: ^bb23
    %55 = arith.trunci %54 : i32 to i8
    memref.store %55, %view[%5, %9, %45, %7] : memref<1x16x16x8xi8>
    %56 = arith.addi %11, %c4 : index
    %57 = memref.load %arg0[%5, %7, %9, %56] : memref<1x8x16x16xf32>
    %58 = arith.mulf %57, %cst : f32
    %59 = math.roundeven %58 : f32
    %60 = arith.fptosi %59 : f32 to i32
    %61 = arith.subi %60, %c-128_i32 : i32
    %62 = arith.cmpi ult, %61, %c256_i32 : i32
    cf.cond_br %62, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    cf.br ^bb27(%60 : i32)
  ^bb26:  // pred: ^bb24
    %63 = arith.cmpi slt, %60, %c-128_i32 : i32
    %64 = arith.select %63, %c-128_i32, %c127_i32 : i32
    cf.br ^bb27(%64 : i32)
  ^bb27(%65: i32):  // 2 preds: ^bb25, ^bb26
    cf.br ^bb28
  ^bb28:  // pred: ^bb27
    %66 = arith.trunci %65 : i32 to i8
    memref.store %66, %view[%5, %9, %56, %7] : memref<1x16x16x8xi8>
    %67 = arith.addi %11, %c5 : index
    %68 = memref.load %arg0[%5, %7, %9, %67] : memref<1x8x16x16xf32>
    %69 = arith.mulf %68, %cst : f32
    %70 = math.roundeven %69 : f32
    %71 = arith.fptosi %70 : f32 to i32
    %72 = arith.subi %71, %c-128_i32 : i32
    %73 = arith.cmpi ult, %72, %c256_i32 : i32
    cf.cond_br %73, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    cf.br ^bb31(%71 : i32)
  ^bb30:  // pred: ^bb28
    %74 = arith.cmpi slt, %71, %c-128_i32 : i32
    %75 = arith.select %74, %c-128_i32, %c127_i32 : i32
    cf.br ^bb31(%75 : i32)
  ^bb31(%76: i32):  // 2 preds: ^bb29, ^bb30
    cf.br ^bb32
  ^bb32:  // pred: ^bb31
    %77 = arith.trunci %76 : i32 to i8
    memref.store %77, %view[%5, %9, %67, %7] : memref<1x16x16x8xi8>
    %78 = arith.addi %11, %c6 : index
    %79 = memref.load %arg0[%5, %7, %9, %78] : memref<1x8x16x16xf32>
    %80 = arith.mulf %79, %cst : f32
    %81 = math.roundeven %80 : f32
    %82 = arith.fptosi %81 : f32 to i32
    %83 = arith.subi %82, %c-128_i32 : i32
    %84 = arith.cmpi ult, %83, %c256_i32 : i32
    cf.cond_br %84, ^bb33, ^bb34
  ^bb33:  // pred: ^bb32
    cf.br ^bb35(%82 : i32)
  ^bb34:  // pred: ^bb32
    %85 = arith.cmpi slt, %82, %c-128_i32 : i32
    %86 = arith.select %85, %c-128_i32, %c127_i32 : i32
    cf.br ^bb35(%86 : i32)
  ^bb35(%87: i32):  // 2 preds: ^bb33, ^bb34
    cf.br ^bb36
  ^bb36:  // pred: ^bb35
    %88 = arith.trunci %87 : i32 to i8
    memref.store %88, %view[%5, %9, %78, %7] : memref<1x16x16x8xi8>
    %89 = arith.addi %11, %c7 : index
    %90 = memref.load %arg0[%5, %7, %9, %89] : memref<1x8x16x16xf32>
    %91 = arith.mulf %90, %cst : f32
    %92 = math.roundeven %91 : f32
    %93 = arith.fptosi %92 : f32 to i32
    %94 = arith.subi %93, %c-128_i32 : i32
    %95 = arith.cmpi ult, %94, %c256_i32 : i32
    cf.cond_br %95, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    cf.br ^bb39(%93 : i32)
  ^bb38:  // pred: ^bb36
    %96 = arith.cmpi slt, %93, %c-128_i32 : i32
    %97 = arith.select %96, %c-128_i32, %c127_i32 : i32
    cf.br ^bb39(%97 : i32)
  ^bb39(%98: i32):  // 2 preds: ^bb37, ^bb38
    cf.br ^bb40
  ^bb40:  // pred: ^bb39
    %99 = arith.trunci %98 : i32 to i8
    memref.store %99, %view[%5, %9, %89, %7] : memref<1x16x16x8xi8>
    %100 = arith.addi %11, %c8 : index
    cf.br ^bb7(%100 : index)
  ^bb41:  // pred: ^bb7
    %101 = arith.addi %9, %c1 : index
    cf.br ^bb5(%101 : index)
  ^bb42:  // pred: ^bb5
    %102 = arith.addi %7, %c1 : index
    cf.br ^bb3(%102 : index)
  ^bb43:  // pred: ^bb3
    %103 = arith.addi %5, %c1 : index
    cf.br ^bb1(%103 : index)
  ^bb44:  // pred: ^bb1
    %104 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_4 = memref.view %104[%c23104][] : memref<106176xi8> to memref<1x16x16x16xi8>
    gemmlir.conv2d_i8(%view, %2, %view_4) bias(%1 : memref<16xi32>) {act = #gemmlir.act<relu>, gemmlir.no_flush_after, padding = 1 : i64, scale = 0.00155569159 : f32} : (memref<1x16x16x8xi8>, memref<3x3x8x16xi8>, memref<1x16x16x16xi8>)
    %105 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_5 = memref.view %105[%c27200][] : memref<106176xi8> to memref<1x18x18x16xi8>
    %base_buffer, %offset, %sizes:4, %strides:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [1, 1, 18, 16], strides: [5184, 288, 16, 1] : memref<i8> to memref<1x1x18x16xi8, strided<[5184, 288, 16, 1]>>
    gemmlir.memset(%reinterpret_cast) {value = 0 : i8} : memref<1x1x18x16xi8, strided<[5184, 288, 16, 1]>>
    %base_buffer_6, %offset_7, %sizes_8:4, %strides_9:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [4896], sizes: [1, 1, 18, 16], strides: [5184, 288, 16, 1] : memref<i8> to memref<1x1x18x16xi8, strided<[5184, 288, 16, 1], offset: 4896>>
    gemmlir.memset(%reinterpret_cast_10) {value = 0 : i8} : memref<1x1x18x16xi8, strided<[5184, 288, 16, 1], offset: 4896>>
    %base_buffer_11, %offset_12, %sizes_13:4, %strides_14:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [288], sizes: [1, 16, 1, 16], strides: [5184, 288, 16, 1] : memref<i8> to memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    cf.br ^bb45(%c0 : index)
  ^bb45(%106: index):  // 2 preds: ^bb44, ^bb55
    %107 = arith.cmpi slt, %106, %c1 : index
    cf.cond_br %107, ^bb46, ^bb56
  ^bb46:  // pred: ^bb45
    cf.br ^bb47(%c0 : index)
  ^bb47(%108: index):  // 2 preds: ^bb46, ^bb54
    %109 = arith.cmpi slt, %108, %c16 : index
    cf.cond_br %109, ^bb48, ^bb55
  ^bb48:  // pred: ^bb47
    cf.br ^bb49(%c0 : index)
  ^bb49(%110: index):  // 2 preds: ^bb48, ^bb53
    %111 = arith.cmpi slt, %110, %c1 : index
    cf.cond_br %111, ^bb50, ^bb54
  ^bb50:  // pred: ^bb49
    cf.br ^bb51(%c0 : index)
  ^bb51(%112: index):  // 2 preds: ^bb50, ^bb52
    %113 = arith.cmpi slt, %112, %c16 : index
    cf.cond_br %113, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %112] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %114 = arith.addi %112, %c1 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %114] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %115 = arith.addi %112, %c2 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %115] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %116 = arith.addi %112, %c3 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %116] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %117 = arith.addi %112, %c4 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %117] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %118 = arith.addi %112, %c5 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %118] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %119 = arith.addi %112, %c6 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %119] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %120 = arith.addi %112, %c7 : index
    memref.store %c0_i8, %reinterpret_cast_15[%106, %108, %110, %120] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 288>>
    %121 = arith.addi %112, %c8 : index
    cf.br ^bb51(%121 : index)
  ^bb53:  // pred: ^bb51
    %122 = arith.addi %110, %c1 : index
    cf.br ^bb49(%122 : index)
  ^bb54:  // pred: ^bb49
    %123 = arith.addi %108, %c1 : index
    cf.br ^bb47(%123 : index)
  ^bb55:  // pred: ^bb47
    %124 = arith.addi %106, %c1 : index
    cf.br ^bb45(%124 : index)
  ^bb56:  // pred: ^bb45
    %base_buffer_16, %offset_17, %sizes_18:4, %strides_19:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_20 = memref.reinterpret_cast %base_buffer_16 to offset: [560], sizes: [1, 16, 1, 16], strides: [5184, 288, 16, 1] : memref<i8> to memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    cf.br ^bb57(%c0 : index)
  ^bb57(%125: index):  // 2 preds: ^bb56, ^bb67
    %126 = arith.cmpi slt, %125, %c1 : index
    cf.cond_br %126, ^bb58, ^bb68
  ^bb58:  // pred: ^bb57
    cf.br ^bb59(%c0 : index)
  ^bb59(%127: index):  // 2 preds: ^bb58, ^bb66
    %128 = arith.cmpi slt, %127, %c16 : index
    cf.cond_br %128, ^bb60, ^bb67
  ^bb60:  // pred: ^bb59
    cf.br ^bb61(%c0 : index)
  ^bb61(%129: index):  // 2 preds: ^bb60, ^bb65
    %130 = arith.cmpi slt, %129, %c1 : index
    cf.cond_br %130, ^bb62, ^bb66
  ^bb62:  // pred: ^bb61
    cf.br ^bb63(%c0 : index)
  ^bb63(%131: index):  // 2 preds: ^bb62, ^bb64
    %132 = arith.cmpi slt, %131, %c16 : index
    cf.cond_br %132, ^bb64, ^bb65
  ^bb64:  // pred: ^bb63
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %131] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %133 = arith.addi %131, %c1 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %133] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %134 = arith.addi %131, %c2 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %134] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %135 = arith.addi %131, %c3 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %135] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %136 = arith.addi %131, %c4 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %136] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %137 = arith.addi %131, %c5 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %137] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %138 = arith.addi %131, %c6 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %138] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %139 = arith.addi %131, %c7 : index
    memref.store %c0_i8, %reinterpret_cast_20[%125, %127, %129, %139] : memref<1x16x1x16xi8, strided<[5184, 288, 16, 1], offset: 560>>
    %140 = arith.addi %131, %c8 : index
    cf.br ^bb63(%140 : index)
  ^bb65:  // pred: ^bb63
    %141 = arith.addi %129, %c1 : index
    cf.br ^bb61(%141 : index)
  ^bb66:  // pred: ^bb61
    %142 = arith.addi %127, %c1 : index
    cf.br ^bb59(%142 : index)
  ^bb67:  // pred: ^bb59
    %143 = arith.addi %125, %c1 : index
    cf.br ^bb57(%143 : index)
  ^bb68:  // pred: ^bb57
    %base_buffer_21, %offset_22, %sizes_23:4, %strides_24:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [304], sizes: [1, 16, 16, 16], strides: [5184, 288, 16, 1] : memref<i8> to memref<1x16x16x16xi8, strided<[5184, 288, 16, 1], offset: 304>>
    memref.copy %view_4, %reinterpret_cast_25 : memref<1x16x16x16xi8> to memref<1x16x16x16xi8, strided<[5184, 288, 16, 1], offset: 304>>
    %144 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_26 = memref.view %144[%c32384][] : memref<106176xi8> to memref<1x16x16x16xi32>
    %145 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_27 = memref.view %145[%c48768][] : memref<106176xi8> to memref<1x16x16x3x3x16xi8>
    %base_buffer_28, %offset_29, %sizes_30:4, %strides_31:4 = memref.extract_strided_metadata %view_5 : memref<1x18x18x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_32 = memref.reinterpret_cast %base_buffer_28 to offset: [0], sizes: [1, 16, 16, 3, 48], strides: [5184, 288, 16, 288, 1] : memref<i8> to memref<1x16x16x3x48xi8, strided<[5184, 288, 16, 288, 1]>>
    %base_buffer_33, %offset_34, %sizes_35:6, %strides_36:6 = memref.extract_strided_metadata %view_27 : memref<1x16x16x3x3x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_37 = memref.reinterpret_cast %base_buffer_33 to offset: [0], sizes: [1, 16, 16, 3, 48], strides: [36864, 2304, 144, 48, 1] : memref<i8> to memref<1x16x16x3x48xi8>
    cf.br ^bb69(%c0 : index)
  ^bb69(%146: index):  // 2 preds: ^bb68, ^bb79
    %147 = arith.cmpi slt, %146, %c1 : index
    cf.cond_br %147, ^bb70, ^bb80
  ^bb70:  // pred: ^bb69
    cf.br ^bb71(%c0 : index)
  ^bb71(%148: index):  // 2 preds: ^bb70, ^bb78
    %149 = arith.cmpi slt, %148, %c16 : index
    cf.cond_br %149, ^bb72, ^bb79
  ^bb72:  // pred: ^bb71
    cf.br ^bb73(%c0 : index)
  ^bb73(%150: index):  // 2 preds: ^bb72, ^bb77
    %151 = arith.cmpi slt, %150, %c16 : index
    cf.cond_br %151, ^bb74, ^bb78
  ^bb74:  // pred: ^bb73
    cf.br ^bb75(%c0 : index)
  ^bb75(%152: index):  // 2 preds: ^bb74, ^bb76
    %153 = arith.cmpi slt, %152, %c3 : index
    cf.cond_br %153, ^bb76, ^bb77
  ^bb76:  // pred: ^bb75
    %154 = vector.load %reinterpret_cast_32[%146, %148, %150, %152, %c0] {alignment = 8 : i64} : memref<1x16x16x3x48xi8, strided<[5184, 288, 16, 288, 1]>>, vector<48xi8>
    vector.store %154, %reinterpret_cast_37[%146, %148, %150, %152, %c0] {alignment = 8 : i64} : memref<1x16x16x3x48xi8>, vector<48xi8>
    %155 = arith.addi %152, %c1 : index
    cf.br ^bb75(%155 : index)
  ^bb77:  // pred: ^bb75
    %156 = arith.addi %150, %c1 : index
    cf.br ^bb73(%156 : index)
  ^bb78:  // pred: ^bb73
    %157 = arith.addi %148, %c1 : index
    cf.br ^bb71(%157 : index)
  ^bb79:  // pred: ^bb71
    %158 = arith.addi %146, %c1 : index
    cf.br ^bb69(%158 : index)
  ^bb80:  // pred: ^bb69
    %base_buffer_38, %offset_39, %sizes_40:6, %strides_41:6 = memref.extract_strided_metadata %view_27 : memref<1x16x16x3x3x16xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_42 = memref.reinterpret_cast %base_buffer_38 to offset: [0], sizes: [256, 144], strides: [144, 1] : memref<i8> to memref<256x144xi8>
    %base_buffer_43, %offset_44, %sizes_45:4, %strides_46:4 = memref.extract_strided_metadata %view_26 : memref<1x16x16x16xi32> -> memref<i32>, index, index, index, index, index, index, index, index, index
    %reinterpret_cast_47 = memref.reinterpret_cast %base_buffer_43 to offset: [0], sizes: [256, 16], strides: [16, 1] : memref<i32> to memref<256x16xi32>
    gemmlir.matmul_i8(%reinterpret_cast_42, %0, %reinterpret_cast_47) : (memref<256x144xi8> x memref<144x16xi8>) -> memref<256x16xi32> {accumulate = false, gemmlir.no_flush_after}
    %159 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_48 = memref.view %159[%c85696][] : memref<106176xi8> to memref<1x16x16x16xf32>
    cf.br ^bb81(%c0 : index)
  ^bb81(%160: index):  // 2 preds: ^bb80, ^bb91
    %161 = arith.cmpi slt, %160, %c1 : index
    cf.cond_br %161, ^bb82, ^bb92
  ^bb82:  // pred: ^bb81
    cf.br ^bb83(%c0 : index)
  ^bb83(%162: index):  // 2 preds: ^bb82, ^bb90
    %163 = arith.cmpi slt, %162, %c16 : index
    cf.cond_br %163, ^bb84, ^bb91
  ^bb84:  // pred: ^bb83
    cf.br ^bb85(%c0 : index)
  ^bb85(%164: index):  // 2 preds: ^bb84, ^bb89
    %165 = arith.cmpi slt, %164, %c16 : index
    cf.cond_br %165, ^bb86, ^bb90
  ^bb86:  // pred: ^bb85
    cf.br ^bb87(%c0 : index)
  ^bb87(%166: index):  // 2 preds: ^bb86, ^bb88
    %167 = arith.cmpi slt, %166, %c16 : index
    cf.cond_br %167, ^bb88, ^bb89
  ^bb88:  // pred: ^bb87
    %168 = memref.load %view_4[%c0, %162, %164, %166] : memref<1x16x16x16xi8>
    %169 = memref.load %view_26[%c0, %162, %164, %166] : memref<1x16x16x16xi32>
    %170 = memref.load %3[%166] : memref<16xf32>
    %171 = arith.sitofp %168 : i8 to f32
    %172 = arith.sitofp %169 : i32 to f32
    %173 = math.fma %172, %cst_3, %170 : f32
    %174 = math.fma %171, %cst_2, %173 : f32
    %175 = arith.maxnumf %174, %cst_1 : f32
    memref.store %175, %view_48[%160, %162, %164, %166] : memref<1x16x16x16xf32>
    %176 = arith.addi %166, %c1 : index
    %177 = memref.load %view_4[%c0, %162, %164, %176] : memref<1x16x16x16xi8>
    %178 = memref.load %view_26[%c0, %162, %164, %176] : memref<1x16x16x16xi32>
    %179 = memref.load %3[%176] : memref<16xf32>
    %180 = arith.sitofp %177 : i8 to f32
    %181 = arith.sitofp %178 : i32 to f32
    %182 = math.fma %181, %cst_3, %179 : f32
    %183 = math.fma %180, %cst_2, %182 : f32
    %184 = arith.maxnumf %183, %cst_1 : f32
    memref.store %184, %view_48[%160, %162, %164, %176] : memref<1x16x16x16xf32>
    %185 = arith.addi %166, %c2 : index
    %186 = memref.load %view_4[%c0, %162, %164, %185] : memref<1x16x16x16xi8>
    %187 = memref.load %view_26[%c0, %162, %164, %185] : memref<1x16x16x16xi32>
    %188 = memref.load %3[%185] : memref<16xf32>
    %189 = arith.sitofp %186 : i8 to f32
    %190 = arith.sitofp %187 : i32 to f32
    %191 = math.fma %190, %cst_3, %188 : f32
    %192 = math.fma %189, %cst_2, %191 : f32
    %193 = arith.maxnumf %192, %cst_1 : f32
    memref.store %193, %view_48[%160, %162, %164, %185] : memref<1x16x16x16xf32>
    %194 = arith.addi %166, %c3 : index
    %195 = memref.load %view_4[%c0, %162, %164, %194] : memref<1x16x16x16xi8>
    %196 = memref.load %view_26[%c0, %162, %164, %194] : memref<1x16x16x16xi32>
    %197 = memref.load %3[%194] : memref<16xf32>
    %198 = arith.sitofp %195 : i8 to f32
    %199 = arith.sitofp %196 : i32 to f32
    %200 = math.fma %199, %cst_3, %197 : f32
    %201 = math.fma %198, %cst_2, %200 : f32
    %202 = arith.maxnumf %201, %cst_1 : f32
    memref.store %202, %view_48[%160, %162, %164, %194] : memref<1x16x16x16xf32>
    %203 = arith.addi %166, %c4 : index
    %204 = memref.load %view_4[%c0, %162, %164, %203] : memref<1x16x16x16xi8>
    %205 = memref.load %view_26[%c0, %162, %164, %203] : memref<1x16x16x16xi32>
    %206 = memref.load %3[%203] : memref<16xf32>
    %207 = arith.sitofp %204 : i8 to f32
    %208 = arith.sitofp %205 : i32 to f32
    %209 = math.fma %208, %cst_3, %206 : f32
    %210 = math.fma %207, %cst_2, %209 : f32
    %211 = arith.maxnumf %210, %cst_1 : f32
    memref.store %211, %view_48[%160, %162, %164, %203] : memref<1x16x16x16xf32>
    %212 = arith.addi %166, %c5 : index
    %213 = memref.load %view_4[%c0, %162, %164, %212] : memref<1x16x16x16xi8>
    %214 = memref.load %view_26[%c0, %162, %164, %212] : memref<1x16x16x16xi32>
    %215 = memref.load %3[%212] : memref<16xf32>
    %216 = arith.sitofp %213 : i8 to f32
    %217 = arith.sitofp %214 : i32 to f32
    %218 = math.fma %217, %cst_3, %215 : f32
    %219 = math.fma %216, %cst_2, %218 : f32
    %220 = arith.maxnumf %219, %cst_1 : f32
    memref.store %220, %view_48[%160, %162, %164, %212] : memref<1x16x16x16xf32>
    %221 = arith.addi %166, %c6 : index
    %222 = memref.load %view_4[%c0, %162, %164, %221] : memref<1x16x16x16xi8>
    %223 = memref.load %view_26[%c0, %162, %164, %221] : memref<1x16x16x16xi32>
    %224 = memref.load %3[%221] : memref<16xf32>
    %225 = arith.sitofp %222 : i8 to f32
    %226 = arith.sitofp %223 : i32 to f32
    %227 = math.fma %226, %cst_3, %224 : f32
    %228 = math.fma %225, %cst_2, %227 : f32
    %229 = arith.maxnumf %228, %cst_1 : f32
    memref.store %229, %view_48[%160, %162, %164, %221] : memref<1x16x16x16xf32>
    %230 = arith.addi %166, %c7 : index
    %231 = memref.load %view_4[%c0, %162, %164, %230] : memref<1x16x16x16xi8>
    %232 = memref.load %view_26[%c0, %162, %164, %230] : memref<1x16x16x16xi32>
    %233 = memref.load %3[%230] : memref<16xf32>
    %234 = arith.sitofp %231 : i8 to f32
    %235 = arith.sitofp %232 : i32 to f32
    %236 = math.fma %235, %cst_3, %233 : f32
    %237 = math.fma %234, %cst_2, %236 : f32
    %238 = arith.maxnumf %237, %cst_1 : f32
    memref.store %238, %view_48[%160, %162, %164, %230] : memref<1x16x16x16xf32>
    %239 = arith.addi %166, %c8 : index
    cf.br ^bb87(%239 : index)
  ^bb89:  // pred: ^bb87
    %240 = arith.addi %164, %c1 : index
    cf.br ^bb85(%240 : index)
  ^bb90:  // pred: ^bb85
    %241 = arith.addi %162, %c1 : index
    cf.br ^bb83(%241 : index)
  ^bb91:  // pred: ^bb83
    %242 = arith.addi %160, %c1 : index
    cf.br ^bb81(%242 : index)
  ^bb92:  // pred: ^bb81
    %243 = memref.get_global @__gemmlir_arena_forward_0 : memref<106176xi8>
    %view_49 = memref.view %243[%c102080][] : memref<106176xi8> to memref<1x8x8x16xf32>
    cf.br ^bb93(%c0 : index)
  ^bb93(%244: index):  // 2 preds: ^bb92, ^bb103
    %245 = arith.cmpi slt, %244, %c1 : index
    cf.cond_br %245, ^bb94, ^bb104
  ^bb94:  // pred: ^bb93
    cf.br ^bb95(%c0 : index)
  ^bb95(%246: index):  // 2 preds: ^bb94, ^bb102
    %247 = arith.cmpi slt, %246, %c8 : index
    cf.cond_br %247, ^bb96, ^bb103
  ^bb96:  // pred: ^bb95
    cf.br ^bb97(%c0 : index)
  ^bb97(%248: index):  // 2 preds: ^bb96, ^bb101
    %249 = arith.cmpi slt, %248, %c8 : index
    cf.cond_br %249, ^bb98, ^bb102
  ^bb98:  // pred: ^bb97
    cf.br ^bb99(%c0 : index)
  ^bb99(%250: index):  // 2 preds: ^bb98, ^bb100
    %251 = arith.cmpi slt, %250, %c16 : index
    cf.cond_br %251, ^bb100, ^bb101
  ^bb100:  // pred: ^bb99
    memref.store %cst_0, %view_49[%244, %246, %248, %250] : memref<1x8x8x16xf32>
    %252 = arith.addi %250, %c1 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %252] : memref<1x8x8x16xf32>
    %253 = arith.addi %250, %c2 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %253] : memref<1x8x8x16xf32>
    %254 = arith.addi %250, %c3 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %254] : memref<1x8x8x16xf32>
    %255 = arith.addi %250, %c4 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %255] : memref<1x8x8x16xf32>
    %256 = arith.addi %250, %c5 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %256] : memref<1x8x8x16xf32>
    %257 = arith.addi %250, %c6 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %257] : memref<1x8x8x16xf32>
    %258 = arith.addi %250, %c7 : index
    memref.store %cst_0, %view_49[%244, %246, %248, %258] : memref<1x8x8x16xf32>
    %259 = arith.addi %250, %c8 : index
    cf.br ^bb99(%259 : index)
  ^bb101:  // pred: ^bb99
    %260 = arith.addi %248, %c1 : index
    cf.br ^bb97(%260 : index)
  ^bb102:  // pred: ^bb97
    %261 = arith.addi %246, %c1 : index
    cf.br ^bb95(%261 : index)
  ^bb103:  // pred: ^bb95
    %262 = arith.addi %244, %c1 : index
    cf.br ^bb93(%262 : index)
  ^bb104:  // pred: ^bb93
    cf.br ^bb105(%c0 : index)
  ^bb105(%263: index):  // 2 preds: ^bb104, ^bb115
    %264 = arith.cmpi slt, %263, %c1 : index
    cf.cond_br %264, ^bb106, ^bb116
  ^bb106:  // pred: ^bb105
    cf.br ^bb107(%c0 : index)
  ^bb107(%265: index):  // 2 preds: ^bb106, ^bb114
    %266 = arith.cmpi slt, %265, %c8 : index
    cf.cond_br %266, ^bb108, ^bb115
  ^bb108:  // pred: ^bb107
    cf.br ^bb109(%c0 : index)
  ^bb109(%267: index):  // 2 preds: ^bb108, ^bb113
    %268 = arith.cmpi slt, %267, %c8 : index
    cf.cond_br %268, ^bb110, ^bb114
  ^bb110:  // pred: ^bb109
    cf.br ^bb111(%c0 : index)
  ^bb111(%269: index):  // 2 preds: ^bb110, ^bb112
    %270 = arith.cmpi slt, %269, %c16 : index
    cf.cond_br %270, ^bb112, ^bb113
  ^bb112:  // pred: ^bb111
    %271 = memref.load %view_49[%263, %265, %267, %269] : memref<1x8x8x16xf32>
    %c2_50 = arith.constant 2 : index
    %272 = arith.muli %265, %c2_50 overflow<nsw> : index
    %273 = arith.addi %272, %c0 : index
    %c2_51 = arith.constant 2 : index
    %274 = arith.muli %267, %c2_51 overflow<nsw> : index
    %275 = arith.addi %274, %c0 : index
    %276 = memref.load %view_48[%263, %273, %275, %269] : memref<1x16x16x16xf32>
    %277 = arith.maximumf %271, %276 : f32
    %c2_52 = arith.constant 2 : index
    %278 = arith.muli %265, %c2_52 overflow<nsw> : index
    %279 = arith.addi %278, %c0 : index
    %c2_53 = arith.constant 2 : index
    %280 = arith.muli %267, %c2_53 overflow<nsw> : index
    %281 = arith.addi %280, %c1 : index
    %282 = memref.load %view_48[%263, %279, %281, %269] : memref<1x16x16x16xf32>
    %283 = arith.maximumf %277, %282 : f32
    %c2_54 = arith.constant 2 : index
    %284 = arith.muli %265, %c2_54 overflow<nsw> : index
    %285 = arith.addi %284, %c1 : index
    %c2_55 = arith.constant 2 : index
    %286 = arith.muli %267, %c2_55 overflow<nsw> : index
    %287 = arith.addi %286, %c0 : index
    %288 = memref.load %view_48[%263, %285, %287, %269] : memref<1x16x16x16xf32>
    %289 = arith.maximumf %283, %288 : f32
    %c2_56 = arith.constant 2 : index
    %290 = arith.muli %265, %c2_56 overflow<nsw> : index
    %291 = arith.addi %290, %c1 : index
    %c2_57 = arith.constant 2 : index
    %292 = arith.muli %267, %c2_57 overflow<nsw> : index
    %293 = arith.addi %292, %c1 : index
    %294 = memref.load %view_48[%263, %291, %293, %269] : memref<1x16x16x16xf32>
    %295 = arith.maximumf %289, %294 : f32
    memref.store %295, %view_49[%263, %265, %267, %269] : memref<1x8x8x16xf32>
    %296 = arith.addi %269, %c1 : index
    %297 = memref.load %view_49[%263, %265, %267, %296] : memref<1x8x8x16xf32>
    %c2_58 = arith.constant 2 : index
    %298 = arith.muli %265, %c2_58 overflow<nsw> : index
    %299 = arith.addi %298, %c0 : index
    %c2_59 = arith.constant 2 : index
    %300 = arith.muli %267, %c2_59 overflow<nsw> : index
    %301 = arith.addi %300, %c0 : index
    %302 = memref.load %view_48[%263, %299, %301, %296] : memref<1x16x16x16xf32>
    %303 = arith.maximumf %297, %302 : f32
    %c2_60 = arith.constant 2 : index
    %304 = arith.muli %265, %c2_60 overflow<nsw> : index
    %305 = arith.addi %304, %c0 : index
    %c2_61 = arith.constant 2 : index
    %306 = arith.muli %267, %c2_61 overflow<nsw> : index
    %307 = arith.addi %306, %c1 : index
    %308 = memref.load %view_48[%263, %305, %307, %296] : memref<1x16x16x16xf32>
    %309 = arith.maximumf %303, %308 : f32
    %c2_62 = arith.constant 2 : index
    %310 = arith.muli %265, %c2_62 overflow<nsw> : index
    %311 = arith.addi %310, %c1 : index
    %c2_63 = arith.constant 2 : index
    %312 = arith.muli %267, %c2_63 overflow<nsw> : index
    %313 = arith.addi %312, %c0 : index
    %314 = memref.load %view_48[%263, %311, %313, %296] : memref<1x16x16x16xf32>
    %315 = arith.maximumf %309, %314 : f32
    %c2_64 = arith.constant 2 : index
    %316 = arith.muli %265, %c2_64 overflow<nsw> : index
    %317 = arith.addi %316, %c1 : index
    %c2_65 = arith.constant 2 : index
    %318 = arith.muli %267, %c2_65 overflow<nsw> : index
    %319 = arith.addi %318, %c1 : index
    %320 = memref.load %view_48[%263, %317, %319, %296] : memref<1x16x16x16xf32>
    %321 = arith.maximumf %315, %320 : f32
    memref.store %321, %view_49[%263, %265, %267, %296] : memref<1x8x8x16xf32>
    %322 = arith.addi %269, %c2 : index
    %323 = memref.load %view_49[%263, %265, %267, %322] : memref<1x8x8x16xf32>
    %c2_66 = arith.constant 2 : index
    %324 = arith.muli %265, %c2_66 overflow<nsw> : index
    %325 = arith.addi %324, %c0 : index
    %c2_67 = arith.constant 2 : index
    %326 = arith.muli %267, %c2_67 overflow<nsw> : index
    %327 = arith.addi %326, %c0 : index
    %328 = memref.load %view_48[%263, %325, %327, %322] : memref<1x16x16x16xf32>
    %329 = arith.maximumf %323, %328 : f32
    %c2_68 = arith.constant 2 : index
    %330 = arith.muli %265, %c2_68 overflow<nsw> : index
    %331 = arith.addi %330, %c0 : index
    %c2_69 = arith.constant 2 : index
    %332 = arith.muli %267, %c2_69 overflow<nsw> : index
    %333 = arith.addi %332, %c1 : index
    %334 = memref.load %view_48[%263, %331, %333, %322] : memref<1x16x16x16xf32>
    %335 = arith.maximumf %329, %334 : f32
    %c2_70 = arith.constant 2 : index
    %336 = arith.muli %265, %c2_70 overflow<nsw> : index
    %337 = arith.addi %336, %c1 : index
    %c2_71 = arith.constant 2 : index
    %338 = arith.muli %267, %c2_71 overflow<nsw> : index
    %339 = arith.addi %338, %c0 : index
    %340 = memref.load %view_48[%263, %337, %339, %322] : memref<1x16x16x16xf32>
    %341 = arith.maximumf %335, %340 : f32
    %c2_72 = arith.constant 2 : index
    %342 = arith.muli %265, %c2_72 overflow<nsw> : index
    %343 = arith.addi %342, %c1 : index
    %c2_73 = arith.constant 2 : index
    %344 = arith.muli %267, %c2_73 overflow<nsw> : index
    %345 = arith.addi %344, %c1 : index
    %346 = memref.load %view_48[%263, %343, %345, %322] : memref<1x16x16x16xf32>
    %347 = arith.maximumf %341, %346 : f32
    memref.store %347, %view_49[%263, %265, %267, %322] : memref<1x8x8x16xf32>
    %348 = arith.addi %269, %c3 : index
    %349 = memref.load %view_49[%263, %265, %267, %348] : memref<1x8x8x16xf32>
    %c2_74 = arith.constant 2 : index
    %350 = arith.muli %265, %c2_74 overflow<nsw> : index
    %351 = arith.addi %350, %c0 : index
    %c2_75 = arith.constant 2 : index
    %352 = arith.muli %267, %c2_75 overflow<nsw> : index
    %353 = arith.addi %352, %c0 : index
    %354 = memref.load %view_48[%263, %351, %353, %348] : memref<1x16x16x16xf32>
    %355 = arith.maximumf %349, %354 : f32
    %c2_76 = arith.constant 2 : index
    %356 = arith.muli %265, %c2_76 overflow<nsw> : index
    %357 = arith.addi %356, %c0 : index
    %c2_77 = arith.constant 2 : index
    %358 = arith.muli %267, %c2_77 overflow<nsw> : index
    %359 = arith.addi %358, %c1 : index
    %360 = memref.load %view_48[%263, %357, %359, %348] : memref<1x16x16x16xf32>
    %361 = arith.maximumf %355, %360 : f32
    %c2_78 = arith.constant 2 : index
    %362 = arith.muli %265, %c2_78 overflow<nsw> : index
    %363 = arith.addi %362, %c1 : index
    %c2_79 = arith.constant 2 : index
    %364 = arith.muli %267, %c2_79 overflow<nsw> : index
    %365 = arith.addi %364, %c0 : index
    %366 = memref.load %view_48[%263, %363, %365, %348] : memref<1x16x16x16xf32>
    %367 = arith.maximumf %361, %366 : f32
    %c2_80 = arith.constant 2 : index
    %368 = arith.muli %265, %c2_80 overflow<nsw> : index
    %369 = arith.addi %368, %c1 : index
    %c2_81 = arith.constant 2 : index
    %370 = arith.muli %267, %c2_81 overflow<nsw> : index
    %371 = arith.addi %370, %c1 : index
    %372 = memref.load %view_48[%263, %369, %371, %348] : memref<1x16x16x16xf32>
    %373 = arith.maximumf %367, %372 : f32
    memref.store %373, %view_49[%263, %265, %267, %348] : memref<1x8x8x16xf32>
    %374 = arith.addi %269, %c4 : index
    %375 = memref.load %view_49[%263, %265, %267, %374] : memref<1x8x8x16xf32>
    %c2_82 = arith.constant 2 : index
    %376 = arith.muli %265, %c2_82 overflow<nsw> : index
    %377 = arith.addi %376, %c0 : index
    %c2_83 = arith.constant 2 : index
    %378 = arith.muli %267, %c2_83 overflow<nsw> : index
    %379 = arith.addi %378, %c0 : index
    %380 = memref.load %view_48[%263, %377, %379, %374] : memref<1x16x16x16xf32>
    %381 = arith.maximumf %375, %380 : f32
    %c2_84 = arith.constant 2 : index
    %382 = arith.muli %265, %c2_84 overflow<nsw> : index
    %383 = arith.addi %382, %c0 : index
    %c2_85 = arith.constant 2 : index
    %384 = arith.muli %267, %c2_85 overflow<nsw> : index
    %385 = arith.addi %384, %c1 : index
    %386 = memref.load %view_48[%263, %383, %385, %374] : memref<1x16x16x16xf32>
    %387 = arith.maximumf %381, %386 : f32
    %c2_86 = arith.constant 2 : index
    %388 = arith.muli %265, %c2_86 overflow<nsw> : index
    %389 = arith.addi %388, %c1 : index
    %c2_87 = arith.constant 2 : index
    %390 = arith.muli %267, %c2_87 overflow<nsw> : index
    %391 = arith.addi %390, %c0 : index
    %392 = memref.load %view_48[%263, %389, %391, %374] : memref<1x16x16x16xf32>
    %393 = arith.maximumf %387, %392 : f32
    %c2_88 = arith.constant 2 : index
    %394 = arith.muli %265, %c2_88 overflow<nsw> : index
    %395 = arith.addi %394, %c1 : index
    %c2_89 = arith.constant 2 : index
    %396 = arith.muli %267, %c2_89 overflow<nsw> : index
    %397 = arith.addi %396, %c1 : index
    %398 = memref.load %view_48[%263, %395, %397, %374] : memref<1x16x16x16xf32>
    %399 = arith.maximumf %393, %398 : f32
    memref.store %399, %view_49[%263, %265, %267, %374] : memref<1x8x8x16xf32>
    %400 = arith.addi %269, %c5 : index
    %401 = memref.load %view_49[%263, %265, %267, %400] : memref<1x8x8x16xf32>
    %c2_90 = arith.constant 2 : index
    %402 = arith.muli %265, %c2_90 overflow<nsw> : index
    %403 = arith.addi %402, %c0 : index
    %c2_91 = arith.constant 2 : index
    %404 = arith.muli %267, %c2_91 overflow<nsw> : index
    %405 = arith.addi %404, %c0 : index
    %406 = memref.load %view_48[%263, %403, %405, %400] : memref<1x16x16x16xf32>
    %407 = arith.maximumf %401, %406 : f32
    %c2_92 = arith.constant 2 : index
    %408 = arith.muli %265, %c2_92 overflow<nsw> : index
    %409 = arith.addi %408, %c0 : index
    %c2_93 = arith.constant 2 : index
    %410 = arith.muli %267, %c2_93 overflow<nsw> : index
    %411 = arith.addi %410, %c1 : index
    %412 = memref.load %view_48[%263, %409, %411, %400] : memref<1x16x16x16xf32>
    %413 = arith.maximumf %407, %412 : f32
    %c2_94 = arith.constant 2 : index
    %414 = arith.muli %265, %c2_94 overflow<nsw> : index
    %415 = arith.addi %414, %c1 : index
    %c2_95 = arith.constant 2 : index
    %416 = arith.muli %267, %c2_95 overflow<nsw> : index
    %417 = arith.addi %416, %c0 : index
    %418 = memref.load %view_48[%263, %415, %417, %400] : memref<1x16x16x16xf32>
    %419 = arith.maximumf %413, %418 : f32
    %c2_96 = arith.constant 2 : index
    %420 = arith.muli %265, %c2_96 overflow<nsw> : index
    %421 = arith.addi %420, %c1 : index
    %c2_97 = arith.constant 2 : index
    %422 = arith.muli %267, %c2_97 overflow<nsw> : index
    %423 = arith.addi %422, %c1 : index
    %424 = memref.load %view_48[%263, %421, %423, %400] : memref<1x16x16x16xf32>
    %425 = arith.maximumf %419, %424 : f32
    memref.store %425, %view_49[%263, %265, %267, %400] : memref<1x8x8x16xf32>
    %426 = arith.addi %269, %c6 : index
    %427 = memref.load %view_49[%263, %265, %267, %426] : memref<1x8x8x16xf32>
    %c2_98 = arith.constant 2 : index
    %428 = arith.muli %265, %c2_98 overflow<nsw> : index
    %429 = arith.addi %428, %c0 : index
    %c2_99 = arith.constant 2 : index
    %430 = arith.muli %267, %c2_99 overflow<nsw> : index
    %431 = arith.addi %430, %c0 : index
    %432 = memref.load %view_48[%263, %429, %431, %426] : memref<1x16x16x16xf32>
    %433 = arith.maximumf %427, %432 : f32
    %c2_100 = arith.constant 2 : index
    %434 = arith.muli %265, %c2_100 overflow<nsw> : index
    %435 = arith.addi %434, %c0 : index
    %c2_101 = arith.constant 2 : index
    %436 = arith.muli %267, %c2_101 overflow<nsw> : index
    %437 = arith.addi %436, %c1 : index
    %438 = memref.load %view_48[%263, %435, %437, %426] : memref<1x16x16x16xf32>
    %439 = arith.maximumf %433, %438 : f32
    %c2_102 = arith.constant 2 : index
    %440 = arith.muli %265, %c2_102 overflow<nsw> : index
    %441 = arith.addi %440, %c1 : index
    %c2_103 = arith.constant 2 : index
    %442 = arith.muli %267, %c2_103 overflow<nsw> : index
    %443 = arith.addi %442, %c0 : index
    %444 = memref.load %view_48[%263, %441, %443, %426] : memref<1x16x16x16xf32>
    %445 = arith.maximumf %439, %444 : f32
    %c2_104 = arith.constant 2 : index
    %446 = arith.muli %265, %c2_104 overflow<nsw> : index
    %447 = arith.addi %446, %c1 : index
    %c2_105 = arith.constant 2 : index
    %448 = arith.muli %267, %c2_105 overflow<nsw> : index
    %449 = arith.addi %448, %c1 : index
    %450 = memref.load %view_48[%263, %447, %449, %426] : memref<1x16x16x16xf32>
    %451 = arith.maximumf %445, %450 : f32
    memref.store %451, %view_49[%263, %265, %267, %426] : memref<1x8x8x16xf32>
    %452 = arith.addi %269, %c7 : index
    %453 = memref.load %view_49[%263, %265, %267, %452] : memref<1x8x8x16xf32>
    %c2_106 = arith.constant 2 : index
    %454 = arith.muli %265, %c2_106 overflow<nsw> : index
    %455 = arith.addi %454, %c0 : index
    %c2_107 = arith.constant 2 : index
    %456 = arith.muli %267, %c2_107 overflow<nsw> : index
    %457 = arith.addi %456, %c0 : index
    %458 = memref.load %view_48[%263, %455, %457, %452] : memref<1x16x16x16xf32>
    %459 = arith.maximumf %453, %458 : f32
    %c2_108 = arith.constant 2 : index
    %460 = arith.muli %265, %c2_108 overflow<nsw> : index
    %461 = arith.addi %460, %c0 : index
    %c2_109 = arith.constant 2 : index
    %462 = arith.muli %267, %c2_109 overflow<nsw> : index
    %463 = arith.addi %462, %c1 : index
    %464 = memref.load %view_48[%263, %461, %463, %452] : memref<1x16x16x16xf32>
    %465 = arith.maximumf %459, %464 : f32
    %c2_110 = arith.constant 2 : index
    %466 = arith.muli %265, %c2_110 overflow<nsw> : index
    %467 = arith.addi %466, %c1 : index
    %c2_111 = arith.constant 2 : index
    %468 = arith.muli %267, %c2_111 overflow<nsw> : index
    %469 = arith.addi %468, %c0 : index
    %470 = memref.load %view_48[%263, %467, %469, %452] : memref<1x16x16x16xf32>
    %471 = arith.maximumf %465, %470 : f32
    %c2_112 = arith.constant 2 : index
    %472 = arith.muli %265, %c2_112 overflow<nsw> : index
    %473 = arith.addi %472, %c1 : index
    %c2_113 = arith.constant 2 : index
    %474 = arith.muli %267, %c2_113 overflow<nsw> : index
    %475 = arith.addi %474, %c1 : index
    %476 = memref.load %view_48[%263, %473, %475, %452] : memref<1x16x16x16xf32>
    %477 = arith.maximumf %471, %476 : f32
    memref.store %477, %view_49[%263, %265, %267, %452] : memref<1x8x8x16xf32>
    %478 = arith.addi %269, %c8 : index
    cf.br ^bb111(%478 : index)
  ^bb113:  // pred: ^bb111
    %479 = arith.addi %267, %c1 : index
    cf.br ^bb109(%479 : index)
  ^bb114:  // pred: ^bb109
    %480 = arith.addi %265, %c1 : index
    cf.br ^bb107(%480 : index)
  ^bb115:  // pred: ^bb107
    %481 = arith.addi %263, %c1 : index
    cf.br ^bb105(%481 : index)
  ^bb116:  // pred: ^bb105
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16x8x8xf32>
    cf.br ^bb117(%c0 : index)
  ^bb117(%482: index):  // 2 preds: ^bb116, ^bb127
    %483 = arith.cmpi slt, %482, %c1 : index
    cf.cond_br %483, ^bb118, ^bb128
  ^bb118:  // pred: ^bb117
    cf.br ^bb119(%c0 : index)
  ^bb119(%484: index):  // 2 preds: ^bb118, ^bb126
    %485 = arith.cmpi slt, %484, %c8 : index
    cf.cond_br %485, ^bb120, ^bb127
  ^bb120:  // pred: ^bb119
    cf.br ^bb121(%c0 : index)
  ^bb121(%486: index):  // 2 preds: ^bb120, ^bb125
    %487 = arith.cmpi slt, %486, %c16 : index
    cf.cond_br %487, ^bb122, ^bb126
  ^bb122:  // pred: ^bb121
    cf.br ^bb123(%c0 : index)
  ^bb123(%488: index):  // 2 preds: ^bb122, ^bb124
    %489 = arith.cmpi slt, %488, %c8 : index
    cf.cond_br %489, ^bb124, ^bb125
  ^bb124:  // pred: ^bb123
    %490 = memref.load %view_49[%482, %484, %488, %486] : memref<1x8x8x16xf32>
    memref.store %490, %alloc[%482, %486, %484, %488] : memref<1x16x8x8xf32>
    %491 = arith.addi %488, %c1 : index
    %492 = memref.load %view_49[%482, %484, %491, %486] : memref<1x8x8x16xf32>
    memref.store %492, %alloc[%482, %486, %484, %491] : memref<1x16x8x8xf32>
    %493 = arith.addi %488, %c2 : index
    %494 = memref.load %view_49[%482, %484, %493, %486] : memref<1x8x8x16xf32>
    memref.store %494, %alloc[%482, %486, %484, %493] : memref<1x16x8x8xf32>
    %495 = arith.addi %488, %c3 : index
    %496 = memref.load %view_49[%482, %484, %495, %486] : memref<1x8x8x16xf32>
    memref.store %496, %alloc[%482, %486, %484, %495] : memref<1x16x8x8xf32>
    %497 = arith.addi %488, %c4 : index
    cf.br ^bb123(%497 : index)
  ^bb125:  // pred: ^bb123
    %498 = arith.addi %486, %c1 : index
    cf.br ^bb121(%498 : index)
  ^bb126:  // pred: ^bb121
    %499 = arith.addi %484, %c1 : index
    cf.br ^bb119(%499 : index)
  ^bb127:  // pred: ^bb119
    %500 = arith.addi %482, %c1 : index
    cf.br ^bb117(%500 : index)
  ^bb128:  // pred: ^bb117
    return %alloc : memref<1x16x8x8xf32>
  }
}

