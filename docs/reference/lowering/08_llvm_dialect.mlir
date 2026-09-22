module attributes {llvm.data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", llvm.target_triple = "riscv64-unknown-linux-gnu", torch.debug_module_name = "Block"} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @tiled_matmul_auto(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, f32, i32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32)
  llvm.func @memrefCopy(i64, !llvm.ptr, !llvm.ptr)
  llvm.func @gemmlir_memset(!llvm.ptr, i32, i64)
  llvm.func @gemmlir_flush()
  llvm.func @tiled_conv_stride_auto(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, f32, i32, i32, i32, i32)
  llvm.mlir.global private @__gemmlir_arena_forward_0() {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<106176 x i8> {
    %0 = llvm.mlir.undef : !llvm.array<106176 x i8>
    llvm.return %0 : !llvm.array<106176 x i8>
  }
  llvm.mlir.global private constant @__constant_16xf32(dense<[-0.0932260155, -0.0050244038, -0.0149413012, 0.292073935, 0.0149224252, 0.24778448, 0.249534339, -0.10057988, -0.0420088544, 0.459543973, -0.305139065, -0.245570511, 0.427169263, -0.234537929, -0.161344975, -0.167135388]> : tensor<16xf32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<16 x f32>
  llvm.mlir.global private constant @__constant_3x3x8x16xi8(dense<"0xFF29AAB1ECB4493FE7AF44764A55E22F1A5D3432BDEEACE4DA5ED4CB04ECF7B4BE26AA3E0F51471A22F219C6C94CBAFFA6C4D9003C46E749484FD43843831699C7FA0A543E50BDD93F2EDC1D3F7F3A0FCE05343A39AA28494861C5CED78358EF41E6C7D049B0352F30A43436210B33E01DF5511D4636F7F321CF0594C5E8F4C23457D3CBFC9106D8BAE7C221E095DFA7E34F34B825B7EF130FCFF063FE87BAD4D635ADC026572ECA214FC4D5474131F4C3625634D1D9ABC8DFF1F77B1481A3255410ECE251F3E511DF463F054A5312A93D0F4C36C26CCFB91209EED6271C95B2C7CDFED032DA30041417CA5C0A6374C535032303F038B83AD3A7D20239F53C3BB05B4152CAA5C7181F38F3D3FA5C152CED98A4E3E24E3A1A1D2543A90B88234F234BF12E236456D4BEE0BD222C11D7F5E746239E0136DEC3DDFCE6DA3510E3022BD544C435BF26EF1BF0DD4F45B854963919414E235629B8B6A049201BBBF5AC12DB04A0CAE103B718BF54062518CDF5F4F70CBC2399D1260BA2DDC7189DB38CB85CFFAE1E59E11F4AD6CECDDF8814F2A3143CDED5ABAEDC3CE6ED003A3145CF512701B8F5E32CDA465205AA29BA54E8DAD2AA49250D13C72846BD43E0609BC72F3E1BDA0C5755DC42ABC6E7330B33D1D529BD2544D7C34AEA48C0EA0C126430B4AA3E1650EAE5D407F5CE18B4D7B2010415FFAE10EC4C3F1864E7524BFF771DDA15D0F91C3D13EAEA56D0B1F61C3D31BFEEB5F110C948073BEA08704ACAB44BEC99D5D5B732C745489D43142A2A606054E63DAC07E5BC00DDE400DC23B34A8B05C02E9B2A48320A26330D0C35418BECFC0604DE593FD5D1CCB4E75112AD2FA3BCEAB8C72B0BEDE71C0D5405BA56F25D1742ED5126AC4F2B05E6AE8CB59687A21AA5B764FE6ABE19C1BA14AA10ADAC56D8EFEC01D293DE2BF54BC57BD3B3BAD049BC241EF26448EC44F1E836C406125AC19CF50107C01BEDDB5A3FEEBD1CB3D1CE5F222ED408F63D0B1F32D2F6704D923ECDBDE44EC0B90E42181C0742A066A4CE16BEA531680B26EE57AF562E078E873C630C11E2D0B0E9CF05044BB7E58C3CFE0AA725EC5A2BEBBBAA4FEC0ACDF73504D0D95415C019F9BDB147B8FBCE36E4F034AA221F250500C0EE28CF08A5F6AED3FED5EF0BC12ECFF9BC47EBFB596F291048E9CFB6CB1ECE4129B498C581E9AF6131CB1E45EE2DFAF30FB62F1CF5D6452CBCC859C8062BD0E2CFB5CC301830F75E42EA542C293DCE491C0F7B334AE3984DBFAF03C99BB22E13D2BDEAF21744F62728EE0D125D0417BB0BF516E298F8030A16FA4CDF7129DCE3214867461BDA38BCB14CCBBAA90CE5F9F508FF1927E190A5A8ECBCB368AB030AC14C32E445CA08279CC853EB60F7F0BBD02E36BED5BD1727FB43BE1ABAC446B94234BEBCD75936B563FFDEC2B02BFAD446001943A7634BF79FEDA83EB3ADEBFA113337B4CE69F93AC2ED5BBE5CB2EE4760C30FBA9A8E9158AE54DF223ABB4BCC2838F8D674FAA1A5AFB30D074CC60EC5FB13D1171CBB8ABAE6B1C5592EB6E4DD421E10F8DB4D8D0DC2211CE10542EFFB242CE2C94A686FC64BDEF4E10648EFC7F309361D001DBDDCF80C1E142A333E3E571BF82EAA0C8D"> : tensor<3x3x8x16xi8>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<3 x array<3 x array<8 x array<16 x i8>>>>
  llvm.mlir.global private constant @__constant_16xi32(dense<[-1550, -2474, 1465, -2521, 2753, 5854, -5913, 1243, -4467, -7415, -5871, -7936, -1411, 13854, -5680, -558]> : tensor<16xi32>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<16 x i32>
  llvm.mlir.global private constant @__constant_144x16xi8(dense<"0x9720D01F1DE1F345A1BCAFB2DE16BF21D9259CFAD9AB043949EFCDE796E709AFD71ADD2C050EE2D9560CAEBB5B06EF2AB4C53605E5307D13C7C952B9AD1D3DCF6D3FA60049A70CF8119FEDD39249C7B1D5155037281B3146C76C4E4CB21CE61DB0DAC84004FF194E28B3102FCE1237EC8512DA1C4A04C5F20624FBD928E3FD05F3BDC6070EF2B71C0CEA010643D4FFDEDD1B382849E322B5ED171BBCC1E332152DFE561620BD11FDECFCD3D2501FCCB677C9FCC650EDE7503A42000A4BBCCE2E8C14C92548B7DCF2019132E4340BDEDE6B4AACB8B65C23C9AC214406C959BC1150DAEF3E45E392A81A1ED8DDF4F94CC46802E4BE592962F8F516B8C864AED4ED02DCEEF6B52051FDA6C9D5AE19D02C52C1D949524FC86AB9216ADEEC1DCB1E3562362AF2D3573BF1F50F0D5B24E10145EAB15EE1CE2A3E1FC268CC0D04FEE7499323D0B9D240D4C0E49A06276CD5F81041052BEDEB26EFA303A505E8C923D9EE78513C0ED61D4E393C502409A1D90BCD362FD6E14A28CE105957EED7E2C037FB3F44A4D70025CCEFA8284D16A118D31E3EF7E7CDBFDB9EAE142CAF0317B83C408CE3F1EDB74DD35B9DB557ED43CCFCA7391960F7AC269405EBCBCDFA5143BBFD0D0725D254C943CBDAEF00D26DA80DD3F04AB52825A3AC249CBBD2FFD14E1AD555D8AC133447AA231203C113281D30A7DE3A2B42D5D2400E1EE1AFA8773E0BC76244E5BC0F1AA4D53A199D54334DCF0369C351002FA4C9D1BD18A6E34423B059A7ACD204AA583122A7BB9ED866DFDD04CE38DE15F01B16EAA6C815EEDA3513E8FBD554C6FAF6F3F2F230DFD8A707F4F3E14AC9F64E4720FB17F2D32275320F18F54BEB393403A052EACF3EB26212B9013E51EBCB3128ABB3F41ED9BDC853E259ABD0E328BD169CA254A603026CDC26399E15D84F2FC472BF9FBB19F28A3CE8EFBACC554FE7E05F4B61A29FF897F7250490DDBBC92CB8BCC1070B1A1840C20C576B3AD540061B684B28011DA94512DA126904ACE11BCF3719200AB1209C2AD156A3ACC7E14D34C1CDFBA0CC1B73A6B941A9DEBEDACF58132229474610DFF0434D95BCA8214DC9AA10C52F434009D9F7B434B444BEBC18DD5B180FF62CEDCD0516A7C4F4F7D0F9BCF4FA531B361EE9ECA965F605BDE0DA53DAFF9B3335084C04ED49E3224C59AEFDCB61F3DB3FE9DDB63535E14F3BCDE320C52D3DE8F05A5516C9AD4FA9B4CA0353E4FABB16A3392CCEE4B2CBDB2025AE964CABA33C0F62E521DE3F36A30E1658954238B8E713514637FE31543F3B2F4681E8CE0F3DD13919D00E8E3D00E008D45048319D0AD10DF0E5313F2B26D5A8005C4FCCFF34FCEB26DFCAA5DD18504C16502CA7DBB13EEF57FDBF8C3215E25B3C67242F42A225E3DE2AE5F2082ADC3011FAF8352C5FD5E3ACFD4D55D20AEA06BF2040F76DE1EE2FD8D7C6C8F4EE363F0993B249D4630820150BE7A504DFFA50B75413E4F2073242B628EFD93FE313EEA35437073BA3CC404017A786E1141934FFCB1B1F31205BA11BC8B13726613E204B6D31FDBDA9483EBA11D770EA0900E3CB7CAFA366F5E1094512B8C8F85BC5E8A66AC650162ED3481D17381F2D370FE8AD66BBD2DE5CF32844D9F346AFEDB8BC55715D5733F9DA00AE1C311DC9570F42E50D3725300C31513CB31DA7FAF5B60BBE17F7C4B29FEAD753CBF430F85A2AA7DDFC10AB3400F495DF0022291E6314D6335F1BCDB7B1FC101AF651BDAB14BFDAF193F6ECBF24C3F4ABD34B123AA32E030E171A6144ED3A4900B7DF55D329001D2C9EA7BF469EB2771D08A96E021FF250B8B2E71FD73AB178E6D11B14523429FF1FE0F1D8461B140423CC320214A20346E1F8B343A9BA014B1BB2ECB0EA642DD20CBDE7F9BCF315491EE1AF66E14A2ED1CD72C258FD0639DA302C1B744942CA22F9860905B101D40AB130B6BB4DFEB1B2ADC1C2AF5F35560C421CB65C1CEDC531B7F05C5C16AE05CF39FF17B4F102CB2DC49E2F2C92204CCFE8B14A4FFBA42517DDF5D9FB60F7F85ACA1FFEC850C6D9DFD3DD02B7D0FDFDC7DFC7516107EFC2C62EF832BDB356208D5AF33937D1224BDBF28E1626B2FACEB4004AF8E5CED52221CE6BF4A720A628C9CC40F92022C70F5031DB0D54D753A8F9E23ED4D9F9F9F12FBC34D9B4DC5C0024EE4ADCEB2CD72732E4031A195820E8D72D2C30FFF0A2E5FCBE7A31F5C9BAF45BC914E522FBEFCE44DCE0D20398EB2014D43E319AD25CF70ED0DBD961EEA1FD9235E306FD160139C8219E4B0D9AF5016B4944248EBB2AC04A4AD0D2E62FEF3C9ADC2BC733DB5142A708164EBBB65EBEAFF54CDF122D4CCA2D02FCDBEAA236280BFC304B6601FA12C900EE5CDE4D19F6180EC6D094E0EA1D3340E2FCD0AC4D210ACB2A466730DFB4A4B6053361B2252A545849F085C55027F45232AEF505C0B4260DB9C82112F7AEF431D302AB4AF3DA38EDDBE3DB0F9EC4DB3A3BF9C6FC9C4655584F2D40D84B45C93B873D5B9102DDD23FF10FE7AF1210261E81AC614E02F02F2B49123445FC1FF8DBC6FA50C618A85201C7A6CDFCE7D5E7C3CD38D14CF4F703DE00BD040FAB0FC8ECFD481DE3A0AAE4C13AC4E9E7CCC0CCF8AD0CEDDD49D233E1EC0FAC17C220DF5E2F25DB1E582BCBCF442DAD06352E5908AA44412E1ECD5A30D4DA93E3B2FF342FBFD1073CA433313129C6E628172720C4C1D6D4134527E6B525447B0FCF33E210DB1ADFAAFBF4B10936EC984AF7CF495757FFA9C8A84F47DBF20DE74A2044E8C972E1009EE956EFC81D124DB4023C1AF3AC11B7682BB1FFB1B7AB7CED19F70AAC0F3632DC4B19A3184C4B730036EF2E06AC190B3DCD0645AD053FCEB2CD1F58D9F10CDE5BF3C9491AFF31A4B5B4494F42CBDBE537BD262C17F021883900DF18531CF6A3AD612067321446B9C2EC44E7D0FFF0581AADFEB3E945C0E2F33A02FC24590C1B4A512B1902221861B7AB00E3F43717BFF7A8234F10B83153CB622EE30431B04A35DFABF022C7E1CC52A8440F42C2B6224353F0EECC380D8C2023DA063F1BF4D8F322430E1D1D0B09E0F13FFDE259F32DC412E3F607EE0ACF0746E6D4522FF931FE20AC4E4BF4AF3849DCFF12FC6CC34021D2D52906B1A748432CDB24AA0EDAC409614513DBBE516620B500B9D0F73963503900B410C11D663BC610BF1994EB08F949BC53D8C2193BBDC725FB047AB8044AAB2F96A911BDDABCD6425BF66117DEE6E3C995B1CE1698532DCFE90BDE39F20BB1B7CFD94C23"> : tensor<144x16xi8>) {addr_space = 0 : i32, alignment = 64 : i64} : !llvm.array<144 x array<16 x i8>>
  llvm.mlir.global private @global_seed(0 : i64) {addr_space = 0 : i32} : i64
  llvm.func @forward(%arg0: !llvm.ptr) -> !llvm.ptr {
    %0 = llvm.mlir.constant(64 : index) : i64
    %1 = llvm.mlir.constant(1024 : index) : i64
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.constant(144 : i64) : i64
    %4 = llvm.mlir.constant(16 : i64) : i64
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %7 = llvm.mlir.constant(36864 : index) : i64
    %8 = llvm.mlir.constant(48 : index) : i64
    %9 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %10 = llvm.mlir.constant(4 : i64) : i64
    %11 = llvm.mlir.constant(304 : index) : i64
    %12 = llvm.mlir.constant(288 : i64) : i64
    %13 = llvm.mlir.constant(5184 : index) : i64
    %14 = llvm.mlir.constant(288 : index) : i64
    %15 = llvm.mlir.constant(0 : i32) : i32
    %16 = llvm.mlir.constant(0.00155569159 : f32) : f32
    %17 = llvm.mlir.constant(false) : i1
    %18 = llvm.mlir.constant(3 : i32) : i32
    %19 = llvm.mlir.constant(8 : i32) : i32
    %20 = llvm.mlir.constant(16 : i32) : i32
    %21 = llvm.mlir.constant(1 : i32) : i32
    %22 = llvm.mlir.constant(4096 : index) : i64
    %23 = llvm.mlir.addressof @__gemmlir_arena_forward_0 : !llvm.ptr
    %24 = llvm.mlir.addressof @__constant_16xf32 : !llvm.ptr
    %25 = llvm.mlir.addressof @__constant_3x3x8x16xi8 : !llvm.ptr
    %26 = llvm.mlir.constant(128 : index) : i64
    %27 = llvm.mlir.addressof @__constant_16xi32 : !llvm.ptr
    %28 = llvm.mlir.constant(3735928559 : index) : i64
    %29 = llvm.mlir.addressof @__constant_144x16xi8 : !llvm.ptr
    %30 = llvm.mlir.zero : !llvm.ptr
    %31 = llvm.mlir.constant(2304 : index) : i64
    %32 = llvm.mlir.constant(144 : index) : i64
    %33 = llvm.mlir.constant(2.12572231E-5 : f32) : f32
    %34 = llvm.mlir.constant(0.024088297 : f32) : f32
    %35 = llvm.mlir.constant(0 : i8) : i8
    %36 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %37 = llvm.mlir.constant(0xFF800000 : f32) : f32
    %38 = llvm.mlir.constant(3 : index) : i64
    %39 = llvm.mlir.constant(3.488010e+01 : f32) : f32
    %40 = llvm.mlir.constant(-128 : i32) : i32
    %41 = llvm.mlir.constant(256 : i32) : i32
    %42 = llvm.mlir.constant(127 : i32) : i32
    %43 = llvm.mlir.constant(4 : index) : i64
    %44 = llvm.mlir.constant(2 : index) : i64
    %45 = llvm.mlir.constant(5 : index) : i64
    %46 = llvm.mlir.constant(6 : index) : i64
    %47 = llvm.mlir.constant(7 : index) : i64
    %48 = llvm.mlir.constant(16 : index) : i64
    %49 = llvm.mlir.constant(256 : index) : i64
    %50 = llvm.mlir.constant(8 : index) : i64
    %51 = llvm.mlir.constant(2048 : index) : i64
    %52 = llvm.mlir.constant(1 : index) : i64
    %53 = llvm.mlir.constant(0 : index) : i64
    %54 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
    %55 = llvm.getelementptr %29[0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<144 x array<16 x i8>>
    %56 = llvm.getelementptr %27[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x i32>
    %57 = llvm.getelementptr %25[0, 0, 0, 0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x array<3 x array<8 x array<16 x i8>>>>
    %58 = llvm.getelementptr %24[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x f32>
    %59 = llvm.getelementptr %23[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<106176 x i8>
    %60 = llvm.getelementptr %59[2048] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.br ^bb1(%53 : i64)
  ^bb1(%61: i64):  // 2 preds: ^bb0, ^bb43
    %62 = llvm.icmp "slt" %61, %52 : i64
    llvm.cond_br %62, ^bb2, ^bb44
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%53 : i64)
  ^bb3(%63: i64):  // 2 preds: ^bb2, ^bb42
    %64 = llvm.icmp "slt" %63, %50 : i64
    llvm.cond_br %64, ^bb4, ^bb43
  ^bb4:  // pred: ^bb3
    llvm.br ^bb5(%53 : i64)
  ^bb5(%65: i64):  // 2 preds: ^bb4, ^bb41
    %66 = llvm.icmp "slt" %65, %48 : i64
    llvm.cond_br %66, ^bb6, ^bb42
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%53 : i64)
  ^bb7(%67: i64):  // 2 preds: ^bb6, ^bb40
    %68 = llvm.icmp "slt" %67, %48 : i64
    llvm.cond_br %68, ^bb8, ^bb41
  ^bb8:  // pred: ^bb7
    %69 = llvm.mul %61, %51 overflow<nsw, nuw> : i64
    %70 = llvm.mul %63, %49 overflow<nsw, nuw> : i64
    %71 = llvm.add %69, %70 overflow<nsw, nuw> : i64
    %72 = llvm.mul %65, %48 overflow<nsw, nuw> : i64
    %73 = llvm.add %71, %72 overflow<nsw, nuw> : i64
    %74 = llvm.add %73, %67 overflow<nsw, nuw> : i64
    %75 = llvm.getelementptr inbounds|nuw %arg0[%74] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %76 = llvm.load %75 : !llvm.ptr -> f32
    %77 = llvm.fmul %76, %39 : f32
    %78 = llvm.intr.roundeven(%77) : (f32) -> f32
    %79 = llvm.fptosi %78 : f32 to i32
    %80 = llvm.sub %79, %40 : i32
    %81 = llvm.icmp "ult" %80, %41 : i32
    llvm.cond_br %81, ^bb9, ^bb10
  ^bb9:  // pred: ^bb8
    llvm.br ^bb11(%79 : i32)
  ^bb10:  // pred: ^bb8
    %82 = llvm.icmp "slt" %79, %40 : i32
    %83 = llvm.select %82, %40, %42 : i1, i32
    llvm.br ^bb11(%83 : i32)
  ^bb11(%84: i32):  // 2 preds: ^bb9, ^bb10
    llvm.br ^bb12
  ^bb12:  // pred: ^bb11
    %85 = llvm.trunc %84 : i32 to i8
    %86 = llvm.mul %65, %26 overflow<nsw, nuw> : i64
    %87 = llvm.add %69, %86 overflow<nsw, nuw> : i64
    %88 = llvm.mul %67, %50 overflow<nsw, nuw> : i64
    %89 = llvm.add %87, %88 overflow<nsw, nuw> : i64
    %90 = llvm.add %89, %63 overflow<nsw, nuw> : i64
    %91 = llvm.getelementptr inbounds|nuw %60[%90] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %85, %91 : i8, !llvm.ptr
    %92 = llvm.add %67, %52 : i64
    %93 = llvm.add %73, %92 overflow<nsw, nuw> : i64
    %94 = llvm.getelementptr inbounds|nuw %arg0[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %95 = llvm.load %94 : !llvm.ptr -> f32
    %96 = llvm.fmul %95, %39 : f32
    %97 = llvm.intr.roundeven(%96) : (f32) -> f32
    %98 = llvm.fptosi %97 : f32 to i32
    %99 = llvm.sub %98, %40 : i32
    %100 = llvm.icmp "ult" %99, %41 : i32
    llvm.cond_br %100, ^bb13, ^bb14
  ^bb13:  // pred: ^bb12
    llvm.br ^bb15(%98 : i32)
  ^bb14:  // pred: ^bb12
    %101 = llvm.icmp "slt" %98, %40 : i32
    %102 = llvm.select %101, %40, %42 : i1, i32
    llvm.br ^bb15(%102 : i32)
  ^bb15(%103: i32):  // 2 preds: ^bb13, ^bb14
    llvm.br ^bb16
  ^bb16:  // pred: ^bb15
    %104 = llvm.trunc %103 : i32 to i8
    %105 = llvm.mul %92, %50 overflow<nsw, nuw> : i64
    %106 = llvm.add %87, %105 overflow<nsw, nuw> : i64
    %107 = llvm.add %106, %63 overflow<nsw, nuw> : i64
    %108 = llvm.getelementptr inbounds|nuw %60[%107] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %104, %108 : i8, !llvm.ptr
    %109 = llvm.add %67, %44 : i64
    %110 = llvm.add %73, %109 overflow<nsw, nuw> : i64
    %111 = llvm.getelementptr inbounds|nuw %arg0[%110] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.fmul %112, %39 : f32
    %114 = llvm.intr.roundeven(%113) : (f32) -> f32
    %115 = llvm.fptosi %114 : f32 to i32
    %116 = llvm.sub %115, %40 : i32
    %117 = llvm.icmp "ult" %116, %41 : i32
    llvm.cond_br %117, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    llvm.br ^bb19(%115 : i32)
  ^bb18:  // pred: ^bb16
    %118 = llvm.icmp "slt" %115, %40 : i32
    %119 = llvm.select %118, %40, %42 : i1, i32
    llvm.br ^bb19(%119 : i32)
  ^bb19(%120: i32):  // 2 preds: ^bb17, ^bb18
    llvm.br ^bb20
  ^bb20:  // pred: ^bb19
    %121 = llvm.trunc %120 : i32 to i8
    %122 = llvm.mul %109, %50 overflow<nsw, nuw> : i64
    %123 = llvm.add %87, %122 overflow<nsw, nuw> : i64
    %124 = llvm.add %123, %63 overflow<nsw, nuw> : i64
    %125 = llvm.getelementptr inbounds|nuw %60[%124] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %121, %125 : i8, !llvm.ptr
    %126 = llvm.add %67, %38 : i64
    %127 = llvm.add %73, %126 overflow<nsw, nuw> : i64
    %128 = llvm.getelementptr inbounds|nuw %arg0[%127] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %129 = llvm.load %128 : !llvm.ptr -> f32
    %130 = llvm.fmul %129, %39 : f32
    %131 = llvm.intr.roundeven(%130) : (f32) -> f32
    %132 = llvm.fptosi %131 : f32 to i32
    %133 = llvm.sub %132, %40 : i32
    %134 = llvm.icmp "ult" %133, %41 : i32
    llvm.cond_br %134, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    llvm.br ^bb23(%132 : i32)
  ^bb22:  // pred: ^bb20
    %135 = llvm.icmp "slt" %132, %40 : i32
    %136 = llvm.select %135, %40, %42 : i1, i32
    llvm.br ^bb23(%136 : i32)
  ^bb23(%137: i32):  // 2 preds: ^bb21, ^bb22
    llvm.br ^bb24
  ^bb24:  // pred: ^bb23
    %138 = llvm.trunc %137 : i32 to i8
    %139 = llvm.mul %126, %50 overflow<nsw, nuw> : i64
    %140 = llvm.add %87, %139 overflow<nsw, nuw> : i64
    %141 = llvm.add %140, %63 overflow<nsw, nuw> : i64
    %142 = llvm.getelementptr inbounds|nuw %60[%141] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %138, %142 : i8, !llvm.ptr
    %143 = llvm.add %67, %43 : i64
    %144 = llvm.add %73, %143 overflow<nsw, nuw> : i64
    %145 = llvm.getelementptr inbounds|nuw %arg0[%144] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %146 = llvm.load %145 : !llvm.ptr -> f32
    %147 = llvm.fmul %146, %39 : f32
    %148 = llvm.intr.roundeven(%147) : (f32) -> f32
    %149 = llvm.fptosi %148 : f32 to i32
    %150 = llvm.sub %149, %40 : i32
    %151 = llvm.icmp "ult" %150, %41 : i32
    llvm.cond_br %151, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    llvm.br ^bb27(%149 : i32)
  ^bb26:  // pred: ^bb24
    %152 = llvm.icmp "slt" %149, %40 : i32
    %153 = llvm.select %152, %40, %42 : i1, i32
    llvm.br ^bb27(%153 : i32)
  ^bb27(%154: i32):  // 2 preds: ^bb25, ^bb26
    llvm.br ^bb28
  ^bb28:  // pred: ^bb27
    %155 = llvm.trunc %154 : i32 to i8
    %156 = llvm.mul %143, %50 overflow<nsw, nuw> : i64
    %157 = llvm.add %87, %156 overflow<nsw, nuw> : i64
    %158 = llvm.add %157, %63 overflow<nsw, nuw> : i64
    %159 = llvm.getelementptr inbounds|nuw %60[%158] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %155, %159 : i8, !llvm.ptr
    %160 = llvm.add %67, %45 : i64
    %161 = llvm.add %73, %160 overflow<nsw, nuw> : i64
    %162 = llvm.getelementptr inbounds|nuw %arg0[%161] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %163 = llvm.load %162 : !llvm.ptr -> f32
    %164 = llvm.fmul %163, %39 : f32
    %165 = llvm.intr.roundeven(%164) : (f32) -> f32
    %166 = llvm.fptosi %165 : f32 to i32
    %167 = llvm.sub %166, %40 : i32
    %168 = llvm.icmp "ult" %167, %41 : i32
    llvm.cond_br %168, ^bb29, ^bb30
  ^bb29:  // pred: ^bb28
    llvm.br ^bb31(%166 : i32)
  ^bb30:  // pred: ^bb28
    %169 = llvm.icmp "slt" %166, %40 : i32
    %170 = llvm.select %169, %40, %42 : i1, i32
    llvm.br ^bb31(%170 : i32)
  ^bb31(%171: i32):  // 2 preds: ^bb29, ^bb30
    llvm.br ^bb32
  ^bb32:  // pred: ^bb31
    %172 = llvm.trunc %171 : i32 to i8
    %173 = llvm.mul %160, %50 overflow<nsw, nuw> : i64
    %174 = llvm.add %87, %173 overflow<nsw, nuw> : i64
    %175 = llvm.add %174, %63 overflow<nsw, nuw> : i64
    %176 = llvm.getelementptr inbounds|nuw %60[%175] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %172, %176 : i8, !llvm.ptr
    %177 = llvm.add %67, %46 : i64
    %178 = llvm.add %73, %177 overflow<nsw, nuw> : i64
    %179 = llvm.getelementptr inbounds|nuw %arg0[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %180 = llvm.load %179 : !llvm.ptr -> f32
    %181 = llvm.fmul %180, %39 : f32
    %182 = llvm.intr.roundeven(%181) : (f32) -> f32
    %183 = llvm.fptosi %182 : f32 to i32
    %184 = llvm.sub %183, %40 : i32
    %185 = llvm.icmp "ult" %184, %41 : i32
    llvm.cond_br %185, ^bb33, ^bb34
  ^bb33:  // pred: ^bb32
    llvm.br ^bb35(%183 : i32)
  ^bb34:  // pred: ^bb32
    %186 = llvm.icmp "slt" %183, %40 : i32
    %187 = llvm.select %186, %40, %42 : i1, i32
    llvm.br ^bb35(%187 : i32)
  ^bb35(%188: i32):  // 2 preds: ^bb33, ^bb34
    llvm.br ^bb36
  ^bb36:  // pred: ^bb35
    %189 = llvm.trunc %188 : i32 to i8
    %190 = llvm.mul %177, %50 overflow<nsw, nuw> : i64
    %191 = llvm.add %87, %190 overflow<nsw, nuw> : i64
    %192 = llvm.add %191, %63 overflow<nsw, nuw> : i64
    %193 = llvm.getelementptr inbounds|nuw %60[%192] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %189, %193 : i8, !llvm.ptr
    %194 = llvm.add %67, %47 : i64
    %195 = llvm.add %73, %194 overflow<nsw, nuw> : i64
    %196 = llvm.getelementptr inbounds|nuw %arg0[%195] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %197 = llvm.load %196 : !llvm.ptr -> f32
    %198 = llvm.fmul %197, %39 : f32
    %199 = llvm.intr.roundeven(%198) : (f32) -> f32
    %200 = llvm.fptosi %199 : f32 to i32
    %201 = llvm.sub %200, %40 : i32
    %202 = llvm.icmp "ult" %201, %41 : i32
    llvm.cond_br %202, ^bb37, ^bb38
  ^bb37:  // pred: ^bb36
    llvm.br ^bb39(%200 : i32)
  ^bb38:  // pred: ^bb36
    %203 = llvm.icmp "slt" %200, %40 : i32
    %204 = llvm.select %203, %40, %42 : i1, i32
    llvm.br ^bb39(%204 : i32)
  ^bb39(%205: i32):  // 2 preds: ^bb37, ^bb38
    llvm.br ^bb40
  ^bb40:  // pred: ^bb39
    %206 = llvm.trunc %205 : i32 to i8
    %207 = llvm.mul %194, %50 overflow<nsw, nuw> : i64
    %208 = llvm.add %87, %207 overflow<nsw, nuw> : i64
    %209 = llvm.add %208, %63 overflow<nsw, nuw> : i64
    %210 = llvm.getelementptr inbounds|nuw %60[%209] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %206, %210 : i8, !llvm.ptr
    %211 = llvm.add %67, %50 : i64
    llvm.br ^bb7(%211 : i64)
  ^bb41:  // pred: ^bb7
    %212 = llvm.add %65, %52 : i64
    llvm.br ^bb5(%212 : i64)
  ^bb42:  // pred: ^bb5
    %213 = llvm.add %63, %52 : i64
    llvm.br ^bb3(%213 : i64)
  ^bb43:  // pred: ^bb3
    %214 = llvm.add %61, %52 : i64
    llvm.br ^bb1(%214 : i64)
  ^bb44:  // pred: ^bb1
    %215 = llvm.inttoptr %28 : i64 to !llvm.ptr
    %216 = llvm.insertvalue %215, %54[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %217 = llvm.getelementptr %59[23104] : (!llvm.ptr) -> !llvm.ptr, i8
    %218 = llvm.insertvalue %217, %216[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %219 = llvm.insertvalue %53, %218[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %220 = llvm.insertvalue %48, %219[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %221 = llvm.insertvalue %52, %220[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %222 = llvm.insertvalue %48, %221[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %223 = llvm.insertvalue %48, %222[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %224 = llvm.insertvalue %48, %223[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %225 = llvm.insertvalue %49, %224[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %226 = llvm.insertvalue %52, %225[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %227 = llvm.insertvalue %22, %226[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    llvm.inline_asm has_side_effects is_align_stack asm_dialect = att ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"  : () -> ()
    llvm.call @gemmlir_flush() : () -> ()
    llvm.call @tiled_conv_stride_auto(%21, %20, %20, %19, %20, %20, %20, %21, %21, %21, %21, %18, %19, %20, %20, %17, %17, %17, %17, %17, %60, %57, %56, %217, %21, %16, %15, %15, %15, %21) : (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i32, f32, i32, i32, i32, i32) -> ()
    %228 = llvm.getelementptr %59[27200] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.call @gemmlir_memset(%228, %15, %12) : (!llvm.ptr, i32, i64) -> ()
    %229 = llvm.getelementptr %228[4896] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.call @gemmlir_memset(%229, %15, %12) : (!llvm.ptr, i32, i64) -> ()
    llvm.br ^bb45(%53 : i64)
  ^bb45(%230: i64):  // 2 preds: ^bb44, ^bb55
    %231 = llvm.icmp "slt" %230, %52 : i64
    llvm.cond_br %231, ^bb46, ^bb56
  ^bb46:  // pred: ^bb45
    llvm.br ^bb47(%53 : i64)
  ^bb47(%232: i64):  // 2 preds: ^bb46, ^bb54
    %233 = llvm.icmp "slt" %232, %48 : i64
    llvm.cond_br %233, ^bb48, ^bb55
  ^bb48:  // pred: ^bb47
    llvm.br ^bb49(%53 : i64)
  ^bb49(%234: i64):  // 2 preds: ^bb48, ^bb53
    %235 = llvm.icmp "slt" %234, %52 : i64
    llvm.cond_br %235, ^bb50, ^bb54
  ^bb50:  // pred: ^bb49
    llvm.br ^bb51(%53 : i64)
  ^bb51(%236: i64):  // 2 preds: ^bb50, ^bb52
    %237 = llvm.icmp "slt" %236, %48 : i64
    llvm.cond_br %237, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    %238 = llvm.getelementptr %228[288] : (!llvm.ptr) -> !llvm.ptr, i8
    %239 = llvm.mul %230, %13 overflow<nsw, nuw> : i64
    %240 = llvm.mul %232, %14 overflow<nsw, nuw> : i64
    %241 = llvm.add %239, %240 overflow<nsw, nuw> : i64
    %242 = llvm.mul %234, %48 overflow<nsw, nuw> : i64
    %243 = llvm.add %241, %242 overflow<nsw, nuw> : i64
    %244 = llvm.add %243, %236 overflow<nsw, nuw> : i64
    %245 = llvm.getelementptr inbounds|nuw %238[%244] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %245 : i8, !llvm.ptr
    %246 = llvm.add %236, %52 : i64
    %247 = llvm.add %243, %246 overflow<nsw, nuw> : i64
    %248 = llvm.getelementptr inbounds|nuw %238[%247] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %248 : i8, !llvm.ptr
    %249 = llvm.add %236, %44 : i64
    %250 = llvm.add %243, %249 overflow<nsw, nuw> : i64
    %251 = llvm.getelementptr inbounds|nuw %238[%250] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %251 : i8, !llvm.ptr
    %252 = llvm.add %236, %38 : i64
    %253 = llvm.add %243, %252 overflow<nsw, nuw> : i64
    %254 = llvm.getelementptr inbounds|nuw %238[%253] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %254 : i8, !llvm.ptr
    %255 = llvm.add %236, %43 : i64
    %256 = llvm.add %243, %255 overflow<nsw, nuw> : i64
    %257 = llvm.getelementptr inbounds|nuw %238[%256] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %257 : i8, !llvm.ptr
    %258 = llvm.add %236, %45 : i64
    %259 = llvm.add %243, %258 overflow<nsw, nuw> : i64
    %260 = llvm.getelementptr inbounds|nuw %238[%259] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %260 : i8, !llvm.ptr
    %261 = llvm.add %236, %46 : i64
    %262 = llvm.add %243, %261 overflow<nsw, nuw> : i64
    %263 = llvm.getelementptr inbounds|nuw %238[%262] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %263 : i8, !llvm.ptr
    %264 = llvm.add %236, %47 : i64
    %265 = llvm.add %243, %264 overflow<nsw, nuw> : i64
    %266 = llvm.getelementptr inbounds|nuw %238[%265] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %266 : i8, !llvm.ptr
    %267 = llvm.add %236, %50 : i64
    llvm.br ^bb51(%267 : i64)
  ^bb53:  // pred: ^bb51
    %268 = llvm.add %234, %52 : i64
    llvm.br ^bb49(%268 : i64)
  ^bb54:  // pred: ^bb49
    %269 = llvm.add %232, %52 : i64
    llvm.br ^bb47(%269 : i64)
  ^bb55:  // pred: ^bb47
    %270 = llvm.add %230, %52 : i64
    llvm.br ^bb45(%270 : i64)
  ^bb56:  // pred: ^bb45
    llvm.br ^bb57(%53 : i64)
  ^bb57(%271: i64):  // 2 preds: ^bb56, ^bb67
    %272 = llvm.icmp "slt" %271, %52 : i64
    llvm.cond_br %272, ^bb58, ^bb68
  ^bb58:  // pred: ^bb57
    llvm.br ^bb59(%53 : i64)
  ^bb59(%273: i64):  // 2 preds: ^bb58, ^bb66
    %274 = llvm.icmp "slt" %273, %48 : i64
    llvm.cond_br %274, ^bb60, ^bb67
  ^bb60:  // pred: ^bb59
    llvm.br ^bb61(%53 : i64)
  ^bb61(%275: i64):  // 2 preds: ^bb60, ^bb65
    %276 = llvm.icmp "slt" %275, %52 : i64
    llvm.cond_br %276, ^bb62, ^bb66
  ^bb62:  // pred: ^bb61
    llvm.br ^bb63(%53 : i64)
  ^bb63(%277: i64):  // 2 preds: ^bb62, ^bb64
    %278 = llvm.icmp "slt" %277, %48 : i64
    llvm.cond_br %278, ^bb64, ^bb65
  ^bb64:  // pred: ^bb63
    %279 = llvm.getelementptr %228[560] : (!llvm.ptr) -> !llvm.ptr, i8
    %280 = llvm.mul %271, %13 overflow<nsw, nuw> : i64
    %281 = llvm.mul %273, %14 overflow<nsw, nuw> : i64
    %282 = llvm.add %280, %281 overflow<nsw, nuw> : i64
    %283 = llvm.mul %275, %48 overflow<nsw, nuw> : i64
    %284 = llvm.add %282, %283 overflow<nsw, nuw> : i64
    %285 = llvm.add %284, %277 overflow<nsw, nuw> : i64
    %286 = llvm.getelementptr inbounds|nuw %279[%285] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %286 : i8, !llvm.ptr
    %287 = llvm.add %277, %52 : i64
    %288 = llvm.add %284, %287 overflow<nsw, nuw> : i64
    %289 = llvm.getelementptr inbounds|nuw %279[%288] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %289 : i8, !llvm.ptr
    %290 = llvm.add %277, %44 : i64
    %291 = llvm.add %284, %290 overflow<nsw, nuw> : i64
    %292 = llvm.getelementptr inbounds|nuw %279[%291] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %292 : i8, !llvm.ptr
    %293 = llvm.add %277, %38 : i64
    %294 = llvm.add %284, %293 overflow<nsw, nuw> : i64
    %295 = llvm.getelementptr inbounds|nuw %279[%294] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %295 : i8, !llvm.ptr
    %296 = llvm.add %277, %43 : i64
    %297 = llvm.add %284, %296 overflow<nsw, nuw> : i64
    %298 = llvm.getelementptr inbounds|nuw %279[%297] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %298 : i8, !llvm.ptr
    %299 = llvm.add %277, %45 : i64
    %300 = llvm.add %284, %299 overflow<nsw, nuw> : i64
    %301 = llvm.getelementptr inbounds|nuw %279[%300] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %301 : i8, !llvm.ptr
    %302 = llvm.add %277, %46 : i64
    %303 = llvm.add %284, %302 overflow<nsw, nuw> : i64
    %304 = llvm.getelementptr inbounds|nuw %279[%303] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %304 : i8, !llvm.ptr
    %305 = llvm.add %277, %47 : i64
    %306 = llvm.add %284, %305 overflow<nsw, nuw> : i64
    %307 = llvm.getelementptr inbounds|nuw %279[%306] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %35, %307 : i8, !llvm.ptr
    %308 = llvm.add %277, %50 : i64
    llvm.br ^bb63(%308 : i64)
  ^bb65:  // pred: ^bb63
    %309 = llvm.add %275, %52 : i64
    llvm.br ^bb61(%309 : i64)
  ^bb66:  // pred: ^bb61
    %310 = llvm.add %273, %52 : i64
    llvm.br ^bb59(%310 : i64)
  ^bb67:  // pred: ^bb59
    %311 = llvm.add %271, %52 : i64
    llvm.br ^bb57(%311 : i64)
  ^bb68:  // pred: ^bb57
    %312 = llvm.insertvalue %228, %216[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %313 = llvm.insertvalue %11, %312[2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %314 = llvm.insertvalue %52, %313[3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %315 = llvm.insertvalue %13, %314[4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %316 = llvm.insertvalue %48, %315[3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %317 = llvm.insertvalue %14, %316[4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %318 = llvm.insertvalue %48, %317[3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %319 = llvm.insertvalue %48, %318[4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %320 = llvm.insertvalue %48, %319[3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %321 = llvm.insertvalue %52, %320[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> 
    %322 = llvm.intr.stacksave : !llvm.ptr
    %323 = llvm.alloca %52 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %227, %323 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %324 = llvm.insertvalue %10, %9[0] : !llvm.struct<(i64, ptr)> 
    %325 = llvm.insertvalue %323, %324[1] : !llvm.struct<(i64, ptr)> 
    %326 = llvm.alloca %52 x !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %321, %326 : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>, !llvm.ptr
    %327 = llvm.insertvalue %326, %324[1] : !llvm.struct<(i64, ptr)> 
    %328 = llvm.alloca %52 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %325, %328 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %329 = llvm.alloca %52 x !llvm.struct<(i64, ptr)> : (i64) -> !llvm.ptr
    llvm.store %327, %329 : !llvm.struct<(i64, ptr)>, !llvm.ptr
    %330 = llvm.getelementptr %30[1] : (!llvm.ptr) -> !llvm.ptr, i8
    %331 = llvm.ptrtoint %330 : !llvm.ptr to i64
    llvm.call @memrefCopy(%331, %328, %329) : (i64, !llvm.ptr, !llvm.ptr) -> ()
    llvm.intr.stackrestore %322 : !llvm.ptr
    %332 = llvm.getelementptr %59[32384] : (!llvm.ptr) -> !llvm.ptr, i8
    %333 = llvm.getelementptr %59[48768] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.br ^bb69(%53 : i64)
  ^bb69(%334: i64):  // 2 preds: ^bb68, ^bb79
    %335 = llvm.icmp "slt" %334, %52 : i64
    llvm.cond_br %335, ^bb70, ^bb80
  ^bb70:  // pred: ^bb69
    llvm.br ^bb71(%53 : i64)
  ^bb71(%336: i64):  // 2 preds: ^bb70, ^bb78
    %337 = llvm.icmp "slt" %336, %48 : i64
    llvm.cond_br %337, ^bb72, ^bb79
  ^bb72:  // pred: ^bb71
    llvm.br ^bb73(%53 : i64)
  ^bb73(%338: i64):  // 2 preds: ^bb72, ^bb77
    %339 = llvm.icmp "slt" %338, %48 : i64
    llvm.cond_br %339, ^bb74, ^bb78
  ^bb74:  // pred: ^bb73
    llvm.br ^bb75(%53 : i64)
  ^bb75(%340: i64):  // 2 preds: ^bb74, ^bb76
    %341 = llvm.icmp "slt" %340, %38 : i64
    llvm.cond_br %341, ^bb76, ^bb77
  ^bb76:  // pred: ^bb75
    %342 = llvm.mul %334, %13 : i64
    %343 = llvm.mul %336, %14 : i64
    %344 = llvm.add %342, %343 : i64
    %345 = llvm.mul %338, %48 : i64
    %346 = llvm.add %344, %345 : i64
    %347 = llvm.mul %340, %14 : i64
    %348 = llvm.add %346, %347 : i64
    %349 = llvm.add %348, %53 : i64
    %350 = llvm.getelementptr %228[%349] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %351 = llvm.load %350 {alignment = 8 : i64} : !llvm.ptr -> vector<48xi8>
    %352 = llvm.mul %334, %7 : i64
    %353 = llvm.mul %336, %31 : i64
    %354 = llvm.add %352, %353 : i64
    %355 = llvm.mul %338, %32 : i64
    %356 = llvm.add %354, %355 : i64
    %357 = llvm.mul %340, %8 : i64
    %358 = llvm.add %356, %357 : i64
    %359 = llvm.add %358, %53 : i64
    %360 = llvm.getelementptr %333[%359] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    llvm.store %351, %360 {alignment = 8 : i64} : vector<48xi8>, !llvm.ptr
    %361 = llvm.add %340, %52 : i64
    llvm.br ^bb75(%361 : i64)
  ^bb77:  // pred: ^bb75
    %362 = llvm.add %338, %52 : i64
    llvm.br ^bb73(%362 : i64)
  ^bb78:  // pred: ^bb73
    %363 = llvm.add %336, %52 : i64
    llvm.br ^bb71(%363 : i64)
  ^bb79:  // pred: ^bb71
    %364 = llvm.add %334, %52 : i64
    llvm.br ^bb69(%364 : i64)
  ^bb80:  // pred: ^bb69
    llvm.inline_asm has_side_effects is_align_stack asm_dialect = att ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"  : () -> ()
    llvm.call @gemmlir_flush() : () -> ()
    llvm.call @tiled_matmul_auto(%5, %4, %3, %333, %55, %30, %332, %3, %4, %4, %4, %6, %6, %21, %15, %6, %6, %17, %17, %17, %2, %17, %35, %21) : (i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, f32, f32, i32, i32, f32, f32, i1, i1, i1, i1, i1, i8, i32) -> ()
    %365 = llvm.getelementptr %59[85696] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.br ^bb81(%53 : i64)
  ^bb81(%366: i64):  // 2 preds: ^bb80, ^bb91
    %367 = llvm.icmp "slt" %366, %52 : i64
    llvm.cond_br %367, ^bb82, ^bb92
  ^bb82:  // pred: ^bb81
    llvm.br ^bb83(%53 : i64)
  ^bb83(%368: i64):  // 2 preds: ^bb82, ^bb90
    %369 = llvm.icmp "slt" %368, %48 : i64
    llvm.cond_br %369, ^bb84, ^bb91
  ^bb84:  // pred: ^bb83
    llvm.br ^bb85(%53 : i64)
  ^bb85(%370: i64):  // 2 preds: ^bb84, ^bb89
    %371 = llvm.icmp "slt" %370, %48 : i64
    llvm.cond_br %371, ^bb86, ^bb90
  ^bb86:  // pred: ^bb85
    llvm.br ^bb87(%53 : i64)
  ^bb87(%372: i64):  // 2 preds: ^bb86, ^bb88
    %373 = llvm.icmp "slt" %372, %48 : i64
    llvm.cond_br %373, ^bb88, ^bb89
  ^bb88:  // pred: ^bb87
    %374 = llvm.mul %53, %22 overflow<nsw, nuw> : i64
    %375 = llvm.mul %368, %49 overflow<nsw, nuw> : i64
    %376 = llvm.add %374, %375 overflow<nsw, nuw> : i64
    %377 = llvm.mul %370, %48 overflow<nsw, nuw> : i64
    %378 = llvm.add %376, %377 overflow<nsw, nuw> : i64
    %379 = llvm.add %378, %372 overflow<nsw, nuw> : i64
    %380 = llvm.getelementptr inbounds|nuw %217[%379] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %381 = llvm.load %380 : !llvm.ptr -> i8
    %382 = llvm.getelementptr inbounds|nuw %332[%379] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %383 = llvm.load %382 : !llvm.ptr -> i32
    %384 = llvm.getelementptr inbounds|nuw %58[%372] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %385 = llvm.load %384 : !llvm.ptr -> f32
    %386 = llvm.sitofp %381 : i8 to f32
    %387 = llvm.sitofp %383 : i32 to f32
    %388 = llvm.intr.fma(%387, %33, %385) : (f32, f32, f32) -> f32
    %389 = llvm.intr.fma(%386, %34, %388) : (f32, f32, f32) -> f32
    %390 = llvm.intr.maxnum(%389, %36) : (f32, f32) -> f32
    %391 = llvm.mul %366, %22 overflow<nsw, nuw> : i64
    %392 = llvm.add %391, %375 overflow<nsw, nuw> : i64
    %393 = llvm.add %392, %377 overflow<nsw, nuw> : i64
    %394 = llvm.add %393, %372 overflow<nsw, nuw> : i64
    %395 = llvm.getelementptr inbounds|nuw %365[%394] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %390, %395 : f32, !llvm.ptr
    %396 = llvm.add %372, %52 : i64
    %397 = llvm.add %378, %396 overflow<nsw, nuw> : i64
    %398 = llvm.getelementptr inbounds|nuw %217[%397] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %399 = llvm.load %398 : !llvm.ptr -> i8
    %400 = llvm.getelementptr inbounds|nuw %332[%397] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %401 = llvm.load %400 : !llvm.ptr -> i32
    %402 = llvm.getelementptr inbounds|nuw %58[%396] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %403 = llvm.load %402 : !llvm.ptr -> f32
    %404 = llvm.sitofp %399 : i8 to f32
    %405 = llvm.sitofp %401 : i32 to f32
    %406 = llvm.intr.fma(%405, %33, %403) : (f32, f32, f32) -> f32
    %407 = llvm.intr.fma(%404, %34, %406) : (f32, f32, f32) -> f32
    %408 = llvm.intr.maxnum(%407, %36) : (f32, f32) -> f32
    %409 = llvm.add %393, %396 overflow<nsw, nuw> : i64
    %410 = llvm.getelementptr inbounds|nuw %365[%409] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %408, %410 : f32, !llvm.ptr
    %411 = llvm.add %372, %44 : i64
    %412 = llvm.add %378, %411 overflow<nsw, nuw> : i64
    %413 = llvm.getelementptr inbounds|nuw %217[%412] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %414 = llvm.load %413 : !llvm.ptr -> i8
    %415 = llvm.getelementptr inbounds|nuw %332[%412] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %416 = llvm.load %415 : !llvm.ptr -> i32
    %417 = llvm.getelementptr inbounds|nuw %58[%411] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %418 = llvm.load %417 : !llvm.ptr -> f32
    %419 = llvm.sitofp %414 : i8 to f32
    %420 = llvm.sitofp %416 : i32 to f32
    %421 = llvm.intr.fma(%420, %33, %418) : (f32, f32, f32) -> f32
    %422 = llvm.intr.fma(%419, %34, %421) : (f32, f32, f32) -> f32
    %423 = llvm.intr.maxnum(%422, %36) : (f32, f32) -> f32
    %424 = llvm.add %393, %411 overflow<nsw, nuw> : i64
    %425 = llvm.getelementptr inbounds|nuw %365[%424] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %423, %425 : f32, !llvm.ptr
    %426 = llvm.add %372, %38 : i64
    %427 = llvm.add %378, %426 overflow<nsw, nuw> : i64
    %428 = llvm.getelementptr inbounds|nuw %217[%427] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %429 = llvm.load %428 : !llvm.ptr -> i8
    %430 = llvm.getelementptr inbounds|nuw %332[%427] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %431 = llvm.load %430 : !llvm.ptr -> i32
    %432 = llvm.getelementptr inbounds|nuw %58[%426] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %433 = llvm.load %432 : !llvm.ptr -> f32
    %434 = llvm.sitofp %429 : i8 to f32
    %435 = llvm.sitofp %431 : i32 to f32
    %436 = llvm.intr.fma(%435, %33, %433) : (f32, f32, f32) -> f32
    %437 = llvm.intr.fma(%434, %34, %436) : (f32, f32, f32) -> f32
    %438 = llvm.intr.maxnum(%437, %36) : (f32, f32) -> f32
    %439 = llvm.add %393, %426 overflow<nsw, nuw> : i64
    %440 = llvm.getelementptr inbounds|nuw %365[%439] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %438, %440 : f32, !llvm.ptr
    %441 = llvm.add %372, %43 : i64
    %442 = llvm.add %378, %441 overflow<nsw, nuw> : i64
    %443 = llvm.getelementptr inbounds|nuw %217[%442] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %444 = llvm.load %443 : !llvm.ptr -> i8
    %445 = llvm.getelementptr inbounds|nuw %332[%442] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %446 = llvm.load %445 : !llvm.ptr -> i32
    %447 = llvm.getelementptr inbounds|nuw %58[%441] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %448 = llvm.load %447 : !llvm.ptr -> f32
    %449 = llvm.sitofp %444 : i8 to f32
    %450 = llvm.sitofp %446 : i32 to f32
    %451 = llvm.intr.fma(%450, %33, %448) : (f32, f32, f32) -> f32
    %452 = llvm.intr.fma(%449, %34, %451) : (f32, f32, f32) -> f32
    %453 = llvm.intr.maxnum(%452, %36) : (f32, f32) -> f32
    %454 = llvm.add %393, %441 overflow<nsw, nuw> : i64
    %455 = llvm.getelementptr inbounds|nuw %365[%454] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %453, %455 : f32, !llvm.ptr
    %456 = llvm.add %372, %45 : i64
    %457 = llvm.add %378, %456 overflow<nsw, nuw> : i64
    %458 = llvm.getelementptr inbounds|nuw %217[%457] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %459 = llvm.load %458 : !llvm.ptr -> i8
    %460 = llvm.getelementptr inbounds|nuw %332[%457] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %461 = llvm.load %460 : !llvm.ptr -> i32
    %462 = llvm.getelementptr inbounds|nuw %58[%456] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %463 = llvm.load %462 : !llvm.ptr -> f32
    %464 = llvm.sitofp %459 : i8 to f32
    %465 = llvm.sitofp %461 : i32 to f32
    %466 = llvm.intr.fma(%465, %33, %463) : (f32, f32, f32) -> f32
    %467 = llvm.intr.fma(%464, %34, %466) : (f32, f32, f32) -> f32
    %468 = llvm.intr.maxnum(%467, %36) : (f32, f32) -> f32
    %469 = llvm.add %393, %456 overflow<nsw, nuw> : i64
    %470 = llvm.getelementptr inbounds|nuw %365[%469] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %468, %470 : f32, !llvm.ptr
    %471 = llvm.add %372, %46 : i64
    %472 = llvm.add %378, %471 overflow<nsw, nuw> : i64
    %473 = llvm.getelementptr inbounds|nuw %217[%472] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %474 = llvm.load %473 : !llvm.ptr -> i8
    %475 = llvm.getelementptr inbounds|nuw %332[%472] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %476 = llvm.load %475 : !llvm.ptr -> i32
    %477 = llvm.getelementptr inbounds|nuw %58[%471] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %478 = llvm.load %477 : !llvm.ptr -> f32
    %479 = llvm.sitofp %474 : i8 to f32
    %480 = llvm.sitofp %476 : i32 to f32
    %481 = llvm.intr.fma(%480, %33, %478) : (f32, f32, f32) -> f32
    %482 = llvm.intr.fma(%479, %34, %481) : (f32, f32, f32) -> f32
    %483 = llvm.intr.maxnum(%482, %36) : (f32, f32) -> f32
    %484 = llvm.add %393, %471 overflow<nsw, nuw> : i64
    %485 = llvm.getelementptr inbounds|nuw %365[%484] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %483, %485 : f32, !llvm.ptr
    %486 = llvm.add %372, %47 : i64
    %487 = llvm.add %378, %486 overflow<nsw, nuw> : i64
    %488 = llvm.getelementptr inbounds|nuw %217[%487] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %489 = llvm.load %488 : !llvm.ptr -> i8
    %490 = llvm.getelementptr inbounds|nuw %332[%487] : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %491 = llvm.load %490 : !llvm.ptr -> i32
    %492 = llvm.getelementptr inbounds|nuw %58[%486] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %493 = llvm.load %492 : !llvm.ptr -> f32
    %494 = llvm.sitofp %489 : i8 to f32
    %495 = llvm.sitofp %491 : i32 to f32
    %496 = llvm.intr.fma(%495, %33, %493) : (f32, f32, f32) -> f32
    %497 = llvm.intr.fma(%494, %34, %496) : (f32, f32, f32) -> f32
    %498 = llvm.intr.maxnum(%497, %36) : (f32, f32) -> f32
    %499 = llvm.add %393, %486 overflow<nsw, nuw> : i64
    %500 = llvm.getelementptr inbounds|nuw %365[%499] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %498, %500 : f32, !llvm.ptr
    %501 = llvm.add %372, %50 : i64
    llvm.br ^bb87(%501 : i64)
  ^bb89:  // pred: ^bb87
    %502 = llvm.add %370, %52 : i64
    llvm.br ^bb85(%502 : i64)
  ^bb90:  // pred: ^bb85
    %503 = llvm.add %368, %52 : i64
    llvm.br ^bb83(%503 : i64)
  ^bb91:  // pred: ^bb83
    %504 = llvm.add %366, %52 : i64
    llvm.br ^bb81(%504 : i64)
  ^bb92:  // pred: ^bb81
    %505 = llvm.getelementptr %59[102080] : (!llvm.ptr) -> !llvm.ptr, i8
    llvm.br ^bb93(%53 : i64)
  ^bb93(%506: i64):  // 2 preds: ^bb92, ^bb103
    %507 = llvm.icmp "slt" %506, %52 : i64
    llvm.cond_br %507, ^bb94, ^bb104
  ^bb94:  // pred: ^bb93
    llvm.br ^bb95(%53 : i64)
  ^bb95(%508: i64):  // 2 preds: ^bb94, ^bb102
    %509 = llvm.icmp "slt" %508, %50 : i64
    llvm.cond_br %509, ^bb96, ^bb103
  ^bb96:  // pred: ^bb95
    llvm.br ^bb97(%53 : i64)
  ^bb97(%510: i64):  // 2 preds: ^bb96, ^bb101
    %511 = llvm.icmp "slt" %510, %50 : i64
    llvm.cond_br %511, ^bb98, ^bb102
  ^bb98:  // pred: ^bb97
    llvm.br ^bb99(%53 : i64)
  ^bb99(%512: i64):  // 2 preds: ^bb98, ^bb100
    %513 = llvm.icmp "slt" %512, %48 : i64
    llvm.cond_br %513, ^bb100, ^bb101
  ^bb100:  // pred: ^bb99
    %514 = llvm.mul %506, %1 overflow<nsw, nuw> : i64
    %515 = llvm.mul %508, %26 overflow<nsw, nuw> : i64
    %516 = llvm.add %514, %515 overflow<nsw, nuw> : i64
    %517 = llvm.mul %510, %48 overflow<nsw, nuw> : i64
    %518 = llvm.add %516, %517 overflow<nsw, nuw> : i64
    %519 = llvm.add %518, %512 overflow<nsw, nuw> : i64
    %520 = llvm.getelementptr inbounds|nuw %505[%519] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %520 : f32, !llvm.ptr
    %521 = llvm.add %512, %52 : i64
    %522 = llvm.add %518, %521 overflow<nsw, nuw> : i64
    %523 = llvm.getelementptr inbounds|nuw %505[%522] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %523 : f32, !llvm.ptr
    %524 = llvm.add %512, %44 : i64
    %525 = llvm.add %518, %524 overflow<nsw, nuw> : i64
    %526 = llvm.getelementptr inbounds|nuw %505[%525] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %526 : f32, !llvm.ptr
    %527 = llvm.add %512, %38 : i64
    %528 = llvm.add %518, %527 overflow<nsw, nuw> : i64
    %529 = llvm.getelementptr inbounds|nuw %505[%528] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %529 : f32, !llvm.ptr
    %530 = llvm.add %512, %43 : i64
    %531 = llvm.add %518, %530 overflow<nsw, nuw> : i64
    %532 = llvm.getelementptr inbounds|nuw %505[%531] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %532 : f32, !llvm.ptr
    %533 = llvm.add %512, %45 : i64
    %534 = llvm.add %518, %533 overflow<nsw, nuw> : i64
    %535 = llvm.getelementptr inbounds|nuw %505[%534] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %535 : f32, !llvm.ptr
    %536 = llvm.add %512, %46 : i64
    %537 = llvm.add %518, %536 overflow<nsw, nuw> : i64
    %538 = llvm.getelementptr inbounds|nuw %505[%537] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %538 : f32, !llvm.ptr
    %539 = llvm.add %512, %47 : i64
    %540 = llvm.add %518, %539 overflow<nsw, nuw> : i64
    %541 = llvm.getelementptr inbounds|nuw %505[%540] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %37, %541 : f32, !llvm.ptr
    %542 = llvm.add %512, %50 : i64
    llvm.br ^bb99(%542 : i64)
  ^bb101:  // pred: ^bb99
    %543 = llvm.add %510, %52 : i64
    llvm.br ^bb97(%543 : i64)
  ^bb102:  // pred: ^bb97
    %544 = llvm.add %508, %52 : i64
    llvm.br ^bb95(%544 : i64)
  ^bb103:  // pred: ^bb95
    %545 = llvm.add %506, %52 : i64
    llvm.br ^bb93(%545 : i64)
  ^bb104:  // pred: ^bb93
    llvm.br ^bb105(%53 : i64)
  ^bb105(%546: i64):  // 2 preds: ^bb104, ^bb115
    %547 = llvm.icmp "slt" %546, %52 : i64
    llvm.cond_br %547, ^bb106, ^bb116
  ^bb106:  // pred: ^bb105
    llvm.br ^bb107(%53 : i64)
  ^bb107(%548: i64):  // 2 preds: ^bb106, ^bb114
    %549 = llvm.icmp "slt" %548, %50 : i64
    llvm.cond_br %549, ^bb108, ^bb115
  ^bb108:  // pred: ^bb107
    llvm.br ^bb109(%53 : i64)
  ^bb109(%550: i64):  // 2 preds: ^bb108, ^bb113
    %551 = llvm.icmp "slt" %550, %50 : i64
    llvm.cond_br %551, ^bb110, ^bb114
  ^bb110:  // pred: ^bb109
    llvm.br ^bb111(%53 : i64)
  ^bb111(%552: i64):  // 2 preds: ^bb110, ^bb112
    %553 = llvm.icmp "slt" %552, %48 : i64
    llvm.cond_br %553, ^bb112, ^bb113
  ^bb112:  // pred: ^bb111
    %554 = llvm.mul %546, %1 overflow<nsw, nuw> : i64
    %555 = llvm.mul %548, %26 overflow<nsw, nuw> : i64
    %556 = llvm.add %554, %555 overflow<nsw, nuw> : i64
    %557 = llvm.mul %550, %48 overflow<nsw, nuw> : i64
    %558 = llvm.add %556, %557 overflow<nsw, nuw> : i64
    %559 = llvm.add %558, %552 overflow<nsw, nuw> : i64
    %560 = llvm.getelementptr inbounds|nuw %505[%559] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %561 = llvm.load %560 : !llvm.ptr -> f32
    %562 = llvm.mul %548, %44 overflow<nsw> : i64
    %563 = llvm.mul %550, %44 overflow<nsw> : i64
    %564 = llvm.mul %546, %22 overflow<nsw, nuw> : i64
    %565 = llvm.mul %562, %49 overflow<nsw, nuw> : i64
    %566 = llvm.add %564, %565 overflow<nsw, nuw> : i64
    %567 = llvm.mul %563, %48 overflow<nsw, nuw> : i64
    %568 = llvm.add %566, %567 overflow<nsw, nuw> : i64
    %569 = llvm.add %568, %552 overflow<nsw, nuw> : i64
    %570 = llvm.getelementptr inbounds|nuw %365[%569] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %571 = llvm.load %570 : !llvm.ptr -> f32
    %572 = llvm.intr.maximum(%561, %571) : (f32, f32) -> f32
    %573 = llvm.add %563, %52 : i64
    %574 = llvm.mul %573, %48 overflow<nsw, nuw> : i64
    %575 = llvm.add %566, %574 overflow<nsw, nuw> : i64
    %576 = llvm.add %575, %552 overflow<nsw, nuw> : i64
    %577 = llvm.getelementptr inbounds|nuw %365[%576] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %578 = llvm.load %577 : !llvm.ptr -> f32
    %579 = llvm.intr.maximum(%572, %578) : (f32, f32) -> f32
    %580 = llvm.add %562, %52 : i64
    %581 = llvm.mul %580, %49 overflow<nsw, nuw> : i64
    %582 = llvm.add %564, %581 overflow<nsw, nuw> : i64
    %583 = llvm.add %582, %567 overflow<nsw, nuw> : i64
    %584 = llvm.add %583, %552 overflow<nsw, nuw> : i64
    %585 = llvm.getelementptr inbounds|nuw %365[%584] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %586 = llvm.load %585 : !llvm.ptr -> f32
    %587 = llvm.intr.maximum(%579, %586) : (f32, f32) -> f32
    %588 = llvm.add %582, %574 overflow<nsw, nuw> : i64
    %589 = llvm.add %588, %552 overflow<nsw, nuw> : i64
    %590 = llvm.getelementptr inbounds|nuw %365[%589] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %591 = llvm.load %590 : !llvm.ptr -> f32
    %592 = llvm.intr.maximum(%587, %591) : (f32, f32) -> f32
    llvm.store %592, %560 : f32, !llvm.ptr
    %593 = llvm.add %552, %52 : i64
    %594 = llvm.add %558, %593 overflow<nsw, nuw> : i64
    %595 = llvm.getelementptr inbounds|nuw %505[%594] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %596 = llvm.load %595 : !llvm.ptr -> f32
    %597 = llvm.add %568, %593 overflow<nsw, nuw> : i64
    %598 = llvm.getelementptr inbounds|nuw %365[%597] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %599 = llvm.load %598 : !llvm.ptr -> f32
    %600 = llvm.intr.maximum(%596, %599) : (f32, f32) -> f32
    %601 = llvm.add %575, %593 overflow<nsw, nuw> : i64
    %602 = llvm.getelementptr inbounds|nuw %365[%601] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %603 = llvm.load %602 : !llvm.ptr -> f32
    %604 = llvm.intr.maximum(%600, %603) : (f32, f32) -> f32
    %605 = llvm.add %583, %593 overflow<nsw, nuw> : i64
    %606 = llvm.getelementptr inbounds|nuw %365[%605] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %607 = llvm.load %606 : !llvm.ptr -> f32
    %608 = llvm.intr.maximum(%604, %607) : (f32, f32) -> f32
    %609 = llvm.add %588, %593 overflow<nsw, nuw> : i64
    %610 = llvm.getelementptr inbounds|nuw %365[%609] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %611 = llvm.load %610 : !llvm.ptr -> f32
    %612 = llvm.intr.maximum(%608, %611) : (f32, f32) -> f32
    llvm.store %612, %595 : f32, !llvm.ptr
    %613 = llvm.add %552, %44 : i64
    %614 = llvm.add %558, %613 overflow<nsw, nuw> : i64
    %615 = llvm.getelementptr inbounds|nuw %505[%614] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %616 = llvm.load %615 : !llvm.ptr -> f32
    %617 = llvm.add %568, %613 overflow<nsw, nuw> : i64
    %618 = llvm.getelementptr inbounds|nuw %365[%617] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %619 = llvm.load %618 : !llvm.ptr -> f32
    %620 = llvm.intr.maximum(%616, %619) : (f32, f32) -> f32
    %621 = llvm.add %575, %613 overflow<nsw, nuw> : i64
    %622 = llvm.getelementptr inbounds|nuw %365[%621] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %623 = llvm.load %622 : !llvm.ptr -> f32
    %624 = llvm.intr.maximum(%620, %623) : (f32, f32) -> f32
    %625 = llvm.add %583, %613 overflow<nsw, nuw> : i64
    %626 = llvm.getelementptr inbounds|nuw %365[%625] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %627 = llvm.load %626 : !llvm.ptr -> f32
    %628 = llvm.intr.maximum(%624, %627) : (f32, f32) -> f32
    %629 = llvm.add %588, %613 overflow<nsw, nuw> : i64
    %630 = llvm.getelementptr inbounds|nuw %365[%629] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %631 = llvm.load %630 : !llvm.ptr -> f32
    %632 = llvm.intr.maximum(%628, %631) : (f32, f32) -> f32
    llvm.store %632, %615 : f32, !llvm.ptr
    %633 = llvm.add %552, %38 : i64
    %634 = llvm.add %558, %633 overflow<nsw, nuw> : i64
    %635 = llvm.getelementptr inbounds|nuw %505[%634] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %636 = llvm.load %635 : !llvm.ptr -> f32
    %637 = llvm.add %568, %633 overflow<nsw, nuw> : i64
    %638 = llvm.getelementptr inbounds|nuw %365[%637] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %639 = llvm.load %638 : !llvm.ptr -> f32
    %640 = llvm.intr.maximum(%636, %639) : (f32, f32) -> f32
    %641 = llvm.add %575, %633 overflow<nsw, nuw> : i64
    %642 = llvm.getelementptr inbounds|nuw %365[%641] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %643 = llvm.load %642 : !llvm.ptr -> f32
    %644 = llvm.intr.maximum(%640, %643) : (f32, f32) -> f32
    %645 = llvm.add %583, %633 overflow<nsw, nuw> : i64
    %646 = llvm.getelementptr inbounds|nuw %365[%645] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %647 = llvm.load %646 : !llvm.ptr -> f32
    %648 = llvm.intr.maximum(%644, %647) : (f32, f32) -> f32
    %649 = llvm.add %588, %633 overflow<nsw, nuw> : i64
    %650 = llvm.getelementptr inbounds|nuw %365[%649] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %651 = llvm.load %650 : !llvm.ptr -> f32
    %652 = llvm.intr.maximum(%648, %651) : (f32, f32) -> f32
    llvm.store %652, %635 : f32, !llvm.ptr
    %653 = llvm.add %552, %43 : i64
    %654 = llvm.add %558, %653 overflow<nsw, nuw> : i64
    %655 = llvm.getelementptr inbounds|nuw %505[%654] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %656 = llvm.load %655 : !llvm.ptr -> f32
    %657 = llvm.add %568, %653 overflow<nsw, nuw> : i64
    %658 = llvm.getelementptr inbounds|nuw %365[%657] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %659 = llvm.load %658 : !llvm.ptr -> f32
    %660 = llvm.intr.maximum(%656, %659) : (f32, f32) -> f32
    %661 = llvm.add %575, %653 overflow<nsw, nuw> : i64
    %662 = llvm.getelementptr inbounds|nuw %365[%661] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %663 = llvm.load %662 : !llvm.ptr -> f32
    %664 = llvm.intr.maximum(%660, %663) : (f32, f32) -> f32
    %665 = llvm.add %583, %653 overflow<nsw, nuw> : i64
    %666 = llvm.getelementptr inbounds|nuw %365[%665] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %667 = llvm.load %666 : !llvm.ptr -> f32
    %668 = llvm.intr.maximum(%664, %667) : (f32, f32) -> f32
    %669 = llvm.add %588, %653 overflow<nsw, nuw> : i64
    %670 = llvm.getelementptr inbounds|nuw %365[%669] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %671 = llvm.load %670 : !llvm.ptr -> f32
    %672 = llvm.intr.maximum(%668, %671) : (f32, f32) -> f32
    llvm.store %672, %655 : f32, !llvm.ptr
    %673 = llvm.add %552, %45 : i64
    %674 = llvm.add %558, %673 overflow<nsw, nuw> : i64
    %675 = llvm.getelementptr inbounds|nuw %505[%674] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %676 = llvm.load %675 : !llvm.ptr -> f32
    %677 = llvm.add %568, %673 overflow<nsw, nuw> : i64
    %678 = llvm.getelementptr inbounds|nuw %365[%677] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %679 = llvm.load %678 : !llvm.ptr -> f32
    %680 = llvm.intr.maximum(%676, %679) : (f32, f32) -> f32
    %681 = llvm.add %575, %673 overflow<nsw, nuw> : i64
    %682 = llvm.getelementptr inbounds|nuw %365[%681] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %683 = llvm.load %682 : !llvm.ptr -> f32
    %684 = llvm.intr.maximum(%680, %683) : (f32, f32) -> f32
    %685 = llvm.add %583, %673 overflow<nsw, nuw> : i64
    %686 = llvm.getelementptr inbounds|nuw %365[%685] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %687 = llvm.load %686 : !llvm.ptr -> f32
    %688 = llvm.intr.maximum(%684, %687) : (f32, f32) -> f32
    %689 = llvm.add %588, %673 overflow<nsw, nuw> : i64
    %690 = llvm.getelementptr inbounds|nuw %365[%689] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %691 = llvm.load %690 : !llvm.ptr -> f32
    %692 = llvm.intr.maximum(%688, %691) : (f32, f32) -> f32
    llvm.store %692, %675 : f32, !llvm.ptr
    %693 = llvm.add %552, %46 : i64
    %694 = llvm.add %558, %693 overflow<nsw, nuw> : i64
    %695 = llvm.getelementptr inbounds|nuw %505[%694] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %696 = llvm.load %695 : !llvm.ptr -> f32
    %697 = llvm.add %568, %693 overflow<nsw, nuw> : i64
    %698 = llvm.getelementptr inbounds|nuw %365[%697] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %699 = llvm.load %698 : !llvm.ptr -> f32
    %700 = llvm.intr.maximum(%696, %699) : (f32, f32) -> f32
    %701 = llvm.add %575, %693 overflow<nsw, nuw> : i64
    %702 = llvm.getelementptr inbounds|nuw %365[%701] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %703 = llvm.load %702 : !llvm.ptr -> f32
    %704 = llvm.intr.maximum(%700, %703) : (f32, f32) -> f32
    %705 = llvm.add %583, %693 overflow<nsw, nuw> : i64
    %706 = llvm.getelementptr inbounds|nuw %365[%705] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %707 = llvm.load %706 : !llvm.ptr -> f32
    %708 = llvm.intr.maximum(%704, %707) : (f32, f32) -> f32
    %709 = llvm.add %588, %693 overflow<nsw, nuw> : i64
    %710 = llvm.getelementptr inbounds|nuw %365[%709] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %711 = llvm.load %710 : !llvm.ptr -> f32
    %712 = llvm.intr.maximum(%708, %711) : (f32, f32) -> f32
    llvm.store %712, %695 : f32, !llvm.ptr
    %713 = llvm.add %552, %47 : i64
    %714 = llvm.add %558, %713 overflow<nsw, nuw> : i64
    %715 = llvm.getelementptr inbounds|nuw %505[%714] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %716 = llvm.load %715 : !llvm.ptr -> f32
    %717 = llvm.add %568, %713 overflow<nsw, nuw> : i64
    %718 = llvm.getelementptr inbounds|nuw %365[%717] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %719 = llvm.load %718 : !llvm.ptr -> f32
    %720 = llvm.intr.maximum(%716, %719) : (f32, f32) -> f32
    %721 = llvm.add %575, %713 overflow<nsw, nuw> : i64
    %722 = llvm.getelementptr inbounds|nuw %365[%721] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %723 = llvm.load %722 : !llvm.ptr -> f32
    %724 = llvm.intr.maximum(%720, %723) : (f32, f32) -> f32
    %725 = llvm.add %583, %713 overflow<nsw, nuw> : i64
    %726 = llvm.getelementptr inbounds|nuw %365[%725] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %727 = llvm.load %726 : !llvm.ptr -> f32
    %728 = llvm.intr.maximum(%724, %727) : (f32, f32) -> f32
    %729 = llvm.add %588, %713 overflow<nsw, nuw> : i64
    %730 = llvm.getelementptr inbounds|nuw %365[%729] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %731 = llvm.load %730 : !llvm.ptr -> f32
    %732 = llvm.intr.maximum(%728, %731) : (f32, f32) -> f32
    llvm.store %732, %715 : f32, !llvm.ptr
    %733 = llvm.add %552, %50 : i64
    llvm.br ^bb111(%733 : i64)
  ^bb113:  // pred: ^bb111
    %734 = llvm.add %550, %52 : i64
    llvm.br ^bb109(%734 : i64)
  ^bb114:  // pred: ^bb109
    %735 = llvm.add %548, %52 : i64
    llvm.br ^bb107(%735 : i64)
  ^bb115:  // pred: ^bb107
    %736 = llvm.add %546, %52 : i64
    llvm.br ^bb105(%736 : i64)
  ^bb116:  // pred: ^bb105
    %737 = llvm.getelementptr %30[1024] : (!llvm.ptr) -> !llvm.ptr, f32
    %738 = llvm.ptrtoint %737 : !llvm.ptr to i64
    %739 = llvm.call @malloc(%738) : (i64) -> !llvm.ptr
    llvm.br ^bb117(%53 : i64)
  ^bb117(%740: i64):  // 2 preds: ^bb116, ^bb127
    %741 = llvm.icmp "slt" %740, %52 : i64
    llvm.cond_br %741, ^bb118, ^bb128
  ^bb118:  // pred: ^bb117
    llvm.br ^bb119(%53 : i64)
  ^bb119(%742: i64):  // 2 preds: ^bb118, ^bb126
    %743 = llvm.icmp "slt" %742, %50 : i64
    llvm.cond_br %743, ^bb120, ^bb127
  ^bb120:  // pred: ^bb119
    llvm.br ^bb121(%53 : i64)
  ^bb121(%744: i64):  // 2 preds: ^bb120, ^bb125
    %745 = llvm.icmp "slt" %744, %48 : i64
    llvm.cond_br %745, ^bb122, ^bb126
  ^bb122:  // pred: ^bb121
    llvm.br ^bb123(%53 : i64)
  ^bb123(%746: i64):  // 2 preds: ^bb122, ^bb124
    %747 = llvm.icmp "slt" %746, %50 : i64
    llvm.cond_br %747, ^bb124, ^bb125
  ^bb124:  // pred: ^bb123
    %748 = llvm.mul %740, %1 overflow<nsw, nuw> : i64
    %749 = llvm.mul %742, %26 overflow<nsw, nuw> : i64
    %750 = llvm.add %748, %749 overflow<nsw, nuw> : i64
    %751 = llvm.mul %746, %48 overflow<nsw, nuw> : i64
    %752 = llvm.add %750, %751 overflow<nsw, nuw> : i64
    %753 = llvm.add %752, %744 overflow<nsw, nuw> : i64
    %754 = llvm.getelementptr inbounds|nuw %505[%753] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %755 = llvm.load %754 : !llvm.ptr -> f32
    %756 = llvm.mul %744, %0 overflow<nsw, nuw> : i64
    %757 = llvm.add %748, %756 overflow<nsw, nuw> : i64
    %758 = llvm.mul %742, %50 overflow<nsw, nuw> : i64
    %759 = llvm.add %757, %758 overflow<nsw, nuw> : i64
    %760 = llvm.add %759, %746 overflow<nsw, nuw> : i64
    %761 = llvm.getelementptr inbounds|nuw %739[%760] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %755, %761 : f32, !llvm.ptr
    %762 = llvm.add %746, %52 : i64
    %763 = llvm.mul %762, %48 overflow<nsw, nuw> : i64
    %764 = llvm.add %750, %763 overflow<nsw, nuw> : i64
    %765 = llvm.add %764, %744 overflow<nsw, nuw> : i64
    %766 = llvm.getelementptr inbounds|nuw %505[%765] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %767 = llvm.load %766 : !llvm.ptr -> f32
    %768 = llvm.add %759, %762 overflow<nsw, nuw> : i64
    %769 = llvm.getelementptr inbounds|nuw %739[%768] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %767, %769 : f32, !llvm.ptr
    %770 = llvm.add %746, %44 : i64
    %771 = llvm.mul %770, %48 overflow<nsw, nuw> : i64
    %772 = llvm.add %750, %771 overflow<nsw, nuw> : i64
    %773 = llvm.add %772, %744 overflow<nsw, nuw> : i64
    %774 = llvm.getelementptr inbounds|nuw %505[%773] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %775 = llvm.load %774 : !llvm.ptr -> f32
    %776 = llvm.add %759, %770 overflow<nsw, nuw> : i64
    %777 = llvm.getelementptr inbounds|nuw %739[%776] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %775, %777 : f32, !llvm.ptr
    %778 = llvm.add %746, %38 : i64
    %779 = llvm.mul %778, %48 overflow<nsw, nuw> : i64
    %780 = llvm.add %750, %779 overflow<nsw, nuw> : i64
    %781 = llvm.add %780, %744 overflow<nsw, nuw> : i64
    %782 = llvm.getelementptr inbounds|nuw %505[%781] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %783 = llvm.load %782 : !llvm.ptr -> f32
    %784 = llvm.add %759, %778 overflow<nsw, nuw> : i64
    %785 = llvm.getelementptr inbounds|nuw %739[%784] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %783, %785 : f32, !llvm.ptr
    %786 = llvm.add %746, %43 : i64
    llvm.br ^bb123(%786 : i64)
  ^bb125:  // pred: ^bb123
    %787 = llvm.add %744, %52 : i64
    llvm.br ^bb121(%787 : i64)
  ^bb126:  // pred: ^bb121
    %788 = llvm.add %742, %52 : i64
    llvm.br ^bb119(%788 : i64)
  ^bb127:  // pred: ^bb119
    %789 = llvm.add %740, %52 : i64
    llvm.br ^bb117(%789 : i64)
  ^bb128:  // pred: ^bb117
    llvm.return %739 : !llvm.ptr
  }
}

