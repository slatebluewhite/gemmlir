; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

@__gemmlir_arena_forward_0 = private global [106176 x i8] undef, align 64
@__constant_16xf32 = private constant [16 x float] [float 0xBFB7DDA900000000, float 0xBF74947820000000, float 0xBF8E998B80000000, float 0x3FD2B156E0000000, float 0x3F8E8FA600000000, float 0x3FCFB766E0000000, float 0x3FCFF0BDC0000000, float 0xBFB9BF9A60000000, float 0xBFA5822F40000000, float 0x3FDD692B20000000, float 0xBFD3876600000000, float 0xBFCF6EDAC0000000, float 0x3FDB56BDC0000000, float 0xBFCE0556C0000000, float 0xBFC4A6F3C0000000, float 0xBFC564B140000000], align 64
@__constant_3x3x8x16xi8 = private constant [3 x [3 x [8 x [16 x i8]]]] [[3 x [8 x [16 x i8]]] [[8 x [16 x i8]] [[16 x i8] c"\FF)\AA\B1\EC\B4I?\E7\AFDvJU\E2/", [16 x i8] c"\1A]42\BD\EE\AC\E4\DA^\D4\CB\04\EC\F7\B4", [16 x i8] c"\BE&\AA>\0FQG\1A\22\F2\19\C6\C9L\BA\FF", [16 x i8] c"\A6\C4\D9\00<F\E7IHO\D48C\83\16\99", [16 x i8] c"\C7\FA\0AT>P\BD\D9?.\DC\1D?\7F:\0F", [16 x i8] c"\CE\054:9\AA(IHa\C5\CE\D7\83X\EF", [16 x i8] c"A\E6\C7\D0I\B05/0\A446!\0B3\E0", [16 x i8] c"\1D\F5Q\1DF6\F7\F3!\CF\05\94\C5\E8\F4\C2"], [8 x [16 x i8]] [[16 x i8] c"4W\D3\CB\FC\91\06\D8\BA\E7\C2!\E0\95\DF\A7", [16 x i8] c"\E3O4\B8%\B7\EF\13\0F\CF\F0c\FE\87\BA\D4", [16 x i8] c"\D65\AD\C0&W.\CA!O\C4\D5GA1\F4", [16 x i8] c"\C3bV4\D1\D9\AB\C8\DF\F1\F7{\14\81\A3%", [16 x i8] c"T\10\EC\E2Q\F3\E5\11\DFF?\05JS\12\A9", [16 x i8] c"=\0FL6\C2l\CF\B9\12\09\EE\D6'\1C\95\B2", [16 x i8] c"\C7\CD\FE\D02\DA0\04\14\17\CA\\\0Act\C5", [16 x i8] c"5\03#\03\F08\B8:\D3\A7\D2\029\F5<;"], [8 x [16 x i8]] [[16 x i8] c"\B0[AR\CA\A5\C7\18\1F8\F3\D3\FA\\\15,", [16 x i8] c"\ED\98\A4\E3\E2N:\1A\1D%C\A9\0B\88#O", [16 x i8] c"#K\F1.#dV\D4\BE\E0\BD\22,\11\D7\F5", [16 x i8] c"\E7F#\9E\016\DE\C3\DD\FC\E6\DA5\10\E3\02", [16 x i8] c"+\D5D\C45\BF&\EF\1B\F0\DDOE\B8T\96", [16 x i8] c"9\19AN#V)\B8\B6\A0I \1B\BB\F5\AC", [16 x i8] c"\12\DB\04\A0\CA\E1\03\B7\18\BFT\06%\18\CD\F5", [16 x i8] c"\F4\F7\0C\BC#\99\D1&\0B\A2\DD\C7\18\9D\B3\8C"]], [3 x [8 x [16 x i8]]] [[8 x [16 x i8]] [[16 x i8] c"\B8\\\FF\AE\1EY\E1\1FJ\D6\CE\CD\DF\88\14\F2", [16 x i8] c"\A3\14<\DE\D5\AB\AE\DC<\E6\ED\00:1E\CF", [16 x i8] c"Q'\01\B8\F5\E3,\DAFR\05\AA)\BAT\E8", [16 x i8] c"\DA\D2\AAI%\0D\13\C7(F\BDC\E0`\9B\C7", [16 x i8] c"/>\1B\DA\0CWU\DCB\AB\C6\E73\0B3\D1", [16 x i8] c"\D5)\BD%D\D7\C3J\EAH\C0\EA\0C\12d0", [16 x i8] c"\B4\AA>\16P\EA\E5\D4\07\F5\CE\18\B4\D7\B2\01", [16 x i8] c"\04\15\FF\AE\10\ECL?\18d\E7RK\FFw\1D"], [8 x [16 x i8]] [[16 x i8] c"\DA\15\D0\F9\1C=\13\EA\EAV\D0\B1\F6\1C=1", [16 x i8] c"\BF\EE\B5\F1\10\C9H\07;\EA\08pJ\CA\B4K", [16 x i8] c"\EC\99\D5\D5\B72\C7EH\9DC\14**``", [16 x i8] c"T\E6=\AC\07\E5\BC\00\DD\E4\00\DC#\B3J\8B", [16 x i8] c"\05\C0.\9B*H2\0A&3\0D\0C5A\8B\EC", [16 x i8] c"\FC\06\04\DEY?\D5\D1\CC\B4\E7Q\12\AD/\A3", [16 x i8] c"\BC\EA\B8\C7+\0B\ED\E7\1C\0DT\05\BAV\F2]", [16 x i8] c"\17B\EDQ&\ACO+\05\E6\AE\8C\B5\96\87\A2"], [8 x [16 x i8]] [[16 x i8] c"\1A\A5\B7d\FEj\BE\19\C1\BA\14\AA\10\AD\ACV", [16 x i8] c"\D8\EF\EC\01\D2\93\DE+\F5K\C5{\D3\B3\BA\D0", [16 x i8] c"I\BC$\1E\F2dH\ECD\F1\E86\C4\06\12Z", [16 x i8] c"\C1\9C\F5\01\07\C0\1B\ED\DBZ?\EE\BD\1C\B3\D1", [16 x i8] c"\CE_\22.\D4\08\F6=\0B\1F2\D2\F6pM\92", [16 x i8] c">\CD\BD\E4N\C0\B9\0EB\18\1C\07B\A0f\A4", [16 x i8] c"\CE\16\BE\A51h\0B&\EEW\AFV.\07\8E\87", [16 x i8] c"<c\0C\11\E2\D0\B0\E9\CF\05\04K\B7\E5\8C<"]], [3 x [8 x [16 x i8]]] [[8 x [16 x i8]] [[16 x i8] c"\FE\0A\A7%\ECZ+\EB\BB\AAO\EC\0A\CD\F75", [16 x i8] c"\04\D0\D9T\15\C0\19\F9\BD\B1G\B8\FB\CE6\E4", [16 x i8] c"\F04\AA\22\1F%\05\00\C0\EE(\CF\08\A5\F6\AE", [16 x i8] c"\D3\FE\D5\EF\0B\C1.\CF\F9\BCG\EB\FBYo)", [16 x i8] c"\10H\E9\CF\B6\CB\1E\CEA)\B4\98\C5\81\E9\AF", [16 x i8] c"a1\CB\1EE\EE-\FA\F3\0F\B6/\1C\F5\D6E", [16 x i8] c",\BC\C8Y\C8\06+\D0\E2\CF\B5\CC0\180\F7", [16 x i8] c"^B\EAT,)=\CEI\1C\0F{3J\E3\98"], [8 x [16 x i8]] [[16 x i8] c"M\BF\AF\03\C9\9B\B2.\13\D2\BD\EA\F2\17D\F6", [16 x i8] c"'(\EE\0D\12]\04\17\BB\0B\F5\16\E2\98\F8\03", [16 x i8] c"\0A\16\FAL\DFq)\DC\E3!HgF\1B\DA8", [16 x i8] c"\BC\B1L\CB\BA\A9\0C\E5\F9\F5\08\FF\19'\E1\90", [16 x i8] c"\A5\A8\EC\BC\B3h\AB\03\0A\C1L2\E4E\CA\08", [16 x i8] c"'\9C\C8S\EB`\F7\F0\BB\D0.6\BE\D5\BD\17", [16 x i8] c"'\FBC\BE\1A\BA\C4F\B9B4\BE\BC\D7Y6", [16 x i8] c"\B5c\FF\DE\C2\B0+\FA\D4F\00\19C\A7cK"], [8 x [16 x i8]] [[16 x i8] c"\F7\9F\ED\A8>\B3\AD\EB\FA\1137\B4\CEi\F9", [16 x i8] c":\C2\ED[\BE\\\B2\EEG`\C3\0F\BA\9A\8E\91", [16 x i8] c"X\AET\DF\22:\BBK\CC(8\F8\D6t\FA\A1", [16 x i8] c"\A5\AF\B3\0D\07L\C6\0E\C5\FB\13\D1\17\1C\BB\8A", [16 x i8] c"\BA\E6\B1\C5Y.\B6\E4\DDB\1E\10\F8\DBM\8D", [16 x i8] c"\0D\C2!\1C\E1\05B\EF\FB$,\E2\C9Jho", [16 x i8] c"\C6K\DE\F4\E1\06H\EF\C7\F3\096\1D\00\1D\BD", [16 x i8] c"\DC\F8\0C\1E\14*3>>W\1B\F8.\AA\0C\8D"]]], align 64
@__constant_16xi32 = private constant [16 x i32] [i32 -1379, i32 -2202, i32 1304, i32 -2244, i32 2450, i32 5210, i32 -5263, i32 1106, i32 -3975, i32 -6599, i32 -5225, i32 -7063, i32 -1256, i32 12331, i32 -5055, i32 -496], align 64
@__constant_144x16xi8 = private constant [144 x [16 x i8]] [[16 x i8] c"\97 \D0\1F\1D\E1\F3E\A1\BC\AF\B2\DE\16\BF!", [16 x i8] c"\D9%\9C\FA\D9\AB\049I\EF\CD\E7\96\E7\09\AF", [16 x i8] c"\D7\1A\DD,\05\0E\E2\D9V\0C\AE\BB[\06\EF*", [16 x i8] c"\B4\C56\05\E50}\13\C7\C9R\B9\AD\1D=\CF", [16 x i8] c"m?\A6\00I\A7\0C\F8\11\9F\ED\D3\92I\C7\B1", [16 x i8] c"\D5\15P7(\1B1F\C7lNL\B2\1C\E6\1D", [16 x i8] c"\B0\DA\C8@\04\FF\19N(\B3\10/\CE\127\EC", [16 x i8] c"\85\12\DA\1CJ\04\C5\F2\06$\FB\D9(\E3\FD\05", [16 x i8] c"\F3\BD\C6\07\0E\F2\B7\1C\0C\EA\01\06C\D4\FF\DE", [16 x i8] c"\DD\1B8(I\E3\22\B5\ED\17\1B\BC\C1\E32\15", [16 x i8] c"-\FEV\16 \BD\11\FD\EC\FC\D3\D2P\1F\CC\B6", [16 x i8] c"w\C9\FC\C6P\ED\E7P:B\00\0AK\BC\CE.", [16 x i8] c"\8C\14\C9%H\B7\DC\F2\01\912\E44\0B\DE\DE", [16 x i8] c"kJ\AC\B8\B6\\#\C9\AC!D\06\C9Y\BC\11", [16 x i8] c"P\DA\EF>E\E3\92\A8\1A\1E\D8\DD\F4\F9L\C4", [16 x i8] c"h\02\E4\BEY)b\F8\F5\16\B8\C8d\AE\D4\ED", [16 x i8] c"\02\DC\EE\F6\B5 Q\FD\A6\C9\D5\AE\19\D0,R", [16 x i8] c"\C1\D9IRO\C8j\B9!j\DE\EC\1D\CB\1E5", [16 x i8] c"b6*\F2\D3W;\F1\F5\0F\0D[$\E1\01E", [16 x i8] c"\EA\B1^\E1\CE*>\1F\C2h\CC\0D\04\FE\E7I", [16 x i8] c"\93#\D0\B9\D2@\D4\C0\E4\9A\06'l\D5\F8\10", [16 x i8] c"A\05+\ED\EB&\EF\A3\03\A5\05\E8\C9#\D9\EE", [16 x i8] c"xQ<\0E\D6\1DN9<P$\09\A1\D9\0B\CD", [16 x i8] c"6/\D6\E1J(\CE\10YW\EE\D7\E2\C07\FB", [16 x i8] c"?D\A4\D7\00%\CC\EF\A8(M\16\A1\18\D3\1E", [16 x i8] c">\F7\E7\CD\BF\DB\9E\AE\14,\AF\03\17\B8<@", [16 x i8] c"\8C\E3\F1\ED\B7M\D3[\9D\B5W\EDC\CC\FC\A7", [16 x i8] c"9\19`\F7\AC&\94\05\EB\CB\CD\FAQC\BB\FD", [16 x i8] c"\0D\07%\D2T\C9C\CB\DA\EF\00\D2m\A8\0D\D3", [16 x i8] c"\F0J\B5(%\A3\AC$\9C\BB\D2\FF\D1N\1A\D5", [16 x i8] c"U\D8\AC\134G\AA#\12\03\C1\13(\1D0\A7", [16 x i8] c"\DE:+B\D5\D2@\0E\1E\E1\AF\A8w>\0B\C7", [16 x i8] c"bD\E5\BC\0F\1A\A4\D5:\19\9DT3M\CF\03", [16 x i8] c"i\C3Q\00/\A4\C9\D1\BD\18\A6\E3D#\B0Y", [16 x i8] c"\A7\AC\D2\04\AAX1\22\A7\BB\9E\D8f\DF\DD\04", [16 x i8] c"\CE8\DE\15\F0\1B\16\EA\A6\C8\15\EE\DA5\13\E8", [16 x i8] c"\FB\D5T\C6\FA\F6\F3\F2\F20\DF\D8\A7\07\F4\F3", [16 x i8] c"\E1J\C9\F6NG \FB\17\F2\D3\22u2\0F\18", [16 x i8] c"\F5K\EB94\03\A0R\EA\CF>\B2b\12\B9\01", [16 x i8] c">Q\EB\CB1(\AB\B3\F4\1E\D9\BD\C8S\E2Y", [16 x i8] c"\AB\D0\E3(\BD\16\9C\A2T\A6\03\02l\DC&9", [16 x i8] c"\9E\15\D8O/\C4r\BF\9F\BB\19\F2\8A<\E8\EF", [16 x i8] c"\BA\CCUO\E7\E0_Ka\A2\9F\F8\97\F7%\04", [16 x i8] c"\90\DD\BB\C9,\B8\BC\C1\07\0B\1A\18@\C2\0CW", [16 x i8] c"k:\D5@\06\1BhK(\01\1D\A9E\12\DA\12", [16 x i8] c"i\04\AC\E1\1B\CF7\19 \0A\B1 \9C*\D1V", [16 x i8] c"\A3\AC\C7\E1M4\C1\CD\FB\A0\CC\1Bs\A6\B9A", [16 x i8] c"\A9\DE\BE\DA\CFX\13\22)GF\10\DF\F0CM", [16 x i8] c"\95\BC\A8!M\C9\AA\10\C5/C@\09\D9\F7\B4", [16 x i8] c"4\B4D\BE\BC\18\DD[\18\0F\F6,\ED\CD\05\16", [16 x i8] c"\A7\C4\F4\F7\D0\F9\BC\F4\FAS\1B6\1E\E9\EC\A9", [16 x i8] c"e\F6\05\BD\E0\DAS\DA\FF\9B35\08L\04\ED", [16 x i8] c"I\E3\22LY\AE\FD\CBa\F3\DB?\E9\DD\B65", [16 x i8] c"5\E1O;\CD\E3 \C5-=\E8\F0ZU\16\C9", [16 x i8] c"\ADO\A9\B4\CA\03S\E4\FA\BB\16\A39,\CE\E4", [16 x i8] c"\B2\CB\DB %\AE\96L\AB\A3<\0Fb\E5!\DE", [16 x i8] c"?6\A3\0E\16X\95B8\B8\E7\13QF7\FE", [16 x i8] c"1T?;/F\81\E8\CE\0F=\D19\19\D0\0E", [16 x i8] c"\8E=\00\E0\08\D4PH1\9D\0A\D1\0D\F0\E51", [16 x i8] c"?+&\D5\A8\00\\O\CC\FF4\FC\EB&\DF\CA", [16 x i8] c"\A5\DD\18PL\16P,\A7\DB\B1>\EFW\FD\BF", [16 x i8] c"\8C2\15\E2[<g$/B\A2%\E3\DE*\E5", [16 x i8] c"\F2\08*\DC0\11\FA\F85,_\D5\E3\AC\FDM", [16 x i8] c"U\D2\0A\EA\06\BF @\F7m\E1\EE/\D8\D7\C6", [16 x i8] c"\C8\F4\EE6?\09\93\B2I\D4c\08 \15\0B\E7", [16 x i8] c"\A5\04\DF\FAP\B7T\13\E4\F2\072B\B6(\EF", [16 x i8] c"\D9?\E3\13\EE\A3T7\07;\A3\CC@@\17\A7", [16 x i8] c"\86\E1\14\194\FF\CB\1B\1F1 [\A1\1B\C8\B1", [16 x i8] c"7&a> Km1\FD\BD\A9H>\BA\11\D7", [16 x i8] c"p\EA\09\00\E3\CB|\AF\A3f\F5\E1\09E\12\B8", [16 x i8] c"\C8\F8[\C5\E8\A6j\C6P\16.\D3H\1D\178", [16 x i8] c"\1F-7\0F\E8\ADf\BB\D2\DE\\\F3(D\D9\F3", [16 x i8] c"F\AF\ED\B8\BCUq]W3\F9\DA\00\AE\1C1", [16 x i8] c"\1D\C9W\0FB\E5\0D7%0\0C1Q<\B3\1D", [16 x i8] c"\A7\FA\F5\B6\0B\BE\17\F7\C4\B2\9F\EA\D7S\CB\F4", [16 x i8] c"0\F8Z*\A7\DD\FC\10\AB4\00\F4\95\DF\00\22", [16 x i8] c")\1Ec\14\D63_\1B\CD\B7\B1\FC\10\1A\F6Q", [16 x i8] c"\BD\AB\14\BF\DA\F1\93\F6\EC\BF$\C3\F4\AB\D3K", [16 x i8] c"\12:\A3.\03\0E\17\1AaD\ED:I\00\B7\DF", [16 x i8] c"U\D3)\00\1D,\9E\A7\BFF\9E\B2w\1D\08\A9", [16 x i8] c"n\02\1F\F2P\B8\B2\E7\1F\D7:\B1x\E6\D1\1B", [16 x i8] c"\14R4)\FF\1F\E0\F1\D8F\1B\14\04#\CC2", [16 x i8] c"\02\14\A2\03F\E1\F8\B3C\A9\BA\01K\1B\B2\EC", [16 x i8] c"\B0\EAd-\D2\0C\BD\E7\F9\BC\F3\15I\1E\E1\AF", [16 x i8] c"f\E1J.\D1\CDr\C2X\FD\069\DA0,\1B", [16 x i8] c"tIB\CA\22\F9\86\09\05\B1\01\D4\0A\B10\B6", [16 x i8] c"\BBM\FE\B1\B2\AD\C1\C2\AF_5V\0CB\1C\B6", [16 x i8] c"\\\1C\ED\C51\B7\F0\\\\\16\AE\05\CF9\FF\17", [16 x i8] c"\B4\F1\02\CB-\C4\9E/,\92 L\CF\E8\B1J", [16 x i8] c"O\FB\A4%\17\DD\F5\D9\FB`\F7\F8Z\CA\1F\FE", [16 x i8] c"\C8P\C6\D9\DF\D3\DD\02\B7\D0\FD\FD\C7\DF\C7Q", [16 x i8] c"a\07\EF\C2\C6.\F82\BD\B3V \8DZ\F39", [16 x i8] c"7\D1\22K\DB\F2\8E\16&\B2\FA\CE\B4\00J\F8", [16 x i8] c"\E5\CE\D5\22!\CEk\F4\A7 \A6(\C9\CC@\F9", [16 x i8] c" \22\C7\0FP1\DB\0DT\D7S\A8\F9\E2>\D4", [16 x i8] c"\D9\F9\F9\F1/\BC4\D9\B4\DC\\\00$\EEJ\DC", [16 x i8] c"\EB,\D7'2\E4\03\1A\19X \E8\D7-,0", [16 x i8] c"\FF\F0\A2\E5\FC\BEz1\F5\C9\BA\F4[\C9\14\E5", [16 x i8] c"\22\FB\EF\CED\DC\E0\D2\03\98\EB \14\D4>1", [16 x i8] c"\9A\D2\\\F7\0E\D0\DB\D9a\EE\A1\FD\925\E3\06", [16 x i8] c"\FD\16\019\C8!\9EK\0D\9A\F5\01kID$", [16 x i8] c"\8E\BB*\C0JJ\D0\D2\E6/\EF<\9A\DC+\C7", [16 x i8] c"3\DBQB\A7\08\16N\BB\B6^\BE\AF\F5L\DF", [16 x i8] c"\12-L\CA-\02\FC\DB\EA\A26(\0B\FC0K", [16 x i8] c"f\01\FA\12\C9\00\EE\\\DEM\19\F6\18\0E\C6\D0", [16 x i8] c"\94\E0\EA\1D3@\E2\FC\D0\ACM!\0A\CB*F", [16 x i8] c"g0\DF\B4\A4\B6\053a\B2%*TXI\F0", [16 x i8] c"\85\C5P'\F4R2\AE\F5\05\C0\B4&\0D\B9\C8", [16 x i8] c"!\12\F7\AE\F41\D3\02\ABJ\F3\DA8\ED\DB\E3", [16 x i8] c"\DB\0F\9E\C4\DB:;\F9\C6\FC\9CFUXO-", [16 x i8] c"@\D8KE\C9;\87=[\91\02\DD\D2?\F1\0F", [16 x i8] c"\E7\AF\12\10&\1E\81\ACaN\02\F0/+I\12", [16 x i8] c"4E\FC\1F\F8\DB\C6\FAP\C6\18\A8R\01\C7\A6", [16 x i8] c"\CD\FC\E7\D5\E7\C3\CD8\D1L\F4\F7\03\DE\00\BD", [16 x i8] c"\04\0F\AB\0F\C8\EC\FDH\1D\E3\A0\AA\E4\C1:\C4", [16 x i8] c"\E9\E7\CC\C0\CC\F8\AD\0C\ED\DDI\D23\E1\EC\0F", [16 x i8] c"\AC\17\C2 \DF^/%\DB\1EX+\CB\CFD-", [16 x i8] c"\AD\065.Y\08\AADA.\1E\CDZ0\D4\DA", [16 x i8] c"\93\E3\B2\FF4/\BF\D1\07<\A4311)\C6", [16 x i8] c"\E6(\17' \C4\C1\D6\D4\13E'\E6\B5%D", [16 x i8] c"{\0F\CF3\E2\10\DB\1A\DF\AA\FB\F4\B1\096\EC", [16 x i8] c"\98J\F7\CFIWW\FF\A9\C8\A8OG\DB\F2\0D", [16 x i8] c"\E7J D\E8\C9r\E1\00\9E\E9V\EF\C8\1D\12", [16 x i8] c"M\B4\02<\1A\F3\AC\11\B7h+\B1\FF\B1\B7\AB", [16 x i8] c"|\ED\19\F7\0A\AC\0F62\DCK\19\A3\18LK", [16 x i8] c"s\006\EF.\06\AC\19\0B=\CD\06E\AD\05?", [16 x i8] c"\CE\B2\CD\1FX\D9\F1\0C\DE[\F3\C9I\1A\FF1", [16 x i8] c"\A4\B5\B4IOB\CB\DB\E57\BD&,\17\F0!", [16 x i8] c"\889\00\DF\18S\1C\F6\A3\ADa g2\14F", [16 x i8] c"\B9\C2\ECD\E7\D0\FF\F0X\1A\AD\FE\B3\E9E\C0", [16 x i8] c"\E2\F3:\02\FC$Y\0C\1BJQ+\19\02\22\18", [16 x i8] c"a\B7\AB\00\E3\F47\17\BF\F7\A8#O\10\B81", [16 x i8] c"S\CBb.\E3\041\B0J5\DF\AB\F0\22\C7\E1", [16 x i8] c"\CCR\A8D\0FB\C2\B6\22CS\F0\EE\CC8\0D", [16 x i8] c"\8C #\DA\06?\1B\F4\D8\F3\22C\0E\1D\1D\0B", [16 x i8] c"\09\E0\F1?\FD\E2Y\F3-\C4\12\E3\F6\07\EE\0A", [16 x i8] c"\CF\07F\E6\D4R/\F91\FE \ACNK\F4\AF", [16 x i8] c"8I\DC\FF\12\FCl\C3@!\D2\D5)\06\B1\A7", [16 x i8] c"HC,\DB$\AA\0E\DA\C4\09aE\13\DB\BEQ", [16 x i8] c"f \B5\00\B9\D0\F79cP9\00\B4\10\C1\1D", [16 x i8] c"f;\C6\10\BF\19\94\EB\08\F9I\BCS\D8\C2\19", [16 x i8] c";\BD\C7%\FB\04z\B8\04J\AB/\96\A9\11\BD", [16 x i8] c"\DA\BC\D6B[\F6a\17\DE\E6\E3\C9\95\B1\CE\16", [16 x i8] c"\98S-\CF\E9\0B\DE9\F2\0B\B1\B7\CF\D9L#"], align 64
@global_seed = private global i64 0

declare ptr @malloc(i64)

declare void @tiled_matmul_auto(i64, i64, i64, ptr, ptr, ptr, ptr, i64, i64, i64, i64, float, float, i32, i32, float, float, i1, i1, i1, i1, i1, i8, i32)

declare void @memrefCopy(i64, ptr, ptr)

declare void @gemmlir_memset(ptr, i32, i64)

declare void @gemmlir_flush()

declare void @tiled_conv_stride_auto(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i1, i1, i1, ptr, ptr, ptr, ptr, i32, float, i32, i32, i32, i32)

define ptr @forward(ptr %0) {
  br label %2

2:                                                ; preds = %197, %1
  %3 = phi i64 [ %198, %197 ], [ 0, %1 ]
  %4 = icmp slt i64 %3, 1
  br i1 %4, label %5, label %199

5:                                                ; preds = %2
  br label %6

6:                                                ; preds = %195, %5
  %7 = phi i64 [ %196, %195 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 8
  br i1 %8, label %9, label %197

9:                                                ; preds = %6
  br label %10

10:                                               ; preds = %193, %9
  %11 = phi i64 [ %194, %193 ], [ 0, %9 ]
  %12 = icmp slt i64 %11, 16
  br i1 %12, label %13, label %195

13:                                               ; preds = %10
  br label %14

14:                                               ; preds = %186, %13
  %15 = phi i64 [ %192, %186 ], [ 0, %13 ]
  %16 = icmp slt i64 %15, 16
  br i1 %16, label %17, label %193

17:                                               ; preds = %14
  %18 = mul nuw nsw i64 %3, 2048
  %19 = mul nuw nsw i64 %7, 256
  %20 = add nuw nsw i64 %18, %19
  %21 = mul nuw nsw i64 %11, 16
  %22 = add nuw nsw i64 %20, %21
  %23 = add nuw nsw i64 %22, %15
  %24 = getelementptr inbounds nuw float, ptr %0, i64 %23
  %25 = load float, ptr %24, align 4
  %26 = fmul float %25, 0x403F0B27E0000000
  %27 = call float @llvm.roundeven.f32(float %26)
  %28 = fptosi float %27 to i32
  %29 = sub i32 %28, -128
  %30 = icmp ult i32 %29, 256
  br i1 %30, label %31, label %32

31:                                               ; preds = %17
  br label %35

32:                                               ; preds = %17
  %33 = icmp slt i32 %28, -128
  %34 = select i1 %33, i32 -128, i32 127
  br label %35

35:                                               ; preds = %31, %32
  %36 = phi i32 [ %34, %32 ], [ %28, %31 ]
  br label %37

37:                                               ; preds = %35
  %38 = trunc i32 %36 to i8
  %39 = mul nuw nsw i64 %11, 128
  %40 = add nuw nsw i64 %18, %39
  %41 = mul nuw nsw i64 %15, 8
  %42 = add nuw nsw i64 %40, %41
  %43 = add nuw nsw i64 %42, %7
  %44 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %43
  store i8 %38, ptr %44, align 1
  %45 = add i64 %15, 1
  %46 = add nuw nsw i64 %22, %45
  %47 = getelementptr inbounds nuw float, ptr %0, i64 %46
  %48 = load float, ptr %47, align 4
  %49 = fmul float %48, 0x403F0B27E0000000
  %50 = call float @llvm.roundeven.f32(float %49)
  %51 = fptosi float %50 to i32
  %52 = sub i32 %51, -128
  %53 = icmp ult i32 %52, 256
  br i1 %53, label %54, label %55

54:                                               ; preds = %37
  br label %58

55:                                               ; preds = %37
  %56 = icmp slt i32 %51, -128
  %57 = select i1 %56, i32 -128, i32 127
  br label %58

58:                                               ; preds = %54, %55
  %59 = phi i32 [ %57, %55 ], [ %51, %54 ]
  br label %60

60:                                               ; preds = %58
  %61 = trunc i32 %59 to i8
  %62 = mul nuw nsw i64 %45, 8
  %63 = add nuw nsw i64 %40, %62
  %64 = add nuw nsw i64 %63, %7
  %65 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %64
  store i8 %61, ptr %65, align 1
  %66 = add i64 %15, 2
  %67 = add nuw nsw i64 %22, %66
  %68 = getelementptr inbounds nuw float, ptr %0, i64 %67
  %69 = load float, ptr %68, align 4
  %70 = fmul float %69, 0x403F0B27E0000000
  %71 = call float @llvm.roundeven.f32(float %70)
  %72 = fptosi float %71 to i32
  %73 = sub i32 %72, -128
  %74 = icmp ult i32 %73, 256
  br i1 %74, label %75, label %76

75:                                               ; preds = %60
  br label %79

76:                                               ; preds = %60
  %77 = icmp slt i32 %72, -128
  %78 = select i1 %77, i32 -128, i32 127
  br label %79

79:                                               ; preds = %75, %76
  %80 = phi i32 [ %78, %76 ], [ %72, %75 ]
  br label %81

81:                                               ; preds = %79
  %82 = trunc i32 %80 to i8
  %83 = mul nuw nsw i64 %66, 8
  %84 = add nuw nsw i64 %40, %83
  %85 = add nuw nsw i64 %84, %7
  %86 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %85
  store i8 %82, ptr %86, align 1
  %87 = add i64 %15, 3
  %88 = add nuw nsw i64 %22, %87
  %89 = getelementptr inbounds nuw float, ptr %0, i64 %88
  %90 = load float, ptr %89, align 4
  %91 = fmul float %90, 0x403F0B27E0000000
  %92 = call float @llvm.roundeven.f32(float %91)
  %93 = fptosi float %92 to i32
  %94 = sub i32 %93, -128
  %95 = icmp ult i32 %94, 256
  br i1 %95, label %96, label %97

96:                                               ; preds = %81
  br label %100

97:                                               ; preds = %81
  %98 = icmp slt i32 %93, -128
  %99 = select i1 %98, i32 -128, i32 127
  br label %100

100:                                              ; preds = %96, %97
  %101 = phi i32 [ %99, %97 ], [ %93, %96 ]
  br label %102

102:                                              ; preds = %100
  %103 = trunc i32 %101 to i8
  %104 = mul nuw nsw i64 %87, 8
  %105 = add nuw nsw i64 %40, %104
  %106 = add nuw nsw i64 %105, %7
  %107 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %106
  store i8 %103, ptr %107, align 1
  %108 = add i64 %15, 4
  %109 = add nuw nsw i64 %22, %108
  %110 = getelementptr inbounds nuw float, ptr %0, i64 %109
  %111 = load float, ptr %110, align 4
  %112 = fmul float %111, 0x403F0B27E0000000
  %113 = call float @llvm.roundeven.f32(float %112)
  %114 = fptosi float %113 to i32
  %115 = sub i32 %114, -128
  %116 = icmp ult i32 %115, 256
  br i1 %116, label %117, label %118

117:                                              ; preds = %102
  br label %121

118:                                              ; preds = %102
  %119 = icmp slt i32 %114, -128
  %120 = select i1 %119, i32 -128, i32 127
  br label %121

121:                                              ; preds = %117, %118
  %122 = phi i32 [ %120, %118 ], [ %114, %117 ]
  br label %123

123:                                              ; preds = %121
  %124 = trunc i32 %122 to i8
  %125 = mul nuw nsw i64 %108, 8
  %126 = add nuw nsw i64 %40, %125
  %127 = add nuw nsw i64 %126, %7
  %128 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %127
  store i8 %124, ptr %128, align 1
  %129 = add i64 %15, 5
  %130 = add nuw nsw i64 %22, %129
  %131 = getelementptr inbounds nuw float, ptr %0, i64 %130
  %132 = load float, ptr %131, align 4
  %133 = fmul float %132, 0x403F0B27E0000000
  %134 = call float @llvm.roundeven.f32(float %133)
  %135 = fptosi float %134 to i32
  %136 = sub i32 %135, -128
  %137 = icmp ult i32 %136, 256
  br i1 %137, label %138, label %139

138:                                              ; preds = %123
  br label %142

139:                                              ; preds = %123
  %140 = icmp slt i32 %135, -128
  %141 = select i1 %140, i32 -128, i32 127
  br label %142

142:                                              ; preds = %138, %139
  %143 = phi i32 [ %141, %139 ], [ %135, %138 ]
  br label %144

144:                                              ; preds = %142
  %145 = trunc i32 %143 to i8
  %146 = mul nuw nsw i64 %129, 8
  %147 = add nuw nsw i64 %40, %146
  %148 = add nuw nsw i64 %147, %7
  %149 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %148
  store i8 %145, ptr %149, align 1
  %150 = add i64 %15, 6
  %151 = add nuw nsw i64 %22, %150
  %152 = getelementptr inbounds nuw float, ptr %0, i64 %151
  %153 = load float, ptr %152, align 4
  %154 = fmul float %153, 0x403F0B27E0000000
  %155 = call float @llvm.roundeven.f32(float %154)
  %156 = fptosi float %155 to i32
  %157 = sub i32 %156, -128
  %158 = icmp ult i32 %157, 256
  br i1 %158, label %159, label %160

159:                                              ; preds = %144
  br label %163

160:                                              ; preds = %144
  %161 = icmp slt i32 %156, -128
  %162 = select i1 %161, i32 -128, i32 127
  br label %163

163:                                              ; preds = %159, %160
  %164 = phi i32 [ %162, %160 ], [ %156, %159 ]
  br label %165

165:                                              ; preds = %163
  %166 = trunc i32 %164 to i8
  %167 = mul nuw nsw i64 %150, 8
  %168 = add nuw nsw i64 %40, %167
  %169 = add nuw nsw i64 %168, %7
  %170 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %169
  store i8 %166, ptr %170, align 1
  %171 = add i64 %15, 7
  %172 = add nuw nsw i64 %22, %171
  %173 = getelementptr inbounds nuw float, ptr %0, i64 %172
  %174 = load float, ptr %173, align 4
  %175 = fmul float %174, 0x403F0B27E0000000
  %176 = call float @llvm.roundeven.f32(float %175)
  %177 = fptosi float %176 to i32
  %178 = sub i32 %177, -128
  %179 = icmp ult i32 %178, 256
  br i1 %179, label %180, label %181

180:                                              ; preds = %165
  br label %184

181:                                              ; preds = %165
  %182 = icmp slt i32 %177, -128
  %183 = select i1 %182, i32 -128, i32 127
  br label %184

184:                                              ; preds = %180, %181
  %185 = phi i32 [ %183, %181 ], [ %177, %180 ]
  br label %186

186:                                              ; preds = %184
  %187 = trunc i32 %185 to i8
  %188 = mul nuw nsw i64 %171, 8
  %189 = add nuw nsw i64 %40, %188
  %190 = add nuw nsw i64 %189, %7
  %191 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), i64 %190
  store i8 %187, ptr %191, align 1
  %192 = add i64 %15, 8
  br label %14

193:                                              ; preds = %14
  %194 = add i64 %11, 1
  br label %10

195:                                              ; preds = %10
  %196 = add i64 %7, 1
  br label %6

197:                                              ; preds = %6
  %198 = add i64 %3, 1
  br label %2

199:                                              ; preds = %2
  call void asm sideeffect alignstack ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"()
  call void @gemmlir_flush()
  call void @tiled_conv_stride_auto(i32 1, i32 16, i32 16, i32 8, i32 16, i32 16, i32 16, i32 1, i32 1, i32 1, i32 1, i32 3, i32 8, i32 16, i32 16, i1 false, i1 false, i1 false, i1 false, i1 false, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 2048), ptr @__constant_3x3x8x16xi8, ptr @__constant_16xi32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i32 1, float 0x3F60AE1CC0000000, i32 0, i32 0, i32 0, i32 1)
  call void @gemmlir_memset(ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27200), i32 0, i64 288)
  call void @gemmlir_memset(ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32096), i32 0, i64 288)
  br label %200

200:                                              ; preds = %249, %199
  %201 = phi i64 [ %250, %249 ], [ 0, %199 ]
  %202 = icmp slt i64 %201, 1
  br i1 %202, label %203, label %251

203:                                              ; preds = %200
  br label %204

204:                                              ; preds = %247, %203
  %205 = phi i64 [ %248, %247 ], [ 0, %203 ]
  %206 = icmp slt i64 %205, 16
  br i1 %206, label %207, label %249

207:                                              ; preds = %204
  br label %208

208:                                              ; preds = %245, %207
  %209 = phi i64 [ %246, %245 ], [ 0, %207 ]
  %210 = icmp slt i64 %209, 1
  br i1 %210, label %211, label %247

211:                                              ; preds = %208
  br label %212

212:                                              ; preds = %215, %211
  %213 = phi i64 [ %244, %215 ], [ 0, %211 ]
  %214 = icmp slt i64 %213, 16
  br i1 %214, label %215, label %245

215:                                              ; preds = %212
  %216 = mul nuw nsw i64 %201, 5184
  %217 = mul nuw nsw i64 %205, 288
  %218 = add nuw nsw i64 %216, %217
  %219 = mul nuw nsw i64 %209, 16
  %220 = add nuw nsw i64 %218, %219
  %221 = add nuw nsw i64 %220, %213
  %222 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %221
  store i8 0, ptr %222, align 1
  %223 = add i64 %213, 1
  %224 = add nuw nsw i64 %220, %223
  %225 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %224
  store i8 0, ptr %225, align 1
  %226 = add i64 %213, 2
  %227 = add nuw nsw i64 %220, %226
  %228 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %227
  store i8 0, ptr %228, align 1
  %229 = add i64 %213, 3
  %230 = add nuw nsw i64 %220, %229
  %231 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %230
  store i8 0, ptr %231, align 1
  %232 = add i64 %213, 4
  %233 = add nuw nsw i64 %220, %232
  %234 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %233
  store i8 0, ptr %234, align 1
  %235 = add i64 %213, 5
  %236 = add nuw nsw i64 %220, %235
  %237 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %236
  store i8 0, ptr %237, align 1
  %238 = add i64 %213, 6
  %239 = add nuw nsw i64 %220, %238
  %240 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %239
  store i8 0, ptr %240, align 1
  %241 = add i64 %213, 7
  %242 = add nuw nsw i64 %220, %241
  %243 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27488), i64 %242
  store i8 0, ptr %243, align 1
  %244 = add i64 %213, 8
  br label %212

245:                                              ; preds = %212
  %246 = add i64 %209, 1
  br label %208

247:                                              ; preds = %208
  %248 = add i64 %205, 1
  br label %204

249:                                              ; preds = %204
  %250 = add i64 %201, 1
  br label %200

251:                                              ; preds = %200
  br label %252

252:                                              ; preds = %301, %251
  %253 = phi i64 [ %302, %301 ], [ 0, %251 ]
  %254 = icmp slt i64 %253, 1
  br i1 %254, label %255, label %303

255:                                              ; preds = %252
  br label %256

256:                                              ; preds = %299, %255
  %257 = phi i64 [ %300, %299 ], [ 0, %255 ]
  %258 = icmp slt i64 %257, 16
  br i1 %258, label %259, label %301

259:                                              ; preds = %256
  br label %260

260:                                              ; preds = %297, %259
  %261 = phi i64 [ %298, %297 ], [ 0, %259 ]
  %262 = icmp slt i64 %261, 1
  br i1 %262, label %263, label %299

263:                                              ; preds = %260
  br label %264

264:                                              ; preds = %267, %263
  %265 = phi i64 [ %296, %267 ], [ 0, %263 ]
  %266 = icmp slt i64 %265, 16
  br i1 %266, label %267, label %297

267:                                              ; preds = %264
  %268 = mul nuw nsw i64 %253, 5184
  %269 = mul nuw nsw i64 %257, 288
  %270 = add nuw nsw i64 %268, %269
  %271 = mul nuw nsw i64 %261, 16
  %272 = add nuw nsw i64 %270, %271
  %273 = add nuw nsw i64 %272, %265
  %274 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %273
  store i8 0, ptr %274, align 1
  %275 = add i64 %265, 1
  %276 = add nuw nsw i64 %272, %275
  %277 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %276
  store i8 0, ptr %277, align 1
  %278 = add i64 %265, 2
  %279 = add nuw nsw i64 %272, %278
  %280 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %279
  store i8 0, ptr %280, align 1
  %281 = add i64 %265, 3
  %282 = add nuw nsw i64 %272, %281
  %283 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %282
  store i8 0, ptr %283, align 1
  %284 = add i64 %265, 4
  %285 = add nuw nsw i64 %272, %284
  %286 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %285
  store i8 0, ptr %286, align 1
  %287 = add i64 %265, 5
  %288 = add nuw nsw i64 %272, %287
  %289 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %288
  store i8 0, ptr %289, align 1
  %290 = add i64 %265, 6
  %291 = add nuw nsw i64 %272, %290
  %292 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %291
  store i8 0, ptr %292, align 1
  %293 = add i64 %265, 7
  %294 = add nuw nsw i64 %272, %293
  %295 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27760), i64 %294
  store i8 0, ptr %295, align 1
  %296 = add i64 %265, 8
  br label %264

297:                                              ; preds = %264
  %298 = add i64 %261, 1
  br label %260

299:                                              ; preds = %260
  %300 = add i64 %257, 1
  br label %256

301:                                              ; preds = %256
  %302 = add i64 %253, 1
  br label %252

303:                                              ; preds = %252
  %304 = call ptr @llvm.stacksave.p0()
  %305 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } { ptr inttoptr (i64 3735928559 to ptr), ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 0, [4 x i64] [i64 1, i64 16, i64 16, i64 16], [4 x i64] [i64 4096, i64 256, i64 16, i64 1] }, ptr %305, align 8
  %306 = insertvalue { i64, ptr } { i64 4, ptr poison }, ptr %305, 1
  %307 = alloca { ptr, ptr, i64, [4 x i64], [4 x i64] }, i64 1, align 8
  store { ptr, ptr, i64, [4 x i64], [4 x i64] } { ptr inttoptr (i64 3735928559 to ptr), ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27200), i64 304, [4 x i64] [i64 1, i64 16, i64 16, i64 16], [4 x i64] [i64 5184, i64 288, i64 16, i64 1] }, ptr %307, align 8
  %308 = insertvalue { i64, ptr } { i64 4, ptr poison }, ptr %307, 1
  %309 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %306, ptr %309, align 8
  %310 = alloca { i64, ptr }, i64 1, align 8
  store { i64, ptr } %308, ptr %310, align 8
  call void @memrefCopy(i64 1, ptr %309, ptr %310)
  call void @llvm.stackrestore.p0(ptr %304)
  br label %311

311:                                              ; preds = %351, %303
  %312 = phi i64 [ %352, %351 ], [ 0, %303 ]
  %313 = icmp slt i64 %312, 1
  br i1 %313, label %314, label %353

314:                                              ; preds = %311
  br label %315

315:                                              ; preds = %349, %314
  %316 = phi i64 [ %350, %349 ], [ 0, %314 ]
  %317 = icmp slt i64 %316, 16
  br i1 %317, label %318, label %351

318:                                              ; preds = %315
  br label %319

319:                                              ; preds = %347, %318
  %320 = phi i64 [ %348, %347 ], [ 0, %318 ]
  %321 = icmp slt i64 %320, 16
  br i1 %321, label %322, label %349

322:                                              ; preds = %319
  br label %323

323:                                              ; preds = %326, %322
  %324 = phi i64 [ %346, %326 ], [ 0, %322 ]
  %325 = icmp slt i64 %324, 3
  br i1 %325, label %326, label %347

326:                                              ; preds = %323
  %327 = mul i64 %312, 5184
  %328 = mul i64 %316, 288
  %329 = add i64 %327, %328
  %330 = mul i64 %320, 16
  %331 = add i64 %329, %330
  %332 = mul i64 %324, 288
  %333 = add i64 %331, %332
  %334 = add i64 %333, 0
  %335 = getelementptr i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 27200), i64 %334
  %336 = load <48 x i8>, ptr %335, align 8
  %337 = mul i64 %312, 36864
  %338 = mul i64 %316, 2304
  %339 = add i64 %337, %338
  %340 = mul i64 %320, 144
  %341 = add i64 %339, %340
  %342 = mul i64 %324, 48
  %343 = add i64 %341, %342
  %344 = add i64 %343, 0
  %345 = getelementptr i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 48768), i64 %344
  store <48 x i8> %336, ptr %345, align 8
  %346 = add i64 %324, 1
  br label %323

347:                                              ; preds = %323
  %348 = add i64 %320, 1
  br label %319

349:                                              ; preds = %319
  %350 = add i64 %316, 1
  br label %315

351:                                              ; preds = %315
  %352 = add i64 %312, 1
  br label %311

353:                                              ; preds = %311
  call void asm sideeffect alignstack ".insn r 0x7B, 0x3, 7, x0, x0, x0", "~{memory}"()
  call void @gemmlir_flush()
  call void @tiled_matmul_auto(i64 256, i64 16, i64 144, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 48768), ptr @__constant_144x16xi8, ptr null, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 144, i64 16, i64 16, i64 16, float 1.000000e+00, float 1.000000e+00, i32 1, i32 0, float 1.000000e+00, float 1.000000e+00, i1 false, i1 false, i1 false, i1 true, i1 false, i8 0, i32 1)
  br label %354

354:                                              ; preds = %501, %353
  %355 = phi i64 [ %502, %501 ], [ 0, %353 ]
  %356 = icmp slt i64 %355, 1
  br i1 %356, label %357, label %503

357:                                              ; preds = %354
  br label %358

358:                                              ; preds = %499, %357
  %359 = phi i64 [ %500, %499 ], [ 0, %357 ]
  %360 = icmp slt i64 %359, 16
  br i1 %360, label %361, label %501

361:                                              ; preds = %358
  br label %362

362:                                              ; preds = %497, %361
  %363 = phi i64 [ %498, %497 ], [ 0, %361 ]
  %364 = icmp slt i64 %363, 16
  br i1 %364, label %365, label %499

365:                                              ; preds = %362
  br label %366

366:                                              ; preds = %369, %365
  %367 = phi i64 [ %496, %369 ], [ 0, %365 ]
  %368 = icmp slt i64 %367, 16
  br i1 %368, label %369, label %497

369:                                              ; preds = %366
  %370 = mul nuw nsw i64 %359, 256
  %371 = add nuw nsw i64 0, %370
  %372 = mul nuw nsw i64 %363, 16
  %373 = add nuw nsw i64 %371, %372
  %374 = add nuw nsw i64 %373, %367
  %375 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %374
  %376 = load i8, ptr %375, align 1
  %377 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %374
  %378 = load i32, ptr %377, align 4
  %379 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %367
  %380 = load float, ptr %379, align 4
  %381 = sitofp i8 %376 to float
  %382 = sitofp i32 %378 to float
  %383 = call float @llvm.fma.f32(float %382, float 0x3EF32289E0000000, float %380)
  %384 = call float @llvm.fma.f32(float %381, float 0x3F952CD580000000, float %383)
  %385 = call float @llvm.maxnum.f32(float %384, float 0.000000e+00)
  %386 = mul nuw nsw i64 %355, 4096
  %387 = add nuw nsw i64 %386, %370
  %388 = add nuw nsw i64 %387, %372
  %389 = add nuw nsw i64 %388, %367
  %390 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %389
  store float %385, ptr %390, align 4
  %391 = add i64 %367, 1
  %392 = add nuw nsw i64 %373, %391
  %393 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %392
  %394 = load i8, ptr %393, align 1
  %395 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %392
  %396 = load i32, ptr %395, align 4
  %397 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %391
  %398 = load float, ptr %397, align 4
  %399 = sitofp i8 %394 to float
  %400 = sitofp i32 %396 to float
  %401 = call float @llvm.fma.f32(float %400, float 0x3EF32289E0000000, float %398)
  %402 = call float @llvm.fma.f32(float %399, float 0x3F952CD580000000, float %401)
  %403 = call float @llvm.maxnum.f32(float %402, float 0.000000e+00)
  %404 = add nuw nsw i64 %388, %391
  %405 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %404
  store float %403, ptr %405, align 4
  %406 = add i64 %367, 2
  %407 = add nuw nsw i64 %373, %406
  %408 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %407
  %409 = load i8, ptr %408, align 1
  %410 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %407
  %411 = load i32, ptr %410, align 4
  %412 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %406
  %413 = load float, ptr %412, align 4
  %414 = sitofp i8 %409 to float
  %415 = sitofp i32 %411 to float
  %416 = call float @llvm.fma.f32(float %415, float 0x3EF32289E0000000, float %413)
  %417 = call float @llvm.fma.f32(float %414, float 0x3F952CD580000000, float %416)
  %418 = call float @llvm.maxnum.f32(float %417, float 0.000000e+00)
  %419 = add nuw nsw i64 %388, %406
  %420 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %419
  store float %418, ptr %420, align 4
  %421 = add i64 %367, 3
  %422 = add nuw nsw i64 %373, %421
  %423 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %422
  %424 = load i8, ptr %423, align 1
  %425 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %422
  %426 = load i32, ptr %425, align 4
  %427 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %421
  %428 = load float, ptr %427, align 4
  %429 = sitofp i8 %424 to float
  %430 = sitofp i32 %426 to float
  %431 = call float @llvm.fma.f32(float %430, float 0x3EF32289E0000000, float %428)
  %432 = call float @llvm.fma.f32(float %429, float 0x3F952CD580000000, float %431)
  %433 = call float @llvm.maxnum.f32(float %432, float 0.000000e+00)
  %434 = add nuw nsw i64 %388, %421
  %435 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %434
  store float %433, ptr %435, align 4
  %436 = add i64 %367, 4
  %437 = add nuw nsw i64 %373, %436
  %438 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %437
  %439 = load i8, ptr %438, align 1
  %440 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %437
  %441 = load i32, ptr %440, align 4
  %442 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %436
  %443 = load float, ptr %442, align 4
  %444 = sitofp i8 %439 to float
  %445 = sitofp i32 %441 to float
  %446 = call float @llvm.fma.f32(float %445, float 0x3EF32289E0000000, float %443)
  %447 = call float @llvm.fma.f32(float %444, float 0x3F952CD580000000, float %446)
  %448 = call float @llvm.maxnum.f32(float %447, float 0.000000e+00)
  %449 = add nuw nsw i64 %388, %436
  %450 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %449
  store float %448, ptr %450, align 4
  %451 = add i64 %367, 5
  %452 = add nuw nsw i64 %373, %451
  %453 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %452
  %454 = load i8, ptr %453, align 1
  %455 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %452
  %456 = load i32, ptr %455, align 4
  %457 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %451
  %458 = load float, ptr %457, align 4
  %459 = sitofp i8 %454 to float
  %460 = sitofp i32 %456 to float
  %461 = call float @llvm.fma.f32(float %460, float 0x3EF32289E0000000, float %458)
  %462 = call float @llvm.fma.f32(float %459, float 0x3F952CD580000000, float %461)
  %463 = call float @llvm.maxnum.f32(float %462, float 0.000000e+00)
  %464 = add nuw nsw i64 %388, %451
  %465 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %464
  store float %463, ptr %465, align 4
  %466 = add i64 %367, 6
  %467 = add nuw nsw i64 %373, %466
  %468 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %467
  %469 = load i8, ptr %468, align 1
  %470 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %467
  %471 = load i32, ptr %470, align 4
  %472 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %466
  %473 = load float, ptr %472, align 4
  %474 = sitofp i8 %469 to float
  %475 = sitofp i32 %471 to float
  %476 = call float @llvm.fma.f32(float %475, float 0x3EF32289E0000000, float %473)
  %477 = call float @llvm.fma.f32(float %474, float 0x3F952CD580000000, float %476)
  %478 = call float @llvm.maxnum.f32(float %477, float 0.000000e+00)
  %479 = add nuw nsw i64 %388, %466
  %480 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %479
  store float %478, ptr %480, align 4
  %481 = add i64 %367, 7
  %482 = add nuw nsw i64 %373, %481
  %483 = getelementptr inbounds nuw i8, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 23104), i64 %482
  %484 = load i8, ptr %483, align 1
  %485 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 32384), i64 %482
  %486 = load i32, ptr %485, align 4
  %487 = getelementptr inbounds nuw float, ptr @__constant_16xf32, i64 %481
  %488 = load float, ptr %487, align 4
  %489 = sitofp i8 %484 to float
  %490 = sitofp i32 %486 to float
  %491 = call float @llvm.fma.f32(float %490, float 0x3EF32289E0000000, float %488)
  %492 = call float @llvm.fma.f32(float %489, float 0x3F952CD580000000, float %491)
  %493 = call float @llvm.maxnum.f32(float %492, float 0.000000e+00)
  %494 = add nuw nsw i64 %388, %481
  %495 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %494
  store float %493, ptr %495, align 4
  %496 = add i64 %367, 8
  br label %366

497:                                              ; preds = %366
  %498 = add i64 %363, 1
  br label %362

499:                                              ; preds = %362
  %500 = add i64 %359, 1
  br label %358

501:                                              ; preds = %358
  %502 = add i64 %355, 1
  br label %354

503:                                              ; preds = %354
  br label %504

504:                                              ; preds = %553, %503
  %505 = phi i64 [ %554, %553 ], [ 0, %503 ]
  %506 = icmp slt i64 %505, 1
  br i1 %506, label %507, label %555

507:                                              ; preds = %504
  br label %508

508:                                              ; preds = %551, %507
  %509 = phi i64 [ %552, %551 ], [ 0, %507 ]
  %510 = icmp slt i64 %509, 8
  br i1 %510, label %511, label %553

511:                                              ; preds = %508
  br label %512

512:                                              ; preds = %549, %511
  %513 = phi i64 [ %550, %549 ], [ 0, %511 ]
  %514 = icmp slt i64 %513, 8
  br i1 %514, label %515, label %551

515:                                              ; preds = %512
  br label %516

516:                                              ; preds = %519, %515
  %517 = phi i64 [ %548, %519 ], [ 0, %515 ]
  %518 = icmp slt i64 %517, 16
  br i1 %518, label %519, label %549

519:                                              ; preds = %516
  %520 = mul nuw nsw i64 %505, 1024
  %521 = mul nuw nsw i64 %509, 128
  %522 = add nuw nsw i64 %520, %521
  %523 = mul nuw nsw i64 %513, 16
  %524 = add nuw nsw i64 %522, %523
  %525 = add nuw nsw i64 %524, %517
  %526 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %525
  store float 0xFFF0000000000000, ptr %526, align 4
  %527 = add i64 %517, 1
  %528 = add nuw nsw i64 %524, %527
  %529 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %528
  store float 0xFFF0000000000000, ptr %529, align 4
  %530 = add i64 %517, 2
  %531 = add nuw nsw i64 %524, %530
  %532 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %531
  store float 0xFFF0000000000000, ptr %532, align 4
  %533 = add i64 %517, 3
  %534 = add nuw nsw i64 %524, %533
  %535 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %534
  store float 0xFFF0000000000000, ptr %535, align 4
  %536 = add i64 %517, 4
  %537 = add nuw nsw i64 %524, %536
  %538 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %537
  store float 0xFFF0000000000000, ptr %538, align 4
  %539 = add i64 %517, 5
  %540 = add nuw nsw i64 %524, %539
  %541 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %540
  store float 0xFFF0000000000000, ptr %541, align 4
  %542 = add i64 %517, 6
  %543 = add nuw nsw i64 %524, %542
  %544 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %543
  store float 0xFFF0000000000000, ptr %544, align 4
  %545 = add i64 %517, 7
  %546 = add nuw nsw i64 %524, %545
  %547 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %546
  store float 0xFFF0000000000000, ptr %547, align 4
  %548 = add i64 %517, 8
  br label %516

549:                                              ; preds = %516
  %550 = add i64 %513, 1
  br label %512

551:                                              ; preds = %512
  %552 = add i64 %509, 1
  br label %508

553:                                              ; preds = %508
  %554 = add i64 %505, 1
  br label %504

555:                                              ; preds = %504
  br label %556

556:                                              ; preds = %756, %555
  %557 = phi i64 [ %757, %756 ], [ 0, %555 ]
  %558 = icmp slt i64 %557, 1
  br i1 %558, label %559, label %758

559:                                              ; preds = %556
  br label %560

560:                                              ; preds = %754, %559
  %561 = phi i64 [ %755, %754 ], [ 0, %559 ]
  %562 = icmp slt i64 %561, 8
  br i1 %562, label %563, label %756

563:                                              ; preds = %560
  br label %564

564:                                              ; preds = %752, %563
  %565 = phi i64 [ %753, %752 ], [ 0, %563 ]
  %566 = icmp slt i64 %565, 8
  br i1 %566, label %567, label %754

567:                                              ; preds = %564
  br label %568

568:                                              ; preds = %571, %567
  %569 = phi i64 [ %751, %571 ], [ 0, %567 ]
  %570 = icmp slt i64 %569, 16
  br i1 %570, label %571, label %752

571:                                              ; preds = %568
  %572 = mul nuw nsw i64 %557, 1024
  %573 = mul nuw nsw i64 %561, 128
  %574 = add nuw nsw i64 %572, %573
  %575 = mul nuw nsw i64 %565, 16
  %576 = add nuw nsw i64 %574, %575
  %577 = add nuw nsw i64 %576, %569
  %578 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %577
  %579 = load float, ptr %578, align 4
  %580 = mul nsw i64 %561, 2
  %581 = mul nsw i64 %565, 2
  %582 = mul nuw nsw i64 %557, 4096
  %583 = mul nuw nsw i64 %580, 256
  %584 = add nuw nsw i64 %582, %583
  %585 = mul nuw nsw i64 %581, 16
  %586 = add nuw nsw i64 %584, %585
  %587 = add nuw nsw i64 %586, %569
  %588 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %587
  %589 = load float, ptr %588, align 4
  %590 = call float @llvm.maximum.f32(float %579, float %589)
  %591 = add i64 %581, 1
  %592 = mul nuw nsw i64 %591, 16
  %593 = add nuw nsw i64 %584, %592
  %594 = add nuw nsw i64 %593, %569
  %595 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %594
  %596 = load float, ptr %595, align 4
  %597 = call float @llvm.maximum.f32(float %590, float %596)
  %598 = add i64 %580, 1
  %599 = mul nuw nsw i64 %598, 256
  %600 = add nuw nsw i64 %582, %599
  %601 = add nuw nsw i64 %600, %585
  %602 = add nuw nsw i64 %601, %569
  %603 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %602
  %604 = load float, ptr %603, align 4
  %605 = call float @llvm.maximum.f32(float %597, float %604)
  %606 = add nuw nsw i64 %600, %592
  %607 = add nuw nsw i64 %606, %569
  %608 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %607
  %609 = load float, ptr %608, align 4
  %610 = call float @llvm.maximum.f32(float %605, float %609)
  store float %610, ptr %578, align 4
  %611 = add i64 %569, 1
  %612 = add nuw nsw i64 %576, %611
  %613 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %612
  %614 = load float, ptr %613, align 4
  %615 = add nuw nsw i64 %586, %611
  %616 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %615
  %617 = load float, ptr %616, align 4
  %618 = call float @llvm.maximum.f32(float %614, float %617)
  %619 = add nuw nsw i64 %593, %611
  %620 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %619
  %621 = load float, ptr %620, align 4
  %622 = call float @llvm.maximum.f32(float %618, float %621)
  %623 = add nuw nsw i64 %601, %611
  %624 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %623
  %625 = load float, ptr %624, align 4
  %626 = call float @llvm.maximum.f32(float %622, float %625)
  %627 = add nuw nsw i64 %606, %611
  %628 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %627
  %629 = load float, ptr %628, align 4
  %630 = call float @llvm.maximum.f32(float %626, float %629)
  store float %630, ptr %613, align 4
  %631 = add i64 %569, 2
  %632 = add nuw nsw i64 %576, %631
  %633 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %632
  %634 = load float, ptr %633, align 4
  %635 = add nuw nsw i64 %586, %631
  %636 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %635
  %637 = load float, ptr %636, align 4
  %638 = call float @llvm.maximum.f32(float %634, float %637)
  %639 = add nuw nsw i64 %593, %631
  %640 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %639
  %641 = load float, ptr %640, align 4
  %642 = call float @llvm.maximum.f32(float %638, float %641)
  %643 = add nuw nsw i64 %601, %631
  %644 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %643
  %645 = load float, ptr %644, align 4
  %646 = call float @llvm.maximum.f32(float %642, float %645)
  %647 = add nuw nsw i64 %606, %631
  %648 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %647
  %649 = load float, ptr %648, align 4
  %650 = call float @llvm.maximum.f32(float %646, float %649)
  store float %650, ptr %633, align 4
  %651 = add i64 %569, 3
  %652 = add nuw nsw i64 %576, %651
  %653 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %652
  %654 = load float, ptr %653, align 4
  %655 = add nuw nsw i64 %586, %651
  %656 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %655
  %657 = load float, ptr %656, align 4
  %658 = call float @llvm.maximum.f32(float %654, float %657)
  %659 = add nuw nsw i64 %593, %651
  %660 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %659
  %661 = load float, ptr %660, align 4
  %662 = call float @llvm.maximum.f32(float %658, float %661)
  %663 = add nuw nsw i64 %601, %651
  %664 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %663
  %665 = load float, ptr %664, align 4
  %666 = call float @llvm.maximum.f32(float %662, float %665)
  %667 = add nuw nsw i64 %606, %651
  %668 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %667
  %669 = load float, ptr %668, align 4
  %670 = call float @llvm.maximum.f32(float %666, float %669)
  store float %670, ptr %653, align 4
  %671 = add i64 %569, 4
  %672 = add nuw nsw i64 %576, %671
  %673 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %672
  %674 = load float, ptr %673, align 4
  %675 = add nuw nsw i64 %586, %671
  %676 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %675
  %677 = load float, ptr %676, align 4
  %678 = call float @llvm.maximum.f32(float %674, float %677)
  %679 = add nuw nsw i64 %593, %671
  %680 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %679
  %681 = load float, ptr %680, align 4
  %682 = call float @llvm.maximum.f32(float %678, float %681)
  %683 = add nuw nsw i64 %601, %671
  %684 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %683
  %685 = load float, ptr %684, align 4
  %686 = call float @llvm.maximum.f32(float %682, float %685)
  %687 = add nuw nsw i64 %606, %671
  %688 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %687
  %689 = load float, ptr %688, align 4
  %690 = call float @llvm.maximum.f32(float %686, float %689)
  store float %690, ptr %673, align 4
  %691 = add i64 %569, 5
  %692 = add nuw nsw i64 %576, %691
  %693 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %692
  %694 = load float, ptr %693, align 4
  %695 = add nuw nsw i64 %586, %691
  %696 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %695
  %697 = load float, ptr %696, align 4
  %698 = call float @llvm.maximum.f32(float %694, float %697)
  %699 = add nuw nsw i64 %593, %691
  %700 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %699
  %701 = load float, ptr %700, align 4
  %702 = call float @llvm.maximum.f32(float %698, float %701)
  %703 = add nuw nsw i64 %601, %691
  %704 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %703
  %705 = load float, ptr %704, align 4
  %706 = call float @llvm.maximum.f32(float %702, float %705)
  %707 = add nuw nsw i64 %606, %691
  %708 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %707
  %709 = load float, ptr %708, align 4
  %710 = call float @llvm.maximum.f32(float %706, float %709)
  store float %710, ptr %693, align 4
  %711 = add i64 %569, 6
  %712 = add nuw nsw i64 %576, %711
  %713 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %712
  %714 = load float, ptr %713, align 4
  %715 = add nuw nsw i64 %586, %711
  %716 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %715
  %717 = load float, ptr %716, align 4
  %718 = call float @llvm.maximum.f32(float %714, float %717)
  %719 = add nuw nsw i64 %593, %711
  %720 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %719
  %721 = load float, ptr %720, align 4
  %722 = call float @llvm.maximum.f32(float %718, float %721)
  %723 = add nuw nsw i64 %601, %711
  %724 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %723
  %725 = load float, ptr %724, align 4
  %726 = call float @llvm.maximum.f32(float %722, float %725)
  %727 = add nuw nsw i64 %606, %711
  %728 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %727
  %729 = load float, ptr %728, align 4
  %730 = call float @llvm.maximum.f32(float %726, float %729)
  store float %730, ptr %713, align 4
  %731 = add i64 %569, 7
  %732 = add nuw nsw i64 %576, %731
  %733 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %732
  %734 = load float, ptr %733, align 4
  %735 = add nuw nsw i64 %586, %731
  %736 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %735
  %737 = load float, ptr %736, align 4
  %738 = call float @llvm.maximum.f32(float %734, float %737)
  %739 = add nuw nsw i64 %593, %731
  %740 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %739
  %741 = load float, ptr %740, align 4
  %742 = call float @llvm.maximum.f32(float %738, float %741)
  %743 = add nuw nsw i64 %601, %731
  %744 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %743
  %745 = load float, ptr %744, align 4
  %746 = call float @llvm.maximum.f32(float %742, float %745)
  %747 = add nuw nsw i64 %606, %731
  %748 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 85696), i64 %747
  %749 = load float, ptr %748, align 4
  %750 = call float @llvm.maximum.f32(float %746, float %749)
  store float %750, ptr %733, align 4
  %751 = add i64 %569, 8
  br label %568

752:                                              ; preds = %568
  %753 = add i64 %565, 1
  br label %564

754:                                              ; preds = %564
  %755 = add i64 %561, 1
  br label %560

756:                                              ; preds = %560
  %757 = add i64 %557, 1
  br label %556

758:                                              ; preds = %556
  %759 = call ptr @malloc(i64 4096)
  br label %760

760:                                              ; preds = %819, %758
  %761 = phi i64 [ %820, %819 ], [ 0, %758 ]
  %762 = icmp slt i64 %761, 1
  br i1 %762, label %763, label %821

763:                                              ; preds = %760
  br label %764

764:                                              ; preds = %817, %763
  %765 = phi i64 [ %818, %817 ], [ 0, %763 ]
  %766 = icmp slt i64 %765, 8
  br i1 %766, label %767, label %819

767:                                              ; preds = %764
  br label %768

768:                                              ; preds = %815, %767
  %769 = phi i64 [ %816, %815 ], [ 0, %767 ]
  %770 = icmp slt i64 %769, 16
  br i1 %770, label %771, label %817

771:                                              ; preds = %768
  br label %772

772:                                              ; preds = %775, %771
  %773 = phi i64 [ %814, %775 ], [ 0, %771 ]
  %774 = icmp slt i64 %773, 8
  br i1 %774, label %775, label %815

775:                                              ; preds = %772
  %776 = mul nuw nsw i64 %761, 1024
  %777 = mul nuw nsw i64 %765, 128
  %778 = add nuw nsw i64 %776, %777
  %779 = mul nuw nsw i64 %773, 16
  %780 = add nuw nsw i64 %778, %779
  %781 = add nuw nsw i64 %780, %769
  %782 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %781
  %783 = load float, ptr %782, align 4
  %784 = mul nuw nsw i64 %769, 64
  %785 = add nuw nsw i64 %776, %784
  %786 = mul nuw nsw i64 %765, 8
  %787 = add nuw nsw i64 %785, %786
  %788 = add nuw nsw i64 %787, %773
  %789 = getelementptr inbounds nuw float, ptr %759, i64 %788
  store float %783, ptr %789, align 4
  %790 = add i64 %773, 1
  %791 = mul nuw nsw i64 %790, 16
  %792 = add nuw nsw i64 %778, %791
  %793 = add nuw nsw i64 %792, %769
  %794 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %793
  %795 = load float, ptr %794, align 4
  %796 = add nuw nsw i64 %787, %790
  %797 = getelementptr inbounds nuw float, ptr %759, i64 %796
  store float %795, ptr %797, align 4
  %798 = add i64 %773, 2
  %799 = mul nuw nsw i64 %798, 16
  %800 = add nuw nsw i64 %778, %799
  %801 = add nuw nsw i64 %800, %769
  %802 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %801
  %803 = load float, ptr %802, align 4
  %804 = add nuw nsw i64 %787, %798
  %805 = getelementptr inbounds nuw float, ptr %759, i64 %804
  store float %803, ptr %805, align 4
  %806 = add i64 %773, 3
  %807 = mul nuw nsw i64 %806, 16
  %808 = add nuw nsw i64 %778, %807
  %809 = add nuw nsw i64 %808, %769
  %810 = getelementptr inbounds nuw float, ptr getelementptr inbounds nuw (i8, ptr @__gemmlir_arena_forward_0, i64 102080), i64 %809
  %811 = load float, ptr %810, align 4
  %812 = add nuw nsw i64 %787, %806
  %813 = getelementptr inbounds nuw float, ptr %759, i64 %812
  store float %811, ptr %813, align 4
  %814 = add i64 %773, 4
  br label %772

815:                                              ; preds = %772
  %816 = add i64 %769, 1
  br label %768

817:                                              ; preds = %768
  %818 = add i64 %765, 1
  br label %764

819:                                              ; preds = %764
  %820 = add i64 %761, 1
  br label %760

821:                                              ; preds = %760
  ret ptr %759
}

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare ptr @llvm.stacksave.p0() #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.stackrestore.p0(ptr) #0

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.roundeven.f32(float) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn }
attributes #1 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
