"�K
BHostIDLE"IDLE1    @��@A    @��@ahB�
'��?ihB�
'��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     ^�@9     ^�@A     ^�@I     ^�@a�ϦB���?if������?�Unknown�
sHost_FusedMatMul"sequential_3/dense_9/Relu(1     |�@9     |�@A     |�@I     |�@a��o;�?i�'"�3�?�Unknown
}HostMatMul")gradient_tape/sequential_3/dense_9/MatMul(1     `�@9     `�@A     `�@I     `�@a��JҊ��?i�9pMx~�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     p{@9     p{@A     p{@I     p{@a�
�E�|s?iJ��q��?�Unknown
tHost_FusedMatMul"sequential_3/dense_10/Relu(1     @]@9     @]@A     @]@I     @]@ai�sG�T?i���ԯ�?�Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_10/MatMul(1     �W@9     �W@A     �W@I     �W@a�w1ΰP?i͔�L-��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     �T@9     �T@A     �T@I     �T@a<;:��yM?i\c�����?�Unknown
�	HostMatMul",gradient_tape/sequential_3/dense_10/MatMul_1(1     @T@9     @T@A     @T@I     @T@a~w�d��L?i������?�Unknown
i
HostWriteSummary"WriteSummary(1      P@9      P@A      P@I      P@a�w��I�F?iZ�6k��?�Unknown�
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     �Q@9     �Q@A     �O@I     �O@aԕ̣`_F?i=M���?�Unknown
^HostGatherV2"GatherV2(1     �G@9     �G@A     �G@I     �G@a�w1ΰ@?iې;B/��?�Unknown
�HostMatMul",gradient_tape/sequential_3/dense_11/MatMul_1(1     �D@9     �D@A     �D@I     �D@a]Y��=?i����?�Unknown
dHostDataset"Iterator::Model(1     �]@9     �]@A     �A@I     �A@a������8?i��#����?�Unknown
ZHostArgMax"ArgMax(1      @@9      @@A      @@I      @@a�w��I�6?i�1]����?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      >@9      >@A      >@I      >@a7�H.�N5?i���o��?�Unknown
wHost_FusedMatMul"sequential_3/dense_11/BiasAdd(1      =@9      =@A      =@I      =@az,��Ҙ4?i��^���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      ;@9      ;@A      ;@I      ;@a��AC.-3?i&a'nh��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      2@I      2@a��W��)?i��W���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@a7�H.�N%?i-��qV��?�Unknown
rHostSoftmax"sequential_3/dense_11/Softmax(1      .@9      .@A      .@I      .@a7�H.�N%?i�o�[���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      (@9      (@A      (@I      (@a�Y:X�!?ib�r���?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@a���=�h?i�T^���?�Unknown�
~HostMatMul"*gradient_tape/sequential_3/dense_11/MatMul(1      $@9      $@A      $@I      $@a���=�h?i��6����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a��W��?i���1O��?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     @Z@9     @Z@A       @I       @a�w��I�?iLI��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a�w��I�?i�kֺ��?�Unknown
�HostReadVariableOp"+sequential_3/dense_10/MatMul/ReadVariableOp(1       @9       @A       @I       @a�w��I�?i����p��?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1       @9       @A       @I       @a�w��I�?i�P{&��?�Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a�w��I�?i\�VM���?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a�h�� �?i'5[e{��?�Unknown
� HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a�h�� �?i��_}��?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�h�� �?i�Nd����?�Unknown
�"HostBiasAddGrad"7gradient_tape/sequential_3/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a�h�� �?i��h�X��?�Unknown
�#HostBiasAddGrad"6gradient_tape/sequential_3/dense_9/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a�h�� �?iShm����?�Unknown
�$HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a�h�� �?i�qݖ��?�Unknown
�%HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @a�Y:X�?i�,;��?�Unknown
|&HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a�Y:X�?i�x瘧��?�Unknown
V'HostSum"Sum_2(1      @9      @A      @I      @a�Y:X�?i�:��/��?�Unknown
�(HostBiasAddGrad"7gradient_tape/sequential_3/dense_10/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a�Y:X�?ij�\T���?�Unknown
�)HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a���=�h?iD���)��?�Unknown
�*HostReluGrad"+gradient_tape/sequential_3/dense_9/ReluGrad(1      @9      @A      @I      @a���=�h?i�>����?�Unknown
�+HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a���=�h?i��>��?�Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a�w��I�?i��'h��?�Unknown
X-HostCast"Cast_2(1      @9      @A      @I      @a�w��I�?i�8����?�Unknown
�.HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a�w��I�?i�d%���?�Unknown
�/HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      @9      @A      @I      @a�Y:X�?i��)b��?�Unknown
v0HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @a�Y:X�?ip&�W���?�Unknown
V1HostCast"Cast(1      @9      @A      @I      @a�Y:X�?iY������?�Unknown
�2HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @R@9     @R@A      @I      @a�Y:X�?iB蚵.��?�Unknown
�3HostReluGrad",gradient_tape/sequential_3/dense_10/ReluGrad(1      @9      @A      @I      @a�Y:X�?i+Ix�r��?�Unknown
�4HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @a�Y:X�?i�U���?�Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a�w��I��>i@����?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a�w��I��>i��|���?�Unknown
X7HostCast"Cast_3(1       @9       @A       @I       @a�w��I��>i�kq?��?�Unknown
X8HostEqual"Equal(1       @9       @A       @I       @a�w��I��>i���l��?�Unknown
T9HostMul"Mul(1       @9       @A       @I       @a�w��I��>iɗ7Z���?�Unknown
s:HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a�w��I��>i�-�����?�Unknown
b;HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a�w��I��>i��^C���?�Unknown
�<HostReadVariableOp",sequential_3/dense_10/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a�w��I��>i�Y�"��?�Unknown
�=HostReadVariableOp",sequential_3/dense_11/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a�w��I��>i��,P��?�Unknown
�>HostReadVariableOp"+sequential_3/dense_11/MatMul/ReadVariableOp(1       @9       @A       @I       @a�w��I��>i~��}��?�Unknown
�?HostReadVariableOp"+sequential_3/dense_9/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a�w��I��>io����?�Unknown
�@HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a�w��I��>i`�@����?�Unknown
�AHostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a�w��I��>iQG����?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a�w��I��>iI���?�Unknown
XCHostCast"Cast_1(1      �?9      �?A      �?I      �?a�w��I��>iA�gs3��?�Unknown
XDHostCast"Cast_4(1      �?9      �?A      �?I      �?a�w��I��>i9��-J��?�Unknown
aEHostIdentity"Identity(1      �?9      �?A      �?I      �?a�w��I��>i1s��`��?�Unknown�
`FHostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a�w��I��>i)>E�w��?�Unknown
uGHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a�w��I��>i!	�\���?�Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�w��I��>i�����?�Unknown
wIHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a�w��I��>i�"ѻ��?�Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a�w��I��>i	jl����?�Unknown
�KHostReadVariableOp"*sequential_3/dense_9/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a�w��I��>i5�E���?�Unknown
�LHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a�w��I��>i�������?�Unknown*�J
uHostFlushSummaryWriter"FlushSummaryWriter(1     ^�@9     ^�@A     ^�@I     ^�@ak�I؏��?ik�I؏��?�Unknown�
sHost_FusedMatMul"sequential_3/dense_9/Relu(1     |�@9     |�@A     |�@I     |�@au��r�d�?i��#��?�Unknown
}HostMatMul")gradient_tape/sequential_3/dense_9/MatMul(1     `�@9     `�@A     `�@I     `�@a
� 0�[�?i�ԃk�p�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1     p{@9     p{@A     p{@I     p{@a�!���?i�F���i�?�Unknown
tHost_FusedMatMul"sequential_3/dense_10/Relu(1     @]@9     @]@A     @]@I     @]@a�n��Ր?i?��К��?�Unknown
~HostMatMul"*gradient_tape/sequential_3/dense_10/MatMul(1     �W@9     �W@A     �W@I     �W@aܾ^���?i:M��\�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1     �T@9     �T@A     �T@I     �T@a	A�L\�?i>DW��?�Unknown
�HostMatMul",gradient_tape/sequential_3/dense_10/MatMul_1(1     @T@9     @T@A     @T@I     @T@a��a�O�?i,�u'��?�Unknown
i	HostWriteSummary"WriteSummary(1      P@9      P@A      P@I      P@a���	�j�?i⸜�=c�?�Unknown�
�
HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1     �Q@9     �Q@A     �O@I     �O@aϊ�I!�?i?�����?�Unknown
^HostGatherV2"GatherV2(1     �G@9     �G@A     �G@I     �G@aܾ^��{?i���a���?�Unknown
�HostMatMul",gradient_tape/sequential_3/dense_11/MatMul_1(1     �D@9     �D@A     �D@I     �D@aS����w?i����?�Unknown
dHostDataset"Iterator::Model(1     �]@9     �]@A     �A@I     �A@a�}���$t?i�u&V9�?�Unknown
ZHostArgMax"ArgMax(1      @@9      @@A      @@I      @@a���	�jr?i"�+^�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1      >@9      >@A      >@I      >@a�",	Dq?id_4����?�Unknown
wHost_FusedMatMul"sequential_3/dense_11/BiasAdd(1      =@9      =@A      =@I      =@aA�݈��p?iF���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1      ;@9      ;@A      ;@I      ;@aѤ�	o?iĝV�(��?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      8@9      8@A      2@I      2@a6��d?i��a����?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      .@9      .@A      .@I      .@a�",	Da?i��j%��?�Unknown
rHostSoftmax"sequential_3/dense_11/Softmax(1      .@9      .@A      .@I      .@a�",	Da?i��si��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      (@9      (@A      (@I      (@aH��[?iON{9�?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      $@9      $@A      $@I      $@a��:\W?i�k����?�Unknown�
~HostMatMul"*gradient_tape/sequential_3/dense_11/MatMul(1      $@9      $@A      $@I      $@a��:\W?i'��h>�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      "@9      "@A      "@I      "@a6��T?i	
�k�'�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1     @Z@9     @Z@A       @I       @a���	�jR?i`���0�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1       @9       @A       @I       @a���	�jR?i�Җ:�?�Unknown
�HostReadVariableOp"+sequential_3/dense_10/MatMul/ReadVariableOp(1       @9       @A       @I       @a���	�jR?i��s:C�?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1       @9       @A       @I       @a���	�jR?ie���oL�?�Unknown
gHostStridedSlice"strided_slice(1       @9       @A       @I       @a���	�jR?i��#�U�?�Unknown
`HostGatherV2"
GatherV2_1(1      @9      @A      @I      @a՗�ZP?i�ǩг]�?�Unknown
�HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a՗�ZP?iT�}�e�?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a՗�ZP?i W�*�m�?�Unknown
�!HostBiasAddGrad"7gradient_tape/sequential_3/dense_11/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a՗�ZP?i잶��u�?�Unknown
�"HostBiasAddGrad"6gradient_tape/sequential_3/dense_9/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a՗�ZP?i�溄�}�?�Unknown
�#HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1      @9      @A      @I      @a՗�ZP?i�.�1���?�Unknown
�$HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      @A      @I      @aH��K?i���3��?�Unknown
|%HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @aH��K?i��5͓�?�Unknown
V&HostSum"Sum_2(1      @9      @A      @I      @aH��K?iG0�7���?�Unknown
�'HostBiasAddGrad"7gradient_tape/sequential_3/dense_10/BiasAdd/BiasAddGrad(1      @9      @A      @I      @aH��K?i���9���?�Unknown
�(HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1      @9      @A      @I      @a��:\G?i>�А^��?�Unknown
�)HostReluGrad"+gradient_tape/sequential_3/dense_9/ReluGrad(1      @9      @A      @I      @a��:\G?i������?�Unknown
�*HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a��:\G?i��>��?�Unknown
v+HostAssignAddVariableOp"AssignAddVariableOp_4(1      @9      @A      @I      @a���	�jB?i�y��{��?�Unknown
X,HostCast"Cast_2(1      @9      @A      @I      @a���	�jB?i �ۖ��?�Unknown
�-HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      @9      @A      @I      @a���	�jB?i+^�B���?�Unknown
�.HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      @9      @A      @I      @aH��;?i�3�C%��?�Unknown
v/HostAssignAddVariableOp"AssignAddVariableOp_2(1      @9      @A      @I      @aH��;?im	�D���?�Unknown
V0HostCast"Cast(1      @9      @A      @I      @aH��;?i��E��?�Unknown
�1HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     @R@9     @R@A      @I      @aH��;?i���F���?�Unknown
�2HostReluGrad",gradient_tape/sequential_3/dense_10/ReluGrad(1      @9      @A      @I      @aH��;?iP��G���?�Unknown
�3HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      @9      @A      @I      @aH��;?i�_�Hi��?�Unknown
t4HostAssignAddVariableOp"AssignAddVariableOp(1       @9       @A       @I       @a���	�j2?i�Ꞷ��?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_1(1       @9       @A       @I       @a���	�j2?i�����?�Unknown
X6HostCast"Cast_3(1       @9       @A       @I       @a���	�j2?i3�JQ��?�Unknown
X7HostEqual"Equal(1       @9       @A       @I       @a���	�j2?iID��?�Unknown
T8HostMul"Mul(1       @9       @A       @I       @a���	�j2?i_}�����?�Unknown
s9HostReadVariableOp"SGD/Cast/ReadVariableOp(1       @9       @A       @I       @a���	�j2?iu��L9��?�Unknown
b:HostDivNoNan"div_no_nan_1(1       @9       @A       @I       @a���	�j2?i�����?�Unknown
�;HostReadVariableOp",sequential_3/dense_10/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a���	�j2?i�(�����?�Unknown
�<HostReadVariableOp",sequential_3/dense_11/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a���	�j2?i�a�N!��?�Unknown
�=HostReadVariableOp"+sequential_3/dense_11/MatMul/ReadVariableOp(1       @9       @A       @I       @a���	�j2?i͚��n��?�Unknown
�>HostReadVariableOp"+sequential_3/dense_9/BiasAdd/ReadVariableOp(1       @9       @A       @I       @a���	�j2?i�������?�Unknown
�?HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1       @9       @A       @I       @a���	�j2?i��P	��?�Unknown
�@HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1       @9       @A       @I       @a���	�j2?iF��V��?�Unknown
vAHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a���	�j"?i���Q}��?�Unknown
XBHostCast"Cast_1(1      �?9      �?A      �?I      �?a���	�j"?i%�����?�Unknown
XCHostCast"Cast_4(1      �?9      �?A      �?I      �?a���	�j"?i������?�Unknown
aDHostIdentity"Identity(1      �?9      �?A      �?I      �?a���	�j"?i;��R���?�Unknown�
`EHostDivNoNan"
div_no_nan(1      �?9      �?A      �?I      �?a���	�j"?i�T����?�Unknown
uFHostReadVariableOp"div_no_nan/ReadVariableOp(1      �?9      �?A      �?I      �?a���	�j"?iQ���>��?�Unknown
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1      �?9      �?A      �?I      �?a���	�j"?i܍�Se��?�Unknown
wHHostReadVariableOp"div_no_nan_1/ReadVariableOp(1      �?9      �?A      �?I      �?a���	�j"?ig*�����?�Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a���	�j"?i�������?�Unknown
�JHostReadVariableOp"*sequential_3/dense_9/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a���	�j"?i}c�T���?�Unknown
�KHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1      �?9      �?A      �?I      �?a���	�j"?i     �?�Unknown