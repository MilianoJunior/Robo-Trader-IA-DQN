��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.12v2.3.0-54-gfcc4b966f18��
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*U
shared_nameFDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
�
XCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	�*
dtype0
�
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*S
shared_nameDBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
�
VCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:�*
dtype0
�
6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*G
shared_name86CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
�
JCategoricalQNetwork/CategoricalQNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*E
shared_name64CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
�
HCategoricalQNetwork/CategoricalQNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias*
_output_shapes	
:�*
dtype0
�
ConstConst*
_output_shapes
:3*
dtype0*�
value�B�3"�  z�  p�  f�  \�  R�  H�  >�  4�  *�   �  �  �  �  ��  ��  ��  ��  ��  ��  p�  H�   �  ��  ��   �       B  �B  �B   C  HC  pC  �C  �C  �C  �C  �C  �C  D  D  D   D  *D  4D  >D  HD  RD  \D  fD  pD  zD

NoOpNoOp
�
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
	3


0
1
 
��
VARIABLE_VALUEDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUE4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE

ref
1

ref
1

_wrapped_policy
 


_q_network
b

_q_network
	variables
trainable_variables
regularization_losses
	keras_api
t
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
	3

0
1
2
	3
 
�

layers
metrics
layer_metrics
	variables
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
n
 _postprocessing_layers
!	variables
"trainable_variables
#regularization_losses
$	keras_api
h

kernel
	bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api

0
1
2
	3

0
1
2
	3
 
�

)layers
*metrics
+layer_metrics
	variables
trainable_variables
,non_trainable_variables
-layer_regularization_losses
regularization_losses

0
 
 
 
 

.0
/1

0
1

0
1
 
�

0layers
1metrics
2layer_metrics
!	variables
"trainable_variables
3non_trainable_variables
4layer_regularization_losses
#regularization_losses

0
	1

0
	1
 
�

5layers
6metrics
7layer_metrics
%	variables
&trainable_variables
8non_trainable_variables
9layer_regularization_losses
'regularization_losses

0
1
 
 
 
 
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

kernel
bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api

.0
/1
 
 
 
 
 
 
 
 
 
 
 
 
�

Blayers
Cmetrics
Dlayer_metrics
:	variables
;trainable_variables
Enon_trainable_variables
Flayer_regularization_losses
<regularization_losses

0
1

0
1
 
�

Glayers
Hmetrics
Ilayer_metrics
>	variables
?trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
@regularization_losses
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������

action_0/observationPlaceholder*+
_output_shapes
:���������*
dtype0* 
shape:���������
j
action_0/rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0/step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_typeDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_1/biasConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_169315748
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_169315760
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_169315782
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_169315775
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpXCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpVCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpJCategoricalQNetwork/CategoricalQNetwork/dense_1/kernel/Read/ReadVariableOpHCategoricalQNetwork/CategoricalQNetwork/dense_1/bias/Read/ReadVariableOpConst_1*
Tin
	2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_save_169315917
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableDCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernelBCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias6CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel4CategoricalQNetwork/CategoricalQNetwork/dense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *.
f)R'
%__inference__traced_restore_169315942��
�
g
-__inference_function_with_signature_169315767
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference_<lambda>_1693155122
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�
�
%__inference__traced_restore_169315942
file_prefix
assignvariableop_variable[
Wassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelY
Uassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasM
Iassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernelK
Gassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_bias

identity_6��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*,
_output_shapes
::::::*
dtypes

2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpWassignvariableop_1_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpUassignvariableop_2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpIassignvariableop_3_categoricalqnetwork_categoricalqnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpGassignvariableop_4_categoricalqnetwork_categoricalqnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_5Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_5�

Identity_6IdentityIdentity_5:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4*
T0*
_output_shapes
: 2

Identity_6"!

identity_6Identity_6:output:0*)
_input_shapes
: :::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_4:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�s
�
+__inference_polymorphic_action_fn_169315873
time_step_step_type
time_step_reward
time_step_discount
time_step_observation`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x

identity_1��
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_observationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd�
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu�
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp�
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������28
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul�
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp�
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������29
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd�
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   3   2#
!CategoricalQNetwork/Reshape/shape�
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
Sum�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueQ
ShapeShapetime_step_step_type*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0^time_step_discount^time_step_observation^time_step_reward^time_step_step_type*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1U
Shape_1Shapetime_step_step_type*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mul[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
	Greater/yt
GreaterGreaterepsilon_rng/Mul:z:0Greater/y:output:0*
T0*#
_output_shapes
:���������2	
Greater}
SelectSelectGreater:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2g

Identity_1Identityclip_by_value_2:z:0*
T0*#
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*m
_input_shapes\
Z:���������:���������:���������:���������:::::3:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:b^
+
_output_shapes
:���������
/
_user_specified_nametime_step/observation
�
a
'__inference_signature_wrapper_169315775
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1693157672
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�
/
-__inference_function_with_signature_169315778�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *'
f"R 
__inference_<lambda>_1693155152
PartitionedCall*
_input_shapes 
�
9
'__inference_signature_wrapper_169315760

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1693157552
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�q
�
+__inference_polymorphic_action_fn_169315606
	step_type

reward
discount
observation`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x

identity_1��
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapeobservationNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd�
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu�
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp�
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������28
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul�
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp�
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������29
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd�
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   3   2#
!CategoricalQNetwork/Reshape/shape�
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
Sum�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueG
ShapeShape	step_type*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0	^discount^observation^reward
^step_type*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1K
Shape_1Shape	step_type*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mul[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
	Greater/yt
GreaterGreaterepsilon_rng/Mul:z:0Greater/y:output:0*
T0*#
_output_shapes
:���������2	
Greater}
SelectSelectGreater:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2g

Identity_1Identityclip_by_value_2:z:0*
T0*#
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*m
_input_shapes\
Z:���������:���������:���������:���������:::::3:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:XT
+
_output_shapes
:���������
%
_user_specified_nameobservation
�
1
__inference__raise_169315618��Assert/Assert�
Assert/ConstConst*
_output_shapes
: *
dtype0*H
value?B= B7EpsilonGreedyPolicy does not support distributions yet.2
Assert/Constt
Assert/Assert/conditionConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
Assert/Assert/condition�
Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*H
value?B= B7EpsilonGreedyPolicy does not support distributions yet.2
Assert/Assert/data_0�
Assert/AssertAssert Assert/Assert/condition:output:0Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2
Assert/Assert*
_input_shapes 2
Assert/AssertAssert/Assert
�
)
'__inference_signature_wrapper_169315782�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1693157782
PartitionedCall*
_input_shapes 
�
9
'__inference_get_initial_state_169315754

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
5
 
__inference_<lambda>_169315515*
_input_shapes 
�
N
__inference_<lambda>_169315512
readvariableop_resource
identity	�p
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpY
IdentityIdentityReadVariableOp:value:0*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
�

�
-__inference_function_with_signature_169315728
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *4
f/R-
+__inference_polymorphic_action_fn_1693157152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:���������:���������:���������:���������:::::322
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:���������
'
_user_specified_name0/observation
�
9
'__inference_get_initial_state_169315506

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�

�
'__inference_signature_wrapper_169315748
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin
2	*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *6
f1R/
-__inference_function_with_signature_1693157282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*m
_input_shapes\
Z:���������:���������:���������:���������:::::322
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:ZV
+
_output_shapes
:���������
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
�
?
-__inference_function_with_signature_169315755

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference_get_initial_state_1693157542
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
�
"__inference__traced_save_169315917
file_prefix'
#savev2_variable_read_readvariableop	c
_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableopa
]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopU
Qsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopS
Osavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableop
savev2_const_1

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b73a2be0d430460984b5eb377ed993b8/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop_savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_kernel_read_readvariableop]savev2_categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_bias_read_readvariableopQsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_kernel_read_readvariableopOsavev2_categoricalqnetwork_categoricalqnetwork_dense_1_bias_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *
dtypes

2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*>
_input_shapes-
+: : :	�:�:
��:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :%!

_output_shapes
:	�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:

_output_shapes
: 
�
�
1__inference_polymorphic_distribution_fn_169315619
	step_type

reward
discount
observation��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *%
f R
__inference__raise_1693156182
StatefulPartitionedCall*W
_input_shapesF
D:���������:���������:���������:���������22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:XT
+
_output_shapes
:���������
%
_user_specified_nameobservation
�q
�
+__inference_polymorphic_action_fn_169315715
	time_step
time_step_1
time_step_2
time_step_3`
\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resourcea
]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceR
Ncategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resourceS
Ocategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource	
mul_x

identity_1��
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const�
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3NCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������2I
GCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape�
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp\categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02U
SCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/flatten/Reshape:output:0[CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul�
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp]categoricalqnetwork_categoricalqnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02V
TCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAddBiasAddNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/MatMul:product:0\CategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2G
ECategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd�
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/ReluReluNCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2D
BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu�
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpNcategoricalqnetwork_categoricalqnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02G
ECategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp�
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMulMatMulPCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/Relu:activations:0MCategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������28
6CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul�
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpOcategoricalqnetwork_categoricalqnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02H
FCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp�
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAddBiasAdd@CategoricalQNetwork/CategoricalQNetwork/dense_1/MatMul:product:0NCategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������29
7CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd�
!CategoricalQNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"����   3   2#
!CategoricalQNetwork/Reshape/shape�
CategoricalQNetwork/ReshapeReshape@CategoricalQNetwork/CategoricalQNetwork/dense_1/BiasAdd:output:0*CategoricalQNetwork/Reshape/shape:output:0*
T0*+
_output_shapes
:���������32
CategoricalQNetwork/Reshapey
SoftmaxSoftmax$CategoricalQNetwork/Reshape:output:0*
T0*+
_output_shapes
:���������32	
Softmaxa
mulMulmul_xSoftmax:softmax:0*
T0*+
_output_shapes
:���������32
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������2
Sum/reduction_indicesl
SumSummul:z:0Sum/reduction_indices:output:0*
T0*'
_output_shapes
:���������2
Sum�
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2%
#Categorical_1/mode/ArgMax/dimension�
Categorical_1/mode/ArgMaxArgMaxSum:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2
Categorical_1/mode/ArgMax�
Categorical_1/mode/CastCast"Categorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2
Categorical_1/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x�
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeCategorical_1/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1�
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs�
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToCategorical_1/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_valueG
ShapeShape	time_step*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slicee
shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB 2
shape_as_tensor\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis�
concatConcatV2strided_slice:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concatj
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : 2
random_uniform/minj
random_uniform/maxConst*
_output_shapes
: *
dtype0*
value	B :2
random_uniform/max�
random_uniformRandomUniformIntconcat:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*#
_output_shapes
:���������2
random_uniform�
IdentityIdentityrandom_uniform:output:0
^time_step^time_step_1^time_step_2^time_step_3*
T0*#
_output_shapes
:���������2

Identityx
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_1/Minimum/y�
clip_by_value_1/MinimumMinimumIdentity:output:0"clip_by_value_1/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1/Minimumh
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_1/y�
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_1K
Shape_1Shape	time_step*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
strided_slice_1g
epsilon_rng/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2
epsilon_rng/ming
epsilon_rng/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
epsilon_rng/max�
epsilon_rng/RandomUniformRandomUniformstrided_slice_1:output:0*
T0*#
_output_shapes
:���������*
dtype02
epsilon_rng/RandomUniform�
epsilon_rng/MulMul"epsilon_rng/RandomUniform:output:0epsilon_rng/max:output:0*
T0*#
_output_shapes
:���������2
epsilon_rng/Mul[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *���=2
	Greater/yt
GreaterGreaterepsilon_rng/Mul:z:0Greater/y:output:0*
T0*#
_output_shapes
:���������2	
Greater}
SelectSelectGreater:z:0clip_by_value:z:0clip_by_value_1:z:0*
T0*#
_output_shapes
:���������2
Selectx
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value_2/Minimum/y�
clip_by_value_2/MinimumMinimumSelect:output:0"clip_by_value_2/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2/Minimumh
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value_2/y�
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value_2g

Identity_1Identityclip_by_value_2:z:0*
T0*#
_output_shapes
:���������2

Identity_1"!

identity_1Identity_1:output:0*m
_input_shapes\
Z:���������:���������:���������:���������:::::3:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:VR
+
_output_shapes
:���������
#
_user_specified_name	time_step"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0/discount:0���������
B
0/observation1
action_0/observation:0���������
0
0/reward$
action_0/reward:0���������
6
0/step_type'
action_0/step_type:0���������6
action,
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:�o
�

train_step
metadata
model_variables
_all_assets

signatures

Laction
Mdistribution
Nget_initial_state
Oget_metadata
Pget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
=
0
1
2
	3"
trackable_tuple_wrapper
.

0
1"
trackable_list_wrapper
`

Qaction
Rget_initial_state
Sget_train_step
Tget_metadata"
signature_map
W:U	�2DCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/kernel
Q:O�2BCategoricalQNetwork/CategoricalQNetwork/EncodingNetwork/dense/bias
J:H
��26CategoricalQNetwork/CategoricalQNetwork/dense_1/kernel
C:A�24CategoricalQNetwork/CategoricalQNetwork/dense_1/bias
1
ref
1"
trackable_tuple_wrapper
1
ref
1"
trackable_tuple_wrapper
3
_wrapped_policy"
_generic_user_object
"
_generic_user_object
.

_q_network"
_generic_user_object
�

_q_network
	variables
trainable_variables
regularization_losses
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "CategoricalQNetwork", "name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "QNetwork", "name": "CategoricalQNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
�

layers
metrics
layer_metrics
	variables
trainable_variables
non_trainable_variables
layer_regularization_losses
regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
�
 _postprocessing_layers
!	variables
"trainable_variables
#regularization_losses
$	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�

kernel
	bias
%	variables
&trainable_variables
'regularization_losses
(	keras_api
[__call__
*\&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 153, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 700}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 700]}}
<
0
1
2
	3"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
�

)layers
*metrics
+layer_metrics
	variables
trainable_variables
,non_trainable_variables
-layer_regularization_losses
regularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

0layers
1metrics
2layer_metrics
!	variables
"trainable_variables
3non_trainable_variables
4layer_regularization_losses
#regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

5layers
6metrics
7layer_metrics
%	variables
&trainable_variables
8non_trainable_variables
9layer_regularization_losses
'regularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
]__call__
*^&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
___call__
*`&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 700, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 12]}}
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

Blayers
Cmetrics
Dlayer_metrics
:	variables
;trainable_variables
Enon_trainable_variables
Flayer_regularization_losses
<regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�

Glayers
Hmetrics
Ilayer_metrics
>	variables
?trainable_variables
Jnon_trainable_variables
Klayer_regularization_losses
@regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
+__inference_polymorphic_action_fn_169315606
+__inference_polymorphic_action_fn_169315873�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_polymorphic_distribution_fn_169315619�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_get_initial_state_169315506�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"B 
__inference_<lambda>_169315515
"B 
__inference_<lambda>_169315512
]B[
'__inference_signature_wrapper_169315748
0/discount0/observation0/reward0/step_type
9B7
'__inference_signature_wrapper_169315760
batch_size
+B)
'__inference_signature_wrapper_169315775
+B)
'__inference_signature_wrapper_169315782
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
	J
Const=
__inference_<lambda>_169315512�

� 
� "� 	6
__inference_<lambda>_169315515�

� 
� "� T
'__inference_get_initial_state_169315506)"�
�
�

batch_size 
� "� �
+__inference_polymorphic_action_fn_169315606�	a���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������8
observation)�&
observation���������
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
+__inference_polymorphic_action_fn_169315873�	a���
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������B
observation3�0
time_step/observation���������
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
1__inference_polymorphic_distribution_fn_169315619����
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������8
observation)�&
observation���������
� 
� "
 �
'__inference_signature_wrapper_169315748�	a���
� 
���
.

0/discount �

0/discount���������
<
0/observation+�(
0/observation���������
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"+�(
&
action�
action���������b
'__inference_signature_wrapper_16931576070�-
� 
&�#
!

batch_size�

batch_size "� [
'__inference_signature_wrapper_1693157750�

� 
� "�

int64�
int64 	?
'__inference_signature_wrapper_169315782�

� 
� "� 