	`��"��@`��"��@!`��"��@	�ۢ�@�ۢ�@!�ۢ�@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$`��"��@��/�$�?A9��v���?Y?5^�I�?*	      k@2F
Iterator::Model!�rh���?!�%�p	J@)��MbX�?1�>C���F@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�V-�?!p	�\@@)�V-�?1p	�\@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥛?!�Ł�(@)�I+��?1"���F$@:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1��?!�9�s�@)y�&1��?1�9�s�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�x?!�zX=�@)�~j�t�x?1�zX=�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!�����n@){�G�zt?1�����n@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�ۢ�@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��/�$�?��/�$�?!��/�$�?      ��!       "      ��!       *      ��!       2	9��v���?9��v���?!9��v���?:      ��!       B      ��!       J	?5^�I�??5^�I�?!?5^�I�?R      ��!       Z	?5^�I�??5^�I�?!?5^�I�?JCPU_ONLYY�ۢ�@b 