	�C�l��	@�C�l��	@!�C�l��	@	��dy�2@��dy�2@!��dy�2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�C�l��	@�n����?A1�Zd�?Y\���(\�?*	     �@2F
Iterator::Model�G�z�?!�^B{	�T@)��K7��?1����KT@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ʡE��?!
�%���&@)��ʡE��?1
�%���&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!�q�q@)��~j�t�?1Lh/���@:Preprocessing2U
Iterator::Model::ParallelMapV2�� �rh�?!_B{	�%@)�� �rh�?1_B{	�%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;�O��n�?!VUUUUU�?);�O��n�?1VUUUUU�?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!C{	�%��?){�G�zt?1C{	�%��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 18.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t24.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9��dy�2@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�n����?�n����?!�n����?      ��!       "      ��!       *      ��!       2	1�Zd�?1�Zd�?!1�Zd�?:      ��!       B      ��!       J	\���(\�?\���(\�?!\���(\�?R      ��!       Z	\���(\�?\���(\�?!\���(\�?JCPU_ONLYY��dy�2@b 