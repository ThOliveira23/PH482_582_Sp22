	���S�6@���S�6@!���S�6@	@�(� 3�?@�(� 3�?!@�(� 3�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���S�6@�x�&1�?A=
ףp�6@Y���S㥫?*	      b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatej�t��?!�{a��M@){�G�z�?1a���K@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!��=��4@)V-��?1      4@:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1��?!,�4�rO#@)y�&1��?1,�4�rO#@:Preprocessing2F
Iterator::Model���S㥛?!Y�i��2@)9��v���?1��FX�!@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�x?!���=�@)�~j�t�x?1���=�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!j��FXN@)����Mb`?1|a���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����MbP?!|a���?)����MbP?1|a���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9B�(� 3�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�x�&1�?�x�&1�?!�x�&1�?      ��!       "      ��!       *      ��!       2	=
ףp�6@=
ףp�6@!=
ףp�6@:      ��!       B      ��!       J	���S㥫?���S㥫?!���S㥫?R      ��!       Z	���S㥫?���S㥫?!���S㥫?JCPU_ONLYYB�(� 3�?b 