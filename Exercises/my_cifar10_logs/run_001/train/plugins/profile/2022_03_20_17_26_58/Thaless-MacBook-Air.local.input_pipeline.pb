	F����8+@F����8+@!F����8+@	/��#'�?/��#'�?!/��#'�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$F����8+@bX9���?A��Q�^*@YX9��v��?*	     �o@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?5^�I�?!��y���K@)?5^�I�?1��y���K@:Preprocessing2F
Iterator::Model{�G�z�?!������?@)D�l����?1�a�a;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!(��(��$@)�������?1v]�u]�#@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!]�u]�u@)�I+��?1]�u]�u@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{�G�zt?!�������?){�G�zt?1�������?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����MbP?!Y�eY�e�?)����MbP?1Y�eY�e�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9/��#'�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	bX9���?bX9���?!bX9���?      ��!       "      ��!       *      ��!       2	��Q�^*@��Q�^*@!��Q�^*@:      ��!       B      ��!       J	X9��v��?X9��v��?!X9��v��?R      ��!       Z	X9��v��?X9��v��?!X9��v��?JCPU_ONLYY/��#'�?b 