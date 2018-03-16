# DenseVec

Map like collection with usize indices that stores values contiguosly in memory so iterating through them is as fast as iterating a vector.

The index uses a sparse vector so using very high numbers will lead to high memory usage

The API mimics that of HashMap and implements all it's features + some methods that are useful in this case like get_unchecked(_mut).