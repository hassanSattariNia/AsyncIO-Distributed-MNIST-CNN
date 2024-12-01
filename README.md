AsyncIO-Distributed-MNIST-CNN
AsyncIO-Distributed-MNIST-CNN is a CNN partitioned and executed in a distributed environment. Each partition runs on a separate client. Forward and backward passes are manually managed using Async IO, with clients accessing only their own computation graph parts.
